# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os

import torch
import numpy as np
from typing import Dict, List, Optional
import psutil
from ..envs import envs
from ..kvflow import KVFlow, Task, get_accelerator_device
from ..utils.profiler import (
    profile_scope,
    profile_cross_scope,
    profile_func,
    start_profiling,
    stop_profiling,
)
from .api import register as mgmt_register, start_mgmt_server, stop_mgmt_server

logger = logging.getLogger(__name__)


def _get_default_cache_size_gb() -> float:
    available_bytes = psutil.virtual_memory().available
    return available_bytes / (10 * 1024**3)


def _bind_pools(
    kv_caches: Optional[Dict[str, torch.Tensor | list]],
) -> tuple[Optional[Dict[str, torch.Tensor]], set[str]]:
    """Normalize bound caches to one dim-0 block pool per layer.

    - A bare Tensor passes through unchanged (attention family: vLLM
      already lays them out block-leading and page-contiguous).
    - A list/tuple of views sharing one storage (mamba family: conv +
      ssm over the same raw buffer, both as_strided with dim 0 = block
      index and stride(0) covering a whole padded page) collapses into
      ONE byte plane: an int8 (num_blocks, page_bytes) view over the
      shared storage. The first view's row stride is authoritative for
      the page width -- downstream parts must agree, or binding fails
      closed.

    Returns (normalized_pools, opaque_layer_names). Opaque layers came
    from multi-dtype fusion: their bytes carry no single numeric
    meaning, so per-call policies must never apply lossy transforms.
    """
    if kv_caches is None:
        return None, set()

    pools: Dict[str, torch.Tensor] = {}
    opaque: set[str] = set()
    for layer_name, entry in kv_caches.items():
        if not isinstance(entry, (list, tuple)):
            pools[layer_name] = entry
            continue

        if len(entry) == 0:
            raise ValueError(f"Layer {layer_name}: empty state list")
        first = entry[0]
        page_bytes = first.stride(0) * first.element_size()
        num_blocks = first.shape[0]
        device = first.device

        total_needed = page_bytes * num_blocks
        storage_size = first.untyped_storage().size()
        if total_needed > storage_size:
            raise ValueError(
                f"Layer {layer_name}: unified pages need {total_needed} "
                f"bytes but the shared storage has {storage_size}")

        base = torch.empty(
            0, dtype=torch.uint8, device=device
        ).set_(first.untyped_storage())
        pool = torch.as_strided(
            base,
            size=(num_blocks, page_bytes),
            stride=(page_bytes, 1),
            storage_offset=0,
        )
        # Fail closed if sibling parts disagree on the page geometry:
        # each view's dim-0 stride, in bytes, must equal the first's.
        for part in entry[1:]:
            part_row_bytes = part.stride(0) * part.element_size()
            if part.stride(0) and part_row_bytes != page_bytes:
                raise ValueError(
                    f"Layer {layer_name}: part strides disagree "
                    f"({part_row_bytes} != {page_bytes} bytes); refusing "
                    "to bind a misaligned multi-dtype pool")
        pools[layer_name] = pool
        opaque.add(layer_name)
        logger.info(
            "Bound layer %s as %d opaque int8 pages of %d bytes",
            layer_name, num_blocks, page_bytes)
    return pools, opaque


class KVStore:
    LABEL = "kv"

    def __init__(
        self,
        model_name: str,
        kv_caches: Optional[Dict[str, torch.Tensor]] = None,
        layer_names: Optional[List[str]] = None,
        rank: int = 0,
        tp_size: int = 1,
    ):
        """Bind row-addressable KV pools.

        ``kv_caches`` maps a LAYER name to either a single GPU pool or
        a LIST of pools sharing one storage (a mamba-style recurrent
        layer: one conv view + one ssm view, both as_strided with
        dim 0 = block index and a common padded page stride).

        Binding normalizes everything to ONE dim-0 block pool per
        layer: list entries collapse into a single uint8/int8 page
        view spanning the shared storage. Thereafter every bound pool
        is homogeneous within its group call; entry flags mark which
        layers must never see numeric-loss transforms.
        """

        if kv_caches is None and layer_names is None:
            raise ValueError(
                "At least one of kv_caches or layer_names must be provided"
            )

        if kv_caches is not None and layer_names is not None:
            kv_keys = set(kv_caches.keys())
            layer_set = set(layer_names)
            if kv_keys != layer_set:
                raise ValueError(
                    f"kv_caches keys {kv_keys} must match layer_names {layer_set}"
                )

        self.kv_caches, self._opaque_layers = _bind_pools(kv_caches)
        self.rank = rank
        self.tp_size = tp_size

        if kv_caches is not None:
            self.layer_names = list(self.kv_caches.keys())
        else:
            self.layer_names = layer_names

        self.skip_compression_count = min(
            envs.IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS, len(self.layer_names)
        )
        self._skip_order = {ln: i for i, ln in
                            enumerate(self.layer_names)}
        # Operator-level lossy request; opaque layers veto it per entry
        # inside _entry_flags.
        self.lossy_trunc = (
            os.getenv("IAXL_KV_LOSSY_TRUNC", "0").strip() not in ("", "0")
        )

        self.has_only_mode = kv_caches is None

        if self.has_only_mode:
            final_persist_dir = f"{model_name}_rank0"
        else:
            final_persist_dir = f"{model_name}_rank{rank}"

        if self.has_only_mode:
            pool_size_gb = 0.0
        else:
            env_size = envs.IAXL_DDR_POOL_SIZE_GB
            if env_size is not None:
                pool_size_gb = env_size
            else:
                pool_size_gb = _get_default_cache_size_gb()

        self.tensorzip = KVFlow(
            persist_dir=final_persist_dir,
            cache_size_gb=pool_size_gb,
            rank=rank,
        )
        start_profiling()

        logger.info(
            "KVStore initialized successfully: "
            "model_name=%s, rank=%d, has_only_mode=%s, "
            "num_layers=%d, "
            "pool_size_gb=%.2f, persist_dir=%s",
            model_name,
            self.rank,
            self.has_only_mode,
            len(self.layer_names),
            pool_size_gb,
            final_persist_dir,
        )

        role = "controller" if self.has_only_mode else "worker"
        if role == "worker":
            mgmt_register("GET", "/v1/cache/status", lambda params, s=self: s.status())
            mgmt_register(
                "GET",
                "/v1/health",
                lambda params, s=self: {"status": "ok", "rank": s.rank},
            )
            mgmt_register(
                "POST",
                "/v1/cache/persist",
                lambda body, s=self: {
                    "rank": s.rank,
                    "result": s.persist(int(body.get("count", 10))),
                },
            )
            mgmt_register(
                "POST",
                "/v1/cache/evict",
                lambda body, s=self: {
                    "rank": s.rank,
                    "result": s.evict(int(body.get("count", 10))),
                },
            )
            mgmt_register(
                "GET", "/v1/cache/metrics", lambda params, s=self: s.metrics(params)
            )
        self._mgmt_server = start_mgmt_server(
            role=role, rank=self.rank, num_workers=self.tp_size
        )

    def _entry_flags(self, layer_names: List[str]) -> List[int]:
        """Per-entry codec bits aligned to the tensors dict order.

        bit0 = compress; bit1 = lossy-trunc. Compression follows the
        operator's skip-prefix; lossy is structurally refused for
        opaque (multi-dtype fused) layers no matter what else asks.
        """
        flags: List[int] = []
        for ln in layer_names:
            compress = int(self._skip_order.get(ln, len(layer_names))
                           >= self.skip_compression_count)
            lossy = 0 if ln in self._opaque_layers else self.lossy_trunc
            flags.append(compress | (lossy << 1))
        return flags

    def put(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
    ) -> Dict[str, Task]:
        """Write blocks to the store, one entry per bound pool.

        ``layer_names`` selects among the pools bound at construction
        (bare layer names; multi-part layers are already unified).
        ``label`` is the store-side namespace: callers that keep several
        independent block spaces (one per KV cache group) pass their
        own; the default keeps every existing key byte for byte.

        A call carrying an explicit ``label`` is treated as complete on
        its own -- the caller passes that namespace's whole layer set in
        one call -- so the block is finalized here rather than waiting
        for a "last layer" that this namespace defines differently.
        """
        if self.has_only_mode:
            raise RuntimeError(
                "put() not available in has-only mode (kv_caches not provided)"
            )

        if layer_names is None:
            layer_names = self.layer_names

        tensors = {name: self.kv_caches[name] for name in layer_names}

        # Per-entry codec flags: bit0 compress, bit1 lossy-trunc.
        # Opaque (multi-dtype fused) layers hard-refuse lossy; the
        # operator's skip-compression prefix still applies by
        # registration order.
        result = self.tensorzip.put(
            label=label or self.LABEL,
            tensors=tensors,
            chunk_indices=block_indices,
            chunk_labels=block_hashs,
            description=description,
            entry_flags=self._entry_flags(layer_names),
        )

        if label is not None or self.layer_names[-1] in layer_names:
            self.tensorzip.put_finish(label or self.LABEL, block_hashs)
            self.tensorzip.record_flush()

        return result

    def put_wait(
        self,
        put_results: Dict[str, Task],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:

        if self.has_only_mode:
            raise RuntimeError(
                "put_wait() not available in has-only mode (kv_caches not provided)"
            )

        return self.tensorzip.put_wait(
            put_results,
            tensor_dict_keys=layer_names,
            wait=wait,
        )

    def get(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        label: Optional[str] = None,
    ) -> Dict[str, Task]:
        """Read blocks back into the bound pools; see ``put`` for the
        naming and namespace contract. Results are keyed by layer name
        so a caller can wait one layer at a time and overlap the rest
        with compute."""
        if self.has_only_mode:
            raise RuntimeError(
                "get() not available in has-only mode (kv_caches not provided)"
            )

        if layer_names is None:
            layer_names = self.layer_names

        tensors = {name: self.kv_caches[name] for name in layer_names}
        block_shape = tuple(next(iter(tensors.values())).shape[1:])

        return self.tensorzip.get(
            label=label or self.LABEL,
            tensors=tensors,
            block_shape=block_shape,
            chunk_indices=block_indices,
            chunk_labels=block_hashs,
            description=description,
        )

    def get_wait(
        self,
        get_results: Dict[str, Task],
        layer_names: Optional[List[str]] = None,
        wait: bool = True,
    ) -> bool:

        if self.has_only_mode:
            raise RuntimeError(
                "get_wait() not available in has-only mode (kv_caches not provided)"
            )

        return self.tensorzip.get_wait(
            get_results,
            tensor_dict_keys=layer_names,
            wait=wait,
        )

    def has(self, block_hashs: Optional[List[str]] = None,
            label: Optional[str] = None) -> List[bool]:
        """Presence, truncated at the first miss.

        The truncation is prefix semantics: a cached prefix is only
        usable up to its first hole, so nothing past one is worth
        reporting. ``label`` selects the namespace, as in ``put``.
        """
        if not block_hashs:
            self.tensorzip.record_flush()
            return []

        results = self.tensorzip.has(
            label=label or self.LABEL,
            chunk_labels=block_hashs,
        )

        mask = np.array(results, dtype=np.bool_)
        idx = np.argmin(mask)
        if not mask[idx]:
            mask[idx + 1 :] = False
            results = mask.tolist()

        return results

    def stop(self):
        stop_mgmt_server(self)
        self.tensorzip.stop()

    def status(self) -> dict:
        status = self.tensorzip.status()
        status["rank"] = self.rank
        status["num_layers"] = len(self.layer_names)
        return status

    def metrics(self, params: Optional[dict] = None) -> dict:
        from ..torch_ext import (
            metrics_set_enabled,
            metrics_reset,
            metrics_read,
        )

        params = params or {}
        if "enable" in params:
            metrics_set_enabled(
                str(params["enable"]).lower() in ("1", "true", "yes", "on")
            )
        if str(params.get("reset", "")).lower() in ("1", "true", "yes", "on"):
            metrics_reset()

        result = metrics_read()
        result["rank"] = self.rank
        return result

    def persist(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {"error": "persist() not available in has-only mode (scheduler)"}
        return self.tensorzip.persist(max_count)

    def evict(self, max_count: int) -> dict:
        if self.has_only_mode:
            return {"error": "evict() not available in has-only mode (scheduler)"}
        return self.tensorzip.evict(max_count)

    def get_persist_candidates(self, max_count: int) -> List[str]:
        return self.tensorzip.get_persist_candidates(max_count)

    def get_evict_candidates(self, max_count: int) -> List[str]:
        return self.tensorzip.get_evict_candidates(max_count)
