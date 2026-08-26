# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
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


class KVStore:
    LABEL = "kv"

    def __init__(
        self,
        model_name: str,
        block_dim: Optional[int] = None,
        kv_caches: Optional[Dict[str, torch.Tensor]] = None,
        layer_names: Optional[List[str]] = None,
        rank: int = 0,
        tp_size: int = 1,
    ):

        if kv_caches is None and layer_names is None:
            raise ValueError(
                "At least one of kv_caches or layer_names must be provided"
            )

        if kv_caches is not None and block_dim is None:
            raise ValueError("block_dim is required when kv_caches is provided")

        if kv_caches is not None and layer_names is not None:
            kv_keys = set(kv_caches.keys())
            layer_set = set(layer_names)
            if kv_keys != layer_set:
                raise ValueError(
                    f"kv_caches keys {kv_keys} must match layer_names {layer_set}"
                )

        self.kv_caches = kv_caches
        self.block_dim = block_dim
        self.rank = rank
        self.tp_size = tp_size

        if kv_caches is not None:
            self.layer_names = list(kv_caches.keys())
        else:
            self.layer_names = layer_names

        self.skip_compression_count = min(
            envs.IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS, len(self.layer_names)
        )

        self.has_only_mode = kv_caches is None

        if kv_caches:
            first_tensor = next(iter(kv_caches.values()))
            self.kvcache_shape = list(first_tensor.shape)

            self.block_shape = list(first_tensor.shape)
            self.block_shape[block_dim] = 1
            self.block_shape = tuple(
                self.block_shape[i]
                for i in range(len(self.block_shape))
                if i != block_dim
            )
        else:
            self.kvcache_shape = None
            self.block_shape = None

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
            "num_layers=%d, block_dim=%s, block_shape=%s, "
            "pool_size_gb=%.2f, persist_dir=%s",
            model_name,
            self.rank,
            self.has_only_mode,
            len(self.layer_names),
            self.block_dim,
            self.block_shape,
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

    def put(
        self,
        block_indices: List[int],
        block_hashs: List[str],
        layer_names: Optional[List[str]] = None,
        description: str = "",
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Task]:
        """Write blocks to the store.

        The last two arguments exist for callers whose tensors are not
        this store's own ``kv_caches``:

        - ``tensors``: write THESE instead of the bound caches. Needed
          when a layer is not a single tensor (a recurrent layer is a
          conv state plus an ssm state over one storage) and must be
          presented as one uniform page view, because the engine
          requires every tensor in a call to share shape and dtype.
        - ``label``: the store-side namespace. Callers that keep several
          independent block spaces (one per KV cache group, per rank)
          pass their own; the default keeps every existing key byte for
          byte.

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

        if tensors is None:
            tensors = {name: self.kv_caches[name] for name in layer_names}

        result = self.tensorzip.put(
            label=label or self.LABEL,
            tensors=tensors,
            chunk_dim=self.block_dim,
            chunk_indices=block_indices,
            chunk_labels=block_hashs,
            description=description,
            skip_compression_count=self.skip_compression_count,
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
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        label: Optional[str] = None,
    ) -> Dict[str, Task]:
        """Read blocks back into GPU memory; see ``put`` for the last
        two arguments. Results are keyed by layer name so a caller can
        wait one layer at a time and overlap the rest with compute."""
        if self.has_only_mode:
            raise RuntimeError(
                "get() not available in has-only mode (kv_caches not provided)"
            )

        if layer_names is None:
            layer_names = self.layer_names

        if tensors is None:
            tensors = {name: self.kv_caches[name] for name in layer_names}

        return self.tensorzip.get(
            label=label or self.LABEL,
            tensors=tensors,
            chunk_dim=self.block_dim,
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
        status["kvcache_shape"] = self.kvcache_shape
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
