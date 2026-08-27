# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""KVShrink external KV cache connector (hybrid GDN/Mamba aware)."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterator, Optional

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.parallel_state import (
    get_world_group,
    model_parallel_is_initialized,
)
import vllm.envs as envs
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

if TYPE_CHECKING:
    from .async_load_config import AsyncLoadLayerConfig
    from vllm.config import ModelConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

from iaxl import KVStore, generate_block_hashs, setup_root_logger
from iaxl.kvflow.flow import Task

from .async_load_config import load_async_load_layer_config_from_env
setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str

SCHEMA_VERSION = 4

@dataclass
class ReqMeta:
    """All transfer instructions for one request in one step."""
    group_ops: tuple[GroupTransferMeta, ...] = ()
    external_hit_tokens: int = 0
    is_async: bool = False
    async_load_layers: int = -1

@dataclass
class ReqGroupState:
    """Per-group mutable state for one request (scheduler side)."""
    block_ids: list[int] = field(default_factory=list)
    next_stored_chunk_idx: int = 0

@dataclass
class ReqState:
    live_source: Any = None
    block_hashes: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    snapshot_boundary: int = 0
    groups: tuple[ReqGroupState, ...] = ()
    pending_load_tokens: int = 0
    last_known_progress: int = 0
    is_async: bool = False
    async_load_layers: int = -1
    async_plan_emitted: bool = False

@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker transfer plan."""
    reqs_to_load: RequestMetadata = field(default_factory=RequestMetadata)
    reqs_to_save: RequestMetadata = field(default_factory=RequestMetadata)

def group_label(namespace: str, group_idx: int, rank: int) -> str:
    """Store namespace for one group's block space on one rank."""
    return f"{namespace}_g{int(group_idx)}_r{int(rank)}"

def lookup_boundary(store: "KVStore", key: "CacheKey") -> bool:
    """Is this boundary present in the store?"""
    try:
        present = store.has([key.hash_str],
                            label=group_label(key.namespace,
                                              key.group_idx, key.rank))
        return bool(present and present[0])
    except Exception:
        logger.exception("lookup error; treating as MISS")
        return False

class Canonicalizer:
    """Builds canonical (num_blocks, page_size_bytes) int8 views per layer."""

    def __init__(self, layer_infos: dict[str, LayerPageInfo], num_blocks: int):
        """Record per-layer descriptors and the GLOBAL block-pool size
        (shared across groups); views are built by ``register``."""
        self._layer_infos: dict[str, LayerPageInfo] = layer_infos
        self._num_blocks: int = num_blocks
        self._views: dict[str, torch.Tensor] = {}

    def register(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        """Build the canonical (num_blocks, page_size_bytes) int8 views
        over the vLLM kv_caches, handling Mamba list-of-tensors storage
        and split-K/V attention layouts. Raises ValueError on any
        descriptor/storage mismatch (fail closed). Called once per layer
        set."""
        for layer_name, info in self._layer_infos.items():
            raw = kv_caches[layer_name]
            if isinstance(raw, (list, tuple)):
                first = raw[0]
                tensor = torch.empty(
                    0, dtype=torch.int8, device=first.device
                ).set_(first.untyped_storage())
            else:
                tensor = torch.empty(
                    0, dtype=torch.int8, device=raw.device
                ).set_(raw.untyped_storage())

            if (not isinstance(raw, (list, tuple))
                    and self._is_split_kv_layout(raw, info)):
                N = info.num_blocks
                half = info.page_size_bytes // 2
                storage = raw.untyped_storage()
                base = torch.empty(
                    0, dtype=torch.int8, device=raw.device).set_(storage)
                flat = base.view(2, N, half)
                k_view, v_view = flat.unbind(0)
                self._views[layer_name] = (k_view, v_view)
                logger.info(
                    "Canonicalized split-K/V layer %s: N=%d half_page=%d",
                    layer_name, N, half)
                continue

            stride = info.page_size_bytes
            needed = stride * (info.num_blocks - 1) + info.page_size_bytes
            if needed > storage_size_bytes(raw):
                raise ValueError(
                    f"Layer {layer_name}: descriptor requires {needed} bytes "
                    f"but storage has {storage_size_bytes(raw)}")
            view = torch.as_strided(
                tensor,
                size=(info.num_blocks, info.page_size_bytes),
                stride=(stride, 1),
                storage_offset=0,
            )
            self._views[layer_name] = view
        logger.info(
            "Canonicalized %d layers, %d blocks, page=%d bytes",
            len(self._views), self._num_blocks,
            next(iter(self._layer_infos.values())).page_size_bytes)

    @staticmethod
    def _is_split_kv_layout(
        raw: torch.Tensor, info: LayerPageInfo
    ) -> bool:
        """True when num_blocks lives in a non-leading physical dim
        (K/V-split layout). Mirrors the physical-to-logical stride
        mapping in vLLM's offloading worker."""
        if raw.dim() < 2 or raw.shape[0] != 2:
            return False
        strides = raw.stride()
        physical_to_logical = sorted(
            range(len(strides)), key=lambda i: strides[i], reverse=True)
        try:
            logical_nb_dim = list(raw.shape).index(info.num_blocks)
        except ValueError:
            return False
        physical_pos = physical_to_logical.index(logical_nb_dim)
        return physical_pos != 0

    def page_view_parts(self, layer_name: str) -> dict[str, torch.Tensor]:
        """Full-pool canonical page views for the KVFlow chunk engine."""
        v = self._views[layer_name]
        if isinstance(v, tuple):
            return {"k": v[0], "v": v[1]}
        return {"page": v}

def storage_size_bytes(t: torch.Tensor | list[torch.Tensor]) -> int:
    """Bytes of the underlying untyped storage for a kv_cache entry;
    Mamba entries (list/tuple of tensors sharing one storage) report
    the first tensor's storage. Used for descriptor bounds checks."""
    if isinstance(t, (list, tuple)):
        return t[0].untyped_storage().size()
    return t.untyped_storage().size()

@dataclass(frozen=True)
class LayerPageInfo:
    """Canonical page info for one layer (as seen by the connector)."""
    num_blocks: int
    page_size_bytes: int

@dataclass(frozen=True)
class GroupInfo:
    """One vLLM KV cache group: a frozen snapshot of its storage spec."""

    group_idx: int
    kind: str
    layer_names: tuple[str, ...]
    block_size: int
    mamba_align_size: Optional[int]
    spec: object = None

@dataclass(frozen=True)
class CacheKey:
    """Logical key for one page, or for a whole boundary."""
    namespace: str
    rank: int
    block_hash: object
    group_idx: int
    layer_name: str

    @property
    def hash_str(self) -> str:
        """Stable string form for paths / JSON (bytes -> hex)."""
        h = self.block_hash
        if isinstance(h, bytes):
            return h.hex()
        return str(h)

    @property
    def boundary_key(self) -> tuple[str, int, str, int]:
        """Isolation-safe identity for a boundary (namespace/rank/hash/group)."""
        return (self.namespace, self.rank, self.hash_str, self.group_idx)

def make_boundary_key(namespace: str, rank: int,
                      group_idx: int, block_hash: object) -> "CacheKey":
    """A group's key at one block hash, with no layer: the address the
    hit policy asks about and the address the save/load builders expand
    into per-layer page keys."""
    return CacheKey(
        namespace=namespace, rank=rank,
        block_hash=block_hash, group_idx=group_idx, layer_name="")

@dataclass
class GroupTransferMeta:
    """Per-group transfer instructions for one request (one step)."""
    group_idx: int
    keys: tuple[CacheKey, ...] = ()
    gpu_block_ids: tuple[int, ...] = ()

class KVShrinkParseError(ValueError):
    """Raised when the vLLM cache config cannot be parsed safely
    (unknown spec or inconsistent layout). The parse never
    guesses (fail closed)."""
    pass

def validate_codec_env() -> None:
    """Refuse to start when the codec is configured to lose bits."""
    lossy = os.getenv("IAXL_KV_LOSSY_TRUNC", "0").strip()
    if lossy in ("", "0"):
        return
    raise KVShrinkParseError(
        f"IAXL_KV_LOSSY_TRUNC={lossy!r} is not supported for models with "
        "GDN/Mamba layers: state pages are stored as opaque bytes, so the "
        "truncation would corrupt the recurrent state itself and produce "
        "wrong output with no error. Set IAXL_KV_LOSSY_TRUNC=0.")

def compute_namespace(
    model_id: str,
    model_revision: str,
    tokenizer_revision: str,
    cache_dtype: str,
    kv_schema_version: int,
    tp_size: int,
) -> str:
    """Stable cache namespace: sha256 over model identity, cache dtype,
    schema version and tp size, truncated to 16 hex chars. The schema
    version is an input so that changing the page layout renames the
    namespace, rather than reading old pages under the new layout."""
    raw = "|".join([
        model_id, model_revision, tokenizer_revision, cache_dtype,
        str(kv_schema_version), str(tp_size),
    ])
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _spec_kind(spec: object) -> str:
    """Mamba or attention; unknown spec types raise KVShrinkParseError
    (fail closed). Sliding-window specs are AttentionSpec subclasses and
    are intentionally NOT distinguished: their block layout is the
    attention one, and the hit-rule difference lives in vLLM's spec
    registry, which is consulted directly."""
    if isinstance(spec, MambaSpec):
        return "mamba"
    if isinstance(spec, AttentionSpec):
        return "attention"
    raise KVShrinkParseError(
        f"Unsupported KV cache spec {type(spec).__name__}")

def _iter_layer_specs(group_spec: Any) -> Iterator[tuple[str, object]]:
    """Yield (layer_name, spec) pairs, expanding UniformTypeKVCacheSpecs."""
    spec = group_spec.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        per_layer = spec.kv_cache_specs
        for name in group_spec.layer_names:
            yield name, per_layer[name]
    else:
        for name in group_spec.layer_names:
            yield name, spec

def parse_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> tuple[list[GroupInfo], dict[str, LayerPageInfo], int]:
    """Return (groups, layer_infos, num_blocks)."""
    num_blocks = kv_cache_config.num_blocks
    groups: list[GroupInfo] = []
    layer_infos: dict[str, LayerPageInfo] = {}

    for g_idx, g in enumerate(kv_cache_config.kv_cache_groups):
        kind = None
        block_size = None
        per_layer_specs: list[tuple[str, object]] = list(_iter_layer_specs(g))

        for name, spec in per_layer_specs:
            sk = _spec_kind(spec)
            if kind is None:
                kind = sk
            elif sk != kind:
                raise KVShrinkParseError(
                    f"Group {g_idx} mixes spec kinds {kind} and {sk}")
            bs = int(spec.block_size)
            if block_size is None:
                block_size = bs
            elif bs != block_size:
                raise KVShrinkParseError(
                    f"Group {g_idx} layers have differing block sizes")

        mamba_align = None
        if kind == "mamba":
            mamba_mode = per_layer_specs[0][1].mamba_cache_mode
            if mamba_mode != "align":
                raise KVShrinkParseError(
                    f"Group {g_idx} has mamba_cache_mode={mamba_mode!r}, "
                    "but the external cache requires 'align'. Start vLLM "
                    "with --enable-prefix-caching --mamba-cache-mode align "
                    "(vLLM forces the mode to 'none' when prefix caching "
                    "is disabled, and disables prefix caching by default "
                    "for hybrid models)")
            mamba_align = block_size
        group = GroupInfo(
            group_idx=g_idx,
            kind=kind,
            layer_names=tuple(g.layer_names),
            block_size=block_size,
            mamba_align_size=mamba_align,
            spec=per_layer_specs[0][1],
        )

        groups.append(group)
        for name, spec in per_layer_specs:
            layer_infos[name] = LayerPageInfo(
                num_blocks=num_blocks,
                page_size_bytes=int(spec.page_size_bytes),
            )

    sizes = {g.block_size for g in groups}
    if len(sizes) != 1:
        raise KVShrinkParseError(
            f"KV cache groups have different block sizes {sorted(sizes)}; "
            "the external cache addresses every group with one block hash "
            "and cannot express that")
    return groups, layer_infos, num_blocks

def save_enabled() -> bool:
    """Production save is ON by default; KVSHRINK_SAVE=0 disables it and
    KVSHRINK_DEBUG_AUTOSAVE=1 force-enables it."""
    return (os.getenv("KVSHRINK_SAVE", "1") != "0"
            or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")

class _StoreAsBlockPool:
    """The one thing vLLM's matching code needs that we must supply."""

    __slots__ = ("_present",)

    null_block = object()

    def __init__(self, present: Callable[[int, object], bool]):
        self._present: Callable[[int, object], bool] = present

    def get_cached_block(
        self, block_hash: object, kv_cache_group_ids: list[int],
    ) -> Optional[list[object]]:
        blocks = []
        for group_id in kv_cache_group_ids:
            if not self._present(group_id, block_hash):
                return None
            blocks.append(_StoreAsBlockPool.null_block)
        return blocks

class HybridHitPolicy:
    """Fixed-point multi-group hit detection (pure function, testable)."""

    def __init__(
        self,
        groups: list[GroupInfo],
        present: Callable[[int, object], bool],
        hash_block_size: int,
        num_computed_tokens: int,
    ):
        """Configure the policy for one request: the groups, a
        store-presence predicate ``present(group_idx, block_hash)`` and
        the request's computed tokens. Orders groups attention-first
        (tighter initial bound) and takes the global mamba alignment as
        the minimum across mamba groups."""
        self._groups: list[GroupInfo] = groups
        self._present: Callable[[int, object], bool] = present
        self._hash_block_size: int = hash_block_size
        self._num_computed: int = num_computed_tokens
        self._ordered: list[GroupInfo] = sorted(
            groups, key=lambda g: 0 if g.kind == "attention" else 1)
        self._mamba_align: Optional[int] = None
        for g in groups:
            if g.kind == "mamba":
                a = g.mamba_align_size
                self._mamba_align = a if self._mamba_align is None \
                    else min(self._mamba_align, a)

    def _lookup(self, group: GroupInfo, block_hashes: list[int],
                candidate: int) -> int:
        """How far this group alone is restorable, in tokens."""
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
        manager_cls = KVCacheSpecRegistry.get_manager_class(group.spec)
        max_length = min(candidate,
                         len(block_hashes) * group.block_size)
        blocks = manager_cls.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_length,
            kv_cache_group_ids=[group.group_idx],
            block_pool=_StoreAsBlockPool(self._present),
            kv_cache_spec=group.spec,
            drop_eagle_block=False,
            alignment_tokens=(self._mamba_align or group.block_size),
        )
        return len(blocks[0]) * group.block_size

    def find_longest_cache_hit(
        self, block_hashes: list[int], max_length: int
    ) -> int:
        """Fixed-point convergence over all groups. Returns the
        restorable boundary in tokens; 0 = miss.
        """
        candidate = max_length
        if self._mamba_align is not None:
            a = self._mamba_align
            candidate = min(candidate - 1, (candidate - 1) // a * a)

        while True:
            changed = False
            for group in self._ordered:
                hit = self._lookup(group, block_hashes, candidate)
                if hit < candidate:
                    candidate = hit
                    changed = True
                if candidate <= self._num_computed:
                    return 0
            if not changed:
                break
        return candidate if candidate > self._num_computed else 0

class _AsyncLoad:
    """One request's cross-step load."""

    __slots__ = ("layer_tasks", "gate_layers", "released")

    def __init__(
        self, layer_tasks: dict[str, Task],
        gate_layers: set[str],
    ):
        self.layer_tasks: dict[str, Task] = layer_tasks
        self.gate_layers: set[str] = gate_layers
        self.released: bool = False

@dataclass
class _SaveCandidate:
    """One boundary's pages accumulated across this step's save plans
    (cross-request dedup by boundary key). ``req_ids`` tracks every
    contributing request: the put reads their GPU blocks, so their
    block freeing is deferred until the write lands."""
    group_idx: int
    pages: dict[str, tuple[CacheKey, int]] = field(default_factory=dict)
    req_ids: set[str] = field(default_factory=set)

class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
    """KVShrink external KV cache connector."""

    @classmethod
    def requires_piecewise_for_cudagraph(
        cls, extra_config: dict[str, Any]
    ) -> bool:
        return True

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self.vllm_config: VllmConfig = vllm_config
        self.model_config: ModelConfig = vllm_config.model_config
        self.tp_size: int = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers: int = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.vllm_device: str = vllm_config.device_config.device_type
        self.rank: int = get_world_group().rank if model_parallel_is_initialized() else 0

        self._async_load_layer_config: AsyncLoadLayerConfig = (
            load_async_load_layer_config_from_env(
                num_layers=self.num_layers,
            )
        )

        self.kvstore: Optional[KVStore] = None

        self._req_states: dict[str, ReqState] = {}
        self._async_load_pending: set[str] = set()

        self._async_loads: dict[str, "_AsyncLoad"] = {}
        self._current_put_tasks: dict[str, list[dict[str, Task]]] = {}
        self._deferred_finished_req_ids: set[str] = set()

        if kv_cache_config is not None:
            self._init_kv_stack(vllm_config, role, kv_cache_config)
        else:
            self._bind_cpu_affinity()
            self._bind_intel_accel()

    def _init_kv_stack(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Build the hybrid stack for this role."""
        pc = vllm_config.parallel_config
        tp_size = pc.tensor_parallel_size
        if pc.pipeline_parallel_size != 1:
            raise RuntimeError(
                "kvshrink hybrid: pipeline parallelism is not supported "
                f"(pipeline_parallel_size={pc.pipeline_parallel_size}); "
                "each rank would persist only its own layers' pages. "
                "Set pipeline_parallel_size=1 or the KV connector.")
        if role == KVConnectorRole.WORKER:
            rank = pc.rank
        else:
            rank = 0
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        block_sizes = sorted({int(g.kv_cache_spec.block_size)
                              for g in kv_cache_config.kv_cache_groups})
        hash_block_size = math.gcd(*block_sizes)
        self._hash_block_size = hash_block_size
        self._namespace = compute_namespace(
            model_id=model_config.model,
            model_revision=model_config.revision or "",
            tokenizer_revision=str(
                model_config.tokenizer_revision or ""),
            cache_dtype=cache_config.cache_dtype,
            kv_schema_version=SCHEMA_VERSION,
            tp_size=tp_size,
        )
        validate_codec_env()

        groups, layer_infos, num_blocks = parse_kv_cache_config(
            kv_cache_config)

        for g, parsed in zip(kv_cache_config.kv_cache_groups, groups):
            if parsed.kind != "mamba":
                continue
            spec_blocks = g.kv_cache_spec.num_speculative_blocks
            if spec_blocks:
                raise RuntimeError(
                    "kvshrink hybrid: speculative decoding is not "
                    f"supported (group has num_speculative_blocks="
                    f"{spec_blocks}); the external GDN snapshot only "
                    "restores the non-speculative state slot. Disable "
                    "speculative decoding or the KV connector.")
        self._groups = groups
        self._layer_infos = layer_infos
        self._rank = rank
        self._attention_layers = tuple(
            ln for g in groups if g.kind != "mamba"
            for ln in g.layer_names)
        recurrent = any(g.kind == "mamba" for g in groups)
        self._block_hash_source = self._choose_block_hash_source(recurrent)

        if role == KVConnectorRole.SCHEDULER:
            self.kvstore = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(i) for i in range(self.num_layers)],
                tp_size=self.tp_size,
            )
        else:
            self._canon = Canonicalizer(layer_infos, num_blocks)
            self._labels = [
                group_label(self._namespace, g.group_idx, rank)
                for g in groups]
            self._layer_group = {
                ln: g.group_idx for g in groups for ln in g.layer_names}
            self._attn_layer_group = {
                ln: g.group_idx for g in groups if g.kind != "mamba"
                for ln in g.layer_names}

        logger.info(
            "kvshrink hybrid path enabled (%s role, %d groups, %d layers, "
            "hash_block_size=%d, namespace=%s, tp=%d rank=%d)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            len(groups), len(layer_infos), hash_block_size,
            self._namespace, tp_size, rank)
        logger.info(
            "kvshrink groups: %s",
            [(g.group_idx, g.kind, g.block_size) for g in groups])

    @staticmethod
    def _choose_block_hash_source(recurrent: bool) -> str:
        """Which block-identity scheme to key the cache with."""
        choice = (os.getenv("KVSHRINK_BLOCK_HASH_SOURCE") or "auto").lower()
        if choice == "auto":
            return "vllm" if recurrent else "legacy"
        if choice not in ("vllm", "legacy"):
            raise ValueError(
                "KVSHRINK_BLOCK_HASH_SOURCE must be auto, vllm or "
                f"legacy; got {choice!r}")
        return choice

    def _bind_cpu_affinity(self) -> None:
        if self.vllm_device == "cpu":
            return

        omp_bind = envs.VLLM_CPU_OMP_THREADS_BIND
        if not omp_bind or omp_bind in ("all", "auto"):
            raise ValueError(
                "VLLM_CPU_OMP_THREADS_BIND must assign CPUs to each worker"
            )

        worker_cpu_specs = omp_bind.split("|")
        if len(worker_cpu_specs) < self.tp_size:
            raise ValueError(
                f"VLLM_CPU_OMP_THREADS_BIND has {len(worker_cpu_specs)} entries, "
                f"but tensor parallel size is {self.tp_size}"
            )

        cpu_ids: set[int] = set()
        for part in worker_cpu_specs[self.rank].split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = map(int, part.split("-", maxsplit=1))
                if start > end:
                    raise ValueError(f"Invalid CPU range: {part}")
                cpu_ids.update(range(start, end + 1))
            else:
                cpu_ids.add(int(part))

        if not cpu_ids:
            raise ValueError(f"No CPUs configured for rank {self.rank}")
        os.sched_setaffinity(0, cpu_ids)
        logger.info("Bound rank %d to CPUs %s", self.rank, sorted(cpu_ids))

    def _bind_intel_accel(self) -> None:
        for source, target in (
            ("KVSHRINK_QAT_DEVICES", "IAXL_QAT_DEVICES"),
            ("KVSHRINK_DSA_DEVICES", "IAXL_DSA_WQS"),
        ):
            spec = os.getenv(source)
            if not spec:
                continue
            devices = spec.split("|")
            if len(devices) <= self.rank:
                raise ValueError(
                    f"{source} has {len(devices)} entries, but rank is {self.rank}"
                )
            os.environ[target] = devices[self.rank]
            logger.info("Bound rank %d: %s=%s", self.rank, target, devices[self.rank])

    def _track_new_request(
        self, req_id: str, block_hashes: list[int],
        num_computed_tokens: int,
        request: Optional["Request"] = None,
    ) -> None:
        """Register a fresh ReqState. Internal: called by us from
        get_num_new_matched_tokens / build_load_meta /
        update_state_after_alloc when a request first becomes visible
        (vLLM has no dedicated "new request" connector hook)."""
        live_source = None
        if request is not None:
            live_source = (request.block_hashes
                           if self._block_hash_source == "vllm"
                           else request.all_token_ids)
        self._req_states[req_id] = ReqState(
            live_source=live_source,
            block_hashes=list(block_hashes),
            num_computed_tokens=num_computed_tokens,
            groups=tuple(
                ReqGroupState() for _ in self._groups),
        )

    def take_async_load_plans(
        self, already_emitted: set[str]
    ) -> dict[str, ReqMeta]:
        """Load plans for requests vLLM parked, drained exactly once."""
        plans = {}
        for req_id in sorted(self._async_load_pending - already_emitted):
            state = self._req_states[req_id]
            meta = self._build_load_meta_from_state(
                req_id, state, scheduled_tokens=0)
            state.async_plan_emitted = True
            state.is_async = False
            if meta is not None and meta.group_ops:
                plans[req_id] = meta
            else:
                logger.warning(
                    "async req=%s has no restorable pages; dropping the "
                    "plan so it is recomputed rather than left waiting",
                    req_id)
        self._async_load_pending -= (self._async_load_pending
                                     - already_emitted)
        return plans

    def _request_block_hashes(self, request: "Request") -> list[Any]:
        """This request's block identities, in block order."""
        if self._block_hash_source == "vllm":
            return list(request.block_hashes)
        tokens = request.all_token_ids
        return [str(h) for h in generate_block_hashs(
            tokens[:-1], self._hash_block_size)]

    def on_request_finished(self, req_id: str) -> None:
        """vLLM trigger: core frees the request ->
        connector.request_finished -> here. Drop the ReqState;
        committed boundaries are content-addressed and stay."""
        self._req_states.pop(req_id, None)
        self._async_load_pending.discard(req_id)

    def on_cached_request(
        self, req_id: str, new_block_ids: tuple[list[int], ...],
        resumed: bool, num_computed_tokens: Optional[int],
    ) -> None:
        """vLLM trigger: every scheduling pass, for each running
        (cached) request, via connector.build_connector_meta.
        """
        state = self._req_states[req_id]
        if state.live_source is not None:
            if self._block_hash_source == "vllm":
                live = state.live_source
            else:
                live = [str(h) for h in generate_block_hashs(
                    state.live_source[:-1], self._hash_block_size)]
            if len(live) > len(state.block_hashes):
                state.block_hashes.extend(live[len(state.block_hashes):])
        old_progress = max(state.num_computed_tokens,
                           state.last_known_progress)
        regression = (num_computed_tokens is not None
                      and num_computed_tokens < old_progress)
        if num_computed_tokens is not None:
            state.num_computed_tokens = num_computed_tokens
            state.last_known_progress = num_computed_tokens
        if resumed or regression:
            safe_n = num_computed_tokens or 0
            for g_idx, group in enumerate(self._groups):
                gstate = state.groups[g_idx]
                safe = safe_n // group.block_size
                if gstate.next_stored_chunk_idx > safe:
                    if os.getenv("KVSHRINK_DEBUG_LOG"):
                        logger.info(
                            "cursor rollback req=%s g%d: %d -> %d "
                            "(progress %d -> %s, resumed=%s)",
                            req_id, g_idx, gstate.next_stored_chunk_idx,
                            safe, old_progress, num_computed_tokens,
                            resumed)
                    gstate.next_stored_chunk_idx = safe
        if new_block_ids:
            for gstate, ids in zip(state.groups, new_block_ids):
                if resumed:
                    gstate.block_ids = list(ids) if ids else []
                elif ids:
                    gstate.block_ids.extend(ids)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """External lookup; returns (hit_tokens, has_async_load)."""
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        block_hashes = self._request_block_hashes(request)
        self._track_new_request(
            request.request_id, block_hashes,
            num_computed_tokens, request=request)
        policy = HybridHitPolicy(
            self._groups, self._present, self._hash_block_size,
            num_computed_tokens)
        boundary = policy.find_longest_cache_hit(
            block_hashes, request.num_tokens)
        state = self._req_states[request.request_id]
        if boundary and state.block_hashes:
            state.snapshot_boundary = boundary
        else:
            boundary = 0
        external = max(0, boundary - num_computed_tokens)
        use_async = self._decide_async(request.request_id, external)
        logger.debug(
            "req=%s external_hit=%d boundary=%d async=%s",
            request.request_id, external, boundary, use_async)
        return external, use_async

    def _decide_async(self, req_id: str, external: int) -> bool:
        """Should this request's pages stream in while the GPU runs
        OTHER requests, instead of stalling a forward step on us?
        """
        if external <= 0 or self._async_load_layer_config is None:
            return False
        state = self._req_states[req_id]
        selected = self._async_load_layer_config.select(
            len(self._req_states))
        if selected == 0:
            return False
        state.is_async = True
        if selected < 0 or selected > len(self._attention_layers):
            state.async_load_layers = -1
        else:
            state.async_load_layers = selected
        return True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record the allocated block tables per group (after alloc)."""
        req_id = request.request_id
        state = self._req_states.get(req_id)
        if state is None:
            self._track_new_request(
                req_id, self._request_block_hashes(request), 0,
                request=request)
            state = self._req_states[req_id]
        state.num_computed_tokens = (
            state.num_computed_tokens + num_external_tokens)
        state.pending_load_tokens = num_external_tokens
        all_block_ids = blocks.get_block_ids()
        for g_idx, ids in enumerate(all_block_ids):
            state.groups[g_idx].block_ids = list(ids)
        if (state.is_async and not state.async_plan_emitted
                and num_external_tokens > 0):
            self._async_load_pending.add(req_id)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "update_state req=%s per-group block_ids: %s hashes=%d",
                req_id, [[b for b in g.block_ids] for g in state.groups],
                len(state.block_hashes))

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        self.on_request_finished(request.request_id)
        return True, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """SupportsHMA entry point (v0.23.0 calls this instead of
        ``request_finished`` whenever the hybrid memory allocator is on,
        which is the default for every model).
        """
        return self.request_finished(request, [])

    def build_load_meta(
        self, new_req: "NewRequestData", scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Build the LOAD ReqMeta for a NewRequestData entry."""
        req_id = new_req.req_id
        state = self._req_states[req_id]
        return self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)

    def build_resumed_load_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Build the LOAD ReqMeta for a PREEMPTION-RESUMED request."""
        state = self._req_states[req_id]
        meta = self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)
        if state.pending_load_tokens > 0:
            npages = sum(len(op.keys) for op in meta.group_ops)
            if npages == 0:
                raise RuntimeError(
                    "kvshrink resumed request has accepted external "
                    "tokens but no restorable pages (req="
                    f"{req_id} pending={state.pending_load_tokens} "
                    f"boundary={state.snapshot_boundary} "
                    f"sched={scheduled_tokens}): refusing to enter "
                    "forward with unrestored state")
        return meta

    def _build_load_meta_from_state(
        self, req_id: str, state: ReqState, scheduled_tokens: int,
    ) -> ReqMeta:
        """The snapshot_boundary recorded by get_num_new_matched_tokens
        is the AUTHORITATIVE restore boundary for this alloc/load. NEVER recompute here: after
        update_state_after_alloc the locally-computed counter already
        includes external tokens, and a fresh lookup would be polluted.
        Missing/expired boundary must
        FAIL CLOSED (boundary 0), not guess."""
        boundary = state.snapshot_boundary
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "TAIL req=%s snapshot_boundary=%d computed_before_fwd=%d "
                "external=%d",
                req_id, state.snapshot_boundary,
                state.num_computed_tokens,
                state.snapshot_boundary - state.num_computed_tokens)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            ids = state.groups[g_idx].block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            if group.kind == "attention":
                gran = group.block_size
                num_hash = boundary // gran
                for i in range(num_hash):
                    if i >= len(state.block_hashes):
                        break
                    blk_hash = state.block_hashes[i]
                    key = self._boundary_key(group, blk_hash)
                    if not lookup_boundary(self.kvstore, key):
                        break
                    if i < len(ids):
                        for layer_name in group.layer_names:
                            keys.append(self._page_key(key, layer_name))
                            gpu_ids.append(ids[i])
            elif group.kind == "mamba":
                if state.block_hashes and boundary > 0:
                    idx = boundary // group.block_size - 1
                    if 0 <= idx < len(state.block_hashes):
                        blk_hash = state.block_hashes[idx]
                        key = self._boundary_key(group, blk_hash)
                        if lookup_boundary(self.kvstore, key):
                            bs = group.block_size
                            curr_idx = (boundary + scheduled_tokens -
                                        1) // bs

                            if scheduled_tokens <= 0 and not state.is_async:
                                raise RuntimeError(
                                    "kvshrink mamba external HIT with "
                                    "scheduled_tokens=0 "
                                    f"(req={req_id} boundary={boundary}): "
                                    "production hits must schedule >= 1 "
                                    "token; refusing to build load meta")
                            if not (0 <= curr_idx < len(ids)
                                    and ids[curr_idx] != 0):
                                raise RuntimeError(
                                    "kvshrink mamba load curr slot "
                                    f"invalid (req={req_id} "
                                    f"boundary={boundary} "
                                    f"sched={scheduled_tokens} "
                                    f"table_idx={curr_idx} "
                                    f"table={ids}): refusing to enter "
                                    "forward with unrestored state")
                            gpu_block = ids[curr_idx]
                            for layer_name in group.layer_names:
                                keys.append(self._page_key(
                                    key, layer_name))
                                gpu_ids.append(gpu_block)
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids)))
        return ReqMeta(
            external_hit_tokens=boundary - state.num_computed_tokens,
            group_ops=tuple(group_ops),
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def build_save_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Production save: INCREMENTAL per-group page persistence."""
        state = self._req_states[req_id]
        progress = state.num_computed_tokens + scheduled_tokens
        state.last_known_progress = max(state.last_known_progress,
                                        progress)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            gstate = state.groups[g_idx]
            ids = gstate.block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            if group.kind == "attention":
                num_hash = min(progress // group.block_size, len(ids),
                               len(state.block_hashes))
                start = gstate.next_stored_chunk_idx
                for i in range(start, num_hash):
                    blk_hash = state.block_hashes[i]
                    for layer_name in group.layer_names:
                        keys.append(self._page_key(
                            self._boundary_key(group, blk_hash),
                            layer_name))
                        gpu_ids.append(ids[i])
                if num_hash > start:
                    gstate.next_stored_chunk_idx = num_hash
            elif group.kind == "mamba":
                if state.block_hashes:
                    block_pos = None
                    for pos in range(len(ids) - 1, -1, -1):
                        if ids[pos] != 0:
                            block_pos = pos
                            break
                    if block_pos is None:
                        if os.getenv("KVSHRINK_DEBUG_LOG"):
                            logger.info(
                                "save mamba g%d: no non-null block in "
                                "ids=%s", g_idx, ids)
                    elif (progress > 0
                          and progress % group.block_size == 0):
                        idx = progress // group.block_size - 1
                        if (idx >= gstate.next_stored_chunk_idx
                                and idx < len(state.block_hashes)):
                            blk_hash = state.block_hashes[idx]
                            for layer_name in group.layer_names:
                                keys.append(self._page_key(
                                    self._boundary_key(group, blk_hash),
                                    layer_name))
                                gpu_ids.append(ids[block_pos])
                            gstate.next_stored_chunk_idx = idx + 1
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids)))
        return ReqMeta(group_ops=tuple(group_ops))

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Assemble this pass's hybrid plans."""
        meta = KVShrinkConnectorMetadata()
        debug = bool(os.getenv("KVSHRINK_DEBUG_LOG"))
        save_on = save_enabled()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            req_meta = self.build_load_meta(
                new_req, num_sched.get(new_req.req_id, 0))
            if debug:
                logger.info(
                    "LOADMETA req=%s ops=%d computed_before_fwd=%d "
                    "num_scheduled_tokens=%s",
                    new_req.req_id, len(req_meta.group_ops),
                    new_req.num_computed_tokens,
                    num_sched.get(new_req.req_id))
                for op in req_meta.group_ops:
                    logger.info(
                        "LOADMETA  g%d kind=%s keys=%d gpu_ids=%d",
                        op.group_idx, self._groups[op.group_idx].kind,
                        len(op.keys), len(op.gpu_block_ids))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.reqs_to_load.requests[new_req.req_id] = req_meta
            if save_on:
                save_meta = self.build_save_meta(
                    new_req.req_id, num_sched.get(new_req.req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[new_req.req_id] = save_meta

        for req_id, req_meta in self.take_async_load_plans(
                set(meta.reqs_to_load.requests)).items():
            if debug:
                logger.info(
                    "LOADMETA(async) req=%s ops=%d layers=%s",
                    req_id, len(req_meta.group_ops),
                    req_meta.async_load_layers)
            meta.reqs_to_load.requests[req_id] = req_meta

        cr = scheduler_output.scheduled_cached_reqs
        for req_id in cr.resumed_req_ids:
            req_meta = self.build_resumed_load_meta(
                req_id, num_sched.get(req_id, 0))
            if debug:
                logger.info(
                    "LOADMETA(resumed) req=%s ops=%d",
                    req_id, len(req_meta.group_ops))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.reqs_to_load.requests[req_id] = req_meta

        if save_on:
            resumed = cr.resumed_req_ids
            new_bids = cr.new_block_ids
            ncts = cr.num_computed_tokens
            for i, req_id in enumerate(cr.req_ids):
                self.on_cached_request(
                    req_id, new_bids[i], req_id in resumed, ncts[i])
                save_meta = self.build_save_meta(
                    req_id, num_sched.get(req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[req_id] = save_meta

        if debug:
            logger.info(
                "build_connector_meta: %d load reqs, %d save reqs",
                len(meta.reqs_to_load.requests),
                len(meta.reqs_to_save.requests))
        return meta

    def _boundary_key(self, group: GroupInfo, block_hash: object) -> CacheKey:
        """This rank's boundary key for one group at one block hash."""
        return make_boundary_key(self._namespace, self._rank,
                                 group.group_idx, block_hash)

    def _present(self, group_idx: int, block_hash: object) -> bool:
        """Store-presence predicate handed to the hit policy, which
        plans against boundary addresses without seeing store details."""
        return lookup_boundary(
            self.kvstore,
            make_boundary_key(self._namespace, self._rank,
                              group_idx, block_hash))

    @staticmethod
    def _page_key(boundary_key: CacheKey, layer_name: str) -> CacheKey:
        """Expand a boundary key to ONE layer's page key: same
        namespace/rank/hash/group as the boundary, plus the layer
        name. This is the exact page address the worker must move."""
        return replace(boundary_key, layer_name=layer_name)

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        static_context = self.vllm_config.compilation_config.static_forward_context
        for layer in static_context.values():
            get_backend = getattr(layer, "get_attn_backend", None)
            if get_backend is not None:
                if "FLASHINFER" in get_backend().get_name().upper():
                    raise RuntimeError("FlashInfer is not supported")
                break

        from vllm.model_executor.models.utils import extract_layer_index

        self.register(kv_caches,
                      sorted(kv_caches, key=extract_layer_index))

        views = self._flat_views(
            {ln: self._canon.page_view_parts(ln)
             for ln in self._layer_names})
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            block_dim=0,
            kv_caches=views,
            rank=self._rank,
            tp_size=self.tp_size,
        )
        logger.info("Registered %d KV cache layers (%d page views)",
                    len(kv_caches), len(views))

    def register(
        self,
        kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
        execution_order: list[str],
    ) -> None:
        """Bind canonical page views and record which layers recur."""
        self._layer_names = list(execution_order)
        self._canon.register(kv_caches)
        self._mamba_layers = frozenset(
            ln for g in self._groups if g.kind == "mamba"
            for ln in g.layer_names)
        self._attn_order = tuple(
            ln for ln in execution_order if ln in self._attn_layer_group)
        segments: dict[str, tuple[str, ...]] = {}
        pending: list[str] = []
        for ln in execution_order:
            if ln in self._mamba_layers:
                pending.append(ln)
            elif ln in self._attn_layer_group and pending:
                segments[ln] = tuple(pending)
                pending = []
        self._mamba_save_segments = segments
        logger.info(
            "kvshrink hybrid worker registered: %d layers, %d attention "
            "hook points, %d recurrent layers (namespace tp=%d rank=%d)",
            len(self._layer_infos), len(self._attn_order),
            len(self._mamba_layers), self.tp_size, self._rank)

    def _metadata(self) -> KVShrinkConnectorMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError(
                "kvshrink hybrid worker received "
                f"{type(metadata).__name__}; expected "
                "KVShrinkConnectorMetadata")
        return metadata

    def _worker_key(self, key: CacheKey) -> CacheKey:
        """Remap a scheduler-built key (rank 0) to this worker's own
        rank: each TP rank persists and loads its OWN shard under its
        own rank path. Without this, TP>1 workers overwrite each
        other's pages under the shared rank-0 key."""
        if key.rank == self._rank:
            return key
        return replace(key, rank=self._rank)

    @staticmethod
    def _flat_views(
        layer_views: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {f"{ln}#{part}": view
                for ln, parts in layer_views.items()
                for part, view in parts.items()}

    def _wait_load(self, tasks: dict[str, Task]) -> None:
        """Host-block until these reads land; an incomplete transfer is
        fatal -- forward is about to read these blocks."""
        if tasks and not self.kvstore.get_wait(get_results=tasks,
                                               wait=True):
            raise RuntimeError(
                "kvshrink load failed: get_wait reported an incomplete "
                "transfer; forward would read unrestored blocks")

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        self.start_load(self._metadata())

    def start_load(self, metadata: KVShrinkConnectorMetadata) -> int:
        """Submit ALL of this step's loads, then host-block on the GDN
        ones. Attention layers stay pipelined: vLLM calls a hook on
        entry to each of them, so their pages are waited for exactly
        when they are about to be read.
        """
        self._load_tasks = {}
        self._saved_layers = set()
        self._step_save_pages = 0
        npages = 0
        _t0 = time.monotonic()
        by_group: dict[int, tuple[list[tuple[int, str]], set[str]]] = {}
        for req_id, req_meta in metadata.reqs_to_load.requests.items():
            if req_meta.is_async:
                tasks: dict[str, Task] = {}
                for op in req_meta.group_ops:
                    if not op.keys:
                        continue
                    t, n = self._submit_group_load(op)
                    tasks.update(t)
                    npages += n
                self._register_async_load(req_id, req_meta, tasks)
            else:
                for op in req_meta.group_ops:
                    if not op.keys:
                        continue
                    entries, layers = by_group.setdefault(
                        op.group_idx, ([], set()))
                    entries.extend(self._op_entries(op))
                    layers.update(key.layer_name for key in op.keys)
        for g_idx, (entries, layers) in by_group.items():
            tasks, n = self._submit_group_load_entries(
                g_idx, entries, layers)
            self._load_tasks.update(tasks)
            npages += n
        recurrent = {k: self._load_tasks.pop(k)
                     for k in list(self._load_tasks)
                     if k.rsplit("#", 1)[0] in self._mamba_layers}
        if recurrent:
            self._wait_load(recurrent)
        if npages:
            logger.info(
                "start_load_kv: %d pages loaded "
                "elapsed_ms=%.3f (rank %d/%d)", npages,
                (time.monotonic() - _t0) * 1e3, self._rank, self.tp_size)
        return npages

    def _register_async_load(
        self, req_id: str, req_meta: ReqMeta,
        tasks: dict[str, Task],
    ) -> None:
        """Track one request's cross-step load and compute its release
        gate.
        """
        if not tasks:
            return
        layers = {k.rsplit("#", 1)[0] for k in tasks}
        recurrent = layers - self._attn_layer_group.keys()
        n = req_meta.async_load_layers
        if n is None or n < 0:
            gate = layers
        else:
            prefix = [ln for ln in self._attn_order if ln in layers][:n]
            gate = recurrent | set(prefix)
        self._async_loads[req_id] = _AsyncLoad(tasks, gate)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "async load req=%s layers=%d gate=%d (recurrent=%d) "
                "requested_prefix=%s", req_id, len(layers), len(gate),
                len(recurrent), n)

    def poll_finished_loads(self) -> set[str]:
        """Report async requests whose gate layers have landed."""
        finished: set[str] = set()
        for req_id, entry in list(self._async_loads.items()):
            if entry.released:
                continue
            gate = {k: t for k, t in entry.layer_tasks.items()
                    if k.rsplit("#", 1)[0] in entry.gate_layers}
            if gate and not self.kvstore.get_wait(get_results=gate,
                                                  wait=False):
                continue
            self._wait_load(gate)
            for k in gate:
                del entry.layer_tasks[k]
            entry.released = True
            finished.add(req_id)
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)
        return finished

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.wait_layer_load(layer_name)

    def wait_layer_load(self, layer_name: str) -> None:
        """Attention-layer entry hook: wait this layer's pages."""
        keys = [k for k in self._load_tasks
                if k.rsplit("#", 1)[0] == layer_name]
        if keys:
            self._wait_load({k: self._load_tasks.pop(k) for k in keys})
        for req_id, entry in list(self._async_loads.items()):
            if not entry.released:
                continue
            keys = [k for k in entry.layer_tasks
                    if k.rsplit("#", 1)[0] == layer_name]
            if keys:
                self._wait_load(
                    {k: entry.layer_tasks.pop(k) for k in keys})
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)

    @staticmethod
    def _op_entries(op: GroupTransferMeta) -> list[tuple[int, str]]:
        """One (gpu_block_id, chunk_label) per block: keys expand per
        block x layer (scheduler invariant), so collapse by label."""
        seen: dict[str, int] = {}
        for key, gpu in zip(op.keys, op.gpu_block_ids):
            seen.setdefault(key.hash_str, gpu)
        return [(gpu, h) for h, gpu in seen.items()]

    def _submit_group_load(
        self, op: GroupTransferMeta
    ) -> tuple[dict[str, Task], int]:
        """Submit one GroupTransferMeta as one engine get; returns the
        flat tasks dict and the page count."""
        return self._submit_group_load_entries(
            op.group_idx, self._op_entries(op),
            {key.layer_name for key in op.keys})

    def _submit_group_load_entries(
        self, g_idx: int, entries: list[tuple[int, str]],
        layer_names: set[str],
    ) -> tuple[dict[str, Task], int]:
        """One engine get covering ``layer_names`` for the blocks in
        ``entries``; returns the flat tasks dict ("layer#part" ->
        Task) and the page count."""
        tensors = self._flat_views(
            {ln: self._canon.page_view_parts(ln) for ln in layer_names})
        tasks = self.kvstore.get(block_indices=[gpu for gpu, _ in entries],
                                 block_hashs=[h for _, h in entries],
                                 layer_names=list(tensors),
                                 tensors=tensors,
                                 label=self._labels[g_idx])
        return tasks, len(entries) * len(tensors)

    def _gather_save_candidates(
        self, metadata: KVShrinkConnectorMetadata
    ) -> dict[tuple[str, int, int, str, int], _SaveCandidate]:
        """Batch-level boundary candidates with cross-request dedup.
        Returns boundary_key -> accumulated per-layer pages."""
        candidates: dict[tuple[str, int, int, str, int], _SaveCandidate] = {}
        for req_id, req_meta in metadata.reqs_to_save.requests.items():
            for op in req_meta.group_ops:
                for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
                    key = self._worker_key(key)
                    cand = candidates.get(key.boundary_key)
                    if cand is None:
                        cand = _SaveCandidate(op.group_idx)
                        candidates[key.boundary_key] = cand
                    cand.pages[key.layer_name] = (key, gpu_block_id)
                    cand.req_ids.add(req_id)
        return candidates

    def _submit_group_layers_save(
        self, g_idx: int, layer_names: list[str],
        entries: list[tuple[int, str]],
    ) -> dict[str, Task]:
        """Submit ONE async engine put covering ``layer_names`` for the
        blocks in ``entries`` (list of (gpu_block_id, chunk_label), same
        order for every layer -- scheduler invariant). Async D2H+zip on
        the engine's put_stream, self-gated on the compute stream so it
        reads final values. Returns the engine tasks dict."""
        tensors = self._flat_views(
            {ln: self._canon.page_view_parts(ln) for ln in layer_names})
        return self.kvstore.put(block_indices=[gpu for gpu, _ in entries],
                                block_hashs=[h for _, h in entries],
                                layer_names=list(tensors),
                                tensors=tensors,
                                label=self._labels[g_idx])

    def _track_put(self, tasks: dict[str, Task], req_ids: set[str]) -> None:
        """Attribute one submitted put to every request whose GPU blocks
        it reads; get_finished defers those requests' block freeing
        until the write lands (same contract as main)."""
        for rid in req_ids:
            self._current_put_tasks.setdefault(rid, []).append(tasks)

    def _submit_layers_save(
        self, g_idx: int, layer_names: list[str],
        metadata: KVShrinkConnectorMetadata,
    ) -> None:
        """Submit ONE async engine put covering ``layer_names`` of one
        group for this step's complete boundary candidates. No waiting:
        the tasks join ``_current_put_tasks`` and drain in
        get_finished. Partial-boundary candidates are skipped here
        exactly as submit_saves skips them."""
        expected = sorted(self._groups[g_idx].layer_names)
        entries: list[tuple[int, str]] = []
        req_ids: set[str] = set()
        for cand in self._gather_save_candidates(metadata).values():
            if cand.group_idx != g_idx:
                continue
            if sorted(cand.pages) != expected:
                continue
            key, gpu_block_id = cand.pages[layer_names[0]]
            entries.append((gpu_block_id, key.hash_str))
            req_ids |= cand.req_ids
        if not entries:
            return
        tasks = self._submit_group_layers_save(g_idx, layer_names,
                                               entries)
        self._track_put(tasks, req_ids)
        self._saved_layers.update(layer_names)
        self._step_save_pages += len(entries) * len(layer_names)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Pipelined save. vLLM calls this on exit of EVERY attention
        layer during forward (kv_transfer_utils decorator).
        """
        if self._connector_metadata is None:
            return
        if os.getenv("KVSHRINK_SAVE_PIPELINED", "1") == "0":
            return
        if not save_enabled():
            return
        metadata = self._metadata()
        segment = self._mamba_save_segments.get(layer_name)
        if segment:
            by_group: dict[int, list[str]] = {}
            for ln in segment:
                by_group.setdefault(self._layer_group[ln], []).append(ln)
            for seg_g_idx, seg_layers in by_group.items():
                self._submit_layers_save(seg_g_idx, seg_layers, metadata)
        self._submit_layers_save(
            self._attn_layer_group[layer_name], [layer_name], metadata)

    def wait_for_save(self) -> None:
        if self.kvstore is None:
            return

        if not save_enabled():
            return
        metadata = self._metadata()
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("wait_for_save worker: reqs_to_save=%d",
                        len(metadata.reqs_to_save.requests))
        pages, boundaries = self.submit_saves(metadata)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("chunk_save: %d pages, %d boundaries",
                        pages, boundaries)
        self.debug_dump_state()

    def submit_saves(
        self, metadata: KVShrinkConnectorMetadata
    ) -> tuple[int, int]:
        """Post-forward save SUBMISSION: every layer not already
        pipelined through save_kv_layer (trailing mamba segment,
        attention layers whose hook never fired) goes out here, one
        engine put per group. Nothing is waited: the writes drain in
        get_finished, which is also where finished requests' blocks are
        released (the deferred-freeing contract, same as main).
        Returns (pages, boundaries) submitted this step."""
        candidates = self._gather_save_candidates(metadata)
        per_group: dict[int, dict[str, Any]] = {}
        nbound = 0
        for bkey, cand in candidates.items():
            _, _, blk_hash, g_idx = bkey
            expected = sorted(self._groups[g_idx].layer_names)
            if sorted(cand.pages) != expected:
                logger.warning(
                    "chunk_save skip commit g%d h=%s: expected %d "
                    "layers, stored %d (%s)", g_idx, blk_hash,
                    len(expected), len(cand.pages),
                    set(expected) ^ set(cand.pages))
                continue
            nbound += 1
            blob = per_group.setdefault(g_idx, {"entries": [],
                                                "req_ids": set()})
            key, gpu_block_id = cand.pages[expected[0]]
            blob["entries"].append((gpu_block_id, key.hash_str))
            blob["req_ids"] |= cand.req_ids
        for g_idx, blob in per_group.items():
            remaining = [ln for ln in self._groups[g_idx].layer_names
                         if ln not in self._saved_layers]
            if not remaining:
                continue
            tasks = self._submit_group_layers_save(
                g_idx, remaining, blob["entries"])
            self._track_put(tasks, blob["req_ids"])
            self._saved_layers.update(remaining)
            self._step_save_pages += len(blob["entries"]) * len(remaining)
        if self._step_save_pages:
            logger.info(
                "chunk_save: %d pages submitted, %d boundaries "
                "(rank %d/%d)", self._step_save_pages, nbound,
                self._rank, self.tp_size)
        return self._step_save_pages, nbound

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Report transfers that completed since the last step."""
        if self.kvstore is None:
            return None, None
        finished_recving = self.poll_finished_loads()

        self._deferred_finished_req_ids.update(finished_req_ids)
        completed: set[str] = set()
        for req_id in self._deferred_finished_req_ids:
            entry = self._async_loads.get(req_id)
            if entry is not None:
                tasks = entry.layer_tasks
                if tasks and not self.kvstore.get_wait(
                        get_results=tasks, wait=False):
                    continue
                self._wait_load(tasks)
                self._async_loads.pop(req_id, None)

            tasks = self._current_put_tasks.get(req_id)
            if tasks is None:
                completed.add(req_id)
                continue
            while tasks:
                if all(t.ctx is None for t in tasks[0].values()):
                    tasks.pop(0)
                    continue
                if not self.kvstore.put_wait(tasks[0], wait=False):
                    break
                tasks.pop(0)
            if not tasks:
                del self._current_put_tasks[req_id]
                completed.add(req_id)

        self._deferred_finished_req_ids.difference_update(completed)
        return (completed or None), (finished_recving or None)

    def debug_dump_state(self) -> None:
        """KVSHRINK_DEBUG_DUMP=1: log sha256 of the first layer page of
        every mamba group at gpu blocks 0..9, so cold-vs-hot GPU states
        can be compared byte-exactly."""
        if not os.getenv("KVSHRINK_DEBUG_DUMP"):
            return
        for group in self._groups:
            if group.kind != "mamba":
                continue
            ln = group.layer_names[0]
            page_view = self._canon.page_view_parts(ln)["page"]
            for blk in range(10):
                page = page_view[blk]
                h = hashlib.sha256(
                    page.cpu().numpy().tobytes()).hexdigest()
                logger.info("DUMP g%d block=%d sha=%s",
                            group.group_idx, blk, h[:16])
