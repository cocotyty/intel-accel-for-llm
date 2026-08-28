# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


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
    # block identity list and GPU block table, per group.
    live_source: Any = None
    block_hashes: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    snapshot_boundary: int = 0
    groups: tuple[ReqGroupState, ...] = ()
    # External tokens accepted this pass (drives the load plan).
    pending_load_tokens: int = 0
    # Last authoritative progress seen by the save path
    # (num_computed + scheduled of the last save plan). Used for
    # fail-closed regression detection: any drop below this value rolls
    # save cursors back even if the resumed flag is missing.
    last_known_progress: int = 0
    # Async load bookkeeping: while is_async, the request is
    # parked and its plan ships via take_async_load_plans.
    is_async: bool = False
    async_load_layers: int = -1
    # Whether the async load plan was already handed out.
    async_plan_emitted: bool = False


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

    def add_request(
        self,
        req_id: ReqId,
        group_ops: tuple[GroupTransferMeta, ...] = (),
        is_async: bool = False,
        async_load_layers: int = -1,
        external_hit_tokens: int = 0,
    ) -> None:
        self.requests[req_id] = ReqMeta(
            group_ops=group_ops,
            external_hit_tokens=external_hit_tokens,
            is_async=is_async,
            async_load_layers=async_load_layers,
        )


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker transfer plan."""
    reqs_to_load: RequestMetadata = field(default_factory=RequestMetadata)
    reqs_to_save: RequestMetadata = field(default_factory=RequestMetadata)


# ======================================================================
# hybrid layout vocabulary: groups, keys, boundaries
# ======================================================================


def group_label(group_idx: int) -> str:
    """Store label for one group's block space."""
    return f"g{int(group_idx)}"


def lookup_boundary(store: "KVStore", key: "CacheKey") -> bool:
    """Is this boundary present in the store? Presence is per
    # (label, hash) chunk; a boundary is present when the store has
    # it under the group's label."""
    try:
        present = store.has([key.hash_str],
                            label=group_label(key.group_idx))
        return bool(present and present[0])
    except Exception:  # pragma: no cover - fail closed to MISS
        logger.exception("lookup error; treating as MISS")
        return False


# ======================================================================
# hybrid layout vocabulary
# ======================================================================
@dataclass(frozen=True)
class GroupInfo:
    """One vLLM KV cache group: a frozen snapshot of its storage
    # contract (kind, layers, block size, mamba alignment)."""

    group_idx: int
    kind: str  # "attention" | "mamba"
    layer_names: tuple[str, ...]
    block_size: int  # tokens per block for this group
    mamba_align_size: Optional[int]  # offload chunk alignment for mamba
    # vLLM's own spec for this group, kept so the hit policy can hand it
    # back to vLLM's matching code instead of reimplementing it.
    spec: object = None


@dataclass(frozen=True)
class CacheKey:
    """Logical key for one page, or for a whole boundary
    # (layer_name empty)."""
    block_hash: object  # int (unit tests) or bytes/str (vLLM)
    group_idx: int
    layer_name: str  # "" addresses the boundary, not one layer

    @property
    def hash_str(self) -> str:
        """Stable string form for paths / JSON (bytes -> hex)."""
        h = self.block_hash
        if isinstance(h, bytes):
            return h.hex()
        return str(h)

    @property
    def boundary_key(self) -> tuple[str, int]:
        """Identity of a boundary: hash and group."""
        return (self.hash_str, self.group_idx)


def make_boundary_key(group_idx: int,
                      block_hash: object) -> "CacheKey":
    """A group's key at one block hash, with no layer: the address the
    hit policy asks about and the address the save/load builders expand
    into per-layer page keys."""
    return CacheKey(
        block_hash=block_hash, group_idx=group_idx, layer_name="")


@dataclass
class GroupTransferMeta:
    """Per-group transfer instructions for one request (one step).
    # keys expands per (block, layer); gpu_block_ids is per block."""
    group_idx: int
    keys: tuple[CacheKey, ...] = ()
    gpu_block_ids: tuple[int, ...] = ()


# ======================================================================
# parse: vLLM KVCacheConfig -> hybrid groups
# ======================================================================
class KVShrinkParseError(ValueError):
    """Raised when the vLLM cache config cannot be parsed safely
    (unknown spec or inconsistent layout). The parse never
    guesses (fail closed)."""
    pass


def validate_codec_env() -> None:
    """Warn when the codec is configured to lose bits (the opaque
    # mamba pools are exempted structurally, so this only flags the
    # operator request)."""
    lossy = os.getenv("IAXL_KV_LOSSY_TRUNC", "0").strip()
    if lossy in ("", "0"):
        return
    logger.warning(
        "IAXL_KV_LOSSY_TRUNC=%s is enabled. On hybrid models the flag "
        "gates per pool entry: attention layers honor it, while mamba "
        "state is exempted structurally (lossy bit hard-cleared by "
        "KVStore).", lossy)


def _spec_kind(spec: object) -> str:
    """Mamba or attention; unknown specs raise KVShrinkParseError."""
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
) -> tuple[list[GroupInfo], int]:
    """Return (groups, num_blocks). Per-layer geometry is not
    # parsed: KVStore binds pools from the live tensors, which carry
    # their own layout. Raises KVShrinkParseError on unknown specs or
    # mixed block sizes."""
    num_blocks = kv_cache_config.num_blocks
    groups: list[GroupInfo] = []

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
        # GDN snapshots are addressed by an aligned boundary.
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

    # One block size for the whole model: hash i names block i in
    # EVERY group, which is what lets one hash address a boundary
    # across groups.
    sizes = {g.block_size for g in groups}
    if len(sizes) != 1:
        raise KVShrinkParseError(
            f"KV cache groups have different block sizes {sorted(sizes)}; "
            "the external cache addresses every group with one block hash "
            "and cannot express that")
    return groups, num_blocks


def save_enabled() -> bool:
    """Production save is ON by default; KVSHRINK_SAVE=0 disables it and
    KVSHRINK_DEBUG_AUTOSAVE=1 force-enables it."""
    return (os.getenv("KVSHRINK_SAVE", "1") != "0"
            or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")


# ======================================================================
# longest-hit policy
# ======================================================================
class _StoreAsBlockPool:
    """The one thing vLLM's matching code needs from us."""

    __slots__ = ("_present",)

    # Stands in for a skipped block. vLLM inserts it as padding and only
    # ever counts it, so it needs no identity beyond being a value.
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
        """Configure the policy for one request."""
        self._groups: list[GroupInfo] = groups
        self._present: Callable[[int, object], bool] = present
        self._hash_block_size: int = hash_block_size
        self._num_computed: int = num_computed_tokens
        # full attention first (tighter initial bound)
        self._ordered: list[GroupInfo] = sorted(
            groups, key=lambda g: 0 if g.kind == "attention" else 1)
        self._mamba_align: Optional[int] = None
        for g in groups:
            if g.kind == "mamba":
                a = g.mamba_align_size
                self._mamba_align = a if self._mamba_align is None \
                    else min(self._mamba_align, a)

    # ------------------------------------------------------------------
    def _lookup(self, group: GroupInfo, block_hashes: list[int],
                candidate: int) -> int:
        """How far this group alone is restorable, in tokens."""
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
        manager_cls = KVCacheSpecRegistry.get_manager_class(group.spec)
        # vLLM indexes the hash list directly out to max_length, so the
        # caller owes it a length its own hashes cover. Its scheduler
        # gets this for free (the bound comes from the same request);
        # ours can be a boundary the request has not reached.
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

    # ------------------------------------------------------------------
    def find_longest_cache_hit(
        self, block_hashes: list[int], max_length: int
    ) -> int:
        """Fixed-point convergence over all groups; returns the
        # restorable prefix in tokens (the snapshot boundary)."""
        candidate = max_length
        if self._mamba_align is not None:
            # the last prompt token is always recomputed (logprobs + state)
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


# ======================================================================
# worker bookkeeping
# ======================================================================


############################################################
# Connector
############################################################

class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
    """KVShrink external KV cache connector (hybrid GDN/Mamba aware)."""

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
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.vllm_device = vllm_config.device_config.device_type
        self.rank = get_world_group().rank if model_parallel_is_initialized() else 0

        self._req_states: dict[ReqId, ReqState] = {}
        # Async requests whose load plan has not been emitted yet (a
        # parked request never appears in the scheduler output).
        self._async_load_pending: set[str] = set()
        self._current_get_tasks: Optional[dict[str, Any]] = None
        self._current_put_tasks: dict[ReqId, list[dict[str, Any]]] = {}
        self._deferred_finished_req_ids: set[ReqId] = set()
        self._last_layer_name: Optional[str] = None
        # Ordered worker-side layer names (populated in register_kv_caches),
        # used to select the first N layers for async early-start.
        self._layer_names: list[str] = []
        # Async load bookkeeping (worker side).
        # Per-request tasks still loading across scheduler steps.
        self._pending_load_tasks: dict[ReqId, dict[str, Any]] = {}
        # Early-start layer count selected for each pending async request.
        self._pending_load_layers: dict[ReqId, int] = {}
        # Tasks early-promoted (first N layers done) whose remaining layers are
        # waited on-demand in wait_for_layer_load during the prefill forward.
        self._early_promoted_tasks: dict[ReqId, dict[str, Any]] = {}
        # Early-promoted tasks active for the current forward pass.
        self._active_promoted_tasks: dict[ReqId, dict[str, Any]] = {}

        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
        )

        if role == KVConnectorRole.SCHEDULER:
            self.kvstore: Optional[KVStore] = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(index) for index in range(self.num_layers)],
                tp_size=self.tp_size,
            )
        else:
            self.kvstore = None
            self._bind_cpu_affinity()
            self._bind_intel_accel()

        # Hybrid additions: group layout parsing for both roles.
        if kv_cache_config is not None:
            self._init_kv_stack(vllm_config, role, kv_cache_config)

    ############################################################
    # Construction
    ############################################################

    def _init_kv_stack(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Build the hybrid stack for this role."""
        pc = vllm_config.parallel_config
        tp_size = pc.tensor_parallel_size
        # Fail-closed: pipeline parallelism shards LAYERS across ranks,
        # so one rank holds half the model's KV and its pages alone name
        # only half a block. Every key would silently address a partial
        # state. Nothing here can degrade safely, so refuse at startup.
        if pc.pipeline_parallel_size != 1:
            raise RuntimeError(
                "kvshrink hybrid: pipeline parallelism is not supported "
                f"(pipeline_parallel_size={pc.pipeline_parallel_size}); "
                "each rank would persist only its own layers' pages. "
                "Set pipeline_parallel_size=1 or the KV connector.")
        if role == KVConnectorRole.WORKER:
            # parallel_config.rank is authoritative (TP rank).
            rank = pc.rank
        else:
            # Scheduler keys carry no rank: each process talks to its
            # own per-rank store directory, and the controller opens
            # the rank0 one.
            rank = 0

        # Block-hash granularity, per v0.23.0's resolve_kv_cache_block_sizes:
        # the GCD of the groups' block sizes (every group's block size is
        # divisible by it). Single group -> that group's block size.
        block_sizes = sorted({int(g.kv_cache_spec.block_size)
                              for g in kv_cache_config.kv_cache_groups})
        hash_block_size = math.gcd(*block_sizes)
        self._hash_block_size = hash_block_size
        # Fail-closed: a lossy codec would corrupt GDN state (see the
        # function for why this path cannot tolerate what the
        # attention-only path is designed to).
        validate_codec_env()

        groups, num_blocks = parse_kv_cache_config(
            kv_cache_config)

        # Fail-closed: speculative decoding widens the GDN state
        # gate beyond what this path was verified for.
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
        # Authoritative TP rank for addressing/labels. Kept separate from
        # self.rank (a world-group read guarded on init order) so
        # rank-sensitive code paths have one stable source.
        self._rank = rank
        # Attention layers in group order, used to size the
        # early-release prefix. Mamba layers are deliberately absent:
        # they are never partially released (see _decide_async).
        self._attention_layers = tuple(
            ln for g in groups if g.kind != "mamba"
            for ln in g.layer_names)
        # A recurrent group changes only which block hashes we ask
        # about (see _choose_block_hash_source); the storage below is
        # the same.
        recurrent = any(g.kind == "mamba" for g in groups)
        self._block_hash_source = self._choose_block_hash_source(recurrent)

        if role != KVConnectorRole.SCHEDULER:
            # One store label per group (see the lookup-vocabulary
            # section for why groups cannot share one).
            self._labels = [group_label(g.group_idx) for g in groups]
            # layer_name -> group idx, every cached layer.
            self._layer_group = {
                ln: g.group_idx for g in groups for ln in g.layer_names}
            # attention layer_name -> group idx (mamba layers map out).
            self._attn_layer_group = {
                ln: g.group_idx for g in groups if g.kind != "mamba"
                for ln in g.layer_names}

        logger.info(
            "kvshrink hybrid path enabled (%s role, %d groups, "
            "hash_block_size=%d, tp=%d rank=%d)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            len(groups), hash_block_size, tp_size, rank)
        logger.info(
            "kvshrink groups: %s",
            [(g.group_idx, g.kind, g.block_size) for g in groups])

    @staticmethod
    def _choose_block_hash_source(recurrent: bool) -> str:
        """Which block-identity scheme keys the cache: vLLM block
        # hashes (default) or the legacy token-hash fallback."""
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

    def _store(self) -> KVStore:
        if self.kvstore is None:
            raise RuntimeError("KVStore has not been initialized")
        return self.kvstore

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def _track_new_request(
        self, req_id: str, block_hashes: list[int],
        num_computed_tokens: int,
        request: Optional["Request"] = None,
    ) -> None:
        """Register a fresh ReqState."""
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
        # Downgrade to synchronous once released.
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
        """Every pass, for each running request: sync the block
        # tables and hashes from upstream; on resume (or progress
        # regression) roll the save cursor back so boundaries emitted
        # before a preemption get re-emitted (overwrite is idempotent)."""
        state = self._req_states[req_id]
        # Adopt block hashes vLLM has appended since we registered
        # (decode completes blocks too; without this, generated tokens
        # are never offloaded). Only ever extends -- hashes are
        # content-addressed and append-only.
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
            # fail-closed: a missing progress on resume is treated as
            # N=0 (roll everything back) rather than skipping the
            # rollback.
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
                    # upstream semantics: for resumed requests
                    # new_block_ids IS the table (replace), per group --
                    # including an EMPTY list, which clears stale blocks
                    gstate.block_ids = list(ids) if ids else []
                elif ids:
                    gstate.block_ids.extend(ids)

    # ------------------------------------------------------------------
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
        # Restorable boundary in tokens; 0 = miss. The policy already
        # gated on live chunk presence (engine Record), so a nonzero
        # boundary is complete by construction; only record it.
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

    # ------------------------------------------------------------------
    def _decide_async(self, req_id: str, external: int) -> bool:
        """Async when there are external tokens to stream and the
        # concurrency-tuned layer count is nonzero. The gate always
        # covers every recurrent layer regardless of that count."""
        if external <= 0 or self._async_load_layer_config is None:
            return False
        state = self._req_states[req_id]
        selected = self._async_load_layer_config.select(
            len(self._req_states))
        if selected == 0:
            return False
        state.is_async = True
        # Clamp: asking for more leading layers than exist would never
        # be satisfiable and would hang the request in
        # WAITING_FOR_REMOTE_KVS forever.
        if selected < 0 or selected > len(self._attention_layers):
            state.async_load_layers = -1  # require every layer
        else:
            state.async_load_layers = selected
        return True

    # ------------------------------------------------------------------
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record the allocated block tables per group and the
        # external-token count the core accepted (drives the load
        # plan). For async requests this is also where the load plan
        # is emitted, since a parked request never appears in
        # build_connector_meta."""
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
        # The ONLY moment we hear about an async request: it is
        # parked, so build_connector_meta never sees it scheduled.
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
        # True = defer freeing to get_finished(): async puts (and an
        # in-flight async load) may still be reading these blocks, so
        # the worker names the request in finished_sending once they
        # land. Committed boundaries are content-addressed and outlive
        # the request; they are never deleted here.
        self.on_request_finished(request.request_id)
        return True, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """SupportsHMA entry point (v0.23 calls this for hybrid models)."""
        return self.request_finished(request, [])

    # ------------------------------------------------------------------
    def build_load_meta(
        self, new_req: "NewRequestData", scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Load plan for a NewRequestData entry."""
        req_id = new_req.req_id
        state = self._req_states[req_id]
        return self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)

    def build_resumed_load_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Load plan for a PREEMPTION-RESUMED request: v1 carries
        # them in scheduled_cached_reqs.resumed_req_ids, not in
        # scheduled_new_reqs, so they need their own loop."""
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
        """Build one request's load plan from its recorded state."""
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
                    # v0.21 hashes are per complete block: hash i == block i
                    if i < len(ids):
                        # one page key + gpu block per layer (full expansion)
                        for layer_name in group.layer_names:
                            keys.append(self._page_key(key, layer_name))
                            gpu_ids.append(ids[i])
            elif group.kind == "mamba":
                # Load the snapshot into the CURR state block only
                # (v0.23 align mode pins execution to column 0 = the
                # block holding the last scheduled token, for both
                # prefill and decode).
                if state.block_hashes and boundary > 0:
                    # hash index of the snapshot AT boundary:
                    # hash[i] covers [i*bs, (i+1)*bs) -> snapshot at
                    # boundary lives at hash[boundary//bs - 1]
                    idx = boundary // group.block_size - 1
                    if 0 <= idx < len(state.block_hashes):
                        blk_hash = state.block_hashes[idx]
                        key = self._boundary_key(group, blk_hash)
                        if lookup_boundary(self.kvstore, key):
                            bs = group.block_size
                            # CURR running-state index (align mode):
                            # (num_computed + num_scheduled - 1) // bs.
                            # Kernels gather exactly this one column, so
                            # it is the only slot forward ever reads.
                            curr_idx = (boundary + scheduled_tokens -
                                        1) // bs

                            # Fail-closed: a HIT already committed
                            # num_computed_tokens=boundary; a skipped
                            # slot would let forward read unrestored
                            # state.
                            if scheduled_tokens <= 0 and not state.is_async:
                                # Sync restore with no scheduled tokens
                                # means no forward, so the slot would stay
                                # unrestored while the core already
                                # credited the tokens: fail-stop. (Async is
                                # correct here: vLLM's prev->curr copy
                                # carries the snapshot in at schedule time.)
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
        """Incremental save plan: for each request, every group,
        # emit the boundaries completed since the last pass. The
        # worker executes it after forward, when the GPU pages hold
        # state up to computed+scheduled tokens. A partial boundary
        # (not all layers of the group) is never emitted."""
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
            # Save the running-state block: the last non-null block
            # in the table (the one forward just wrote).
                if state.block_hashes:
                    block_pos = None
                    for pos in range(len(ids) - 1, -1, -1):
                        if ids[pos] != 0:  # 0 = null block placeholder
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
        """Assemble this pass's load/save plans."""
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
                meta.reqs_to_load.add_request(
                    new_req.req_id, req_meta.group_ops, req_meta.is_async,
                    req_meta.async_load_layers, req_meta.external_hit_tokens)
            if save_on:
                save_meta = self.build_save_meta(
                    new_req.req_id, num_sched.get(new_req.req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.add_request(
                        new_req.req_id, save_meta.group_ops)

            # ASYNC requests are parked by vLLM; their plans were
            # emitted at allocation time (update_state_after_alloc).
        for req_id, req_meta in self.take_async_load_plans(
                set(meta.reqs_to_load.requests)).items():
            if debug:
                logger.info(
                    "LOADMETA(async) req=%s ops=%d layers=%s",
                    req_id, len(req_meta.group_ops),
                    req_meta.async_load_layers)
            meta.reqs_to_load.add_request(
                req_id, req_meta.group_ops, req_meta.is_async,
                req_meta.async_load_layers, req_meta.external_hit_tokens)

        # PREEMPTION-RESUMED requests ride scheduled_cached_reqs.
        # resumed_req_ids, NOT scheduled_new_reqs. Their external-hit
        # tokens were accepted this same pass, so without a load plan
        # here the worker would never restore the pages while the core
        # already skips recompute -- silent garbage output.
        cr = scheduler_output.scheduled_cached_reqs
        for req_id in cr.resumed_req_ids:
            req_meta = self.build_resumed_load_meta(
                req_id, num_sched.get(req_id, 0))
            if debug:
                logger.info(
                    "LOADMETA(resumed) req=%s ops=%d",
                    req_id, len(req_meta.group_ops))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.reqs_to_load.add_request(
                req_id, req_meta.group_ops, req_meta.is_async,
                req_meta.async_load_layers, req_meta.external_hit_tokens)

        # Running requests cross boundaries in later steps too (chunked
        # prefill tails, decode-time crossings): sync their tables first,
        # then emit incremental saves.
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
                    meta.reqs_to_save.add_request(req_id,
                                                  save_meta.group_ops)

        if debug:
            logger.info(
                "build_connector_meta: %d load reqs, %d save reqs",
                len(meta.reqs_to_load.requests),
                len(meta.reqs_to_save.requests))
        return meta

    def _boundary_key(self, group: GroupInfo, block_hash: object) -> CacheKey:
        """Boundary key for one group at one block hash."""
        return make_boundary_key(group.group_idx, block_hash)

    def _present(self, group_idx: int, block_hash: object) -> bool:
        """Store-presence predicate handed to the hit policy, which
        plans against boundary addresses without seeing store details."""
        return lookup_boundary(
            self.kvstore,
            make_boundary_key(group_idx, block_hash))

    @staticmethod
    def _page_key(boundary_key: CacheKey, layer_name: str) -> CacheKey:
        """Expand a boundary key to ONE layer's page key: same hash and
        group as the boundary, plus the layer name. This is the exact
        page address the worker must move."""
        return replace(boundary_key, layer_name=layer_name)

    ############################################################
    # Worker Side Methods
    ############################################################

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

        # Execution order feeds the async release gate.
        self.register(sorted(kv_caches, key=extract_layer_index))

        # The store binds the RAW kv_caches directly.
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            kv_caches=kv_caches,
        # _rank: captured in _init_kv_stack.
            rank=self._rank,
            tp_size=self.tp_size,
        )
        logger.info("Registered %d KV cache layers",
                    len(self._store().layer_names))

    def register(
        self,
        execution_order: list[str],
    ) -> None:
        """Record the model's execution order; only the attention
        # order is used, by the async release gate."""
        # Ordered layer names for the block store layout and the two
        # order-sensitive derived sets.
        self._layer_names = list(execution_order)
        # All GDN layers: waited as one barrier in start_load before any
        # attention layer runs.
        self._mamba_layers = frozenset(
            ln for g in self._groups if g.kind == "mamba"
            for ln in g.layer_names)
        # Attention layers in model execution order, used by the async
        # release gate ("the first N layers" means nothing otherwise).
        self._attn_order = tuple(
            ln for ln in execution_order if ln in self._attn_layer_group)
        # Save pipelining segments: the mamba layers between attention
        # layer i-1 and attention layer i are final when i's save hook
        # fires, so they ride that hook (the trailing segment is
        # submitted by wait_for_save).
        segments: dict[str, tuple[str, ...]] = {}
        pending: list[str] = []
        for ln in execution_order:
            if ln in self._mamba_layers:
                pending.append(ln)
            elif ln in self._attn_layer_group and pending:
                segments[ln] = tuple(pending)
                pending = []
        self._mamba_save_segments = segments
        # Last attention hook: clears the per-step get tasks (main's
        # cleanup point).
        self._last_layer_name = self._attn_order[-1] if self._attn_order else None
        logger.info(
            "kvshrink hybrid worker registered: %d attention "
            "hook points, %d recurrent layers (tp=%d rank=%d)",
            len(self._attn_order),
            len(self._mamba_layers), self.tp_size, self._rank)

    def _metadata(self) -> KVShrinkConnectorMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError(
                "kvshrink hybrid worker received "
                f"{type(metadata).__name__}; expected "
                "KVShrinkConnectorMetadata")
        return metadata

    # ----------------------------------------------------------
    # load path
    # ----------------------------------------------------------
    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        metadata = self._metadata()

        # A no-forward batch cannot consume promoted tasks layer by layer.
        if forward_context.attn_metadata is not None:
            duplicates = (
                self._active_promoted_tasks.keys()
                & self._early_promoted_tasks.keys()
            )
            if duplicates:
                raise RuntimeError(
                    f"Duplicate promoted load tasks for requests {duplicates}"
                )
            self._active_promoted_tasks.update(self._early_promoted_tasks)
            self._early_promoted_tasks = {}

        # Submits every load, then host-blocks on the recurrent ones;
        # attention layers are waited by their own hooks during forward.
        self.start_load(metadata)

    def start_load(self, metadata: KVShrinkConnectorMetadata) -> int:
        """Submit all of this step's loads, then host-block on the
        # recurrent ones (no hook ever fires for them). Attention
        # pages are waited per layer by the forward hooks."""
        # Per-step reset; _saved_layers/_step_save_pages track
        # what the save hooks have submitted.
        self._current_get_tasks = None
        self._saved_layers = set()
        self._step_save_pages = 0
        npages = 0
        _t0 = time.monotonic()
        # Sync loads merge per group into one engine call each.
        by_group: dict[int, tuple[list[tuple[int, str]], set[str]]] = {}
        for req_id, req_meta in metadata.reqs_to_load.requests.items():
            if req_meta.is_async:
                # Async tasks are parked (no forward this step) and
                # polled across steps in get_finished (main's flow).
                tasks: dict[str, Task] = {}
                for op in req_meta.group_ops:
                    if not op.keys:
                        continue
                    t, n = self._submit_group_load(op)
                    tasks.update(t)
                    npages += n
                if tasks:
                    self._pending_load_tasks[req_id] = tasks
                    self._pending_load_layers[req_id] = (
                        req_meta.async_load_layers)
            else:
                for op in req_meta.group_ops:
                    if not op.keys:
                        continue
                    entries, layers = by_group.setdefault(
                        op.group_idx, ([], set()))
                    entries.extend(self._op_entries(op))
                    layers.update(key.layer_name for key in op.keys)
        merged: dict[str, Task] = {}
        for g_idx, (entries, layers) in by_group.items():
            tasks, n = self._submit_group_load_entries(
                g_idx, entries, layers)
            merged.update(tasks)
            npages += n
        if merged:
            self._current_get_tasks = merged
        # Every recurrent layer, waited before forward begins (main's
        # layer filter reused: these layers have no forward hook).
        recurrent = [ln for ln in merged if ln in self._mamba_layers]
        if recurrent:
            if not self._store().get_wait(
                    get_results=merged, layer_names=recurrent, wait=True):
                raise RuntimeError(
                    "kvshrink load failed: recurrent pages did not land; "
                    "forward would read unrestored state")
        if npages:
            logger.info(
                "start_load_kv: %d pages loaded "
                "elapsed_ms=%.3f (rank %d/%d)", npages,
                (time.monotonic() - _t0) * 1e3, self._rank, self.tp_size)
        return npages

    def wait_for_layer_load(self, layer_name: str) -> None:
        # main's hook verbatim: wait this layer's pages in the merged
        # sync batch and in every promoted async load (recurrent layers
        # were already waited in start_load; waiting a landed layer is
        # a no-op).
        if not self._current_get_tasks and not self._active_promoted_tasks:
            return

        if self._current_get_tasks:
            success = self._store().get_wait(
                get_results=self._current_get_tasks,
                layer_names=[layer_name],
            )
            if not success:
                raise RuntimeError(
                    f"Failed to load KV cache for layer {layer_name}"
                )

        for tasks in self._active_promoted_tasks.values():
            success = self._store().get_wait(
                get_results=tasks,
                layer_names=[layer_name],
            )
            if not success:
                raise RuntimeError(
                    f"Failed to load promoted KV cache for layer {layer_name}"
                )

        if layer_name == self._last_layer_name:
            self._current_get_tasks = None
            self._active_promoted_tasks = {}

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
        ``entries``; returns the tasks dict (layer name -> Task) and
        the page count."""
        tasks = self._store().get(block_indices=[gpu for gpu, _ in entries],
                                 block_hashs=[h for _, h in entries],
                                 layer_names=list(layer_names),
                                 label=self._labels[g_idx])
        return tasks, len(entries) * len(layer_names)

    # ------------------------------------------------------------------
    # save path
    # ------------------------------------------------------------------
    def _gather_save_candidates(
        self, metadata: KVShrinkConnectorMetadata
    ) -> dict[tuple[str, int], dict]:
        """boundary_key -> {"group_idx", "pages", "req_ids"}, with
        # cross-request dedup (shared boundaries are written once)."""
        candidates: dict[tuple[str, int], dict] = {}
        for req_id, req_meta in metadata.reqs_to_save.requests.items():
            for op in req_meta.group_ops:
                for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
                    cand = candidates.get(key.boundary_key)
                    if cand is None:
                        cand = {"group_idx": op.group_idx,
                                "pages": {}, "req_ids": set()}
                        candidates[key.boundary_key] = cand
                    cand["pages"][key.layer_name] = (key, gpu_block_id)
                    cand["req_ids"].add(req_id)
        return candidates

    def _submit_group_layers_save(
        self, g_idx: int, layer_names: list[str],
        entries: list[tuple[int, str]],
    ) -> dict[str, Task]:
        """One async engine put covering ``layer_names`` for the
        # (gpu_block, hash) entries."""
        return self._store().put(block_indices=[gpu for gpu, _ in entries],
                                block_hashs=[h for _, h in entries],
                                layer_names=layer_names,
                                label=self._labels[g_idx])


    def _submit_layers_save(
        self, g_idx: int, layer_names: list[str],
        metadata: KVShrinkConnectorMetadata,
    ) -> None:
        """One async engine put for ``layer_names`` of one group over
        # this step's complete boundary candidates."""
        expected = sorted(self._groups[g_idx].layer_names)
        entries: list[tuple[int, str]] = []
        req_ids: set[str] = set()
        for cand in self._gather_save_candidates(metadata).values():
            if cand["group_idx"] != g_idx:
                continue
            if sorted(cand["pages"]) != expected:
                continue  # partial boundary: skipped at submit time too
            key, gpu_block_id = cand["pages"][layer_names[0]]
            entries.append((gpu_block_id, key.hash_str))
            req_ids |= cand["req_ids"]
        if not entries:
            return
        tasks = self._submit_group_layers_save(g_idx, layer_names,
                                               entries)
        for rid in req_ids:
            self._current_put_tasks.setdefault(rid, []).append(tasks)
        self._saved_layers.update(layer_names)
        self._step_save_pages += len(entries) * len(layer_names)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Submit this attention layer's pages plus the mamba segment
        # before it (their kernels already ran, so the data is final).
        # The trailing segment goes out in wait_for_save. Submission
        # only; the drain lives in get_finished."""
        if self._connector_metadata is None:
            return
        if os.getenv("KVSHRINK_SAVE_PIPELINED", "1") == "0":
            return
        if not save_enabled():
            return
        metadata = self._metadata()
        segment = self._mamba_save_segments.get(layer_name)
        if segment:
            # A segment mixes groups (execution order interleaves the
            # three mamba groups); each group's layers go under their
            # own store label.
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
        """Submit every layer not already pipelined (trailing mamba
        # segment, hooks that never fired), one put per group. The
        # drain lives in get_finished. Returns (pages, boundaries)."""
        candidates = self._gather_save_candidates(metadata)
        per_group: dict[int, dict[str, Any]] = {}
        nbound = 0
        for bkey, cand in candidates.items():
            blk_hash, g_idx = bkey
            expected = sorted(self._groups[g_idx].layer_names)
            if sorted(cand["pages"]) != expected:
                # A partial boundary is skipped (the scheduler's
                # incremental cursor has advanced, so it ages out as a
                # MISS, never wrong data). Log unconditionally: this is
                # the one save-side anomaly that does not fail-stop.
                logger.warning(
                    "chunk_save skip commit g%d h=%s: expected %d "
                    "layers, stored %d (%s)", g_idx, blk_hash,
                    len(expected), len(cand.pages),
                    set(expected) ^ set(cand.pages))
                continue
            nbound += 1
            blob = per_group.setdefault(g_idx, {"entries": [],
                                                "req_ids": set()})
            # entries are identical across a group's layers (same gpu
            # block and hash per boundary, expanded per layer)
            key, gpu_block_id = cand["pages"][expected[0]]
            blob["entries"].append((gpu_block_id, key.hash_str))
            blob["req_ids"] |= cand["req_ids"]
        for g_idx, blob in per_group.items():
            remaining = [ln for ln in self._groups[g_idx].layer_names
                         if ln not in self._saved_layers]
            if not remaining:
                continue
            tasks = self._submit_group_layers_save(
                g_idx, remaining, blob["entries"])
            for rid in blob["req_ids"]:
                self._current_put_tasks.setdefault(rid, []).append(tasks)
            self._saved_layers.update(remaining)
            self._step_save_pages += len(blob["entries"]) * len(remaining)
            # Finalized by its own write; no second phase.
        if self._step_save_pages:
            # Counterpart of the start_load_kv line: without it a run
            # that saves nothing looks exactly like a healthy one.
            logger.info(
                "chunk_save: %d pages submitted, %d boundaries "
                "(rank %d/%d)", self._step_save_pages, nbound,
                self._rank, self.tp_size)
        return self._step_save_pages, nbound

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # Poll asynchronous load tasks submitted in start_load_kv().
        # Hybrid gate (the one semantic delta from main): every recurrent
        # layer in the plan gates the release, whatever the configured
        # count says -- a GDN state is read whole at forward start, so
        # releasing before it lands reads stale memory, silently.
        finished_recving: set[str] = set()
        for req_id in list(self._pending_load_tasks.keys()):
            tasks = self._pending_load_tasks[req_id]
            async_load_layers = self._pending_load_layers[req_id]
            if async_load_layers == -1:
                # Require all layers before marking the load finished.
                if self._store().get_wait(get_results=tasks, wait=False):
                    self._store().get_wait(get_results=tasks, wait=True)
                    del self._pending_load_tasks[req_id]
                    del self._pending_load_layers[req_id]
                    finished_recving.add(req_id)
            else:
                # recurrent layers union the first-N attention prefix
                gate_layers = (
                    [ln for ln in tasks if ln in self._mamba_layers]
                    + [ln for ln in self._attn_order
                       if ln in tasks][:async_load_layers])
                if self._store().get_wait(
                        get_results=tasks, layer_names=gate_layers,
                        wait=False):
                    self._store().get_wait(
                        get_results=tasks, layer_names=gate_layers,
                        wait=True)
                    del self._pending_load_tasks[req_id]
                    del self._pending_load_layers[req_id]
                    # Early promote once the gate layers are loaded; the
                    # remaining layers are waited on-demand in
                    # wait_for_layer_load().
                    self._early_promoted_tasks[req_id] = tasks
                    finished_recving.add(req_id)

        self._deferred_finished_req_ids.update(finished_req_ids)
        completed: set[str] = set()
        for req_id in self._deferred_finished_req_ids:
            # Finished with an async load still in flight: drain it
            # here (its layer hooks will never fire again).
            load_tasks = (self._pending_load_tasks.get(req_id)
                          or self._early_promoted_tasks.get(req_id)
                          or self._active_promoted_tasks.get(req_id))
            if load_tasks is not None:
                if not self._store().get_wait(get_results=load_tasks,
                                             wait=False):
                    continue
                self._store().get_wait(get_results=load_tasks, wait=True)
                self._pending_load_tasks.pop(req_id, None)
                self._pending_load_layers.pop(req_id, None)
                self._early_promoted_tasks.pop(req_id, None)
                self._active_promoted_tasks.pop(req_id, None)

            tasks = self._current_put_tasks.get(req_id)
            if tasks is None:
                completed.add(req_id)
                continue
            while tasks:
                if all(t.ctx is None for t in tasks[0].values()):
                    # A boundary shared with another finished request:
                    # that request's drain already finalized this put
                    # (put_wait sets ctx None on completion).
                    tasks.pop(0)
                    continue
                if not self._store().put_wait(tasks[0], wait=False):
                    break
                tasks.pop(0)
            if not tasks:
                del self._current_put_tasks[req_id]
                completed.add(req_id)

        self._deferred_finished_req_ids.difference_update(completed)
        return (completed or None), (finished_recving or None)

    # ------------------------------------------------------------------
    # debug dump
    # ------------------------------------------------------------------
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
            page_view = self._store().kv_caches[ln]
            for blk in range(10):
                page = page_view[blk]
                h = hashlib.sha256(
                    page.cpu().numpy().tobytes()).hexdigest()
                logger.info("DUMP g%d block=%d sha=%s",
                            group.group_idx, blk, h[:16])
