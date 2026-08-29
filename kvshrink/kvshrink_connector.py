# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator, Optional

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

from iaxl import KVStore, setup_root_logger
from iaxl.kvflow.flow import Task

from .hybrid_hit import HybridHitPolicy
from .async_load_config import (
    load_async_load_layer_config_from_env)
setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str


@dataclass
class ReqMeta:
    """All transfer instructions for one request in one step.

    block_hashes is per block and shared by every group (the mamba
    snapshot is the LAST entry); group_block_ids[n] holds group n's
    destination blocks -- an attention group aligns with
    block_hashes, a mamba group carries exactly its CURR slot."""
    block_hashes: tuple[str, ...] = ()
    group_block_ids: tuple[tuple[int, ...], ...] = ()
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class ReqGroupState:
    """Per-group mutable state for one request (scheduler side)."""
    block_ids: list[int] = field(default_factory=list)
    next_stored_chunk_idx: int = 0


@dataclass
class ReqState:
    # The engine's live block_hashes list (it grows in place as decode
    # completes blocks); block_hashes below is our copy, synced from it
    # each pass in on_cached_request.
    live_source: list = field(default_factory=list)
    block_hashes: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    groups: tuple[ReqGroupState, ...] = ()
    # External tokens accepted this pass (drives the load plan).
    pending_load_tokens: int = 0
    # Last authoritative progress seen by the save path
    # (num_computed + scheduled of the last save plan). Used for
    # fail-closed regression detection: any drop below this value rolls
    # save cursors back even if the resumed flag is missing.
    last_known_progress: int = 0
    # Async load bookkeeping: while is_async, the request is
    # parked and its plan ships from build_connector_meta.
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
        block_hashes: tuple[str, ...] = (),
        group_block_ids: tuple[tuple[int, ...], ...] = (),
        is_async: bool = False,
        async_load_layers: int = -1,
    ) -> None:
        self.requests[req_id] = ReqMeta(
            block_hashes=block_hashes,
            group_block_ids=group_block_ids,
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


def _hash_str(block_hash) -> str:
    """Stable string form for the store (bytes -> hex)."""
    return block_hash.hex() if isinstance(block_hash, bytes) \
        else str(block_hash)


# ======================================================================
# parse: vLLM KVCacheConfig -> hybrid groups
# ======================================================================
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
    """One GroupInfo per vLLM KV cache group. Per-layer geometry is not
    parsed: KVStore binds pools from the live tensors, which carry their
    own layout."""
    groups: list[GroupInfo] = []
    for g_idx, g in enumerate(kv_cache_config.kv_cache_groups):
        spec = list(_iter_layer_specs(g))[0][1]
        kind = "mamba" if isinstance(spec, MambaSpec) else "attention"
        groups.append(GroupInfo(
            group_idx=g_idx,
            kind=kind,
            layer_names=tuple(g.layer_names),
            block_size=int(spec.block_size),
            mamba_align_size=(int(spec.block_size)
                              if kind == "mamba" else None),
            spec=spec,
        ))
    return groups, kv_cache_config.num_blocks


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
        groups, _num_blocks = parse_kv_cache_config(kv_cache_config)
        self._groups = groups
        # Block-hash granularity, per v0.23.0's resolve_kv_cache_block_sizes:
        # the GCD of the groups' block sizes (every group's block size is
        # divisible by it). Single group -> that group's block size.
        self._hash_block_size = math.gcd(*(g.block_size for g in groups))

        # Fail-closed: spec decode moves the GDN running state into
        # per-draft speculative blocks; the boundary block is committed
        # only on acceptance, so a snapshot would persist a draft
        # intermediate state (kvshrink-hybrid.md §5.4).
        for g in groups:
            if g.kind == "mamba" and g.spec.num_speculative_blocks:
                raise RuntimeError(
                    "kvshrink hybrid: speculative decoding is not "
                    f"supported (group has num_speculative_blocks="
                    f"{g.spec.num_speculative_blocks}); the external GDN "
                    "snapshot only restores the non-speculative state "
                    "slot. Disable speculative decoding or the KV "
                    "connector.")

        # Attention layers in group order, used to size the
        # early-release prefix. Mamba layers are deliberately absent:
        # they are never partially released.
        self._attention_layers = tuple(
            ln for g in groups if g.kind != "mamba"
            for ln in g.layer_names)
        if role != KVConnectorRole.SCHEDULER:
            # layer_name -> group idx, every cached layer.
            self._layer_group = {
                ln: g.group_idx for g in groups for ln in g.layer_names}
            # attention layer_name -> group idx (mamba layers map out).
            self._attn_layer_group = {
                ln: g.group_idx for g in groups if g.kind != "mamba"
                for ln in g.layer_names}

        logger.info(
            "kvshrink hybrid path enabled (%s role, tp=%d rank=%d, "
            "hash_block_size=%d, groups=%s)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            tp_size, self.rank, self._hash_block_size,
            [(g.group_idx, g.kind, g.block_size) for g in groups])

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
        live = state.live_source
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
        # This request's block identities, in block order: always the
        # engine's own hashes, so "hash i names block i" is guaranteed
        # by the engine rather than re-derived (kvshrink-hybrid.md §8).
        block_hashes = list(request.block_hashes)
        self._req_states[request.request_id] = ReqState(
            live_source=request.block_hashes,
            block_hashes=list(block_hashes),
            num_computed_tokens=num_computed_tokens,
            groups=tuple(ReqGroupState() for _ in self._groups),
        )
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        policy = HybridHitPolicy(
            self._groups,
            lambda g, h: self._store().has(
                [_hash_str(h)], label=f"g{g}")[0],
            self._hash_block_size, num_computed_tokens)
        # Restorable boundary in tokens; 0 = miss. The policy already
        # gated on live chunk presence (engine Record), so a nonzero
        # boundary is complete by construction; only record it.
        boundary = policy.find_longest_cache_hit(
            block_hashes, request.num_tokens)
        state = self._req_states[request.request_id]
        external = max(0, boundary - num_computed_tokens)
        # Async when there are external tokens to stream and the
        # concurrency-tuned layer count is nonzero.
        use_async = external > 0 and self._async_load_layer_config is not None
        if use_async:
            selected = self._async_load_layer_config.select(
                len(self._req_states))
            use_async = selected != 0
        if use_async:
            state.is_async = True
            # Clamp: more leading layers than exist would hang the
            # request in WAITING_FOR_REMOTE_KVS forever.
            if selected < 0 or selected > len(self._attention_layers):
                state.async_load_layers = -1  # require every layer
            else:
                state.async_load_layers = selected
        logger.debug(
            "req=%s external_hit=%d boundary=%d async=%s",
            request.request_id, external, boundary, use_async)
        return external, use_async

    # ------------------------------------------------------------------
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
        self._req_states.pop(request.request_id, None)
        self._async_load_pending.discard(request.request_id)
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
            npages = sum(len(g) for g in meta.group_block_ids)
            if npages == 0:
                raise RuntimeError(
                    "kvshrink resumed request has accepted external "
                    "tokens but no restorable pages (req="
                    f"{req_id} pending={state.pending_load_tokens} "
                    f"boundary={state.num_computed_tokens} "
                    f"sched={scheduled_tokens}): refusing to enter "
                    "forward with unrestored state")
        return meta

    def _build_load_meta_from_state(
        self, req_id: str, state: ReqState, scheduled_tokens: int,
    ) -> ReqMeta:
        """Build one request's load plan from its recorded state."""
        ext = state.pending_load_tokens
        hashes: list[str] = []
        group_ids: list[tuple[int, ...]] = [() for _ in self._groups]
        if ext > 0:
            nc = state.num_computed_tokens
            owner = next(
                (g for g in self._groups if g.kind == "attention"),
                self._groups[0])
            for g_idx, group in enumerate(self._groups):
                # The group table is always populated by alloc before
                # scheduling; a missing table with ext > 0 is a bug
                # that must index out of range loudly below, not skip
                # silently.
                ids = state.groups[g_idx].block_ids
                if group.kind == "attention":
                    # Load only the external range: the core's own
                    # prefix-hit blocks already hold their data (shared
                    # physical pages). The credit landed
                    # num_computed_tokens exactly on the boundary, so
                    # this range is the last ext tokens.
                    start = (nc - ext) // group.block_size
                    end = nc // group.block_size
                    if group is owner:
                        hashes = [_hash_str(state.block_hashes[i])
                                  for i in range(start, end)]
                    group_ids[g_idx] = tuple(ids[i]
                                             for i in range(start, end))
                elif group.kind == "mamba":
                    # Load the snapshot into the CURR state block only
                    # (v0.23 align mode pins execution to column 0 =
                    # the block holding the last scheduled token, for
                    # both prefill and decode). Its hash is the plan's
                    # last entry: the snapshot lives at the boundary,
                    # which the credit made num_computed_tokens.
                    curr_idx = (nc + scheduled_tokens
                                - 1) // group.block_size

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
                            f"(req={req_id} "
                            f"boundary={nc}): "
                            "production hits must schedule >= 1 "
                            "token; refusing to build load meta")
                    if not (0 <= curr_idx < len(ids)
                            and ids[curr_idx] != 0):
                        raise RuntimeError(
                            "kvshrink mamba load curr slot "
                            f"invalid (req={req_id} "
                            f"boundary={nc} "
                            f"sched={scheduled_tokens} "
                            f"table_idx={curr_idx} "
                            f"table={ids}): refusing to enter "
                            "forward with unrestored state")
                    if group is owner:
                        idx = nc // group.block_size - 1
                        hashes = [_hash_str(state.block_hashes[idx])]
                    group_ids[g_idx] = (ids[curr_idx],)
        return ReqMeta(
            block_hashes=tuple(hashes),
            group_block_ids=tuple(group_ids),
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
        owner = next((g for g in self._groups if g.kind == "attention"),
                     self._groups[0])
        hashes: list[str] = []
        group_ids: list[tuple[int, ...]] = [() for _ in self._groups]
        for g_idx, group in enumerate(self._groups):
            gstate = state.groups[g_idx]
            ids = gstate.block_ids
            if group.kind == "attention":
                num_hash = min(progress // group.block_size, len(ids),
                               len(state.block_hashes))
                start = gstate.next_stored_chunk_idx
                if num_hash > start:
                    if group is owner:
                        hashes = [_hash_str(state.block_hashes[i])
                                  for i in range(start, num_hash)]
                    group_ids[g_idx] = tuple(ids[i] for i in
                                             range(start, num_hash))
                    gstate.next_stored_chunk_idx = num_hash
            else:
                # Save the running-state block: the last non-null block
                # in the table (the one forward just wrote). It fires
                # only at a boundary, whose hash is then the plan's
                # last entry.
                if state.block_hashes:
                    block_pos = None
                    for pos in range(len(ids) - 1, -1, -1):
                        if ids[pos] != 0:  # 0 = null block placeholder
                            block_pos = pos
                            break
                    if (block_pos is not None and progress > 0
                            and progress % group.block_size == 0):
                        idx = progress // group.block_size - 1
                        if (idx >= gstate.next_stored_chunk_idx
                                and idx < len(state.block_hashes)):
                            if group is owner:
                                hashes = [_hash_str(
                                    state.block_hashes[idx])]
                            group_ids[g_idx] = (ids[block_pos],)
                            gstate.next_stored_chunk_idx = idx + 1
        return ReqMeta(
            block_hashes=tuple(hashes),
            group_block_ids=tuple(group_ids),
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Assemble this pass's load/save plans."""
        meta = KVShrinkConnectorMetadata()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            req_meta = self.build_load_meta(
                new_req, num_sched[new_req.req_id])
            if any(req_meta.group_block_ids):
                meta.reqs_to_load.add_request(
                    new_req.req_id, req_meta.block_hashes,
                    req_meta.group_block_ids, req_meta.is_async,
                    req_meta.async_load_layers)
            save_meta = self.build_save_meta(
                new_req.req_id, num_sched[new_req.req_id])
            if any(save_meta.group_block_ids):
                meta.reqs_to_save.add_request(
                    new_req.req_id, save_meta.block_hashes,
                    save_meta.group_block_ids)

            # ASYNC requests are parked by vLLM; their plans were
            # emitted at allocation time (update_state_after_alloc).
        # Load plans for requests vLLM parked, drained exactly once.
        # Every parked request was queued with external tokens
        # accepted, so its plan always carries pages.
        pending = sorted(self._async_load_pending
                         - set(meta.reqs_to_load.requests))
        for req_id in pending:
            state = self._req_states[req_id]
            req_meta = self._build_load_meta_from_state(
                req_id, state, scheduled_tokens=0)
            state.async_plan_emitted = True
            # Downgrade to synchronous once released.
            state.is_async = False
            meta.reqs_to_load.add_request(
                req_id, req_meta.block_hashes, req_meta.group_block_ids,
                req_meta.is_async, req_meta.async_load_layers)
        self._async_load_pending -= set(pending)

        # PREEMPTION-RESUMED requests ride scheduled_cached_reqs.
        # resumed_req_ids, NOT scheduled_new_reqs. Their external-hit
        # tokens were accepted this same pass, so without a load plan
        # here the worker would never restore the pages while the core
        # already skips recompute -- silent garbage output.
        cr = scheduler_output.scheduled_cached_reqs
        for req_id in cr.resumed_req_ids:
            req_meta = self.build_resumed_load_meta(
                req_id, num_sched[req_id])
            if any(req_meta.group_block_ids):
                meta.reqs_to_load.add_request(
                    req_id, req_meta.block_hashes,
                    req_meta.group_block_ids, req_meta.is_async,
                    req_meta.async_load_layers)

        # Running requests cross boundaries in later steps too (chunked
        # prefill tails, decode-time crossings): sync their tables first,
        # then emit incremental saves.
        resumed = cr.resumed_req_ids
        new_bids = cr.new_block_ids
        ncts = cr.num_computed_tokens
        for i, req_id in enumerate(cr.req_ids):
            self.on_cached_request(
                req_id, new_bids[i], req_id in resumed, ncts[i])
            save_meta = self.build_save_meta(
                req_id, num_sched[req_id])
            if any(save_meta.group_block_ids):
                meta.reqs_to_save.add_request(
                    req_id, save_meta.block_hashes,
                    save_meta.group_block_ids)
        return meta

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
            rank=self.rank,
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
            len(self._mamba_layers), self.tp_size, self.rank)

    # ----------------------------------------------------------
    # load path
    # ----------------------------------------------------------
    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

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

        # Submit all of this step's loads, then host-block on the
        # recurrent ones (no hook ever fires for them). Attention
        # pages are waited per layer by the forward hooks.
        # Per-step reset; _saved_layers/_step_save_pages track
        # what the save hooks have submitted.
        self._current_get_tasks = None
        self._saved_layers = set()
        self._step_save_pages = 0
        npages = 0
        _t0 = time.monotonic()
        # One engine get per layer, in execution order: the transfers
        # then queue in the order forward consumes them. Sync and async
        # dispatch identically; they differ only in where the tasks
        # land -- the merged batch the forward hooks wait on, or the
        # per-request dicts get_finished polls for parked requests.
        merged: dict[str, Task] = {}
        async_tasks: dict[str, dict[str, Task]] = {}
        async_gate: dict[str, int] = {}
        npages = 0
        for ln in self._layer_names:
            g_idx = self._layer_group[ln]
            mamba = self._groups[g_idx].kind == "mamba"
            sync_entries: list[tuple[int, str]] = []
            for req_id, req_meta in metadata.reqs_to_load.requests.items():
                gids = req_meta.group_block_ids[g_idx]
                if not gids:
                    continue
                # A mamba entry is the plan's last hash into its single
                # CURR block; an attention group pairs every block with
                # its hash.
                entries = ((gids[0], req_meta.block_hashes[-1]),) if mamba \
                    else tuple(zip(gids, req_meta.block_hashes))
                if not req_meta.is_async:
                    sync_entries.extend(entries)
                    continue
                npages += len(entries)
                async_tasks.setdefault(req_id, {}).update(
                    self._store().get(
                        block_indices=[gpu for gpu, _ in entries],
                        block_hashs=[h for _, h in entries],
                        layer_names=[ln], label=f"g{g_idx}"))
                async_gate[req_id] = req_meta.async_load_layers
            if sync_entries:
                npages += len(sync_entries)
                merged.update(self._store().get(
                    block_indices=[gpu for gpu, _ in sync_entries],
                    block_hashs=[h for _, h in sync_entries],
                    layer_names=[ln], label=f"g{g_idx}"))
        if merged:
            self._current_get_tasks = merged
        for req_id, tasks in async_tasks.items():
            self._pending_load_tasks[req_id] = tasks
            self._pending_load_layers[req_id] = async_gate[req_id]
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
                (time.monotonic() - _t0) * 1e3, self.rank, self.tp_size)
        return 

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

    # ------------------------------------------------------------------
    # save path
    # ------------------------------------------------------------------
    def _gather_save_candidates(
        self, metadata: KVShrinkConnectorMetadata
    ) -> dict[tuple[str, int], dict]:
        """(hash, group) -> {"gpu", "req_ids"}, with cross-request
        # dedup (shared boundaries are written once)."""
        candidates: dict[tuple[str, int], dict] = {}
        for req_id, req_meta in metadata.reqs_to_save.requests.items():
            for g_idx, gids in enumerate(req_meta.group_block_ids):
                if not gids:
                    continue
                if self._groups[g_idx].kind == "mamba":
                    items = ((req_meta.block_hashes[-1], gids[0]),)
                else:
                    items = zip(req_meta.block_hashes, gids)
                for h, gpu in items:
                    cand = candidates.setdefault(
                        (h, g_idx), {"gpu": gpu, "req_ids": set()})
                    cand["gpu"] = gpu
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
                                label=f"g{g_idx}")


    def _submit_layers_save(
        self, g_idx: int, layer_names: list[str],
        metadata: KVShrinkConnectorMetadata,
    ) -> None:
        """One async engine put for ``layer_names`` of one group over
        # this step's complete boundary candidates."""
        entries: list[tuple[int, str]] = []
        req_ids: set[str] = set()
        for (h, cand_g), cand in self._gather_save_candidates(
                metadata).items():
            if cand_g != g_idx:
                continue
            entries.append((cand["gpu"], h))
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
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")
        segment = self._mamba_save_segments.get(layer_name)
        if segment:
            # A segment mixes groups (execution order interleaves the
            # three mamba groups); each layer goes under its own
            # group's store label.
            for ln in segment:
                self._submit_layers_save(
                    self._layer_group[ln], [ln], metadata)
        self._submit_layers_save(
            self._attn_layer_group[layer_name], [layer_name], metadata)

    def wait_for_save(self) -> None:
        if self.kvstore is None:
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")
        pages, boundaries = self.submit_saves(metadata)
        # KVSHRINK_DEBUG_DUMP=1: sha256 of the first mamba page of each
        # group at blocks 0..9, for byte-exact cold-vs-hot comparison.
        if os.getenv("KVSHRINK_DEBUG_DUMP"):
            for group in self._groups:
                if group.kind != "mamba":
                    continue
                for blk in range(10):
                    page = self._store().kv_caches[group.layer_names[0]][blk]
                    h = hashlib.sha256(
                        page.cpu().numpy().tobytes()).hexdigest()
                    logger.info("DUMP g%d block=%d sha=%s",
                                group.group_idx, blk, h[:16])

    def submit_saves(
        self, metadata: KVShrinkConnectorMetadata
    ) -> tuple[int, int]:
        """Submit every layer not already pipelined (trailing mamba
        # segment, hooks that never fired), one put per layer in
        # execution order. The drain lives in get_finished. Returns
        # (pages, boundaries)."""
        candidates = self._gather_save_candidates(metadata)
        nbound = len(candidates)
        for ln in self._layer_names:
            if ln in self._saved_layers:
                continue
            g_idx = self._layer_group[ln]
            entries: list[tuple[int, str]] = []
            req_ids: set[str] = set()
            for (h, cand_g), cand in candidates.items():
                if cand_g != g_idx:
                    continue
                entries.append((cand["gpu"], h))
                req_ids |= cand["req_ids"]
            if not entries:
                continue
            tasks = self._submit_group_layers_save(g_idx, [ln], entries)
            for rid in req_ids:
                self._current_put_tasks.setdefault(rid, []).append(tasks)
            self._saved_layers.add(ln)
            self._step_save_pages += len(entries)
            # Finalized by its own write; no second phase.
        if self._step_save_pages:
            # Counterpart of the start_load_kv line: without it a run
            # that saves nothing looks exactly like a healthy one.
            logger.info(
                "chunk_save: %d pages submitted, %d boundaries "
                "(rank %d/%d)", self._step_save_pages, nbound,
                self.rank, self.tp_size)
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
