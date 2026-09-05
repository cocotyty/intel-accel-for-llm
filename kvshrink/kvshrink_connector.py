# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

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
    next_block_to_save: int = 0


@dataclass
class ReqState:
    # A reference to the vLLM request object's block_hashes list.
    # On decode, vLLM appends a new hash to it in place for every
    # completed block (there is no callback); preemption never
    # truncates it. Append-only and content-addressed, so reading it
    # directly at any point yields the same hashes (plus any newer
    # ones) a copied snapshot would have held.
    live_block_hashes: list = field(default_factory=list)
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
# hybrid layout vocabulary
# ======================================================================
@dataclass(frozen=True)
class GroupInfo:
    """One vLLM KV cache group: a frozen snapshot of its storage
    contract (kind, layers). The block size is shared by every group
    and lives on the connector, not here."""

    group_idx: int
    kind: str  # "attention" | "mamba"
    layer_names: tuple[str, ...]
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
def parse_kv_cache_config(
    kv_cache_config: KVCacheConfig,
) -> tuple[list[GroupInfo], int]:
    """One GroupInfo per vLLM KV cache group, plus the block size.
    Per-layer geometry is not parsed: KVStore binds pools from the
    live tensors, which carry their own layout.

    v0.23 pads hybrid models to a single block size at startup
    (_align_hybrid_block_size): every group shares it, so a mismatch
    here is a broken vLLM contract and fails closed."""
    groups: list[GroupInfo] = []
    sizes: set[int] = set()
    for g_idx, g in enumerate(kv_cache_config.kv_cache_groups):
        spec = g.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs):
            spec = spec.kv_cache_specs[g.layer_names[0]]
        kind = "mamba" if isinstance(spec, MambaSpec) else "attention"
        sizes.add(int(spec.block_size))
        groups.append(GroupInfo(
            group_idx=g_idx,
            kind=kind,
            layer_names=tuple(g.layer_names),
            spec=spec,
        ))
    if len(sizes) != 1:
        raise RuntimeError(
            "kvshrink hybrid: groups have mismatched block sizes "
            f"{sorted(sizes)}; v0.23 pads hybrid models to one block "
            "size at startup, so this cannot be a valid config")
    return groups, sizes.pop()


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

        pc = vllm_config.parallel_config
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
        groups, block_size = parse_kv_cache_config(kv_cache_config)
        self._groups = groups
        # The one block size every group shares (v0.23 pads hybrid
        # models to it at startup); the engine's block hashes are
        # generated at exactly this granularity.
        self._block_size = block_size

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

        # Attention layer count, used to clamp the async early-start
        # prefix. Mamba layers are deliberately absent: they are never
        # partially released.
        self._num_attn_layers = sum(
            len(g.layer_names) for g in groups if g.kind != "mamba")
        if role != KVConnectorRole.SCHEDULER:
            # layer_name -> group idx, every cached layer.
            self._layer_group = {
                ln: g.group_idx for g in groups for ln in g.layer_names}

        logger.info(
            "kvshrink hybrid path enabled (%s role, tp=%d rank=%d, "
            "block_size=%d, groups=%s)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            self.tp_size, self.rank, self._block_size,
            [(g.group_idx, g.kind) for g in groups])

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
        tables and hashes from upstream; on resume (or progress
        regression) roll the save cursor back so boundaries emitted
        before a preemption get re-emitted (overwrite is idempotent)."""
        state = self._req_states[req_id]
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
                safe = safe_n // self._block_size
                if gstate.next_block_to_save > safe:
                    gstate.next_block_to_save = safe
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
        self._req_states[request.request_id] = ReqState(
            live_block_hashes=request.block_hashes,
            num_computed_tokens=num_computed_tokens,
            groups=tuple(ReqGroupState() for _ in self._groups),
        )
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        policy = HybridHitPolicy(
            self._groups,
            lambda g, h: self._store().has(
                [_hash_str(h)], label=f"g{g}")[0],
            self._block_size, num_computed_tokens)
        # Restorable boundary in tokens; 0 = miss. The policy already
        # gated on live chunk presence (engine Record), so a nonzero
        # boundary is complete by construction; only record it.
        boundary = policy.find_longest_cache_hit(
            request.block_hashes,
            request.num_tokens)
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
            if selected < 0 or selected > self._num_attn_layers:
                state.async_load_layers = -1  # require every layer
            else:
                state.async_load_layers = selected
        logger.debug(
            "req=%s external_hit=%d boundary=%d async=%s",
            request.request_id, external, boundary, use_async)
        return external, use_async

    # ------------------------------------------------------------------
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record the allocated block tables per group and the
        external-token count the core accepted (drives the load
        plan). For async requests this is also where the load plan
        is emitted, since a parked request never appears in
        build_connector_meta."""
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
        them in scheduled_cached_reqs.resumed_req_ids, not in
        scheduled_new_reqs, so they need their own loop."""
        state = self._req_states[req_id]
        meta = self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)
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
        """Build one request's load plan from its recorded state. The
        caller only calls this for requests that accepted external
        tokens (state.pending_load_tokens > 0)."""
        ext = state.pending_load_tokens
        nc = state.num_computed_tokens
        hashes: list[str] = []
        group_ids: list[tuple[int, ...]] = [() for _ in self._groups]
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
                start = (nc - ext) // self._block_size
                end = nc // self._block_size
                if group is owner:
                    hashes = [_hash_str(state.live_block_hashes[i])
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
                            - 1) // self._block_size

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
                    idx = nc // self._block_size - 1
                    hashes = [_hash_str(state.live_block_hashes[idx])]
                group_ids[g_idx] = (ids[curr_idx],)
        # The plan reads every group's hit range back from the store;
        # those blocks are already there, so the save cursors skip
        # them instead of re-writing them on the first post-restore
        # save pass (main's existence_cache did this job with a
        # presence snapshot; ours is the restore event itself).
        # For attention, nc//bs is the number of hit blocks; for
        # mamba, the credit landed nc exactly on a boundary, so
        # nc//bs is the slot past that boundary's snapshot.
        skip_to = nc // self._block_size
        for gstate in state.groups:
            if gstate.next_block_to_save < skip_to:
                gstate.next_block_to_save = skip_to
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
        emit the boundaries completed since the last pass. The
        worker executes it after forward, when the GPU pages hold
        state up to computed+scheduled tokens. A partial boundary
        (not all layers of the group) is never emitted."""
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
                num_hash = min(progress // self._block_size, len(ids),
                               len(state.live_block_hashes))
                start = gstate.next_block_to_save
                if num_hash > start:
                    if group is owner:
                        hashes = [_hash_str(state.live_block_hashes[i])
                                  for i in range(start, num_hash)]
                    group_ids[g_idx] = tuple(ids[i] for i in
                                             range(start, num_hash))
                    gstate.next_block_to_save = num_hash
            else:
                # Save the running-state block. align mode pins the
                # kernel to the table's tail column, and the hash list
                # only grows when a block completes, so the newest
                # completed boundary is always len(live)-1 -- no
                # progress arithmetic, no boundary-timing window. The
                # prev column holds the boundary state for the whole
                # next block cycle (released only when the following
                # block completes), so a save that runs a few steps
                # late still reads the right slot. A multi-boundary
                # pass physically materialises only the tail state;
                # the skipped middle boundaries get no entry (same
                # semantics as vLLM's own offloading connector), and
                # the cursor jumps to cover them.
                idx = len(state.live_block_hashes) - 1
                if idx >= gstate.next_block_to_save:
                    if idx >= len(ids):
                        # Resumed request: the engine's hash list is
                        # longer than the freshly replaced table. The
                        # physical slot does not exist yet -- the table
                        # grows back as forward crosses boundaries, so
                        # just wait (the cursor stays put and the save
                        # happens once the slot is real).
                        continue
                    if ids[idx] == 0:
                        raise RuntimeError(
                            "kvshrink mamba save: boundary column "
                            f"is NULL (req={req_id} "
                            f"progress={progress} "
                            f"table_idx={idx} table={ids})")
                    if group is owner:
                        hashes = [_hash_str(state.live_block_hashes[idx])]
                    group_ids[g_idx] = (ids[idx],)
                    gstate.next_block_to_save = idx + 1
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
            if self._req_states[new_req.req_id].pending_load_tokens > 0:
                req_meta = self.build_load_meta(
                    new_req, num_sched[new_req.req_id])
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

        # Load plans for requests vLLM parked (async loads never appear
        # in the scheduler output), drained exactly once. Every parked
        # request was queued with external tokens accepted, so its plan
        # always carries pages.
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
            if self._req_states[req_id].pending_load_tokens > 0:
                req_meta = self.build_resumed_load_meta(
                    req_id, num_sched[req_id])
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
        order is used, by the async release gate."""
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
            ln for ln in execution_order
            if ln not in self._mamba_layers)
        # Save pipelining segments: the mamba layers between attention
        # layer i-1 and attention layer i are final when i's save hook
        # fires, so they ride that hook (the trailing segment is
        # submitted by wait_for_save).
        segments: dict[str, tuple[str, ...]] = {}
        pending: list[str] = []
        for ln in execution_order:
            if ln in self._mamba_layers:
                pending.append(ln)
            elif pending:
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
        # This is the first worker call of the step, so the save
        # bookkeeping resets here -- before any save hook can fire.
        self._current_get_tasks = None
        self._saved_layers = set()
        self._step_save_pages = 0
        npages = 0
        _t0 = time.monotonic()
        # One engine get per layer, in execution order: the engine
        # stream runs transfers FIFO, so submission order must be the
        # order forward consumes the layers. Synchronous (blocking)
        # loads go first -- this pass's forward cannot start without
        # them, while a parked request's pages are not needed now.
        sync_tasks: dict[str, Task] = {}
        for ln in self._layer_names:
            g_idx = self._layer_group[ln]
            mamba = self._groups[g_idx].kind == "mamba"
            sync_block_ids: list[int] = []
            sync_block_hashes: list[str] = []
            for req_id, req_meta in metadata.reqs_to_load.requests.items():
                if req_meta.is_async:
                    continue
                gids = req_meta.group_block_ids[g_idx]
                if not gids:
                    continue
                # A mamba entry is the plan's last hash into its single
                # CURR block; an attention group pairs every block with
                # its hash.
                entries = ((gids[0], req_meta.block_hashes[-1]),) if mamba \
                    else tuple(zip(gids, req_meta.block_hashes))
                sync_block_ids.extend(gpu for gpu, _ in entries)
                sync_block_hashes.extend(h for _, h in entries)
            if sync_block_ids:
                npages += len(sync_block_ids)
                sync_tasks.update(self._store().get(
                    block_indices=sync_block_ids,
                    block_hashs=sync_block_hashes,
                    layer_names=[ln], label=f"g{g_idx}"))
        if sync_tasks:
            self._current_get_tasks = sync_tasks
        # Asynchronous loads per request, in the same layer-major
        # order; tasks land in the per-request dicts get_finished
        # polls for parked requests.
        async_tasks: dict[str, dict[str, Task]] = {}
        for ln in self._layer_names:
            g_idx = self._layer_group[ln]
            mamba = self._groups[g_idx].kind == "mamba"
            for req_id, req_meta in metadata.reqs_to_load.requests.items():
                if not req_meta.is_async:
                    continue
                gids = req_meta.group_block_ids[g_idx]
                if not gids:
                    continue
                entries = ((gids[0], req_meta.block_hashes[-1]),) if mamba \
                    else tuple(zip(gids, req_meta.block_hashes))
                npages += len(entries)
                async_tasks.setdefault(req_id, {}).update(
                    self._store().get(
                        block_indices=[gpu for gpu, _ in entries],
                        block_hashs=[h for _, h in entries],
                        layer_names=[ln], label=f"g{g_idx}"))
        for req_id, tasks in async_tasks.items():
            self._pending_load_tasks[req_id] = tasks
            self._pending_load_layers[req_id] = (
                metadata.reqs_to_load.requests[req_id].async_load_layers)
        # Every recurrent layer, waited before forward begins (main's
        # layer filter reused: these layers have no forward hook).
        recurrent = [ln for ln in sync_tasks if ln in self._mamba_layers]
        if recurrent:
            if not self._store().get_wait(
                    get_results=sync_tasks, layer_names=recurrent, wait=True):
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
        # main's hook verbatim: wait this layer's pages in the sync
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
    def _save_layer(
        self, ln: str, metadata: KVShrinkConnectorMetadata
    ) -> None:
        """One async engine put per request for layer ``ln`` (same
        shape as main's save_kv_layer loop, plus the group label and
        the mamba slot rule)."""
        g_idx = self._layer_group[ln]
        mamba = self._groups[g_idx].kind == "mamba"
        for req_id, req_meta in metadata.reqs_to_save.requests.items():
            gids = req_meta.group_block_ids[g_idx]
            if not gids:
                continue
            # A mamba entry is the plan's last hash into its single
            # CURR block; an attention group pairs every block with
            # its hash.
            if mamba:
                block_ids = [gids[0]]
                block_hashes = [req_meta.block_hashes[-1]]
            else:
                block_ids = list(gids)
                block_hashes = list(req_meta.block_hashes)
            tasks = self._store().put(
                block_indices=block_ids,
                block_hashs=block_hashes,
                layer_names=[ln], label=f"g{g_idx}")
            self._current_put_tasks.setdefault(req_id, []).append(tasks)
            self._step_save_pages += len(block_ids)
        self._saved_layers.add(ln)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Submit this attention layer's pages plus the mamba segment
        before it (their kernels already ran, so the data is final).
        The trailing segment goes out in wait_for_save. Submission
        only; the drain lives in get_finished."""
        if self._connector_metadata is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")
        for ln in self._mamba_save_segments.get(layer_name, ()):
            self._save_layer(ln, metadata)
        self._save_layer(layer_name, metadata)

    def wait_for_save(self) -> None:
        """Submit every layer no hook covered (the trailing mamba
        segment), then log the step's save volume. Submission only;
        the drain lives in get_finished."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")
        for ln in self._layer_names:
            if ln not in self._saved_layers:
                self._save_layer(ln, metadata)
        if self._step_save_pages:
            # Counterpart of the start_load_kv line: without it a run
            # that saves nothing looks exactly like a healthy one.
            nbound = sum(
                len(r.block_hashes)
                for r in metadata.reqs_to_save.requests.values())
            logger.info(
                "chunk_save: %d pages submitted, %d boundaries "
                "(rank %d/%d)", self._step_save_pages, nbound,
                self.rank, self.tp_size)

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
            while tasks and self._store().put_wait(tasks[0], wait=False):
                tasks.pop(0)
            if not tasks:
                self._current_put_tasks.pop(req_id)
                completed.add(req_id)

        self._deferred_finished_req_ids.difference_update(completed)
        return (completed or None), (finished_recving or None)
