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

    block_hashes holds the offered boundaries' keys -- one per
    boundary, shared by every group (hashes belong to the token
    sequence, not to a group). group_block_ids[g] is the FULL offer
    width for every group: an attention group carries one page per
    boundary; a mamba group carries its state column per boundary
    with 0 where no column is real (never-materialized middles of
    multi-boundary chunks). The worker pairs positionally and drops
    the 0 slots."""
    block_hashes: tuple[str, ...] = ()
    group_block_ids: tuple[tuple[int, ...], ...] = ()
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class ReqGroupState:
    """Per-group mutable state for one request (scheduler side).

    block_ids: this group's block table. Save progress itself is NOT
    per group: attention's save cursor IS save_watermark, and mamba
    consumes the same offer."""
    block_ids: list[int] = field(default_factory=list)


@dataclass
class ReqState:
    # A reference to the vLLM request object's block_hashes list:
    # live_block_hashes[i] names block i, and is what every plan
    # addresses. On decode, vLLM appends a new hash to it in place for
    # every completed block (there is no callback); preemption never
    # truncates it. Append-only and content-addressed, so reading it
    # directly at any point yields the same hashes (plus any newer
    # ones) a copied snapshot would have held -- which is why we hold
    # the reference instead of tracking our own copy.
    live_block_hashes: list = field(default_factory=list)
    num_computed_tokens: int = 0
    groups: tuple[ReqGroupState, ...] = ()
    # ---- Layer 1: token/hash space, group-agnostic ----
    # Number of boundaries OFFERED for saving so far -- and it IS the
    # attention group's save cursor: attention consumes every offered
    # boundary (one page each) in the same pass, so its frontier is
    # the watermark read before the raise, not a separate variable.
    # build_save_meta advances it to the token frontier (min of credit
    # and keyed-hash count) and caps it when credit rolls back (the
    # only live trigger is pure-attention spec rejection -- GDN
    # refuses spec at init, where the cap is a structural no-op).
    # The mamba groups need no cursor of their own: their dedup is
    # the same "frontier moved" trigger (end > start), and the
    # restore skip is structural (the tail rule only ever targets
    # the newest boundary).
    save_watermark: int = 0
    # This request's load plan, built in update_state_after_alloc from
    # the block objects the engine hands over there, and handed out
    # once by build_connector_meta. None = nothing to restore.
    load_plan: Optional[ReqMeta] = None
    # Async load decision, made in get_num_new_matched_tokens and
    # consumed when the plan is built: while is_async, the request is
    # parked and its plan ships from build_connector_meta.
    is_async: bool = False
    async_load_layers: int = -1


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

    Requires every group to share one block size. That is a limit of
    this connector, not of vLLM: v0.23 supports heterogeneous groups
    and schedules on lcm(group block sizes) (kv_cache_utils
    get_block_size_config), e.g. DeepSeek V4 mixes 256/64/4. Our plans
    address a single block size throughout, so anything else fails
    closed here rather than mis-addressing later."""
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
            f"{sorted(sizes)}, which this connector does not support "
            "(every plan addresses one block size). vLLM itself allows "
            "them -- it schedules on lcm(group block sizes) -- so this "
            "is our limit, not a broken config.")
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
        # The block size every group shares (parse_kv_cache_config
        # refuses anything else). Every plan here indexes the engine's
        # block hashes at this granularity.
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

    def sync_running_request(
        self, req_id: str, new_block_ids: tuple[list[int], ...],
        resumed: bool, num_computed_tokens: int,
    ) -> None:
        """Every pass, for each running request: pull the engine's
        snapshots into our state -- the credit (Layer 1's input) and
        the per-group block tables (Layer 2's maps). No save decisions
        here; build_save_meta owns the watermark lifecycle."""
        state = self._req_states[req_id]
        state.num_computed_tokens = num_computed_tokens
        if new_block_ids:
            for gstate, ids in zip(state.groups, new_block_ids):
                if resumed:
                    # upstream semantics: for resumed requests
                    # new_block_ids IS the table (replace), per group --
                    # including an EMPTY list, which clears stale blocks
                    gstate.block_ids = list(ids)
                else:
                    gstate.block_ids.extend(ids)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """External lookup; returns (hit_tokens, has_async_load)."""
        # This request's block identities, in block order: derived from
        # the engine's own hashes, so "key i names block i" follows
        # from the engine rather than being re-derived
        # (kvshrink-hybrid.md §8).
        state = ReqState(
            live_block_hashes=request.block_hashes,
            num_computed_tokens=num_computed_tokens,
            groups=tuple(ReqGroupState() for _ in self._groups),
        )
        self._req_states[request.request_id] = state
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
            state.live_block_hashes,
            request.num_tokens)
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
        """Record the allocated block tables per group, and -- while the
        engine still hands us block OBJECTS -- build this request's load
        plan from them.

        This is the only callback that sees KVCacheBlocks. Deriving the
        destination slots later, from token counts, would be
        re-answering a question the engine has already answered (and is
        what the in-tree connectors avoid: mooncake keeps the ids from
        here in _reqs_need_recv, the offloading scheduler builds its
        whole TransferJob here)."""
        req_id = request.request_id
        state = self._req_states[req_id]
        start = state.num_computed_tokens // self._block_size
        state.num_computed_tokens += num_external_tokens
        end = state.num_computed_tokens // self._block_size
        for g_idx, ids in enumerate(blocks.get_block_ids()):
            state.groups[g_idx].block_ids = list(ids)
        if num_external_tokens <= 0:
            # vLLM calls this a SECOND time for an async request, once
            # its transfer lands and the request is promoted back out of
            # WAITING_FOR_REMOTE_KVS. That pass can only carry 0: the
            # promotion left num_computed_tokens non-zero (scheduler.py
            # :822), which skips the branch that asks the connector for
            # external tokens, so num_external_computed_tokens keeps its
            # initial 0. Returning here is what stops a second transfer
            # being queued for a request that is RUNNING by then.
            return

        # The restore range's keys, filled once: hashes belong to the
        # token sequence, not to a group.
        hashes = tuple(
            _hash_str(h) for h in state.live_block_hashes[start:end])
        group_ids: list[tuple[int, ...]] = [() for _ in self._groups]
        for g_idx, group in enumerate(self._groups):
            group_blocks = blocks.blocks[g_idx]
            if group.kind == "attention":
                # Layer 2: map the restore range onto this group's
                # pages. The range covers only the external tokens --
                # the core's own prefix-hit blocks already hold their
                # data (shared physical pages).
                group_ids[g_idx] = tuple(
                    b.block_id for b in group_blocks[start:end])
            else:
                # Layer 2: mamba restore slot and worker timing:
                # In vLLM (gpu_model_runner.py:4167 vs 4276), the engine's
                # `preprocess_mamba` runs BEFORE the connector's
                # `start_load_kv`. In `preprocess_mamba`, vLLM copies
                # prev_state_idx ((num_computed - 1) // bs) to curr_state_idx
                # (len(group_blocks) - 1). Because this copy happens before
                # our external load, loading into prev_state_idx would be
                # futile -- the copy has already finished.
                # Therefore, the external snapshot MUST be loaded directly
                # into the running state slot `group_blocks[-1]`
                # (curr_state_idx), where `_model_forward` will execute.
                # The plan spans the offer [start, end) with 0 sentinels
                # in all preceding positions so the worker's positional
                # zip lands `hashes[-1]` (the hit boundary key) squarely
                # onto `group_blocks[-1].block_id`.
                group_ids[g_idx] = tuple(
                    [0] * (end - start - 1) + [group_blocks[-1].block_id])
        # Layer 1: the restored prefix is covered -- the watermark
        # (= attention's save cursor) jumps past it so the first
        # post-restore save pass offers only newly computed
        # boundaries. end is the number of restored blocks. The mamba
        # groups need no jump: with the watermark at `end` the first
        # pass offers nothing new (end == start), so the restored
        # snapshot is not re-put.
        state.save_watermark = max(state.save_watermark, end)
        state.load_plan = ReqMeta(
            block_hashes=hashes,
            group_block_ids=tuple(group_ids),
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )
        if state.is_async:
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
    def _take_load_plan(self, req_id: str) -> ReqMeta:
        """Pop the request's plan: a plan is emitted exactly once per
        allocation, and a later allocation (resume) builds a fresh one."""
        state = self._req_states[req_id]
        plan = state.load_plan
        state.load_plan = None
        return plan

    def build_load_meta(
        self, req: "NewRequestData" | str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Hand out the plan built at alloc time (test helper)."""
        req_id = req.req_id if hasattr(req, "req_id") else req
        return self._take_load_plan(req_id)

    def build_save_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Incremental save plan, in two layers.

        Layer 1 (token/hash space, group-agnostic): how many
        boundaries are real after this forward. `credit` guards
        prefill -- hashes are pre-computed for the whole prompt at
        construction while the pages lag behind; `hashes` guards spec
        decode -- draft tokens inflate the credit but are appended
        only once accepted, so the min always lands on the accepted
        side. The output is a range offer [start, end), where start
        is the watermark read after the brake and before the raise.

        Layer 2 (per group): map the offered range onto this group's
        physical objects. attention consumes the whole range (one page
        per boundary; its cursor IS the watermark -- nothing sits
        between the offer and the consumption); mamba puts every
        boundary in the range whose state column is real -- the same
        per-boundary granularity, with null columns of multi-boundary
        chunks skipped (their states are never materialized). Both
        share one trigger: a non-empty offer (end > start) -- the
        dedup is derived from the frontier movement, not stored.

        The worker executes the plan after forward, when the GPU pages
        hold state up to computed+scheduled tokens. A partial boundary
        (not all layers of the group) is never emitted."""
        state = self._req_states[req_id]
        # Layer 1: the frontier, and the rollback brake, in one place.
        # Sequence inside a pass: brake, read, raise. The brake
        # rewinds the watermark to the frontier the engine last
        # credits -- a no-op in normal flow (the previous pass's
        # progress IS this pass's credit snapshot) and the only brake
        # when credit rolls back (pure-attention spec rejection: the
        # rolled-back boundaries get re-emitted with the recomputed
        # pages; store overwrite is idempotent). Reading it as `start`
        # right after the brake makes the watermark serve as the
        # attention cursor -- no separate per-group variable. Same
        # recompute-from-arithmetic guarantee as the offloading
        # connector's advance_stored_idx. The brake MUST land before
        # the raise -- the raise would otherwise erase the rewind.
        # current_token_block: the index of the token block currently
        # being computed (= the count of complete boundaries).
        current_token_block = (
            state.num_computed_tokens // self._block_size)
        state.save_watermark = min(
            state.save_watermark, current_token_block)
        start = state.save_watermark
        end = min((state.num_computed_tokens + scheduled_tokens)
                  // self._block_size,
                  len(state.live_block_hashes))
        state.save_watermark = max(state.save_watermark, end)
        # The offer's keys, filled once: hashes belong to the token
        # sequence, not to a group. Both group kinds consume the SAME
        # offer [start, end) -- the rollback brake re-opens it after a
        # credit rollback so the recomputed pages/states overwrite the
        # bad store copies.
        hashes = tuple(
            _hash_str(h) for h in state.live_block_hashes[start:end])
        # Layer 2: map the offered range [start, end) onto each group's
        # block table. If start == end, the slices are empty.
        group_ids = tuple(
            tuple(g.block_ids[start:end]) for g in state.groups)
        return ReqMeta(
            block_hashes=hashes,
            group_block_ids=group_ids,
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Assemble this pass's load/save plans."""
        meta = KVShrinkConnectorMetadata()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            if self._req_states[new_req.req_id].load_plan is not None:
                req_meta = self._take_load_plan(new_req.req_id)
                meta.reqs_to_load.add_request(
                    new_req.req_id, req_meta.block_hashes,
                    req_meta.group_block_ids, req_meta.is_async,
                    req_meta.async_load_layers)
            save_meta = self.build_save_meta(
                new_req.req_id, num_sched[new_req.req_id])
            if save_meta.block_hashes:
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
            req_meta = self._take_load_plan(req_id)
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
            if self._req_states[req_id].load_plan is not None:
                req_meta = self._take_load_plan(req_id)
                meta.reqs_to_load.add_request(
                    req_id, req_meta.block_hashes,
                    req_meta.group_block_ids, req_meta.is_async,
                    req_meta.async_load_layers)

        # Running requests cross boundaries in later steps too (chunked
        # prefill tails, decode-time crossings). The two calls are a
        # LOAD-BEARING PAIR, in this order: the rollback brake inside
        # build_save_meta must observe the credit sync_running_request
        # just pulled in, and it must land on the group cursors before
        # the watermark raise -- reordering or splitting them silently
        # drops the re-emission of rolled-back boundaries.
        resumed = cr.resumed_req_ids
        new_bids = cr.new_block_ids
        ncts = cr.num_computed_tokens
        for i, req_id in enumerate(cr.req_ids):
            self.sync_running_request(
                req_id, new_bids[i], req_id in resumed, ncts[i])
            save_meta = self.build_save_meta(
                req_id, num_sched[req_id])
            if save_meta.block_hashes:
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
            sync_block_ids: list[int] = []
            sync_block_hashes: list[str] = []
            for req_id, req_meta in metadata.reqs_to_load.requests.items():
                if req_meta.is_async:
                    continue
                gids = req_meta.group_block_ids[g_idx]
                if not gids:
                    continue
                # Positional pairing against the shared key list; a 0
                # slot (no real state column) drops out here.
                entries = tuple(
                    (gid, h) for gid, h in zip(gids, req_meta.block_hashes)
                    if gid != 0)
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
            for req_id, req_meta in metadata.reqs_to_load.requests.items():
                if not req_meta.is_async:
                    continue
                gids = req_meta.group_block_ids[g_idx]
                if not gids:
                    continue
                entries = tuple(
                    (gid, h) for gid, h in zip(gids, req_meta.block_hashes)
                    if gid != 0)
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
        shape as main's save_kv_layer loop, plus the group label)."""
        g_idx = self._layer_group[ln]
        for req_id, req_meta in metadata.reqs_to_save.requests.items():
            gids = req_meta.group_block_ids[g_idx]
            # Positional pairing against the shared key list; a 0 slot
            # (no real state column) drops out here.
            pairs = [(gid, h) for gid, h in
                     zip(gids, req_meta.block_hashes) if gid != 0]
            if not pairs:
                continue
            tasks = self._store().put(
                block_indices=[g for g, _ in pairs],
                block_hashs=[h for _, h in pairs],
                layer_names=[ln], label=f"g{g_idx}")
            self._current_put_tasks.setdefault(req_id, []).append(tasks)
            self._step_save_pages += len(pairs)
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
                # Recurrent layers union the first-N attention prefix.
                # All recurrent (Mamba/GDN) layers MUST be waited here before
                # releasing the request: vLLM's `preprocess_mamba` runs once
                # per batch in `gpu_model_runner.py` before any layer forward,
                # and Mamba layers have no per-layer forward hooks.
                # Therefore, only attention layers can be streamed in the
                # background and waited on-demand via `wait_for_layer_load`.
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
