# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from __future__ import annotations

import logging
import os
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
    """Per-request transfer plan: block_hashes across all groups, and
    group_block_ids per group (with 0 sentinels for recurrent chunk middles)."""
    block_hashes: tuple[str, ...] = ()
    group_block_ids: tuple[tuple[int, ...], ...] = ()
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class ReqGroupState:
    """Per-group mutable block table for one request (scheduler side)."""
    block_ids: list[int] = field(default_factory=list)


@dataclass
class ReqState:
    # Reference to the vLLM request's block_hashes list.
    live_block_hashes: list = field(default_factory=list)
    num_computed_tokens: int = 0
    num_prompt_tokens: int = 0
    groups: tuple[ReqGroupState, ...] = ()
    # Highest boundary index (in blocks) offered for saving so far.
    save_watermark: int = 0
    # True if the request is parked waiting for background async KV transfer.
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
    """Snapshot of a vLLM KV cache group storage contract (kind, layers)."""
    group_idx: int
    kind: str  # "attention" | "mamba"
    layer_names: tuple[str, ...]
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
    """Parse vLLM KV cache groups and common block size.
    All groups must share the same block size."""
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


# ======================================================================
# Connector
# ======================================================================

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
        self._reqs_to_load = RequestMetadata()
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

        groups, block_size = parse_kv_cache_config(kv_cache_config)
        self._groups = groups
        # Common block size across all groups.
        self._block_size = block_size

        for g in groups:
            if g.kind == "mamba" and getattr(g.spec, "num_speculative_blocks", 0) > 0:
                logger.info(
                    "kvshrink: mamba speculative decoding enabled with "
                    "num_speculative_blocks=%d (running in prefill-only save mode)",
                    g.spec.num_speculative_blocks,
                )

        # Attention layer count used to clamp the async early-start prefix.
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

    # --- Scheduler Side Methods ---

    def sync_running_request(
        self, req_id: str, new_block_ids: tuple[list[int], ...],
        resumed: bool, num_computed_tokens: int,
    ) -> None:
        """Pull the engine's block tables and computed token count
        into our scheduler state for running prefill chunks."""
        state = self._req_states[req_id]
        state.num_computed_tokens = num_computed_tokens
        if new_block_ids:
            for gstate, ids in zip(state.groups, new_block_ids):
                if resumed:
                    gstate.block_ids = list(ids)
                else:
                    gstate.block_ids.extend(ids)

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """External lookup; returns (hit_tokens, has_async_load)."""
        # Initialize scheduler state for this new request.
        num_prompt = getattr(request, "num_prompt_tokens", 0) or getattr(request, "num_tokens", 0)
        state = ReqState(
            live_block_hashes=request.block_hashes,
            num_computed_tokens=num_computed_tokens,
            num_prompt_tokens=num_prompt,
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
        # Find the longest prefix match across all groups (0 = miss).
        boundary = policy.find_longest_cache_hit(
            state.live_block_hashes,
            request.num_tokens)
        external = max(0, boundary - num_computed_tokens)
        # Stream asynchronously if configured and there are tokens to load.
        use_async = external > 0 and self._async_load_layer_config is not None
        if use_async:
            selected = self._async_load_layer_config.select(
                len(self._req_states))
            use_async = selected != 0
        if use_async:
            state.is_async = True
            # Clamp layer count to available attention layers.
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
        """Record allocated block tables per group and build the load plan
        from the block objects provided by the engine."""
        req_id = request.request_id
        state = self._req_states[req_id]
        start = state.num_computed_tokens // self._block_size
        state.num_computed_tokens += num_external_tokens
        end = state.num_computed_tokens // self._block_size
        for g_idx, ids in enumerate(blocks.get_block_ids()):
            state.groups[g_idx].block_ids = list(ids)
        if num_external_tokens <= 0:
            # Second callback after async promotion carries 0 external tokens; skip.
            return

        # The restore range's keys, filled once: hashes belong to the
        # token sequence, not to a group.
        hashes = tuple(
            _hash_str(h) for h in state.live_block_hashes[start:end])
        group_ids: list[tuple[int, ...]] = [() for _ in self._groups]
        for g_idx, group in enumerate(self._groups):
            group_blocks = blocks.blocks[g_idx]
            if group.kind == "attention":
                # Map the restore range directly onto the allocated attention blocks.
                group_ids[g_idx] = tuple(
                    b.block_id for b in group_blocks[start:end])
            else:
                # Restore Mamba state directly into the execution slot (-1 - num_spec),
                # prepending 0 sentinels so hashes[-1] aligns with the target block ID.
                num_spec = getattr(group.spec, "num_speculative_blocks", 0)
                target_block = group_blocks[-1 - num_spec]
                group_ids[g_idx] = tuple(
                    [0] * (end - start - 1) + [target_block.block_id])
        # Advance watermark past restored prefix so loaded blocks are not re-saved.
        state.save_watermark = max(state.save_watermark, end)
        self._reqs_to_load.add_request(
            req_id,
            block_hashes=hashes,
            group_block_ids=tuple(group_ids),
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Free scheduler state; memory reclamation is handled in get_finished.
        self._req_states.pop(request.request_id, None)
        self._reqs_to_load.requests.pop(request.request_id, None)
        return True, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """SupportsHMA entry point (v0.23 calls this for hybrid models)."""
        return self.request_finished(request, [])

    # ------------------------------------------------------------------
    def build_save_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Build incremental save plan for newly computed prefill blocks
        in range [save_watermark, end)."""
        state = self._req_states[req_id]

        # Prefill-only save policy: decode steps never produce saves.
        if (state.num_prompt_tokens > 0 and state.num_computed_tokens >= state.num_prompt_tokens) or scheduled_tokens <= 1:
            return ReqMeta(group_block_ids=tuple(() for _ in self._groups))

        # Roll back watermark if preemption occurred, then compute new boundary.
        current_token_block = (
            state.num_computed_tokens // self._block_size)
        state.save_watermark = min(
            state.save_watermark, current_token_block)
        start = state.save_watermark
        end = min((state.num_computed_tokens + scheduled_tokens)
                  // self._block_size,
                  len(state.live_block_hashes))
        state.save_watermark = max(state.save_watermark, end)
        # Collect block hashes for the [start, end) range.
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
        meta = KVShrinkConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_save=RequestMetadata(),
        )
        self._reqs_to_load = RequestMetadata()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            save_meta = self.build_save_meta(
                new_req.req_id, num_sched[new_req.req_id])
            if save_meta.block_hashes:
                meta.reqs_to_save.add_request(
                    new_req.req_id, save_meta.block_hashes,
                    save_meta.group_block_ids)

        cr = scheduler_output.scheduled_cached_reqs
        resumed = cr.resumed_req_ids
        new_bids = cr.new_block_ids
        ncts = cr.num_computed_tokens
        for i, req_id in enumerate(cr.req_ids):
            sched_toks = num_sched[req_id]
            if sched_toks <= 1:
                # Prefill-only save policy: decode steps never produce saves.
                continue
            self.sync_running_request(
                req_id, new_bids[i], req_id in resumed, ncts[i])
            save_meta = self.build_save_meta(
                req_id, sched_toks)
            if save_meta.block_hashes:
                meta.reqs_to_save.add_request(
                    req_id, save_meta.block_hashes,
                    save_meta.group_block_ids)
        return meta

    # --- Worker Side Methods ---

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        from vllm.model_executor.models.utils import extract_layer_index

        execution_order = sorted(kv_caches, key=extract_layer_index)
        self._layer_names = list(execution_order)
        self._mamba_layers = frozenset(
            ln for g in self._groups if g.kind == "mamba"
            for ln in g.layer_names)
        self._attn_order = tuple(
            ln for ln in execution_order
            if ln not in self._mamba_layers)
        segments: dict[str, tuple[str, ...]] = {}
        pending: list[str] = []
        for ln in execution_order:
            if ln in self._mamba_layers:
                pending.append(ln)
            elif pending:
                segments[ln] = tuple(pending)
                pending = []
        self._mamba_save_segments = segments
        self._last_layer_name = self._attn_order[-1] if self._attn_order else None

        # The store binds the RAW kv_caches directly.
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            kv_caches=kv_caches,
            rank=self.rank,
            tp_size=self.tp_size,
        )
        logger.info(
            "Registered %d KV cache layers (%d attention, %d recurrent)",
            len(execution_order), len(self._attn_order), len(self._mamba_layers))

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

        # Submit step loads, wait recurrent layers, and reset save bookkeeping.
        self._current_get_tasks = None
        self._saved_layers = set()
        if metadata.reqs_to_load.requests:
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
                    sync_tasks.update(self._store().get(
                        block_indices=sync_block_ids,
                        block_hashs=sync_block_hashes,
                        layer_names=[ln], label=f"g{g_idx}"))
            if sync_tasks:
                self._current_get_tasks = sync_tasks
            # Asynchronous loads per request, polled in get_finished.
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
        return 

    def wait_for_layer_load(self, layer_name: str) -> None:
        # Wait this layer's pages in sync batch and in promoted async loads.
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
        self._saved_layers.add(ln)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Submit attention layer pages plus preceding recurrent layers."""
        if self._connector_metadata is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata) or not metadata.reqs_to_save.requests:
            return
        for ln in self._mamba_save_segments.get(layer_name, ()):
            self._save_layer(ln, metadata)
        self._save_layer(layer_name, metadata)

    def wait_for_save(self) -> None:
        """Submit every layer no forward hook covered (recurrent/mamba layers)."""
        if self._connector_metadata is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata) or not metadata.reqs_to_save.requests:
            return
        for ln in self._layer_names:
            if ln not in self._saved_layers:
                self._save_layer(ln, metadata)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # Poll async load tasks: recurrent layers must land before releasing.
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
                # Recurrent layers and first N attention layers must complete
                # before releasing the request to forward.
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
                    # Early-promote request once gate layers finish loading.
                    self._early_promoted_tasks[req_id] = tasks
                    finished_recving.add(req_id)

        self._deferred_finished_req_ids.update(finished_req_ids)
        completed: set[str] = set()
        for req_id in self._deferred_finished_req_ids:
            # Drain remaining async loads for finished requests.
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
