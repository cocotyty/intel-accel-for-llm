# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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

from .hybrid_hit import HybridHitPolicy
from .async_load_config import load_async_load_layer_config_from_env

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str


@dataclass
class ReqMeta:
    group_block_ids: tuple[tuple[int, ...], ...] = ()
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class ReqGroupState:
    """Per-group mutable block table for one request (scheduler side)."""
    block_ids: list[int] = field(default_factory=list)


@dataclass
class ReqState:
    num_computed_tokens: int = 0
    # Reference to the vLLM request's block_hashes list.
    block_hashes: list = field(default_factory=list)
    num_prompt_tokens: int = 0
    groups: tuple[ReqGroupState, ...] = ()
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

    def add_request(
        self,
        req_id: ReqId,
        group_block_ids: tuple[tuple[int, ...], ...],
        block_hashes: list[str],
        is_async: bool = False,
        async_load_layers: int = -1,
    ) -> None:
        self.requests[req_id] = ReqMeta(
            group_block_ids,
            block_hashes,
            is_async,
            async_load_layers,
        )


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: RequestMetadata
    reqs_to_save: RequestMetadata


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


class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
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
        self._reqs_to_save = RequestMetadata()
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
        self.block_size = block_size

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
            self.tp_size, self.rank, self.block_size,
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

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        if self._req_states.pop(request.request_id, None) is not None:
            logger.warning("Discarded stale state for request %s", request.request_id)

        num_prompt = getattr(request, "num_prompt_tokens", 0) or getattr(request, "num_tokens", 0)
        state = ReqState(
            num_computed_tokens=num_computed_tokens,
            block_hashes=request.block_hashes,
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
            self.block_size, num_computed_tokens)
        matched_tokens = policy.find_longest_cache_hit(
            state.block_hashes, request.num_tokens
        )
        num_new_tokens = max(0, matched_tokens - num_computed_tokens)

        # Decide sync vs async for this request. The load can only be async when
        # there are external tokens to load and async is enabled. Concurrency is
        # approximated by the number of in-flight requests (this one included).
        selected_layers = self._async_load_layer_config.select(
            len(self._req_states)
        )
        # A dynamic-map layer value of 0 selects synchronous loading. It is not
        # an async request that resumes before layer 0.
        use_async = num_new_tokens > 0 and selected_layers != 0
        state.is_async = use_async
        if use_async:
            state.async_load_layers = selected_layers
            # Clamp layer count to available attention layers.
            if selected_layers > self._num_attn_layers:
                state.async_load_layers = -1

        logger.info(
            f"get_num_new_matched_tokens, req-{request.request_id}, "
            f"externally-cached tokens: {num_new_tokens}, "
            f"locally-cached tokens: {num_computed_tokens}, async={use_async}, "
            f"selected_load_layers={selected_layers}, "
            f"async_load_layers={state.async_load_layers}"
        )
        return num_new_tokens, use_async

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        state = self._req_states.get(request.request_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {request.request_id}")

        block_ids = blocks.get_block_ids()
        for g_idx, ids in enumerate(block_ids):
            state.groups[g_idx].block_ids = list(ids)
        if num_external_tokens == 0:
            return
        if num_external_tokens % self.block_size != 0:
            raise ValueError("External token count must be block aligned")

        load_start = state.num_computed_tokens // self.block_size
        load_end = min(
            load_start + num_external_tokens // self.block_size,
            len(state.block_hashes),
            *(len(block_ids[g_idx]) for g_idx, group in enumerate(self._groups)
              if group.kind == "attention"),
        )
        if load_end <= load_start:
            return
        group_ids: list[tuple[int, ...]] = [() for _ in self._groups]
        for g_idx, group in enumerate(self._groups):
            group_blocks = block_ids[g_idx]
            if group.kind == "attention":
                group_ids[g_idx] = tuple(
                    group_blocks[load_start:load_end])
            else:
                # Restore Mamba state directly into the execution slot (-1 - num_spec),
                # prepending 0 sentinels so hashes[-1] aligns with the target block ID.
                num_spec = getattr(group.spec, "num_speculative_blocks", 0)
                group_ids[g_idx] = tuple(
                    [0] * (load_end - load_start - 1)
                    + [group_blocks[-1 - num_spec]])
        state.num_computed_tokens += num_external_tokens
        self._reqs_to_load.add_request(
            request.request_id,
            tuple(group_ids),
            [_hash_str(h) for h in state.block_hashes[load_start:load_end]],
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def _add_request_to_save(
        self, req_id: ReqId, scheduled_tokens: int
    ) -> None:
        state = self._req_states.get(req_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {req_id}")

        # Prefill-only save policy: decode steps never produce saves.
        if scheduled_tokens <= 1 or (
            state.num_prompt_tokens > 0
            and state.num_computed_tokens >= state.num_prompt_tokens
        ):
            return

        start = state.num_computed_tokens // self.block_size
        end = min(
            (state.num_computed_tokens + scheduled_tokens) // self.block_size,
            len(state.block_hashes),
        )
        block_hashes = [_hash_str(h) for h in state.block_hashes[start:end]]
        if block_hashes:
            block_ids = tuple(
                tuple(group.block_ids[start:end]) for group in state.groups
            )
            self._reqs_to_save.add_request(req_id, block_ids, block_hashes)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # True = defer freeing to get_finished() (async load/save may still run).
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

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        for request in scheduler_output.scheduled_new_reqs:
            self._add_request_to_save(
                request.req_id, scheduler_output.num_scheduled_tokens[request.req_id]
            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            block_ids = cached_reqs.new_block_ids[index]
            is_prefill = scheduler_output.num_scheduled_tokens[req_id] > 1
            if not is_prefill:
                continue
            state = self._req_states[req_id]
            state.num_computed_tokens = cached_reqs.num_computed_tokens[index]
            if block_ids and req_id not in cached_reqs.resumed_req_ids:
                for group, ids in zip(state.groups, block_ids):
                    group.block_ids.extend(ids)
            self._add_request_to_save(
                req_id, scheduler_output.num_scheduled_tokens[req_id]
            )

        metadata = KVShrinkConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_save=self._reqs_to_save,
        )
        self._reqs_to_load = RequestMetadata()
        self._reqs_to_save = RequestMetadata()
        return metadata

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        from vllm.model_executor.models.utils import extract_layer_index

        execution_order = sorted(
            [ln for ln in kv_caches if extract_layer_index(ln) < self.num_layers],
            key=extract_layer_index,
        )
        self._layer_names = list(execution_order)
        self._mamba_layers = frozenset(
            ln for g in self._groups if g.kind == "mamba"
            for ln in g.layer_names)
        self._attn_order = tuple(
            ln for ln in execution_order
            if ln not in self._mamba_layers)
        first_attention = (execution_order.index(self._attn_order[0])
                           if self._attn_order else len(execution_order))
        self._leading_mamba_layers = execution_order[:first_attention]
        self._async_load_order = [ln for ln in execution_order if ln in self._mamba_layers]
        self._async_load_order.extend(self._attn_order)
        segments: dict[str, tuple[str, ...]] = {}
        pending: list[str] = []
        for ln in execution_order:
            if ln in self._mamba_layers:
                pending.append(ln)
            elif pending:
                segments[ln] = tuple(pending)
                pending = []
        self._mamba_save_segments = segments
        self._mamba_load_segments = {}
        for index, layer_name in enumerate(self._attn_order):
            start = execution_order.index(layer_name) + 1
            end = (execution_order.index(self._attn_order[index + 1])
                   if index + 1 < len(self._attn_order) else len(execution_order))
            self._mamba_load_segments[layer_name] = execution_order[start:end]
        self._last_layer_name = self._attn_order[-1] if self._attn_order else None

        # The store binds base model kv_caches directly.
        base_kv_caches = {ln: kv_caches[ln] for ln in execution_order}
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            kv_caches=base_kv_caches,
            rank=self.rank,
            tp_size=self.tp_size,
        )
        logger.info(
            "Registered %d KV cache layers (%d attention, %d recurrent)",
            len(execution_order), len(self._attn_order), len(self._mamba_layers))

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

        self._saved_layers = set()
        self._current_get_tasks = None
        if not metadata.reqs_to_load.requests:
            return

        sync_reqs: list[tuple[ReqId, ReqMeta]] = []
        async_reqs: list[tuple[ReqId, ReqMeta]] = []
        for req_id, request in metadata.reqs_to_load.requests.items():
            if len(request.group_block_ids) != len(self._groups) or any(
                block_ids and len(block_ids) != len(request.block_hashes)
                for block_ids in request.group_block_ids
            ):
                raise ValueError(f"Mismatched block metadata for request {req_id}")
            if not request.block_hashes:
                continue
            if request.is_async:
                async_reqs.append((req_id, request))
            else:
                sync_reqs.append((req_id, request))

        # Submit synchronous (blocking) loads first as a single merged batch so
        # they are enqueued ahead of the asynchronous loads for this pass.
        sync_tasks: dict[str, Any] = {}
        for layer_name in self._layer_names:
            group_idx = self._layer_group[layer_name]
            sync_block_ids: list[int] = []
            sync_block_hashes: list[str] = []
            for req_id, request in sync_reqs:
                for block_id, block_hash in zip(
                    request.group_block_ids[group_idx], request.block_hashes
                ):
                    if block_id != 0:
                        sync_block_ids.append(block_id)
                        sync_block_hashes.append(block_hash)
            if sync_block_ids:
                sync_tasks.update(self._store().get(
                    block_indices=sync_block_ids,
                    block_hashs=sync_block_hashes,
                    layer_names=[layer_name],
                    label=f"g{group_idx}",
                ))
        if sync_tasks:
            self._current_get_tasks = sync_tasks

        # Submit asynchronous loads per request; they are polled across
        # scheduler steps in get_finished().
        async_tasks: dict[ReqId, dict[str, Any]] = {}
        # All Mamba prev blocks must land before next-step preprocessing.
        for layer_name in self._async_load_order:
            group_idx = self._layer_group[layer_name]
            for req_id, request in async_reqs:
                pairs = [
                    (block_id, block_hash)
                    for block_id, block_hash in zip(
                        request.group_block_ids[group_idx], request.block_hashes
                    )
                    if block_id != 0
                ]
                if not pairs:
                    continue
                block_ids, block_hashes = zip(*pairs)
                async_tasks.setdefault(req_id, {}).update(self._store().get(
                    block_indices=list(block_ids),
                    block_hashs=list(block_hashes),
                    layer_names=[layer_name],
                    description=req_id,
                    label=f"g{group_idx}",
                ))
        for req_id, request in async_reqs:
            if req_id in async_tasks:
                self._pending_load_tasks[req_id] = async_tasks[req_id]
                self._pending_load_layers[req_id] = request.async_load_layers

        # Sync targets are consumed in layer order; wait only the leading Mamba run.
        if not sync_tasks:
            return
        if self._leading_mamba_layers:
            if not self._store().get_wait(
                get_results=sync_tasks, layer_names=self._leading_mamba_layers, wait=True
            ):
                raise RuntimeError("Failed to load leading recurrent KV cache")
        if not self._attn_order:
            self._current_get_tasks = None

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._current_get_tasks and not self._active_promoted_tasks:
            return

        # Wait for the synchronous (batched) loads for this layer.
        if self._current_get_tasks and layer_name in self._current_get_tasks:
            success = self._store().get_wait(
                get_results=self._current_get_tasks,
                layer_names=[layer_name],
            )
            if not success:
                raise RuntimeError(
                    f"Failed to load KV cache for layer {layer_name}"
                )

        # Wait for the remaining layers of early-promoted async loads. Their
        # first N layers were already finalized in get_finished(); waiting on an
        # already-finalized layer is a no-op.
        for tasks in self._active_promoted_tasks.values():
            if layer_name not in tasks:
                continue
            success = self._store().get_wait(
                get_results=tasks,
                layer_names=[layer_name],
            )
            if not success:
                raise RuntimeError(
                    f"Failed to load promoted KV cache for layer {layer_name}"
                )

        if layer_name == self._last_layer_name:
            self._active_promoted_tasks = {}

    def _save_layer(
        self, layer_name: str, metadata: KVShrinkConnectorMetadata
    ) -> None:
        group_idx = self._layer_group[layer_name]
        for req_id, request in metadata.reqs_to_save.requests.items():
            pairs = [
                (block_id, block_hash)
                for block_id, block_hash in zip(
                    request.group_block_ids[group_idx], request.block_hashes
                )
                if block_id != 0
            ]
            if not pairs:
                continue
            tasks = self._store().put(
                block_indices=[g for g, _ in pairs],
                block_hashs=[h for _, h in pairs],
                layer_names=[layer_name], label=f"g{group_idx}")
            self._current_put_tasks.setdefault(req_id, []).append(tasks)
        self._saved_layers.add(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # The attention post-hook is the last hook before the next Mamba run.
        if self._current_get_tasks:
            recurrent = [
                ln for ln in self._mamba_load_segments.get(layer_name, ())
                if ln in self._current_get_tasks
            ]
            if recurrent and not self._store().get_wait(
                get_results=self._current_get_tasks, layer_names=recurrent, wait=True
            ):
                raise RuntimeError("Failed to load recurrent KV cache")
            if layer_name == self._last_layer_name:
                self._store().get_wait(get_results=self._current_get_tasks, wait=True)
                self._current_get_tasks = None

        if self._connector_metadata is None:
            return

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")
        if not metadata.reqs_to_save.requests:
            return
        for ln in self._mamba_save_segments.get(layer_name, ()):
            self._save_layer(ln, metadata)
        self._save_layer(layer_name, metadata)

    def wait_for_save(self) -> None:
        """Submit every layer no forward hook covered (recurrent/mamba layers)."""
        if self._connector_metadata is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")
        if not metadata.reqs_to_save.requests:
            return
        for ln in self._layer_names:
            if ln not in self._saved_layers:
                self._save_layer(ln, metadata)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        # Poll asynchronous load tasks submitted in start_load_kv().
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
                    get_results=tasks, layer_names=gate_layers, wait=False
                ):
                    self._store().get_wait(
                        get_results=tasks, layer_names=gate_layers, wait=True
                    )
                    del self._pending_load_tasks[req_id]
                    del self._pending_load_layers[req_id]
                    self._early_promoted_tasks[req_id] = tasks
                    finished_recving.add(req_id)

        self._deferred_finished_req_ids.update(finished_req_ids)
        completed: set[str] = set()

        for req_id in self._deferred_finished_req_ids:
            load_tasks = (
                self._pending_load_tasks.get(req_id)
                or self._early_promoted_tasks.get(req_id)
                or self._active_promoted_tasks.get(req_id)
            )
            if load_tasks is not None:
                if not self._store().get_wait(
                    get_results=load_tasks, wait=False
                ):
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
