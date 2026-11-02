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
    get_tensor_model_parallel_rank,
    model_parallel_is_initialized,
)
import vllm.envs as envs
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.core.kv_cache_utils import (
    get_block_hash,
    resolve_block_hashes,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    group_kernel_blocks,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from iaxl import KVStore, setup_root_logger
from iaxl.envs import envs as iaxl_envs
from iaxl.utils.affinity import bind_cpu_affinity, bind_intel_accel

from .async_load_config import load_async_load_layer_config_from_env

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)

ReqId = str

# OMP-Main TaskQueue levels: P0 loads a forward waits on, P1 first N layers, P2 remaining layers.
LOAD_PRIORITY_P0 = 0
LOAD_PRIORITY_P1 = 1
LOAD_PRIORITY_P2 = 2


@dataclass
class GroupInfo:
    """One KV cache group, keeping the scheduler's original group index."""

    group_id: int
    is_mamba: bool
    block_size: int
    layer_names: list[str]
    enabled: bool


@dataclass
class ReqMeta:
    # One block table per KV cache group; 0 marks a block that is not
    # transferred for that group. block_hashes is shared by all groups.
    block_ids: tuple[list[int], ...] = ()
    block_hashes: list[str] = field(default_factory=list)
    is_async: bool = False
    async_load_layers: int = -1

    def for_group(self, group_idx: int) -> tuple[list[int], list[str]]:
        pairs = [
            (block_id, block_hash)
            for block_id, block_hash in zip(
                self.block_ids[group_idx], self.block_hashes
            )
            if block_id != 0
        ]
        return [p[0] for p in pairs], [p[1] for p in pairs]


@dataclass
class ReqState:
    num_computed_tokens: int = 0
    block_hashes: list[str] = field(default_factory=list)
    # Current block table per group, read from the scheduler each step.
    block_ids: list[list[int]] = field(default_factory=list)
    # (group_id, block_hash) already sent to the store for this request.
    saved: set[tuple[int, str]] = field(default_factory=set)
    is_async: bool = False
    async_load_layers: int = -1


@dataclass
class RequestMetadata:
    requests: dict[ReqId, ReqMeta] = field(default_factory=dict)

    def add_request(
        self,
        req_id: ReqId,
        block_ids: tuple[list[int], ...],
        block_hashes: list[str],
        is_async: bool = False,
        async_load_layers: int = -1,
    ) -> None:
        self.requests[req_id] = ReqMeta(
            block_ids,
            block_hashes,
            is_async,
            async_load_layers,
        )


@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    reqs_to_load: RequestMetadata
    reqs_to_save: RequestMetadata
    scheduled_req_ids: set[ReqId] = field(default_factory=set)


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
        self.kv_cache_config = kv_cache_config
        self.model_config = vllm_config.model_config
        # Keep every vLLM group, including scratch groups we do not transfer.
        # `enabled` marks groups that participate in external prefix reuse;
        # keeping the original group_id avoids renumbering block tables.
        self.groups: list[GroupInfo] = []
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            is_mamba = isinstance(spec, MambaSpec)
            if isinstance(spec, UniformTypeKVCacheSpecs):
                is_mamba = isinstance(
                    next(iter(spec.kv_cache_specs.values())), MambaSpec
                )
            enabled = bool(
                group.enable_kv_transfer and group.kv_cache_spec.prefix_cacheable
            )
            self.groups.append(
                GroupInfo(
                    group_id=group_id,
                    is_mamba=is_mamba,
                    block_size=group.kv_cache_spec.block_size,
                    layer_names=list(group.layer_names),
                    enabled=enabled,
                )
            )
        self.has_mamba = any(g.is_mamba and g.enabled for g in self.groups)
        if self.has_mamba and vllm_config.cache_config.mamba_cache_mode != "align":
            raise ValueError(
                "KVShrinkConnector on hybrid models requires "
                "mamba_cache_mode='align', got "
                f"{vllm_config.cache_config.mamba_cache_mode!r}"
            )
        block_sizes = {g.block_size for g in self.groups if g.enabled}
        if len(block_sizes) != 1:
            raise ValueError(
                "KVShrinkConnector requires one block size across enabled "
                f"prefix-cacheable groups, got {sorted(block_sizes)}"
            )
        self.block_size = block_sizes.pop()
        _, self.hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        # Compressed indexer caches have no per-layer load hook. Until we add
        # an explicit side-cache wait, finish all loads before the forward for
        # these models (correctness baseline, no early-start overlap).
        def _has_side_cache(spec) -> bool:
            specs = (
                spec.kv_cache_specs.values()
                if isinstance(spec, UniformTypeKVCacheSpecs)
                else (spec,)
            )
            return any(getattr(item, "tokens_per_state", 1) > 1 for item in specs)

        self._wait_all_loads = any(
            _has_side_cache(group.kv_cache_spec) for group in kv_cache_config.kv_cache_groups
        )
        self._gpu_block_pool = None
        logger.info(
            "KVShrinkConnector(%s): block_size=%d, hash_block_size=%d, "
            "has_mamba=%s, mamba_cache_mode=%r, prefix caching=%s, "
            "enabled_groups=%s",
            role.name,
            self.block_size,
            self.hash_block_size,
            self.has_mamba,
            vllm_config.cache_config.mamba_cache_mode,
            vllm_config.cache_config.enable_prefix_caching,
            [g.group_id for g in self.groups if g.enabled],
        )
        self.num_mamba_layers = sum(
            len(g.layer_names) for g in self.groups if g.is_mamba and g.enabled
        )
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.vllm_device = vllm_config.device_config.device_type
        parallel_config = vllm_config.parallel_config
        # data_parallel_index, not data_parallel_rank (vLLM resets the latter
        # to 0 for dense models).
        self.dp_rank = parallel_config.data_parallel_index
        if self.model_config.is_moe:
            self.dp_size = parallel_config.data_parallel_size
        else:
            self.dp_size = int(os.environ.get("DP_SIZE", "1"))
            logger.warning(
                "Dense model: reading DP size from the DP_SIZE env var (%d) "
                "because vLLM resets data_parallel_size to 1 for dense models.",
                self.dp_size,
            )
        self.tp_rank = (
            get_tensor_model_parallel_rank()
            if model_parallel_is_initialized()
            else 0
        )
        # Globally-unique worker index / total worker count: KVStore/CPU/port identity.
        self.global_rank = self.dp_rank * self.tp_size + self.tp_rank
        self.global_size = self.dp_size * self.tp_size

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
        # TaskQueue bundle id of each async request's remaining layers until they are raised.
        self._load_bundle_ids: dict[ReqId, int] = {}
        # 0 means no bundle in the TaskQueue.
        self._next_load_bundle_id = 1

        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
        )

        if role == KVConnectorRole.SCHEDULER:
            # Scheduler store reads its DP group's tp0; offset mgmt ports by DP
            # group so DP schedulers on one host don't clash.
            iaxl_envs.IAXL_API_CONTROLLER_PORT += self.dp_rank
            iaxl_envs.IAXL_API_WORKER_BASE_PORT += self.dp_rank * self.tp_size
            self.kvstore: Optional[KVStore] = KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(index) for index in range(self.num_layers)],
                rank=self.dp_rank * self.tp_size,
                tp_size=self.tp_size,
            )
        else:
            self.kvstore = None
            if not iaxl_envs.IAXL_RDMA_ENABLE:  # compression/DSA run on the daemon node
                self._bind_cpu_affinity()
                self._bind_intel_accel()

    def _bind_cpu_affinity(self) -> None:
        spec = os.getenv("KVSHRINK_CPU_OMP_THREADS_BIND")
        if spec:
            cpus = spec.split("|")
            if len(cpus) <= self.global_rank:
                raise ValueError(
                    f"KVSHRINK_CPU_OMP_THREADS_BIND has {len(cpus)} entries, "
                    f"but rank is {self.global_rank}"
                )
            os.environ["IAXL_CPU_OMP_THREADS_BIND"] = cpus[self.global_rank]
            logger.info(
                "Bound rank %d: IAXL_CPU_OMP_THREADS_BIND=%s",
                self.global_rank,
                cpus[self.global_rank],
            )
        if self.vllm_device == "cpu":  # vLLM's CPU worker binds itself
            return
        bind_cpu_affinity(
            self.global_rank, self.global_size, envs.VLLM_CPU_OMP_THREADS_BIND
        )

    def _bind_intel_accel(self) -> None:
        bind_intel_accel(self.global_rank)

    def _store(self) -> KVStore:
        if self.kvstore is None:
            raise RuntimeError("KVStore has not been initialized")
        return self.kvstore

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        # Needed to read the exact hash of a mamba boundary hand-off.
        self._gpu_block_pool = gpu_block_pool

    @staticmethod
    def _label(group: GroupInfo) -> str:
        return f"g{group.group_id}"

    @staticmethod
    def _hash_key(block_hash: bytes) -> str:
        # vLLM BlockHash is bytes. IAFL's KV pool rejects path separators such
        # as "\", so never use str(bytes).
        return block_hash.hex()

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

        # vLLM computes the prefix hashes at hash_block_size. Resolve them
        # once at the connector's common block size; every enabled group uses
        # the same token-prefix hash list.
        raw_hashes = list(request.block_hashes)
        if self.hash_block_size == self.block_size:
            resolved_hashes = raw_hashes
        else:
            resolved_hashes = resolve_block_hashes(
                raw_hashes, self.hash_block_size, self.block_size
            )
        resolved_hashes = [self._hash_key(h) for h in resolved_hashes]

        state = ReqState(
            num_computed_tokens=num_computed_tokens,
            block_hashes=resolved_hashes,
        )
        self._req_states[request.request_id] = state

        enabled_groups = [group for group in self.groups if group.enabled]
        if not enabled_groups:
            return 0, False

        matched_tokens: Optional[int] = None
        exists_by_group: dict[int, list[bool]] = {}
        for group in enabled_groups:
            exists = self._store().has(
                resolved_hashes,
                label=self._label(group),
                truncate=not group.is_mamba,
            )
            exists_by_group[group.group_id] = exists
            if group.is_mamba:
                # Mamba states are sparse; find the deepest saved boundary.
                hit_blocks = 0
                for index, present in enumerate(exists):
                    if present:
                        hit_blocks = index + 1
            else:
                hit_blocks = next(
                    (index for index, present in enumerate(exists) if not present),
                    len(exists),
                )
            # The last prompt token must be recomputed for logits.
            max_blocks = max(0, (request.num_tokens - 1) // group.block_size)
            hit_blocks = min(hit_blocks, max_blocks)
            tokens = hit_blocks * group.block_size
            matched_tokens = (
                tokens
                if matched_tokens is None
                else min(matched_tokens, tokens)
            )

        matched_tokens = (
            (matched_tokens or 0) // self.block_size * self.block_size
        )

        # Shrink until every mamba group has a state at the chosen boundary.
        matched_blocks = matched_tokens // self.block_size
        while matched_blocks > 0:
            if all(
                group.is_mamba is False
                or (
                    matched_blocks - 1 < len(exists_by_group[group.group_id])
                    and exists_by_group[group.group_id][matched_blocks - 1]
                )
                for group in enabled_groups
            ):
                break
            matched_blocks -= 1
        matched_tokens = matched_blocks * self.block_size
        num_new_tokens = max(0, matched_tokens - num_computed_tokens)

        # Decide sync vs async for this request. The load can only be async when
        # there are external tokens to load and async is enabled. Concurrency is
        # approximated by the number of in-flight requests (this one included).
        selected_layers = self._async_load_layer_config.select(
            len(self._req_states)
        )
        # Mamba layers lead the worker's layer order and have no layer-load
        # hook, so the early-start window must cover them. -1 keeps its
        # meaning (wait for every layer); 0 asks for a synchronous load, which
        # hybrid cannot do, so it becomes that window as well.
        if self.has_mamba and selected_layers >= 0:
            selected_layers += self.num_mamba_layers
        # A dynamic-map layer value of 0 selects synchronous loading. It is not
        # an async request that resumes before layer 0.
        use_async = num_new_tokens > 0 and selected_layers != 0
        state.is_async = use_async
        if use_async:
            state.async_load_layers = selected_layers

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

        if num_external_tokens == 0:
            return
        if num_external_tokens % self.block_size != 0:
            raise ValueError("External token count must be block aligned")

        block_ids = blocks.get_block_ids()
        shared_hashes = state.block_hashes
        load_start = state.num_computed_tokens // self.block_size
        load_end = load_start + num_external_tokens // self.block_size
        all_ids: list[list[int]] = []
        for group in self.groups:
            if not group.enabled:
                all_ids.append([])
                continue
            ids = list(block_ids[group.group_id])
            if load_end > len(ids) or load_end > len(shared_hashes):
                raise ValueError(
                    f"Load range {load_start}:{load_end} exceeds group "
                    f"{group.group_id} table/hash length "
                    f"({len(ids)}, {len(shared_hashes)})"
                )
            if group.is_mamba:
                # Only the boundary slot carries a saved state.
                all_ids.append(
                    [0] * (load_end - load_start - 1) + [ids[load_end - 1]]
                )
            else:
                all_ids.append(ids[load_start:load_end])

        self._reqs_to_load.add_request(
            request.request_id,
            tuple(all_ids),
            shared_hashes[load_start:load_end],
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def _add_request_to_save(
        self,
        req_id: ReqId,
        num_computed_tokens: int,
        num_scheduled_tokens: int,
        boundary_offloads: list[tuple[int, int, int]],
    ) -> None:
        state = self._req_states.get(req_id)
        if state is None:
            raise RuntimeError(f"Missing state for request {req_id}")

        num_tokens = num_computed_tokens + num_scheduled_tokens
        shared_hashes = state.block_hashes
        all_ids: list[list[int]] = []

        for group in self.groups:
            if not group.enabled:
                all_ids.append([])
                continue

            ids = state.block_ids[group.group_id]
            group_ids = [0] * len(shared_hashes)

            if group.is_mamba:
                # Never resolve mamba state positionally. The scheduler hands
                # us the exact block and its hash for each committed boundary.
                if self._gpu_block_pool is None:
                    logger.warning(
                        "No GPU block pool bound; skipping mamba boundary saves"
                    )
                else:
                    for gid, block_id, boundary_tokens in boundary_offloads:
                        if gid != group.group_id:
                            continue
                        # The saved state covers [0, boundary_tokens), whose
                        # prefix hash is the block ending at that boundary.
                        if (
                            boundary_tokens % group.block_size != 0
                            or boundary_tokens > num_tokens
                        ):
                            continue
                        index = boundary_tokens // group.block_size - 1
                        if index < 0 or index >= len(shared_hashes):
                            continue
                        block = self._gpu_block_pool.blocks[block_id]
                        if block.is_null or block.block_hash is None:
                            continue
                        block_hash = self._hash_key(
                            get_block_hash(block.block_hash)
                        )
                        if block_hash != shared_hashes[index]:
                            logger.warning(
                                "Mamba boundary hash mismatch at %d: %s != %s",
                                index,
                                block_hash,
                                shared_hashes[index],
                            )
                            continue
                        key = (group.group_id, block_hash)
                        if key in state.saved:
                            continue
                        state.saved.add(key)
                        group_ids[index] = block_id
            else:
                limit = min(
                    len(ids),
                    num_tokens // group.block_size,
                    len(shared_hashes),
                )
                for index in range(limit):
                    block_id = ids[index]
                    if block_id == 0:
                        continue
                    block_hash = shared_hashes[index]
                    key = (group.group_id, block_hash)
                    if key in state.saved:
                        continue
                    state.saved.add(key)
                    group_ids[index] = block_id

            all_ids.append(group_ids)

        if any(any(ids) for ids in all_ids):
            self._reqs_to_save.add_request(
                req_id, tuple(all_ids), shared_hashes
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # True = defer freeing to get_finished() (async load/save may still run).
        self._req_states.pop(request.request_id, None)
        return True, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        return self.request_finished(request, [])

    @staticmethod
    def _current_block_ids(scheduler_output: SchedulerOutput, req_id: str):
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if block_state is None:
            return None
        # vLLM 0.33+ resolves block tables lazily through get_block_ids();
        # older releases exposed the raw dict.
        get_block_ids = getattr(block_state, "get_block_ids", None)
        if get_block_ids is not None:
            return get_block_ids(req_id)
        return block_state.block_ids.get(req_id)

    @staticmethod
    def _boundary_offloads(scheduler_output: SchedulerOutput, req_id: str):
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if block_state is None:
            return []
        return list(block_state.boundary_state_offloads.get(req_id, []))

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for request in scheduler_output.scheduled_new_reqs:
            state = self._req_states[request.req_id]
            current = self._current_block_ids(scheduler_output, request.req_id)
            state.block_ids = [
                list(ids)
                for ids in (current if current is not None else request.block_ids)
            ]
            self._add_request_to_save(
                request.req_id,
                request.num_computed_tokens,
                num_scheduled_tokens[request.req_id],
                self._boundary_offloads(scheduler_output, request.req_id),
            )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached_reqs.req_ids):
            if req_id in cached_reqs.resumed_req_ids:
                raise RuntimeError("Resuming from preemption is not supported")

            state = self._req_states[req_id]
            current = self._current_block_ids(scheduler_output, req_id)
            if current is not None:
                state.block_ids = [list(ids) for ids in current]
            self._add_request_to_save(
                req_id,
                cached_reqs.num_computed_tokens[index],
                num_scheduled_tokens[req_id],
                self._boundary_offloads(scheduler_output, req_id),
            )

        # kvshrink loads asynchronously, but its start_load_kv only submits
        # host-side work; running it before the forward keeps the 0.23-style
        # placement (0.29 would otherwise defer it to post_forward).
        scheduler_output.has_sync_kv_loads = True
        metadata = KVShrinkConnectorMetadata(
            reqs_to_load=self._reqs_to_load,
            reqs_to_save=self._reqs_to_save,
            # scheduler_output.num_scheduled_tokens is a dict keyed by req id.
            scheduled_req_ids=set(scheduler_output.num_scheduled_tokens.keys()),
        )
        self._reqs_to_load = RequestMetadata()
        self._reqs_to_save = RequestMetadata()
        return metadata

    ############################################################
    # Worker Side Methods
    ############################################################

    def _view_as_blocks(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """View each layer's KV cache as contiguous ``(num_blocks, page)`` rows
        per logical block, taken from the ``KVCacheConfig`` geometry."""
        num_blocks = self.kv_cache_config.num_blocks
        spec_by_layer: dict[str, Any] = {}
        for group in self.kv_cache_config.kv_cache_groups:
            specs = (
                group.kv_cache_spec.kv_cache_specs
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else {}
            )
            for layer_name in group.layer_names:
                spec_by_layer[layer_name] = specs.get(
                    layer_name, group.kv_cache_spec
                )

        attn_dtype = next(
            (
                spec.dtype
                for spec in spec_by_layer.values()
                if isinstance(spec, AttentionSpec)
            ),
            None,
        )
        assert attn_dtype is not None

        views: dict[str, torch.Tensor] = {}
        for layer_name, cache in kv_caches.items():
            ref = group_kernel_blocks(cache, num_blocks)
            # vLLM hands mamba layers int8 views of the shared buffer (their
            # MambaSpec carries per-state dtypes, so the view is not cast), so
            # offset and stride are in int8 units: rescale them by bytes.
            itemsize = attn_dtype.itemsize
            offset = ref.storage_offset() * ref.element_size()
            stride = ref.stride(0) * ref.element_size()
            page = spec_by_layer[layer_name].page_size_bytes
            assert offset % itemsize == 0 and stride % itemsize == 0
            assert page % itemsize == 0
            if iaxl_envs.IAXL_RDMA_ENABLE and stride != page:
                raise ValueError(
                    "KVShrink remote pool does not support packed block strides; "
                    f"layer {layer_name} has stride={stride}, page={page}. "
                    "Use the local KVStore path or disable RDMA."
                )
            views[layer_name] = torch.tensor(
                [], dtype=attn_dtype, device=ref.device
            ).set_(
                ref.untyped_storage(),
                offset // itemsize,
                (num_blocks, page // itemsize),
                (stride // itemsize, 1),
            )
        return views

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        # Skip draft (MTP) layers; put mamba layers first so the async
        # early-start window covers them.
        mamba_layers = {
            ln
            for group in self.groups
            if group.is_mamba and group.enabled
            for ln in group.layer_names
        }
        kv_caches = {
            ln: kv_caches[ln]
            for ln in sorted(kv_caches, key=lambda ln: ln not in mamba_layers)
            if extract_layer_index(ln) < self.num_layers
        }
        # Skip groups that are not externally cacheable (e.g. the indexer tail
        # ring), but keep their original group_id.
        self.groups = [
            GroupInfo(
                group_id=group.group_id,
                is_mamba=group.is_mamba,
                block_size=group.block_size,
                layer_names=(
                    [ln for ln in group.layer_names if ln in kv_caches]
                    if group.enabled
                    else []
                ),
                enabled=group.enabled,
            )
            for group in self.groups
        ]
        # Keep only enabled layers in the worker data path. Disabled groups
        # keep their slot in self.groups, but must not appear in _layer_names.
        enabled_layers = {
            ln
            for group in self.groups
            if group.enabled
            for ln in group.layer_names
        }
        kv_caches = {
            ln: tensor for ln, tensor in kv_caches.items() if ln in enabled_layers
        }
        self.num_mamba_layers = sum(
            len(group.layer_names)
            for group in self.groups
            if group.is_mamba and group.enabled
        )

        kv_caches = self._view_as_blocks(kv_caches)
        first_kv_cache = next(iter(kv_caches.values()))
        self._last_layer_name = next(reversed(kv_caches))
        self._layer_names = list(kv_caches.keys())
        self.kvstore = KVStore(
            model_name=os.path.basename(self.model_config.model),
            block_dim=0,
            kv_caches=kv_caches,
            rank=self.global_rank,
            tp_size=self.tp_size,
        )
        logger.info(
            "Registered %d KV cache layers (%d mamba) with shape %s",
            len(kv_caches),
            self.num_mamba_layers,
            list(first_kv_cache.shape),
        )

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")

        # vLLM may put layers with different page sizes/dtypes in one group
        # (MLA + indexer). Submit one layer per get() so KVFlow's same-shape /
        # same-dtype contract holds, and use a per-group label as the key
        # namespace.
        self._current_get_tasks = None
        sync_batches: dict[int, tuple[list[int], list[str]]] = {}
        async_reqs: list[tuple[ReqId, ReqMeta]] = []
        for req_id, request in metadata.reqs_to_load.requests.items():
            assert all(
                len(request.block_ids[group.group_id])
                == len(request.block_hashes)
                for group in self.groups
                if group.enabled
            ), req_id
            if request.is_async:
                async_reqs.append((req_id, request))
            else:
                # Hybrid loads are always async: vLLM still zeroes the blocks
                # of a synchronous load on the compute stream.
                assert not self.has_mamba
                for group in self.groups:
                    if not group.enabled:
                        continue
                    block_ids, block_hashes = request.for_group(group.group_id)
                    if not block_ids:
                        continue
                    batch = sync_batches.setdefault(group.group_id, ([], []))
                    batch[0].extend(block_ids)
                    batch[1].extend(block_hashes)
        sync_tasks: dict[str, Any] = {}
        for group_id, (block_ids, block_hashes) in sync_batches.items():
            group = self.groups[group_id]
            for layer_name in group.layer_names:
                sync_tasks.update(
                    self._store().get(
                        block_indices=block_ids,
                        block_hashs=block_hashes,
                        layer_names=[layer_name],
                        label=self._label(group),
                    )
                )
        if sync_tasks:
            self._current_get_tasks = sync_tasks

        # Wait only on promoted requests scheduled in this forward; a no-forward batch waits on none.
        if forward_context.attn_metadata is not None:
            duplicates = (
                self._active_promoted_tasks.keys()
                & self._early_promoted_tasks.keys()
            )
            if duplicates:
                raise RuntimeError(
                    f"Duplicate promoted load tasks for requests {duplicates}"
                )
            scheduled: list[ReqId] = []
            for req_id in self._early_promoted_tasks:
                if req_id in metadata.scheduled_req_ids:
                    scheduled.append(req_id)
            raise_load_bundle_ids: list[int] = []
            for req_id in scheduled:
                self._active_promoted_tasks[req_id] = self._early_promoted_tasks.pop(
                    req_id
                )
                load_bundle_id = self._load_bundle_ids.pop(req_id, None)
                if load_bundle_id is not None:
                    raise_load_bundle_ids.append(load_bundle_id)
            if raise_load_bundle_ids:
                # Before the async gets below; sorting P0 by layer interleaves these with the sync batch.
                self._store().set_bundles_priority(raise_load_bundle_ids, LOAD_PRIORITY_P0)
                self._store().sort_priority_level(LOAD_PRIORITY_P0)

        # Submit asynchronous loads per request; they are polled across
        # scheduler steps in get_finished().
        for req_id, request in async_reqs:
            # First N layers at P1 overtake waiting remaining layers, which stay at P2 until raised.
            priorities: dict[str, int] = {}
            has_remaining_layers = False
            for layer_index, layer_name in enumerate(self._layer_names):
                # -1 promotes only after every layer is loaded.
                if request.async_load_layers == -1 or layer_index < request.async_load_layers:
                    priorities[layer_name] = LOAD_PRIORITY_P1
                else:
                    priorities[layer_name] = LOAD_PRIORITY_P2
                    has_remaining_layers = True
            load_bundle_id = self._next_load_bundle_id
            self._next_load_bundle_id += 1
            tasks: dict[str, Any] = {}
            for group in self.groups:
                if not group.enabled:
                    continue
                block_ids, block_hashes = request.for_group(group.group_id)
                if not block_ids:
                    continue
                for layer_name in group.layer_names:
                    tasks.update(self._store().get(
                        block_indices=block_ids,
                        block_hashs=block_hashes,
                        layer_names=[layer_name],
                        description=req_id,
                        label=self._label(group),
                        priorities=priorities,
                        bundle_id=load_bundle_id,
                    ))
            self._pending_load_tasks[req_id] = tasks
            self._pending_load_layers[req_id] = request.async_load_layers
            if has_remaining_layers:
                self._load_bundle_ids[req_id] = load_bundle_id

        if self._wait_all_loads:
            if self._current_get_tasks and not self._store().get_wait(
                get_results=self._current_get_tasks
            ):
                raise RuntimeError("Failed to load side-cache KV before forward")
            for tasks in self._pending_load_tasks.values():
                if not self._store().get_wait(get_results=tasks):
                    raise RuntimeError("Failed to load side-cache KV before forward")

    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._current_get_tasks and not self._active_promoted_tasks:
            return

        # Wait for the synchronous (batched) loads for this layer.
        if self._current_get_tasks:
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

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        # Save every layer after the forward in wait_for_save(). Indexer side
        # caches have no per-layer hook, and a single path covers both those
        # caches and ordinary attention layers.
        return

    def wait_for_save(self) -> None:
        if self._connector_metadata is None:
            return
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError("Unexpected connector metadata")
        for req_id, request in metadata.reqs_to_save.requests.items():
            for group in self.groups:
                if not group.enabled:
                    continue
                block_ids, block_hashes = request.for_group(group.group_id)
                if not block_ids:
                    continue
                for layer_name in group.layer_names:
                    tasks = self._store().put(
                        block_indices=block_ids,
                        block_hashs=block_hashes,
                        layer_names=[layer_name],
                        label=self._label(group),
                        flush=layer_name == group.layer_names[-1],
                    )
                    self._current_put_tasks.setdefault(req_id, []).append(tasks)

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
                    del self._pending_load_tasks[req_id]
                    del self._pending_load_layers[req_id]
                    finished_recving.add(req_id)
            else:
                # Early promote once the first N layers are loaded; the remaining
                # layers are waited on-demand in wait_for_layer_load().
                first_n_layers = self._layer_names[:async_load_layers]
                if self._store().get_wait(
                    get_results=tasks, layer_names=first_n_layers, wait=False
                ):
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
                self._pending_load_tasks.pop(req_id, None)
                self._pending_load_layers.pop(req_id, None)
                self._early_promoted_tasks.pop(req_id, None)
                self._active_promoted_tasks.pop(req_id, None)
                # Finished before entering a forward: its bundle is never raised.
                self._load_bundle_ids.pop(req_id, None)

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
