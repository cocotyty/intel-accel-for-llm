# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
import math
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
from vllm.v1.kv_cache_interface import KVCacheConfig

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.request import Request

from iaxl import KVStore, setup_root_logger

from .layout import RequestMetadata

from .async_load_config import load_async_load_layer_config_from_env

setup_root_logger(show_pid_tid=False)
logger = logging.getLogger(__name__)



@dataclass
class KVShrinkConnectorMetadata(KVConnectorMetadata):
    """Scheduler -> worker transfer plan.

    ``reqs_to_load`` are LOAD plans (executed before forward),
    ``reqs_to_save`` are SAVE plans (executed after forward). Both map
    request id to ``layout.ReqMeta``; the worker only ever sees this
    object, so each plan is fully self-describing.
    """
    reqs_to_load: RequestMetadata = field(default_factory=RequestMetadata)
    reqs_to_save: RequestMetadata = field(default_factory=RequestMetadata)



def _save_enabled() -> bool:
    """Production save is ON by default; KVSHRINK_SAVE=0 disables it and
    KVSHRINK_DEBUG_AUTOSAVE=1 force-enables it."""
    return (os.getenv("KVSHRINK_SAVE", "1") != "0"
            or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")


class KVShrinkConnector(KVConnectorBase_V1, SupportsHMA):
    """KVShrink external KV cache connector.

    ONE path for every model. vLLM already describes a model as a list
    of KV cache groups, so a pure-attention model is the one-group case
    and a GDN/Mamba model is the two-group case; the scheduler and
    worker are written against groups and need no knowledge of which
    they are serving.

    A recurrent group changes nothing about how bytes are stored: every
    group is a block space in one store, told apart by its namespace.
    """

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
        self.block_size = vllm_config.cache_config.block_size
        self.tp_size = vllm_config.parallel_config.tensor_parallel_size
        self.num_layers = self.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.vllm_device = vllm_config.device_config.device_type
        self.rank = get_world_group().rank if model_parallel_is_initialized() else 0

        # Ordered worker-side layer names (populated in
        # register_kv_caches) for the block store layout.
        self._layer_names: list[str] = []

        self._async_load_layer_config = load_async_load_layer_config_from_env(
            num_layers=self.num_layers,
        )

        # One stack for every model. vLLM already describes any model as
        # a list of KV cache groups, so a pure-attention model is simply
        # the one-group case; the only thing a GDN/Mamba model changes is
        # WHICH store layout is underneath, and that is an adapter
        # choice, not a second code path.
        self._sched = None
        self._worker = None
        self._backend = None
        self._canon = None
        self._groups: list = []
        # Authoritative TP rank, set by _init_kv_stack. Distinct from
        # self.rank, which is read from the world group before
        # distributed init has run and is therefore 0 on every worker.
        self._rank: Optional[int] = None
        self.kvstore: Optional[KVStore] = None

        if kv_cache_config is not None:
            self._init_kv_stack(vllm_config, role, kv_cache_config)
        else:
            self.kvstore = None
            self._bind_cpu_affinity()
            self._bind_intel_accel()

    ############################################################
    # Construction
    ############################################################

    def _init_kv_stack(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        """Build the hybrid stack for this role.

        Scheduler role gets the hit policy + plan builder over a
        READ-ONLY backend (presence checks only, no writer lease, no GPU
        pool). Worker role gets the canonical page views + the transfer
        engine over a WRITER backend holding this rank's single-writer
        """
        from .backend import KVStoreBackend
        from .layout import Canonicalizer
        from .layout import (compute_namespace, parse_kv_cache_config,
                                    validate_codec_env)
        from .layout import SCHEMA_VERSION
        from .scheduler import HybridRequestScheduler
        from .worker import HybridWorker

        pc = vllm_config.parallel_config
        tp_size = int(getattr(pc, "tensor_parallel_size", 1) or 1)
        if role == KVConnectorRole.WORKER:
            # parallel_config.rank, NOT get_world_group(): the connector
            # is constructed before distributed init in the worker
            # processes, so the world group would report rank 0 on every
            # TP rank and the ranks would overwrite each other's shards.
            rank = int(getattr(pc, "rank", 0) or 0)
        else:
            # Scheduler-side keys are always rank 0; each worker verifies
            # its own shard through its own backend.
            rank = 0
        self.kvstore = None
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config

        # Block-hash granularity, per v0.23.0's resolve_kv_cache_block_sizes:
        # the GCD of the groups' block sizes (every group's block size is
        # divisible by it). Single group -> that group's block size.
        block_sizes = sorted({int(g.kv_cache_spec.block_size)
                              for g in kv_cache_config.kv_cache_groups})
        hash_block_size = math.gcd(*block_sizes)
        namespace = compute_namespace(
            model_id=str(getattr(model_config, "model", "model")),
            model_revision=str(getattr(model_config, "revision", None) or ""),
            tokenizer_revision=str(
                getattr(model_config, "tokenizer_revision", None) or ""),
            cache_dtype=str(getattr(cache_config, "cache_dtype", "auto")),
            kv_schema_version=SCHEMA_VERSION,
            tp_size=tp_size,
            pp_size=1,
        )
        # Fail-closed: a lossy codec would corrupt GDN state (see the
        # function for why this path cannot tolerate what the
        # attention-only path is designed to).
        validate_codec_env()

        groups, layer_infos, num_blocks = parse_kv_cache_config(
            kv_cache_config, hash_block_size=hash_block_size)

        # Fail-closed: speculative decoding widens the GDN state gather.
        # v0.23.0's mamba_get_block_table_tensor returns
        # block_table[start : start + 1 + num_speculative_blocks] and the
        # decode path reads all of those columns, but an external
        # snapshot only ever restores column 0 (the block holding this
        # step's last scheduled token). Serving a hit would let the
        # kernel read unrestored speculative slots.
        for g in kv_cache_config.kv_cache_groups:
            spec_blocks = int(
                getattr(g.kv_cache_spec, "num_speculative_blocks", 0) or 0)
            if spec_blocks:
                raise RuntimeError(
                    "kvshrink hybrid: speculative decoding is not "
                    f"supported (group has num_speculative_blocks="
                    f"{spec_blocks}); the external GDN snapshot only "
                    "restores the non-speculative state slot. Disable "
                    "speculative decoding or the KV connector.")
        self._groups = groups
        self._rank = rank
        self._tp_size = tp_size
        # A recurrent group changes only which block hashes we ask
        # about (see _block_hash_source); the storage below is the same.
        recurrent = any(g.kind == "mamba" for g in groups)

        if role == KVConnectorRole.SCHEDULER:
            # Presence-only store: the scheduler asks whether boundaries
            # are readable and never moves bytes.
            self._backend = KVStoreBackend(KVStore(
                model_name=os.path.basename(self.model_config.model),
                layer_names=[str(i) for i in range(self.num_layers)],
                tp_size=self.tp_size,
            ))
            self._backend.register_layout(namespace, tp_size, rank)
            self._sched = HybridRequestScheduler(
                groups, self._backend, hash_block_size, namespace,
                tp_size, rank,
                block_hash_source=self._block_hash_source(recurrent),
                async_load_config=self._async_load_layer_config)
        else:
            # The store needs kv_caches, which arrive later; bound in
            # register_kv_caches.
            self._backend = KVStoreBackend()
            self._backend.register_layout(namespace, tp_size, rank)
            self._canon = Canonicalizer(layer_infos, num_blocks)
            self._worker = HybridWorker(
                groups, layer_infos, num_blocks, self._backend,
                self._canon, rank, tp_size)

        logger.info(
            "kvshrink hybrid path enabled (%s role, %d groups, %d layers, "
            "hash_block_size=%d, namespace=%s, tp=%d rank=%d)",
            "scheduler" if role == KVConnectorRole.SCHEDULER else "worker",
            len(groups), len(layer_infos), hash_block_size, namespace,
            tp_size, rank)
        logger.info(
            "kvshrink groups: %s",
            [(g.group_idx, g.kind, g.block_size) for g in groups])

    @staticmethod
    def _block_hash_source(recurrent: bool) -> str:
        """Which block-identity scheme to key the cache with.

        Defaults preserve what each layout already wrote, because
        switching schemes does not migrate data -- it renames it, and
        every existing entry becomes unreachable until it is written
        again. The block layout keeps its own token-derived hashes; the
        boundary layout keeps vLLM's, which is what it shipped with.

        ``KVSHRINK_BLOCK_HASH_SOURCE=vllm|legacy`` overrides, for
        operators willing to trade one cold warm-up for a single scheme
        across both layouts.
        """
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

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        return self._sched.get_num_new_matched_tokens(
            request, num_computed_tokens)
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._sched.update_state_after_alloc(
            request, blocks, num_external_tokens)
        return
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Saves are SYNCHRONOUS (wait_for_save writes every page
        # inside the step), so no in-flight job
        # can still reference these blocks and vLLM may free them
        # immediately. Returning True would promise a get_finished()
        # ack that never comes -- a deterministic block leak.
        # Committed boundaries are content-addressed and outlive the
        # request; they are never deleted here.
        self._sched.on_request_finished(request.request_id)
        return False, None
    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        """SupportsHMA entry point (v0.23.0 calls this instead of
        ``request_finished`` whenever the hybrid memory allocator is on,
        which is the default for every model).

        Both paths keep their own contract: hybrid frees immediately,
        pure attention keeps deferring to get_finished(). The pure path
        only ever has one KV cache group, so its single block list is
        forwarded unchanged.
        """
        return self.request_finished(request, [])
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        return self._build_connector_meta(scheduler_output)
    def _build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Assemble this pass's hybrid plans.

        Order matters: for cached (running) requests the block-table
        sync (``on_cached_request``) MUST run before ``build_save_meta``,
        so the save plan already sees blocks allocated in the SAME pass.
        """
        meta = KVShrinkConnectorMetadata()
        sched = self._sched
        debug = bool(os.getenv("KVSHRINK_DEBUG_LOG"))
        save_enabled = _save_enabled()
        num_sched = scheduler_output.num_scheduled_tokens

        for new_req in scheduler_output.scheduled_new_reqs:
            req_meta = sched.build_load_meta(
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
            if save_enabled:
                save_meta = sched.build_save_meta(
                    new_req.req_id, num_sched.get(new_req.req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[new_req.req_id] = save_meta

        # ASYNC requests ride NEITHER list: vLLM parks them in
        # WAITING_FOR_REMOTE_KVS, so they are absent from
        # scheduled_new_reqs and from scheduled_cached_reqs alike. Their
        # plan comes from what update_state_after_alloc recorded, and
        # without it the worker would have nothing to transfer and the
        # request would wait forever to be released.
        for req_id, req_meta in sched.take_async_load_plans(
                set(meta.reqs_to_load.requests)).items():
            if debug:
                logger.info(
                    "LOADMETA(async) req=%s ops=%d layers=%s",
                    req_id, len(req_meta.group_ops),
                    req_meta.async_load_layers)
            meta.reqs_to_load.requests[req_id] = req_meta

        # PREEMPTION-RESUMED requests ride scheduled_cached_reqs.
        # resumed_req_ids, NOT scheduled_new_reqs. Their external-hit
        # tokens were accepted this same pass, so without a load plan
        # here the worker would never restore the pages while the core
        # already skips recompute -- silent garbage output.
        cr = scheduler_output.scheduled_cached_reqs
        for req_id in (getattr(cr, "resumed_req_ids", None) or ()):
            req_meta = sched.build_resumed_load_meta(
                req_id, num_sched.get(req_id, 0))
            if req_meta is None:
                continue
            if debug:
                logger.info(
                    "LOADMETA(resumed) req=%s ops=%d",
                    req_id, len(req_meta.group_ops))
            if req_meta.external_hit_tokens > 0 or req_meta.group_ops:
                meta.reqs_to_load.requests[req_id] = req_meta

        # Running requests cross boundaries in later steps too (chunked
        # prefill tails, decode-time crossings): sync their tables first,
        # then emit incremental saves.
        if save_enabled:
            resumed = getattr(cr, "resumed_req_ids", None) or set()
            new_bids = getattr(cr, "new_block_ids", None) or []
            ncts = getattr(cr, "num_computed_tokens", None) or []
            for i, req_id in enumerate(getattr(cr, "req_ids", []) or []):
                sched.on_cached_request(
                    req_id,
                    new_bids[i] if i < len(new_bids) else None,
                    req_id in resumed,
                    ncts[i] if i < len(ncts) else None)
                save_meta = sched.build_save_meta(
                    req_id, num_sched.get(req_id, 0))
                if save_meta.group_ops:
                    meta.reqs_to_save.requests[req_id] = save_meta

        if debug:
            logger.info(
                "build_connector_meta: %d load reqs, %d save reqs",
                len(meta.reqs_to_load.requests),
                len(meta.reqs_to_save.requests))
        return meta

    def _register_layer_caches(
        self, kv_caches: dict[str, torch.Tensor]
    ) -> None:
        """Bind canonical page views, in model execution order.

        The order matters to the async release gate, which holds a
        request until its FIRST N layers have landed -- a statement
        about position, not about names. The ``kv_caches`` dict does not
        carry it: v0.23.0 builds it group by group
        (``_kv_cache_spec_attn_group_iterator``), so mamba and attention
        layers arrive in separate runs. We recover the order the way
        vLLM's own ``bind_kv_cache`` does, from the layer index in the
        layer name, and fail closed if the names do not yield a unique
        order rather than gate on an arbitrary prefix.
        """
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")
        from vllm.model_executor.models.utils import extract_layer_index

        missing = [ln for ln in self._worker._layer_infos
                   if ln not in kv_caches]
        if missing:
            raise RuntimeError(
                f"kvshrink hybrid: layers {sorted(missing)} are in the KV "
                "cache config but absent from kv_caches; refusing to start")

        indexed: dict[int, str] = {}
        for layer_name in kv_caches:
            try:
                idx = extract_layer_index(layer_name)
            except Exception as exc:
                raise RuntimeError(
                    "kvshrink hybrid: cannot derive the execution index of "
                    f"layer {layer_name!r} ({exc}); refusing to start"
                    ) from exc
            if idx in indexed:
                raise RuntimeError(
                    "kvshrink hybrid: layers "
                    f"{indexed[idx]!r} and {layer_name!r} share execution "
                    f"index {idx}; the execution order is ambiguous")
            indexed[idx] = layer_name
        order = [indexed[i] for i in sorted(indexed)]

        self._layer_names = order
        self._worker.register(kv_caches, order)

    def _metadata(self) -> KVShrinkConnectorMetadata:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, KVShrinkConnectorMetadata):
            raise TypeError(
                "kvshrink hybrid worker received "
                f"{type(metadata).__name__}; expected "
                "KVShrinkConnectorMetadata")
        return metadata

    ############################################################
    # Worker Side Methods
    ############################################################

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        if not kv_caches:
            raise ValueError("kv_caches must not be empty")

        static_context = self.vllm_config.compilation_config.static_forward_context
        for layer in static_context.values():
            get_backend = getattr(layer, "get_attn_backend", None)
            if get_backend is not None:
                if "FLASHINFER" in get_backend().get_name().upper():
                    raise RuntimeError("FlashInfer is not supported")
                break

        if self._worker is not None:
            self._register_layer_caches(kv_caches)

        if self._backend is not None:
            # The store is handed the canonical page views, not the raw
            # tensors: every view is a (num_blocks, page_bytes) int8
            # array whose dim-0 rows are blocks, which is the shape the
            # transfer path uses for both attention and GDN. Raw tensors
            # would not do -- a GDN layer is two tensors of different
            # shape and dtype, and the engine requires one shape per
            # call. Views are keyed per part because a split-K/V
            # attention layer contributes two.
            views = {
                f"{ln}::{part}": view
                for ln in self._layer_names
                for part, view in self._canon.page_view_parts(ln)[0].items()
            }
            self.kvstore = KVStore(
                model_name=os.path.basename(self.model_config.model),
                block_dim=0,
                kv_caches=views,
                # _rank, not self.rank: the latter comes from the world
                # group, which reports 0 on every TP rank because the
                # connector is built before distributed init. Two ranks
                # claiming rank 0 collide on the management port and
                # would overwrite each other's shards.
                rank=self._rank if self._rank is not None else self.rank,
                tp_size=self.tp_size,
            )
            self._backend.bind_store(self.kvstore)
            logger.info("Registered %d KV cache layers (%d page views)",
                        len(kv_caches), len(views))

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        # Submits every load, then host-blocks on the recurrent ones;
        # attention layers are waited by their own hooks during forward.
        self._worker.start_load(self._metadata())
        return
    def wait_for_layer_load(self, layer_name: str) -> None:
        # This attention layer's pages. GDN was already waited for
        # before forward began.
        self._worker.wait_layer_load(layer_name)
        return
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        if self._connector_metadata is None:
            return
        # Pipelined attention save: submit this layer's D2H+zip now
        # so it overlaps the remaining layers' compute. GDN groups
        # never reach this hook; they save in wait_for_save.
        self._worker.save_kv_layer(layer_name, self._metadata())
        return
    def wait_for_save(self) -> None:
        if self._worker is None:
            return

        hw = self._worker
        # A sticky load poison and any un-waited load must surface here,
        # before anything is persisted: entering the save path after the
        # forward read unrestored state would commit wrong data.
        hw.raise_load_poison()
        hw.loads_drained_check()
        if not hw.save_enabled():
            return
        metadata = self._metadata()
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("wait_for_save worker: reqs_to_save=%d",
                        len(metadata.reqs_to_save.requests))
        pages, boundaries = hw.wait_save(metadata)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info("chunk_save: %d pages, %d boundaries",
                        pages, boundaries)
        hw.debug_dump_state()

    def shutdown(self) -> None:
        """Release the store backend (Record flush, writer lease)."""
        if self._worker is not None:
            self._worker.shutdown()
        elif self._backend is not None:
            self._backend.close()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Report transfers that completed since the last step.

        Worker-side only; the scheduler role has nothing in flight.
        """
        if self._worker is None:
            return None, None
        # A sticky poison surfaces at every hook entry and is never
        # swallowed by the finish protocol; raising here also stops a
        # poisoned request from being reported as successfully loaded
        # (vLLM aborts it instead, so it cannot hang).
        self._worker.raise_load_poison()
        # Saves complete within the step, so nothing is ever reported as
        # finished-sending. Loads may not: an async request stays parked
        # until we name it here.
        recving = self._worker.poll_finished_loads()
        return None, (recving or None)
