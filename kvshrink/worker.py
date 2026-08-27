# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Worker-side execution engine for the hybrid (GDN) connector path."""
from __future__ import annotations



import logging
import os
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Optional

from .backend import group_label
from .kvshrink_connector import (CacheKey, Canonicalizer, GroupInfo,
                                 GroupTransferMeta, KVShrinkConnectorMetadata,
                                 LayerPageInfo, ReqMeta, save_enabled)

if TYPE_CHECKING:
    import torch

    from iaxl import KVStore
    from iaxl.kvflow.flow import Task

# log under the vllm.* namespace: vLLM only configures the "vllm"
# logger (handler+level); an unconfigured logger would drop INFO
# evidence lines that the GPU probes grep for.
logger = logging.getLogger("vllm." + __name__)


def _now() -> float:
    """Monotonic clock for step-latency accounting: immune to NTP
    steps, so measured durations are never negative."""
    import time as _t
    return _t.monotonic()


class _AsyncLoad:
    """One request's cross-step load."""

    __slots__ = ("layer_tasks", "gate_layers", "released")

    def __init__(
        self, layer_tasks: dict[str, list[dict[str, Task]]],
        gate_layers: set[str],
    ):
        self.layer_tasks: dict[str, list[dict[str, Task]]] = layer_tasks
        self.gate_layers: set[str] = gate_layers
        self.released: bool = False


@dataclass
class _SaveCandidate:
    """One boundary's pages accumulated across this step's save plans
    (cross-request dedup by boundary key)."""
    group_idx: int
    pages: dict[str, tuple[CacheKey, int]] = field(default_factory=dict)


class HybridWorker:
    """Worker-role executor for the hybrid path (see module docstring)."""

    def __init__(self, groups: list[GroupInfo],
                 layer_infos: dict[str, LayerPageInfo], namespace: str,
                 canonicalizer: Canonicalizer, rank: int, tp_size: int):
        """Wire up the worker-role pieces: the group/layer layout, this
        rank's store labels and the canonical page-view builder for
        this rank's block pool, plus this rank's TP identity (the worker
        persists and loads its OWN shard).
        """
        self._groups: list[GroupInfo] = groups
        self._layer_infos: dict[str, LayerPageInfo] = layer_infos
        self._canon: Canonicalizer = canonicalizer
        self.rank: int = rank
        self.tp_size: int = tp_size
        # Store namespace per group (see backend.py's module docstring
        # for why group and rank must be part of it).
        self._labels: list[str] = [
            group_label(namespace, g.group_idx, rank) for g in groups]
        # The store cannot exist until vLLM hands over kv_caches, long
        # after the connector is built; the connector assigns this in
        # register_kv_caches.
        self.store: Optional[KVStore] = None

        self._kv_caches_ref: Optional[dict[str, torch.Tensor]] = None
        # Per-step load tasks: layer_name -> list of per-call engine
        # task dicts. Populated by start_load, popped by the per-layer
        # waits.
        self._load_tasks: dict[str, list[dict[str, Task]]] = {}
        # Pipelined attention saves: layer_name -> (group_idx, tasks).
        self._step_attn_saves: dict[str, tuple[int, dict[str, Task]]] = {}
        # In-flight ASYNC loads: req_id -> _AsyncLoad. Unlike
        # _load_tasks these deliberately OUTLIVE the step that submitted
        # them -- the whole point is that the request is not occupying a
        # forward step while its pages arrive. Entries leave in two
        # stages: released (reported through get_finished, so vLLM may
        # schedule the request again) and then drained (its remaining
        # layers waited by the per-layer hooks during that forward).
        self._async_loads: dict[str, "_AsyncLoad"] = {}

        # attention layer_name -> group idx (mamba layers map out).
        self._attn_layer_group: dict[str, int] = {
            ln: g.group_idx for g in groups if g.kind != "mamba"
            for ln in g.layer_names}
        # All GDN layer names, waited as one barrier in start_load
        # before any attention layer runs (populated in register).
        self._mamba_layers: frozenset[str] = frozenset()

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(
        self, kv_caches: dict[str, torch.Tensor], execution_order: list[str]
    ) -> None:
        """Bind canonical page views and record which layers recur."""
        self._kv_caches_ref = kv_caches
        self._canon.register(kv_caches)

        self._mamba_layers = frozenset(
            ln for g in self._groups if g.kind == "mamba"
            for ln in g.layer_names)
        attn_order = [ln for ln in execution_order
                      if ln in self._attn_layer_group]
        if not attn_order:
            raise RuntimeError(
                "kvshrink hybrid: no attention layers found in the "
                "execution order; nothing would ever wait for a load")
        self._attn_order: tuple[str, ...] = tuple(attn_order)
        logger.info(
            "kvshrink hybrid worker registered: %d layers, %d attention "
            "hook points, %d recurrent layers (namespace tp=%d rank=%d)",
            len(self._layer_infos), len(attn_order),
            len(self._mamba_layers), self.tp_size, self.rank)

    def _worker_key(self, key: CacheKey) -> CacheKey:
        """Remap a scheduler-built key (rank 0) to this worker's own
        rank: each TP rank persists and loads its OWN shard under its
        own rank path. Without this, TP>1 workers overwrite each
        other's pages under the shared rank-0 key."""
        if key.rank == self.rank:
            return key
        return replace(key, rank=self.rank)

    # ------------------------------------------------------------------
    # store transfers
    # ------------------------------------------------------------------
    # A layer contributes one page view, or two when its K and V are
    # separate tensors. The engine takes one flat tensor dict, so part
    # views are flattened under "layer::part" keys (loads regroup the
    # returned tasks by layer with an rsplit on "::").
    @staticmethod
    def _flat_views(
        layer_views: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        return {f"{ln}::{part}": view
                for ln, parts in layer_views.items()
                for part, view in parts.items()}

    def _wait_load(
        self, layer_tasks: dict[str, Task], wait: bool = True
    ) -> bool:
        """Block until the reads land, or with ``wait=False`` report
        whether they have without consuming them -- the poll used to
        decide if an async request may be released, which must not stall
        the step doing the asking.
        """
        if not layer_tasks:
            return True
        if not wait:
            return bool(
                self.store.get_wait(get_results=layer_tasks, wait=False))
        if not self.store.get_wait(get_results=layer_tasks, wait=True):
            raise RuntimeError(
                "kvshrink load failed: get_wait reported an incomplete "
                "transfer; forward would read unrestored blocks")
        return True

    def _wait_store(self, tasks: dict[str, Task]) -> None:
        """Host-block until these writes land. An incomplete write is
        fail-stop, same as an incomplete load: the scheduler's save
        cursor has already advanced past these blocks, so losing them
        silently would skip them for the rest of the process."""
        if not tasks:
            return
        if not self.store.put_wait(put_results=tasks, wait=True):
            raise RuntimeError(
                "kvshrink save failed: put_wait reported an incomplete "
                "transfer; the save cursor has already advanced")

    # ------------------------------------------------------------------
    # load path
    # ------------------------------------------------------------------
    def start_load(self, metadata: KVShrinkConnectorMetadata) -> int:
        """Submit ALL of this step's loads, then host-block on the GDN
        ones. Attention layers stay pipelined: vLLM calls a hook on
        entry to each of them, so their pages are waited for exactly
        when they are about to be read.
        """
        self._load_tasks = {}
        self._step_attn_saves = {}
        npages = 0
        _t0 = _now()
        for req_id, req_meta in metadata.reqs_to_load.requests.items():
            if req_meta.is_async:
                # Async tasks must survive this step and must not be
                # host-blocked by the GDN barrier below (that barrier
                # is for requests about to enter forward; an async one
                # is not): collect them into a private dict.
                tasks: dict[str, list[dict[str, Task]]] = {}
                for op in req_meta.group_ops:
                    npages += self._submit_op_load(op, tasks)
                self._register_async_load(req_id, req_meta, tasks)
            else:
                for op in req_meta.group_ops:
                    npages += self._submit_op_load(op, self._load_tasks)
        # Every GDN layer, waited before forward begins.
        recurrent = [td for ln in self._mamba_layers
                     for td in self._load_tasks.pop(ln, [])]
        if recurrent:
            self._wait_tasks(recurrent)
        if npages:
            logger.info(
                "start_load_kv: %d pages loaded "
                "elapsed_ms=%.3f (rank %d/%d)", npages,
                (_now() - _t0) * 1e3, self.rank, self.tp_size)
        return npages

    def _register_async_load(
        self, req_id: str, req_meta: ReqMeta,
        sink: dict[str, list[dict[str, Task]]],
    ) -> None:
        """Track one request's cross-step load and compute its release
        gate.
        """
        if not sink:
            return
        recurrent = {ln for ln in sink if ln not in self._attn_layer_group}
        n = req_meta.async_load_layers
        if n is None or n < 0:
            gate = set(sink)
        else:
            prefix = [ln for ln in self._attn_order if ln in sink][:n]
            gate = recurrent | set(prefix)
        self._async_loads[req_id] = _AsyncLoad(sink, gate)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "async load req=%s layers=%d gate=%d (recurrent=%d) "
                "requested_prefix=%s", req_id, len(sink), len(gate),
                len(recurrent), n)

    def poll_finished_loads(self) -> set[str]:
        """Report async requests whose gate layers have landed."""
        finished: set[str] = set()
        for req_id, entry in list(self._async_loads.items()):
            if entry.released:
                continue
            gate_tasks = [td for ln in entry.gate_layers
                          for td in entry.layer_tasks.get(ln, ())]
            # Poll each submission separately: a task entry is one
            # engine call's task set, and get_wait expects one
            # such set per call, not a flattened list of them.
            if any(not self._wait_load(td, wait=False)
                   for td in gate_tasks):
                continue  # not landed yet; ask again next step
            # Landed: finalize the gate layers and hand the rest to
            # the per-layer hooks.
            for ln in list(entry.gate_layers):
                tds = entry.layer_tasks.pop(ln, [])
                if tds:
                    self._wait_tasks(tds)
            entry.released = True
            finished.add(req_id)
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)
        return finished

    def wait_layer_load(self, layer_name: str) -> None:
        """Attention-layer entry hook: wait this layer's pages."""
        tds = self._load_tasks.pop(layer_name, [])
        for req_id, entry in list(self._async_loads.items()):
            if not entry.released:
                continue
            tds += entry.layer_tasks.pop(layer_name, [])
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)
        if tds:
            self._wait_tasks(tds)

    def _submit_op_load(
        self, op: GroupTransferMeta,
        sink: dict[str, list[dict[str, Task]]],
    ) -> int:
        """Submit one GroupTransferMeta to the engine (async get).

        Returns the number of (layer, block) pages covered."""
        if not op.keys:
            return 0
        by_layer: dict[str, list[tuple[int, str]]] = {}
        for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
            by_layer.setdefault(key.layer_name, []).append(
                (gpu_block_id, key.hash_str))
        if not by_layer:
            return 0
        # Every layer of the group addresses the same chunk sequence
        # (scheduler invariant: keys expand per block x layer).
        entries = by_layer[next(iter(by_layer))]
        tensors = self._flat_views(
            {ln: self._canon.page_view_parts(ln) for ln in by_layer})
        tasks = self.store.get(block_indices=[gpu for gpu, _ in entries],
                               block_hashs=[h for _, h in entries],
                               layer_names=list(tensors),
                               tensors=tensors,
                               label=self._labels[op.group_idx])
        for key, task in tasks.items():
            sink.setdefault(key.rsplit("::", 1)[0], []).append(
                {key: task})
        return len(entries) * len(tensors)

    def _wait_tasks(self, task_dicts: list[dict[str, Task]]) -> None:
        """Host-block until these engine tasks landed (fail-stop)."""
        for td in task_dicts:
            self._wait_load(td)

    # ------------------------------------------------------------------
    # save path
    # ------------------------------------------------------------------
    def _gather_save_candidates(
        self, metadata: KVShrinkConnectorMetadata
    ) -> dict[tuple[str, int, int, str, int], _SaveCandidate]:
        """Batch-level boundary candidates with cross-request dedup.
        Returns boundary_key -> accumulated per-layer pages."""
        candidates: dict[tuple[str, int, int, str, int], _SaveCandidate] = {}
        for req_meta in metadata.reqs_to_save.requests.values():
            for op in req_meta.group_ops:
                for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
                    key = self._worker_key(key)
                    cand = candidates.get(key.boundary_key)
                    if cand is None:
                        cand = _SaveCandidate(op.group_idx)
                        candidates[key.boundary_key] = cand
                    cand.pages[key.layer_name] = (key, gpu_block_id)
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
        return self.store.put(block_indices=[gpu for gpu, _ in entries],
                              block_hashs=[h for _, h in entries],
                              layer_names=list(tensors),
                              tensors=tensors,
                              label=self._labels[g_idx])

    def save_kv_layer(
        self, layer_name: str, metadata: KVShrinkConnectorMetadata
    ) -> None:
        """Pipelined attention save. vLLM calls this on exit of EVERY
        attention layer during forward (kv_transfer_utils decorator).
        """
        if os.getenv("KVSHRINK_SAVE_PIPELINED", "1") == "0":
            return
        if not save_enabled():
            return
        g_idx = self._attn_layer_group.get(layer_name)
        if g_idx is None:
            return  # not an attention layer we serve (fast path)
        expected = sorted(self._groups[g_idx].layer_names)
        entries: list[tuple[int, str]] = []
        for _bkey, cand in self._gather_save_candidates(metadata).items():
            if cand.group_idx != g_idx:
                continue
            if sorted(cand.pages) != expected:
                continue  # partial boundary: skipped at commit time too
            if layer_name not in cand.pages:
                continue
            key, gpu_block_id = cand.pages[layer_name]
            entries.append((gpu_block_id, key.hash_str))
        if not entries:
            return
        tasks = self._submit_group_layers_save(g_idx, [layer_name],
                                               entries)
        self._step_attn_saves[layer_name] = (g_idx, tasks)

    def wait_save(
        self, metadata: KVShrinkConnectorMetadata
    ) -> tuple[int, int]:
        """Post-forward save: GDN groups submit here; attention groups
        collect their pipelined tasks; then wait for the writes,
        write every page of every group.
        Fail-stop on any anomaly (the scheduler already advanced its
        incremental indices). Returns (pages, boundaries)."""
        if self._kv_caches_ref is None:
            return 0, 0
        _t0 = _now()
        candidates = self._gather_save_candidates(metadata)
        pipelined = os.getenv("KVSHRINK_SAVE_PIPELINED", "1") != "0"
        # group the complete candidates for one engine put per group
        per_group: dict[int, dict[str, list[tuple[int, str]]]] = {}
        nbound = 0
        for bkey, cand in candidates.items():
            namespace, tp_size, rank, blk_hash, g_idx = bkey
            expected = sorted(self._groups[g_idx].layer_names)
            if sorted(cand.pages) != expected:
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
            layers = per_group.setdefault(g_idx, {})
            for layer_name in expected:
                key, gpu_block_id = cand.pages[layer_name]
                layers.setdefault(layer_name, []).append(
                    (gpu_block_id, key.hash_str))
        npages = 0
        for g_idx, layers in per_group.items():
            entries = layers[next(iter(layers))]
            if self._groups[g_idx].kind != "mamba" and pipelined:
                # Attention: save_kv_layer submitted each layer during
                # forward; wait for them here. A layer the
                # decorator never fired for is submitted now -- the
                # plan must be fully covered either way.
                for layer_name in layers:
                    stashed = self._step_attn_saves.pop(layer_name, None)
                    tasks = (stashed[1] if stashed is not None
                             else self._submit_group_layers_save(
                                 g_idx, [layer_name], entries))
                    self._wait_store(tasks)
            else:
                tasks = self._submit_group_layers_save(
                    g_idx, list(layers), entries)
                self._wait_store(tasks)
            chunk_indices = [gpu for gpu, _ in entries]
            npages += len(chunk_indices) * len(layers)
            # A block is finalized by its own write, with every layer of
            # the group in one call, so it is committed the moment the
            # write lands. There is no second phase to publish and
            # therefore nothing that can outlive its data.
        if self._step_attn_saves:
            # Layers whose submit was never consumed by a group above
            # (plan changed mid-step): wait them so pinned staging is
            # released, then drop. Their data is identical to what the
            # group path stored (same labels), so nothing is lost.
            logger.warning(
                "chunk_save: %d stashed layer saves unconsumed (%s); "
                "draining", len(self._step_attn_saves),
                sorted(self._step_attn_saves))
            for _ln, (_g, tasks) in self._step_attn_saves.items():
                self._wait_store(tasks)
            self._step_attn_saves.clear()
        if npages:
            # Counterpart of the start_load_kv line: without it a run
            # that saves nothing looks exactly like a healthy one.
            logger.info(
                "chunk_save: %d pages stored, %d boundaries committed "
                "elapsed_ms=%.3f (rank %d/%d)", npages, nbound,
                (_now() - _t0) * 1e3, self.rank, self.tp_size)
        return npages, nbound

    # ------------------------------------------------------------------
    # debug dump
    # ------------------------------------------------------------------
    def debug_dump_state(self) -> None:
        """KVSHRINK_DEBUG_DUMP=1: log sha256 of the first layer page of
        every mamba group at gpu blocks 0..9, so cold-vs-hot GPU states
        can be compared byte-exactly."""
        if not os.getenv("KVSHRINK_DEBUG_DUMP") \
                or self._kv_caches_ref is None:
            return
        import hashlib
        for group in self._groups:
            if group.kind != "mamba":
                continue
            ln = group.layer_names[0]
            for blk in range(10):
                page = self._canon.get_page(ln, blk)
                h = hashlib.sha256(
                    page.cpu().numpy().tobytes()).hexdigest()
                logger.info("DUMP g%d block=%d sha=%s",
                            group.group_idx, blk, h[:16])
