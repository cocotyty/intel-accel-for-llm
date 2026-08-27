# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Worker-side execution engine for the hybrid (GDN) connector path.

Owns everything the worker role does with a hybrid
KVShrinkConnectorMetadata: canonical page views, load submission,
load submission, pipelined attention save, and the post-forward
save commit. The connector facade (kvshrink_connector.py) only
dispatches.

Load pipelining without any vLLM patch
--------------------------------------
vLLM calls ``wait_for_layer_load`` at every ATTENTION layer's entry
(piecewise cudagraph is forced) but never at GDN layers. So:

- ``start_load``: submit ALL loads (attention pages + GDN snapshots)
  to the engine (async unzip+H2D on the engine's get_stream), then
  host-block ONLY on the LEADING GDN segment -- the GDN layers that
  execute before the first attention layer and therefore have no
  attention hook to ride on.
- ``wait_layer_load(attn_i)``: wait attention layer i's pages AND the
  GDN segment between attn_i and the next attention layer (those GDN
  layers execute after attn_i, so waiting at attn_i's entry is in
  time). Their transfers overlapped the preceding layers' compute --
  this IS the layer pipeline.

GDN snapshots are written into the CURR state slot: v0.23.0's GDN
execution metadata is pinned to the CURR block for both chunked-prefill
and decode, and preprocess_mamba's prev->curr copy runs before
start_load_kv, so a CURR write during forward is always safe and a PREV
write would be dead work.

Save path
---------
Attention groups save PIPELINED: ``save_kv_layer`` submits each layer's
async D2H+zip at that layer's exit (the layer's pages for this step are
final then). GDN groups save in ``wait_save`` (their state is final
only post-forward). Waiting for every write
all happen in ``wait_save``.

Fail-stop contract: any load/save anomaly raises (EngineCore fatal).
Silently dropping a save would lose a boundary permanently (the
scheduler already advanced its incremental indices); entering forward
with unrestored pages would emit wrong tokens (the core already skipped
recompute).
"""
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
    """One request's cross-step load.

    ``layer_tasks`` is drained in two phases and must not be waited all
    at once: the gate layers are waited when the request is released,
    the rest during forward by the layer hooks. Anything still present
    once forward has run means a hook never fired, which would mean the
    model read unrestored memory.

    ``gate_layers`` is what must land before the request may run at all.
    It always contains every recurrent (mamba/GDN) layer: a recurrent
    state is consumed whole at the very start of forward, so there is no
    such thing as releasing a request with half of it. Attention layers
    may be gated on a prefix because each one is waited immediately
    before its own kernels.
    """

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

        Initializes the per-step task bookkeeping (load tasks, stashed
        attention saves) -- the worker is the EXECUTE side; it owns the
        writer lease, while the scheduler only plans against a
        read-only store."""
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
        # Per-step load tasks, "layer::part" -> engine Task: populated
        # by start_load (one merged engine call per group per step),
        # waited and popped per layer -- GDN layers as one barrier in
        # start_load, attention layers at their forward-entry hooks.
        self._load_tasks: dict[str, Task] = {}
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
        """Bind canonical page views and record which layers recur.

        ``execution_order``: all cached layer names in model execution
        order (the connector derives it from static_forward_context or
        the layer-index naming convention, fail-closed). Only the
        attention layers' order is used, by the async release gate --
        "the first N layers" means nothing otherwise.
        """
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

    def _wait_load(self, tasks: dict[str, Task]) -> None:
        """Host-block until these reads land; an incomplete transfer is
        fatal -- forward is about to read these blocks."""
        if tasks and not self.store.get_wait(get_results=tasks, wait=True):
            raise RuntimeError(
                "kvshrink load failed: get_wait reported an incomplete "
                "transfer; forward would read unrestored blocks")

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

        GDN gets no such hook, so it is waited for here, in one barrier.
        That costs the overlap for a request's recurrent state -- one
        block per layer, a few tens of MB in total, against a forward
        of a wholly different order -- and in exchange there is no
        machinery deciding which attention layer is responsible for
        which GDN layer, and no way for a GDN layer to reach forward
        unwaited. Returns the number of (layer, block) pages submitted.
        """
        self._load_tasks = {}
        self._step_attn_saves = {}
        npages = 0
        _t0 = _now()
        # Sync loads merge per group into ONE engine call per step;
        # duplicate labels across requests (two requests loading the
        # same external block into different GPU blocks) are fine --
        # the engine transfers each (label, index) pair independently.
        by_group: dict[int, tuple[list[tuple[int, str]], set[str]]] = {}
        for req_id, req_meta in metadata.reqs_to_load.requests.items():
            if req_meta.is_async:
                # Async tasks must survive this step and must not be
                # host-blocked by the GDN barrier below (that barrier
                # is for requests about to enter forward; an async one
                # is not): collect them into a private dict.
                tasks: dict[str, Task] = {}
                for op in req_meta.group_ops:
                    t, n = self._submit_group_load(op)
                    tasks.update(t)
                    npages += n
                self._register_async_load(req_id, req_meta, tasks)
            else:
                for op in req_meta.group_ops:
                    entries, layers = by_group.setdefault(
                        op.group_idx, ([], set()))
                    entries.extend(self._op_entries(op))
                    layers.update(key.layer_name for key in op.keys)
        for g_idx, (entries, layers) in by_group.items():
            tasks, n = self._submit_group_load_entries(
                g_idx, entries, layers)
            self._load_tasks.update(tasks)
            npages += n
        # Every GDN layer, waited before forward begins.
        recurrent = {k: self._load_tasks.pop(k)
                     for k in list(self._load_tasks)
                     if k.rsplit("::", 1)[0] in self._mamba_layers}
        if recurrent:
            self._wait_load(recurrent)
        if npages:
            logger.info(
                "start_load_kv: %d pages loaded "
                "elapsed_ms=%.3f (rank %d/%d)", npages,
                (_now() - _t0) * 1e3, self.rank, self.tp_size)
        return npages

    def _register_async_load(
        self, req_id: str, req_meta: ReqMeta,
        tasks: dict[str, Task],
    ) -> None:
        """Track one request's cross-step load and compute its release
        gate.

        The gate always includes every recurrent layer present in the
        plan, whatever the configured layer count says. A GDN/Mamba
        state is read whole at the start of forward, so releasing a
        request whose state is still in flight would let the model run
        on stale memory -- silently, with plausible output. Attention
        layers are safe to gate on a prefix because every one of them is
        waited immediately before its own kernels.
        """
        if not tasks:
            return
        layers = {k.rsplit("::", 1)[0] for k in tasks}
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
        """Report async requests whose gate layers have landed.

        Called once per step from the connector's ``get_finished``. vLLM
        keeps an async request parked until we name it here. A failed
        transfer raises out of the poll (EngineCore fatal), same as a
        failed blocking wait.

        Polling is non-blocking on purpose: this runs inside some OTHER
        request's step, and blocking here would reintroduce exactly the
        stall async loading exists to remove.
        """
        finished: set[str] = set()
        for req_id, entry in list(self._async_loads.items()):
            if entry.released:
                continue
            gate = {k: t for k, t in entry.layer_tasks.items()
                    if k.rsplit("::", 1)[0] in entry.gate_layers}
            if gate and not self.store.get_wait(get_results=gate,
                                                wait=False):
                continue  # not landed yet; ask again next step
            # Landed: finalize the gate layers and hand the rest to
            # the per-layer hooks.
            self._wait_load(gate)
            for k in gate:
                del entry.layer_tasks[k]
            entry.released = True
            finished.add(req_id)
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)
        return finished

    def wait_layer_load(self, layer_name: str) -> None:
        """Attention-layer entry hook: wait this layer's pages.

        Drains EVERY released async entry, not just the ones running in
        this batch: the worker is not told which requests a forward step
        covers, and a task waited a step early only costs a wait for
        bytes already on their way, while a task never waited means
        forward read unrestored memory.
        """
        keys = [k for k in self._load_tasks
                if k.rsplit("::", 1)[0] == layer_name]
        if keys:
            self._wait_load({k: self._load_tasks.pop(k) for k in keys})
        for req_id, entry in list(self._async_loads.items()):
            if not entry.released:
                continue
            keys = [k for k in entry.layer_tasks
                    if k.rsplit("::", 1)[0] == layer_name]
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
        ``entries``; returns the flat tasks dict ("layer::part" ->
        Task) and the page count."""
        tensors = self._flat_views(
            {ln: self._canon.page_view_parts(ln) for ln in layer_names})
        tasks = self.store.get(block_indices=[gpu for gpu, _ in entries],
                               block_hashs=[h for _, h in entries],
                               layer_names=list(tensors),
                               tensors=tensors,
                               label=self._labels[g_idx])
        return tasks, len(entries) * len(tensors)

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

        An attention layer's page for this step's tokens is final the
        moment that layer returns, so this layer's D2H+zip can overlap
        the remaining layers' compute instead of adding to the
        post-forward critical path. GDN groups are NOT covered here:
        their layers never call this hook and their state is only final
        after forward -- they save in wait_save.

        This method only SUBMITS. Waiting for the writes
        stay in wait_save. Partial-boundary
        candidates are skipped here exactly as wait_save skips them, so
        the stashed per-layer entries stay aligned with the committable
        boundary list. KVSHRINK_SAVE_PIPELINED=0 disables this path
        (everything then submits in wait_save).
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
