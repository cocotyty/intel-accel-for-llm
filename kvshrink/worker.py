# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Worker-side execution engine for the hybrid (GDN) connector path."""
from __future__ import annotations



import logging
import os
from dataclasses import replace
from typing import Optional

from .layout import CacheKey
from .layout import LookupStatus

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

    def __init__(self, layer_tasks: dict, gate_layers: set):
        self.layer_tasks = layer_tasks
        self.gate_layers = gate_layers
        self.released = False


class HybridWorker:
    """Worker-role executor for the hybrid path (see module docstring)."""

    def __init__(self, groups, layer_infos, num_blocks, backend,
                 canonicalizer, rank: int, tp_size: int):
        self._groups = groups
        self._layer_infos = layer_infos
        self._num_blocks = num_blocks
        self._backend = backend
        self._canon = canonicalizer
        self.rank = rank
        self.tp_size = tp_size

        self._kv_caches_ref = None
        # Per-step load tasks: layer_name -> list of per-call engine
        # task dicts. Populated by start_load, popped by the per-layer
        # waits. A leftover at step end means a wait never ran ->
        # fail-stop (residue check in wait_save).
        self._load_tasks: dict[str, list] = {}
        # Pipelined attention saves: layer_name -> (group_idx, tasks).
        self._step_attn_saves: dict = {}
        # Sticky LOAD poison (allocation-after-HIT failures must
        # fail-stop every later worker hook, never degrade to recompute).
        self._load_poison: Optional[BaseException] = None
        # In-flight ASYNC loads: req_id -> _AsyncLoad. Unlike
        # _load_tasks these deliberately OUTLIVE the step that submitted
        # them -- the whole point is that the request is not occupying a
        # forward step while its pages arrive. Entries leave in two
        # stages: released (reported through get_finished, so vLLM may
        # schedule the request again) and then drained (its remaining
        # layers waited by the per-layer hooks during that forward).
        self._async_loads: dict[str, "_AsyncLoad"] = {}

        # attention layer_name -> group idx (mamba layers map out).
        self._attn_layer_group = {
            ln: g.group_idx for g in groups if g.kind != "mamba"
            for ln in g.layer_names}
        # Piggyback map (built in register): attention layer name ->
        # tuple of GDN layer names that execute after it and before the
        # next attention layer. Plus the leading GDN segment.
        self._mamba_layers: frozenset[str] = frozenset()
        """Wire up the worker-role pieces: the group/layer layout, the
        boundary backend and the canonical page-view builder for this
        rank's block pool, plus this rank's TP identity (the worker
        persists and loads its OWN shard).

        Initializes the per-step task bookkeeping (load tasks, stashed
        attention saves) and the sticky load-poison latch -- the
        worker is the EXECUTE side; it owns the writer lease, while
        the scheduler only plans against a read-only backend."""

    # ------------------------------------------------------------------
    # registration
    # ------------------------------------------------------------------
    def register(self, kv_caches, execution_order: list[str]) -> None:
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
        self._attn_order = tuple(attn_order)
        logger.info(
            "kvshrink hybrid worker registered: %d layers, %d attention "
            "hook points, %d recurrent layers (namespace tp=%d rank=%d)",
            len(self._layer_infos), len(attn_order),
            len(self._mamba_layers), self.tp_size, self.rank)

    # ------------------------------------------------------------------
    # poison (sticky fail-stop)
    # ------------------------------------------------------------------
    def raise_load_poison(self) -> None:
        """Re-raise the sticky load failure so no later hook proceeds
        (fail-closed).
        """
        if self._load_poison is not None:
            raise self._load_poison

    def _poison_load(self, error: BaseException) -> None:
        """Latch a load failure as sticky (first error wins). The
        failure is re-raised by every later worker hook, so a
        partially-loaded step can never enter forward and silently
        emit wrong output."""
        if self._load_poison is None:
            self._load_poison = error
        logger.error("kvshrink load poison: %s", error)

    def _worker_key(self, key: CacheKey) -> CacheKey:
        """Remap a scheduler-built key (rank 0) to this worker's own
        rank: each TP rank persists and loads its OWN shard under its
        own rank path. Without this, TP>1 workers overwrite each
        other's pages under the shared rank-0 key."""
        if key.rank == self.rank:
            return key
        return replace(key, rank=self.rank)

    def _layer_views(self, layer_name: str):
        """Canonical page views over the raw KV tensors of one layer:
        part key -> (num_blocks, page_bytes) GPU view. The chunk
        engine moves rows of these views, indexed by GPU block id."""
        parts, _chunk_dim = self._canon.page_view_parts(layer_name)
        return parts

    # ------------------------------------------------------------------
    # load path
    # ------------------------------------------------------------------
    def start_load(self, metadata) -> int:
        """Submit ALL of this step's loads, then host-block on the GDN
        ones. Attention layers stay pipelined: vLLM calls a hook on
        entry to each of them, so their pages are waited for exactly
        when they are about to be read.
        """
        self.raise_load_poison()
        if self._load_tasks:
            # A previous step's submit aborted midway and its residue
            # was never drained -- refuse to silently drop in-flight
            # engine tasks (fail-stop).
            err = RuntimeError(
                "kvshrink chunk load: stale step residue "
                f"{sorted(self._load_tasks)}: the previous step's load "
                "was never drained (hook path aborted?)")
            self._poison_load(err)
            raise err
        self._load_tasks = {}
        self._step_attn_saves = {}
        npages = 0
        _t0 = _now()
        try:
            for req_meta in metadata.reqs_to_load:
                # Async requests get their OWN sink: their tasks must
                # survive this step, and must not be host-blocked by
                # the GDN barrier below (that barrier is for requests
                # about to enter forward; an async one is not).
                is_async = bool(getattr(req_meta, "is_async", False))
                sink = {} if is_async else self._load_tasks
                for op in getattr(req_meta, "group_ops", []):
                    npages += self._submit_op_load(req_meta, op, sink)
                if is_async:
                    self._register_async_load(req_meta, sink)
        except BaseException as e:
            # Submit-stage failures (pool budget, engine errors) must
            # poison like wait-stage failures: a partially submitted
            # step can never enter forward.
            self._poison_load(e)
            raise
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

    def _register_async_load(self, req_meta, sink: dict) -> None:
        """Track one request's cross-step load and compute its release
        gate.
        """
        if not sink:
            return
        req_id = req_meta.req_id
        if req_id in self._async_loads:
            err = RuntimeError(
                f"kvshrink async load: request {req_id} already has an "
                "in-flight load; a second plan would strand the first")
            self._poison_load(err)
            raise err
        recurrent = {ln for ln in sink if ln not in self._attn_layer_group}
        n = getattr(req_meta, "async_load_layers", -1)
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

    def poll_finished_loads(self) -> set:
        """Report async requests whose gate layers have landed."""
        finished: set = set()
        for req_id, entry in list(self._async_loads.items()):
            if entry.released:
                continue
            gate_tasks = [td for ln in entry.gate_layers
                          for td in entry.layer_tasks.get(ln, ())]
            try:
                # Poll each submission separately: a task entry is one
                # engine call's task set, and the backend expects one
                # such set per call, not a flattened list of them.
                if any(not self._backend.wait_layer_loads(td, wait=False)
                       for td in gate_tasks):
                    continue  # not landed yet; ask again next step
                # Landed: finalize the gate layers and hand the rest to
                # the per-layer hooks.
                for ln in list(entry.gate_layers):
                    tds = entry.layer_tasks.pop(ln, [])
                    if tds:
                        self._wait_tasks(tds)
            except BaseException as e:  # noqa: BLE001 - see docstring
                self._poison_load(e)
                logger.exception(
                    "async load failed for req=%s; reporting it finished "
                    "so it cannot hang, poisoned so it cannot be trusted",
                    req_id)
                self._async_loads.pop(req_id, None)
                finished.add(req_id)
                continue
            entry.released = True
            finished.add(req_id)
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)
        return finished

    def _drain_async_layer(self, layer_name: str) -> list:
        """Collect any released async request's tasks for this layer."""
        tds: list = []
        for req_id, entry in list(self._async_loads.items()):
            if not entry.released:
                continue
            tds += entry.layer_tasks.pop(layer_name, [])
            if not entry.layer_tasks:
                self._async_loads.pop(req_id, None)
        return tds

    def wait_layer_load(self, layer_name: str) -> None:
        """Attention-layer entry hook: wait this layer's pages."""
        self.raise_load_poison()
        tds = self._load_tasks.pop(layer_name, [])
        tds += self._drain_async_layer(layer_name)
        if tds:
            self._wait_tasks(tds)

    def loads_drained_check(self) -> None:
        """Fail-stop if any submitted load was never waited (a hook
        never ran -> forward just read unrestored state)."""
        if self._load_tasks:
            err = RuntimeError(
                "kvshrink load left unrestored layers "
                f"{sorted(self._load_tasks)}: their layer hook never ran")
            self._poison_load(err)
            raise err

    def _submit_op_load(self, req_meta, op, sink) -> int:
        """Submit one GroupTransferMeta to the engine (async get)."""
        group = self._groups[op.group_idx]
        if not op.keys:
            return 0
        if group.kind == "mamba":
            # The scheduler's HIT and this submit are not atomic.
            first = self._worker_key(op.keys[0])
            boundary_key = replace(first, layer_name="")
            if self._backend.lookup_boundary(
                    boundary_key) != LookupStatus.HIT:
                err = RuntimeError(
                    "kvshrink mamba load: boundary vanished after HIT "
                    f"req={req_meta.req_id} boundary="
                    f"{op.snapshot_boundary_tokens}; refusing to enter "
                    "forward with unrestored state")
                self._poison_load(err)
                raise err
        by_layer: dict[str, list] = {}
        for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
            by_layer.setdefault(key.layer_name, []).append(
                (gpu_block_id, key.hash_str))
        if not by_layer:
            return 0
        # Every layer of the group addresses the same chunk sequence
        # (scheduler invariant: keys expand per block x layer).
        any_layer = next(iter(by_layer))
        entries = by_layer[any_layer]
        for layer_name, ent in by_layer.items():
            if ent != entries:
                err = RuntimeError(
                    "kvshrink chunk load: inconsistent op expansion "
                    f"req={req_meta.req_id} group={op.group_idx} "
                    f"layer={layer_name}")
                self._poison_load(err)
                raise err
        views = {ln: self._layer_views(ln) for ln in by_layer}
        # Split into calls with unique chunk labels (one engine call
        # maps chunk_labels 1:1 to chunk_indices).
        calls: list[tuple[list, list]] = []
        slot_of: dict[str, int] = {}
        for gpu_block_id, h in entries:
            c = slot_of.get(h)
            if c is None:
                slot_of[h] = len(calls)
                calls.append(([], []))
                c = slot_of[h]
            calls[c][0].append(gpu_block_id)
            calls[c][1].append(h)
        npages = 0
        for indices, labels in calls:
            tasks = self._backend.submit_group_loads(
                op.group_idx, views, indices, labels)
            for layer_name, td in tasks.items():
                sink.setdefault(layer_name, []).append(td)
                npages += len(indices)
        return npages

    def _wait_tasks(self, task_dicts) -> None:
        """Host-block until these engine tasks landed (fail-stop)."""
        try:
            for td in task_dicts:
                self._backend.wait_layer_loads(td)
        except BaseException as e:
            self._poison_load(e)
            raise

    # ------------------------------------------------------------------
    # save path
    # ------------------------------------------------------------------
    def save_enabled(self) -> bool:
        """Reflects the KVSHRINK_SAVE switch: ON by default; "0"
        disables production saving and KVSHRINK_DEBUG_AUTOSAVE=1
        force-enables it."""
        return (os.getenv("KVSHRINK_SAVE", "1") != "0"
                or os.getenv("KVSHRINK_DEBUG_AUTOSAVE") == "1")

    def _gather_save_candidates(self, metadata) -> dict:
        """Batch-level boundary candidates with cross-request dedup.
        Returns boundary_key -> {"group_idx", "pages": {layer_name:
        (key, gpu_block_id)}, "boundary_tokens"}."""
        candidates: dict[tuple, dict] = {}
        for req_meta in metadata.reqs_to_save:
            for op in req_meta.group_ops:
                for key, gpu_block_id in zip(op.keys, op.gpu_block_ids):
                    key = self._worker_key(key)
                    cand = candidates.get(key.boundary_key)
                    if cand is None:
                        cand = {"group_idx": op.group_idx,
                                "pages": {},
                                "boundary_tokens": None}
                        candidates[key.boundary_key] = cand
                    cand["pages"][key.layer_name] = (key, gpu_block_id)
                    if op.snapshot_boundary_tokens is not None:
                        cand["boundary_tokens"] = \
                            op.snapshot_boundary_tokens
        return candidates

    def _submit_group_layers_save(self, g_idx, layer_names, entries):
        """Submit ONE async engine put covering ``layer_names`` for the
        blocks in ``entries`` (list of (gpu_block_id, chunk_label), same
        order for every layer -- scheduler invariant). Async D2H+zip on
        the engine's put_stream, self-gated on the compute stream so it
        reads final values. Returns the engine tasks dict."""
        chunk_indices = [gpu for gpu, _ in entries]
        chunk_labels = [h for _, h in entries]
        if len(set(chunk_labels)) != len(chunk_labels):
            raise RuntimeError(
                "kvshrink chunk save: duplicate chunk labels in one "
                f"engine call group={g_idx} (candidates dedup broken)")
        views = {ln: self._layer_views(ln) for ln in layer_names}
        tasks = self._backend.submit_group_stores(
            g_idx, views, chunk_indices, chunk_labels)
        return tasks

    def save_kv_layer(self, layer_name: str, metadata) -> None:
        """Pipelined attention save. vLLM calls this on exit of EVERY
        attention layer during forward (kv_transfer_utils decorator).
        """
        if os.getenv("KVSHRINK_SAVE_PIPELINED", "1") == "0":
            return
        if not self.save_enabled():
            return
        g_idx = self._attn_layer_group.get(layer_name)
        if g_idx is None:
            return  # not an attention layer we serve (fast path)
        expected = sorted(self._groups[g_idx].layer_names)
        entries = []
        for _bkey, cand in self._gather_save_candidates(metadata).items():
            if cand["group_idx"] != g_idx:
                continue
            if sorted(cand["pages"]) != expected:
                continue  # partial boundary: skipped at commit time too
            if layer_name not in cand["pages"]:
                continue
            key, gpu_block_id = cand["pages"][layer_name]
            entries.append((gpu_block_id, key.hash_str))
        if not entries:
            return
        tasks = self._submit_group_layers_save(g_idx, [layer_name],
                                               entries)
        self._step_attn_saves[layer_name] = (g_idx, tasks)

    def wait_save(self, metadata) -> tuple[int, int]:
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
        per_group: dict[int, dict] = {}
        for bkey, cand in candidates.items():
            namespace, tp_size, rank, blk_hash, g_idx = bkey
            expected = sorted(self._groups[g_idx].layer_names)
            if sorted(cand["pages"]) != expected:
                # A partial boundary is skipped (the scheduler's
                # incremental cursor has advanced, so it ages out as a
                # MISS, never wrong data). Log unconditionally: this is
                # the one save-side anomaly that does not fail-stop.
                logger.warning(
                    "chunk_save skip commit g%d h=%s: expected %d "
                    "layers, stored %d (%s)", g_idx, blk_hash,
                    len(expected), len(cand["pages"]),
                    set(expected) ^ set(cand["pages"]))
                continue
            gslot = per_group.setdefault(g_idx, {"bnds": [],
                                                 "layers": {}})
            gslot["bnds"].append((bkey, cand))
            for layer_name in expected:
                key, gpu_block_id = cand["pages"][layer_name]
                gslot["layers"].setdefault(layer_name, []).append(
                    (gpu_block_id, key.hash_str))
        npages = 0
        nbound = 0
        for g_idx, gslot in per_group.items():
            layers = gslot["layers"]
            any_layer = next(iter(layers))
            entries = layers[any_layer]
            for layer_name, ent in layers.items():
                if ent != entries:
                    raise RuntimeError(
                        "kvshrink chunk save: inconsistent op expansion "
                        f"group={g_idx} layer={layer_name}")
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
                    self._backend.wait_group_stores(tasks)
            else:
                tasks = self._submit_group_layers_save(
                    g_idx, list(layers), entries)
                self._backend.wait_group_stores(tasks)
            chunk_indices = [gpu for gpu, _ in entries]
            npages += len(chunk_indices) * len(layers)
            # A block is finalized by its own write, with every layer of
            # the group in one call, so it is committed the moment the
            # write lands. There is no second phase to publish and
            # therefore nothing that can outlive its data.
            nbound += len(gslot["bnds"])
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
                self._backend.wait_group_stores(tasks)
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

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Flush and release the backend (Record sync, writer lease),
        then re-raise the first sticky load poison so cleanup never
        masks it."""
        errors = []
        try:
            self._backend.close()
        except BaseException as e:  # pragma: no cover - collect
            errors.append(e)
        if self._load_poison is not None:
            errors.append(self._load_poison)
        if errors:
            raise errors[0]
