# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-side (EngineCore process) KV cache planning.

Three responsibilities that all belong to "deciding what to transfer",
kept together because they are only ever used as one unit:

- the async-load policy, which decides whether a request waits for its
  pages in ``WAITING_FOR_REMOTE_KVS`` (freeing the GPU to run other
  requests) or blocks a forward step;
- ``HybridHitPolicy``, the fixed-point search for the longest boundary
  every KV cache group can serve;
- ``HybridRequestScheduler``, which tracks per-request block tables and
  emits the load/save plans the worker executes.

Nothing here touches GPU memory or the storage engine: this process
only produces plans. The worker is the sole executor, so every plan
must be self-describing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import Optional

from iaxl import generate_block_hashs

from .backend import lookup_boundary
from .kvshrink_connector import (CacheKey, GroupInfo, GroupTransferMeta,
                     ReqMeta, ReqGroupState, ReqState, make_boundary_key)

# ======================================================================
# request scheduler (EngineCore side)
# ======================================================================
# Scheduler-side request state for the hybrid connector.
#
# For each request we:
# 1. run the hit policy (find_longest_cache_hit) against the store,
# 2. after vLLM allocates blocks, record per-group block tables,
# 3. build load metadata: for attention groups, every hit block in the
#    prefix; for mamba groups, the single state snapshot block at the
#    restore boundary (every GDN load is waited at start_load_kv,
#    before forward begins),
# 4. build incremental save metadata and track resume/cursor
#    rollback lifecycle.


# Log under the vllm.* namespace: vLLM's init_logger attaches NO
# handler and relies on propagation to the configured "vllm" parent
# logger, so a bare __name__ logger silently drops every record in the
# EngineCore process.
logger = logging.getLogger("vllm." + __name__)


class HybridRequestScheduler:
    """Scheduler-side request state machine: hit detection + load/save
    plan builder.

    This object owns no vLLM interface methods itself; the connector
    facade (connector.py) holds the vLLM hooks and delegates. The vLLM
    trigger map below says WHEN each entry point fires and WHAT it is
    for; per-function docstrings describe HOW.

    One scheduling pass looks like this:

    1. NEW request arrives -> the core asks the connector
       ``get_num_new_matched_tokens(request, num_computed_tokens)``
       -> :meth:`get_num_new_matched_tokens`.
       Purpose: how many tokens beyond the local prefix-cache hit can be
       treated as already computed thanks to the external store.
       We run the hit policy over the request's block hashes (Record-
       gated, always synchronous), remember the authoritative restore
       point as ``snapshot_boundary``, and return the external token
       count. The core then skips recomputing those tokens.
    2. Block allocation succeeded (same pass) -> the core calls
       ``connector.update_state_after_alloc(request, blocks,
       num_external_computed_tokens)`` -> :meth:`update_state_after_alloc`.
       Purpose: tell us where the GPU blocks landed and how many
       external tokens the core accepted (i.e. will skip recompute for).
       We snapshot per-group block_ids and set ``pending_load_tokens`` --
       the external tokens the worker MUST restore before forward.
    3. End of the pass: the core calls
       ``connector.build_connector_meta(scheduler_output)``. The facade
       asks this object for per-request plans and ships them to the
       worker inside the connector metadata. Four kinds of work:

       a) LOAD plan for NEW requests -> :meth:`build_load_meta`.
          Restores pages up to the snapshot boundary recorded in step 1
          (never re-looks-up: after step 2 the progress counters already
          include external tokens, a fresh lookup would be polluted).
       b) LOAD plan for PREEMPTION-RESUMED requests ->
          :meth:`build_resumed_load_meta`. v1 carries resumed requests
          in ``scheduled_cached_reqs.resumed_req_ids``, NOT in
          ``scheduled_new_reqs``, so they need their own loop --
          missing them would yield garbage output after preemption. Guard: if the
          core accepted external tokens for the request but we find
          zero restorable pages, raise instead of entering forward with
          unrestored KV while the core skips recompute.
       c) SAVE plan for EVERY request scheduled this pass ->
          :meth:`build_save_meta`. Incremental: only blocks/boundaries
          not previously emitted are saved. The worker executes it
          AFTER forward, when the GPU pages hold state up to
          computed+scheduled tokens.
       d) Bookkeeping for RUNNING cached requests ->
          :meth:`on_cached_request`, done before (c). Sync the
          authoritative progress and block tables from upstream. On
          resume (or any progress regression) roll the save cursor
          back, so boundaries emitted before a preemption but never
          provably persisted get re-emitted (safe: overwrite is
          idempotent; skipping them would lose data).
    4. Request teardown -> the core calls
       ``connector.request_finished(request)`` ->
       :meth:`on_request_finished`, which drops the ReqState.

    How per-group block tables (ReqGroupState.block_ids) stay
    current
    ---------------------------------------------------------------
    block_ids is our copy of vLLM's block table for the request, one
    list per KV cache group. A block id is an index into that group's
    GPU block pool (not a raw address; the worker multiplies it by the
    page layout to locate the data).

    vLLM allocates new blocks inside ``kv_cache_manager.allocate_slots``
    during every scheduling pass, whenever a request crosses a block
    boundary (decode: a new block every block_size tokens; chunked
    prefill: at each boundary crossing). Those new block ids reach us
    through TWO channels, depending on which scheduling loop the
    request is in:

    - Requests scheduled from the WAITING queue (new and
      preemption-resumed): immediately after allocate_slots succeeds,
      the core calls ``connector.update_state_after_alloc`` with the
      request's FULL current block table
      (``kv_cache_manager.get_blocks(request_id)``). We replace our
      copy wholesale.
    - RUNNING requests: the core's running loop calls allocate_slots
      but does NOT notify the connector. Instead the newly allocated
      blocks travel inside the SchedulerOutput as
      ``scheduled_cached_reqs.new_block_ids`` (a parallel array to
      ``req_ids``). :meth:`on_cached_request` appends them to our copy
      (or replaces it for resumed requests, where upstream sends the
      full table again).

    Ordering inside build_connector_meta matters:
    :meth:`on_cached_request` (table sync) runs BEFORE
    :meth:`build_save_meta`, so the save plan already sees blocks
    allocated in the SAME pass. The plan is executed by the worker
    after forward (wait_for_save), at which point those blocks
    actually contain the computed KV data.

    One scheduling pass, top to bottom
    ---------------------------------------------------------------
    ::

      vLLM core (scheduler process)        this object
      ===================================  ==============================
      new/resumed request:
        get_num_new_matched_tokens(req)  -> hit lookup; pin
                                            snapshot_boundary
        allocate_slots(req)              (core allocates GPU blocks)
        update_state_after_alloc(req,    -> replace block_ids copies;
          blocks, num_external_tokens)       set pending_load_tokens
      running requests:
        allocate_slots(req)              (no connector callback; new
                                          blocks ride new_block_ids)
      build_connector_meta(sched_output)
        per new request                -> build_load_meta
        per resumed request            -> build_resumed_load_meta
                                          (fail-closed guard)
        per running request            -> on_cached_request FIRST
                                          (sync progress + tables),
                                          then build_save_meta
      ================= pickle: KVShrinkConnectorMetadata ==============
      worker process (per GPU rank)
        start_load_kv                  execute LOAD plans (BEFORE fwd)
        forward                        GPU computes this pass's tokens
        wait_for_save                  execute SAVE plans (AFTER fwd)

    Structure and data flow
    ---------------------------------------------------------------
    ::

      scheduler process                  worker process
      ============================       ============================
      HybridRequestScheduler             connector (worker role)
        ReqState (per request)         start_load_kv
          block_hashes ............        wait_for_save
            content hash per block             |
          snapshot_boundary ......             v
            restore point pinned           BoundaryBackend
            at lookup                          |
          pending_load_tokens ....             v
            external tokens the            TensorZip chunk engine
            worker must restore                |
          groups[g]:                           v
            block_ids ............         disk / host memory
              copy of vLLM's block           (content-addressed
              table, group g's pool           chunk store)
            next_stored_chunk_idx
              save cursor (rolled back
              on resume/regression)

    Save addressing: one logical block, two addresses
    ---------------------------------------------------------------
    ::

      token stream     | block 0 | block 1 | ... | block i |
                       (block_size tokens each)

      block_hashes[i] --> store key     (content-addressed: which
                                         chunks the page is written to
                                         / found under)
      block_ids[i]    --> GPU pool blk  (position-addressed: which
                                         physical block the worker
                                         reads the page from)

      A save op pairs them: keys[k] <-> gpu_block_ids[k].

    Everything else in this file is internal plumbing for the above
    (key builders, hash recompute).
    """

    def __init__(
        self,
        groups: list[GroupInfo],
        store,
        hash_block_size: int,
        namespace: str,
        tp_size: int,
        rank: int,
        async_load_config=None,
        block_hash_source: str = "vllm",
    ):
        """Record the per-group layout, the read-only presence store
        and TP identity, and own the per-request ReqState table plus
        the resume/cursor-rollback counter that spans a request's whole
        scheduling lifecycle.

        This is the DECISION side of the hybrid path: it only plans
        (hit lookup, load/save ReqMeta) against a read-only store;
        the worker executes transfers and owns the page views and this
        rank's writer lease."""
        self._groups = groups
        self._store = store
        self._hash_block_size = hash_block_size
        self._namespace = namespace
        self._tp_size = tp_size
        self._rank = rank
        # Async-load policy (AsyncLoadLayerConfig). None disables it, so
        # every request keeps the old behaviour of occupying a forward
        # step while its pages arrive.
        self._async_load_config = async_load_config
        # Where a block's cache identity comes from. This is a DATA
        # COMPATIBILITY switch, not a behavioural one: the two sources
        # produce different key values, so flipping it makes every
        # previously written entry unreachable (a cold cache, not a
        # corrupt one). Each layout therefore keeps the source it was
        # written with unless an operator says otherwise.
        if block_hash_source not in ("vllm", "legacy"):
            raise ValueError(
                f"unknown block hash source {block_hash_source!r}; "
                "expected 'vllm' or 'legacy'")
        self._block_hash_source = block_hash_source
        # Attention layers in execution order, used to size the
        # early-release prefix. Mamba layers are deliberately absent:
        # they are never partially released (see _decide_async).
        self._attention_layers: tuple[str, ...] = tuple(
            ln for g in groups if g.kind != "mamba" for ln in g.layer_names)
        self._req_states: dict[str, ReqState] = {}
        # Async requests whose load plan has not been emitted yet. See
        # update_state_after_alloc for why this cannot be derived from
        # the scheduler output.
        self._async_load_pending: set[str] = set()

    # ------------------------------------------------------------------
    def on_new_request(
        self, req_id: str, block_hashes: list[int],
        num_computed_tokens: int, request=None,
    ) -> None:
        """Register a fresh ReqState. Internal: called by us from
        get_num_new_matched_tokens / build_load_meta /
        update_state_after_alloc when a request first becomes visible
        (vLLM has no dedicated "new request" connector hook)."""
        live_source = None
        if request is not None:
            live_source = (request.block_hashes
                           if self._block_hash_source == "vllm"
                           else request.all_token_ids)
        self._req_states[req_id] = ReqState(
            live_source=live_source,
            block_hashes=list(block_hashes),
            num_computed_tokens=num_computed_tokens,
            groups=tuple(
                ReqGroupState() for _ in self._groups),
        )

    def take_async_load_plans(self, already_emitted: set) -> dict:
        """Load plans for requests vLLM parked, drained exactly once.

        Emitting twice would submit a second transfer for a request that
        already has one in flight; the worker refuses that outright,
        because the first submission's tasks would be left with nothing
        to drain them.

        ``already_emitted`` lets the caller skip requests whose plan was
        produced by the normal path in this same step (a request can be
        both newly scheduled and pending here if vLLM changed its mind
        between passes).
        """
        plans = {}
        for req_id in sorted(self._async_load_pending - already_emitted):
            state = self._req_states.get(req_id)
            if state is None:
                continue
            meta = self._build_load_meta_from_state(
                req_id, state, scheduled_tokens=0)
            state.async_plan_emitted = True
            # Downgrade to synchronous from here on. Once released, vLLM
            # reschedules the request through the ordinary new-request
            # path, which builds a plan again. Leaving is_async set would
            # make the worker open a SECOND cross-step transfer and
            # report the request finished a second time -- by which point
            # it is RUNNING, and vLLM asserts that a finished-recving
            # request is either parked or done.
            state.is_async = False
            if meta is not None and meta.group_ops:
                plans[req_id] = meta
            else:
                logger.warning(
                    "async req=%s has no restorable pages; dropping the "
                    "plan so it is recomputed rather than left waiting",
                    req_id)
        self._async_load_pending -= (self._async_load_pending
                                     - already_emitted)
        return plans

    def _request_block_hashes(self, request) -> list:
        """This request's block identities, in block order.

        ``legacy`` recomputes them from the token ids the way the
        block-oriented path always has, so caches written by earlier
        versions stay readable. ``vllm`` adopts the engine's own
        prefix-cache hashes, which is what the boundary layout has
        always used and what lets a hit line up with vLLM's local
        prefix cache exactly.

        Both exclude the final token, matching vLLM: a block is only
        identified once its tokens are computed.
        """
        if self._block_hash_source == "vllm":
            return list(request.block_hashes)
        tokens = request.all_token_ids
        return [str(h) for h in generate_block_hashs(
            tokens[:-1], self._hash_block_size)]

    def on_request_finished(self, req_id: str) -> None:
        """vLLM trigger: core frees the request ->
        connector.request_finished -> here. Drop the ReqState;
        committed boundaries are content-addressed and stay."""
        self._req_states.pop(req_id, None)
        self._async_load_pending.discard(req_id)

    def on_cached_request(
        self, req_id: str, new_block_ids, resumed: bool,
        num_computed_tokens: Optional[int],
    ) -> None:
        """vLLM trigger: every scheduling pass, for each running
        (cached) request, via connector.build_connector_meta.

        Track a scheduled CACHED request: sync the authoritative
        num_computed from upstream and extend block tables with newly
        allocated blocks so incremental save targets the right slots.
        ``resumed`` (preemption resume) REPLACES tables per upstream
        CachedRequestData semantics.

        Cursor rollback: the
        incremental save cursor means "proven not to need re-emission in
        THIS request lifecycle", NOT "metadata was once constructed".
        On resume (or any authoritative progress regression, even with a
        missing resumed flag -- fail-closed) every group's cursor rolls
        back to floor(N / block_size): boundaries emitted before a
        preemption but never provably persisted are re-emitted.
        Re-emission is safe (writing a block again is idempotent);
        NOT rolling back permanently skips un-persisted
        boundaries."""
        state = self._req_states.get(req_id)
        if state is None:
            return
        # Adopt block hashes vLLM has appended since we registered
        # (decode completes blocks too; without this, generated tokens
        # are never offloaded). Only ever extends -- hashes are
        # content-addressed and append-only.
        if state.live_source is not None:
            if self._block_hash_source == "vllm":
                live = state.live_source
            else:
                live = [str(h) for h in generate_block_hashs(
                    state.live_source[:-1], self._hash_block_size)]
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
                    if os.getenv("KVSHRINK_DEBUG_LOG"):
                        logger.info(
                            "cursor rollback req=%s g%d: %d -> %d "
                            "(progress %d -> %s, resumed=%s)",
                            req_id, g_idx, gstate.next_stored_chunk_idx,
                            safe, old_progress, num_computed_tokens,
                            resumed)
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
        self, request, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        """External lookup; returns (hit_tokens, has_async_load).

        vLLM trigger: the core calls connector.get_num_new_matched_tokens
        while scheduling a NEW request, BEFORE block allocation, to ask
        how many tokens the external store can vouch for. Our answer is
        added to num_computed_tokens by the core, so it must be backed
        by restorable pages.

        Always synchronous: the chunk-tier lookup is Record-gated and
        never defers (no PENDING/RETRY states exist anymore).
        """
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        block_hashes = self._request_block_hashes(request)
        self.on_new_request(
            request.request_id, block_hashes,
            num_computed_tokens, request=request)
        policy = HybridHitPolicy(
            self._groups, self._present, self._hash_block_size,
            num_computed_tokens)
        # Restorable boundary in tokens; 0 = miss. The policy already
        # gated on live chunk presence (engine Record), so a nonzero
        # boundary is complete by construction; only record it.
        boundary = policy.find_longest_cache_hit(
            block_hashes, request.num_tokens)
        state = self._req_states[request.request_id]
        if boundary and state.block_hashes:
            state.snapshot_boundary = boundary
        else:
            boundary = 0
        external = max(0, boundary - num_computed_tokens)
        use_async = self._decide_async(request.request_id, external)
        logger.debug(
            "req=%s external_hit=%d boundary=%d async=%s",
            request.request_id, external, boundary, use_async)
        return external, use_async

    # ------------------------------------------------------------------
    def _decide_async(self, req_id: str, external: int) -> bool:
        """Should this request's pages stream in while the GPU runs
        OTHER requests, instead of stalling a forward step on us?

        Answering True hands vLLM ``load_kv_async=True``: it parks the
        request in WAITING_FOR_REMOTE_KVS, allocates its blocks anyway
        (so we have somewhere to write), and only reschedules it once
        the worker names it in ``get_finished``. The alternative is what
        we did before -- enter forward immediately and block a layer
        hook until the bytes arrive, which idles the GPU for exactly as
        long as the storage is slow. That cost grows with concurrency
        and will grow again when the tier is remote rather than local
        disk.

        Concurrency is approximated by the number of live request
        states, matching the block-path policy so one knob means one
        thing everywhere.

        Returns False (stay synchronous) when there is nothing to load,
        when no policy is configured, or when the policy selects 0
        layers -- 0 means "synchronous", NOT "release before any layer".
        """
        if external <= 0 or self._async_load_config is None:
            return False
        state = self._req_states.get(req_id)
        if state is None:
            return False
        try:
            selected = self._async_load_config.select(len(self._req_states))
        except Exception:  # pragma: no cover - fail closed to sync
            logger.exception(
                "async load policy failed for req=%s; loading synchronously",
                req_id)
            return False
        if selected == 0:
            return False
        state.is_async = True
        # Clamp: asking for more leading layers than exist would never
        # be satisfiable and would hang the request in
        # WAITING_FOR_REMOTE_KVS forever.
        if selected < 0 or selected > len(self._attention_layers):
            state.async_load_layers = -1  # require every layer
        else:
            state.async_load_layers = selected
        return True

    # ------------------------------------------------------------------
    def update_state_after_alloc(
        self, request, blocks, num_external_tokens: int
    ) -> None:
        """Record the allocated block tables per group (after alloc).

        vLLM trigger: the core calls connector.update_state_after_alloc
        right AFTER successful block allocation (same pass as the
        lookup), passing the num_external_tokens it accepted. Only on
        this path is pending_load_tokens set -- the alloc-failure path
        never calls us, so no pending load obligation can leak.

        Design note -- why this hook only RECORDS facts and defers plan
        building to build_connector_meta (unlike the legacy
        pure-attention connector, which computes the load range inline
        here):

        1. The restore point is fixed at lookup time, not by arithmetic.
           GDN state can only be restored at segment boundaries, so the
           hit policy pins ``snapshot_boundary`` during
           get_num_new_matched_tokens. Dividing
           num_external_tokens by block_size here would not necessarily
           land on a legal boundary; the plan must follow the recorded
           boundary, never a fresh computation.
        2. Multiple KV groups keep separate block tables. The attention
           group and the GDN group have independent block pools; the
           per-group ids recorded here are later assembled by different
           rules (attention per block, mamba = the last non-null
           snapshot slot). The legacy connector can hardcode
           ``get_block_ids()[0]``; we cannot.
        3. New and preemption-resumed requests share one plan builder.
           Resumed requests arrive via resumed_req_ids, not the new
           request path; assembling plans at alloc time would fork the
           logic. Recording facts here and building plans from the
           recorded state in build_connector_meta keeps both entries on
           the same code path (_build_load_meta_from_state).

        ``blocks`` is the result of kv_cache_manager.get_blocks(request_id):
        a tuple of per-group block sequences (KVCacheBlock objects).
        """
        req_id = request.request_id
        state = self._req_states.get(req_id)
        if state is None:
            self.on_new_request(
                req_id, self._request_block_hashes(request), 0,
                request=request)
            state = self._req_states[req_id]
        state.num_computed_tokens = (
            state.num_computed_tokens + num_external_tokens)
        state.pending_load_tokens = num_external_tokens
        all_block_ids = blocks.get_block_ids()
        for g_idx, group in enumerate(self._groups):
            if g_idx >= len(all_block_ids):
                continue
            ids = list(all_block_ids[g_idx])
            state.groups[g_idx].block_ids = ids
        if (state.is_async and not state.async_plan_emitted
                and num_external_tokens > 0):
            # This is the ONLY moment we hear about an async request.
            # vLLM allocates its blocks, calls us here, and then parks it
            # in WAITING_FOR_REMOTE_KVS -- out of scheduled_new_reqs and
            # out of scheduled_cached_reqs. A plan builder that walks
            # those two lists would therefore never emit anything for
            # it, the worker would have nothing to transfer, nothing to
            # report finished, and the request would wait forever for a
            # release that cannot come. Record it here and emit from the
            # record instead.
            self._async_load_pending.add(req_id)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "update_state req=%s per-group block_ids: %s hashes=%d",
                req_id, [[b for b in g.block_ids] for g in state.groups],
                len(state.block_hashes))

    # ------------------------------------------------------------------
    def build_load_meta(self, new_req, scheduled_tokens: int = 0) -> ReqMeta:
        """Build the LOAD ReqMeta for a NewRequestData entry.

        vLLM trigger: connector.build_connector_meta iterates
        ``scheduler_output.scheduled_new_reqs`` at the end of the pass.

        attention groups: all prefix blocks whose boundary hash is HIT;
        mamba groups: the single state snapshot block at the restore
        boundary (written into the CURR state block; see
           _build_load_meta_from_state).
        """
        req_id = new_req.req_id
        state = self._req_states.get(req_id)
        if state is None:
            # A load plan is only ever asked for after we reported
            # external tokens for this request, and that report is what
            # creates the state. No state means the two disagree, and a
            # plan built on a guess would address the wrong blocks.
            raise RuntimeError(
                f"kvshrink: load plan requested for unknown request "
                f"{req_id}; get_num_new_matched_tokens never ran")
        return self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)

    def build_resumed_load_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> Optional[ReqMeta]:
        """Build the LOAD ReqMeta for a PREEMPTION-RESUMED request.

        vLLM v1 carries resumed requests in
        ``scheduled_cached_reqs.resumed_req_ids`` (NOT
        ``scheduled_new_reqs``), so build_connector_meta must ask for
        their load meta explicitly. State (block tables, snapshot
        boundary, pending_load_tokens) was already refreshed this
        scheduling pass by
        get_num_new_matched_tokens + update_state_after_alloc.

        Fail-closed: if the core accepted external tokens
        (pending_load_tokens > 0) the meta MUST carry restorable pages;
        an empty load with pending external tokens means the forward
        would read
        unrestored KV while num_computed_tokens skips recompute. Raise
        instead of silently emitting wrong tokens.
        Returns None when the request was never seen by the connector
        (no external tokens could have been accepted).
        """
        state = self._req_states.get(req_id)
        if state is None:
            return None
        meta = self._build_load_meta_from_state(
            req_id, state, scheduled_tokens)
        if state.pending_load_tokens > 0:
            npages = sum(len(op.keys) for op in meta.group_ops)
            if npages == 0:
                raise RuntimeError(
                    "kvshrink resumed request has accepted external "
                    "tokens but no restorable pages (req="
                    f"{req_id} pending={state.pending_load_tokens} "
                    f"boundary={state.snapshot_boundary} "
                    f"sched={scheduled_tokens}): refusing to enter "
                    "forward with unrestored state")
        return meta

    def _build_load_meta_from_state(
        self, req_id: str, state, scheduled_tokens: int,
    ) -> ReqMeta:
        """The snapshot_boundary recorded by get_num_new_matched_tokens
        is the AUTHORITATIVE restore boundary for this alloc/load. NEVER recompute here: after
        update_state_after_alloc the locally-computed counter already
        includes external tokens, and a fresh lookup would be polluted.
        Missing/expired boundary must
        FAIL CLOSED (boundary 0), not guess."""
        boundary = state.snapshot_boundary
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "TAIL req=%s snapshot_boundary=%d computed_before_fwd=%d "
                "external=%d",
                req_id, state.snapshot_boundary,
                state.num_computed_tokens,
                state.snapshot_boundary - state.num_computed_tokens)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            ids = state.groups[g_idx].block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            if group.kind == "attention":
                gran = group.block_size
                num_hash = boundary // gran
                for i in range(num_hash):
                    if i >= len(state.block_hashes):
                        break
                    blk_hash = state.block_hashes[i]
                    key = self._boundary_key(group, blk_hash)
                    if not lookup_boundary(self._store, key):
                        break
                    # v0.21 hashes are per complete block: hash i == block i
                    if i < len(ids):
                        # one page key + gpu block per layer (full expansion)
                        for layer_name in group.layer_names:
                            keys.append(self._page_key(key, layer_name))
                            gpu_ids.append(ids[i])
            elif group.kind == "mamba":
                # Load the snapshot into the CURR state block ONLY
                # (v0.23.0 semantics, verified upstream): the align-mode
                # block table pins the GDN execution metadata to column 0
                # = the block holding this step's last scheduled token,
                # for BOTH the chunked-prefill and decode paths -- there
                # is no prev/curr distinction at execution time.
                # preprocess_mamba's prev -> curr copy runs BEFORE
                # start_load_kv (execute_model order), and our H2D write
                # lands before forward (waited at start_load_kv), i.e.
                # after the copy and before any GDN layer runs, so CURR
                # is the
                # one correct target. Writing PREV would be dead work:
                # the kernel never reads it that step.
                if state.block_hashes and boundary > 0:
                    # hash index of the snapshot AT boundary:
                    # hash[i] covers [i*bs, (i+1)*bs) -> snapshot at
                    # boundary lives at hash[boundary//bs - 1]
                    idx = boundary // group.block_size - 1
                    if 0 <= idx < len(state.block_hashes):
                        blk_hash = state.block_hashes[idx]
                        key = self._boundary_key(group, blk_hash)
                        if lookup_boundary(self._store, key):
                            bs = group.block_size
                            # CURR running-state index for this step
                            # (upstream align-mode formula):
                            # (num_computed + num_scheduled - 1) // bs
                            # with num_computed == boundary here.
                            #
                            # Why exactly one slot, and why this one:
                            # in align mode the kernels do not scan the
                            # table, they gather a single column --
                            # mamba_get_block_table_tensor computes
                            # start = (seq_lens - 1) // block_size and
                            # mamba_attn then takes column 0 of the
                            # gathered result. So this index is the only
                            # location forward will ever read for this
                            # step. The previous generation wrote both a
                            # prev and a curr slot because the timing of
                            # vLLM's prev->curr copy was unclear; the
                            # v0.23 source settles it, making the second
                            # write dead weight. The flip side is that
                            # there is no fallback slot, so an invalid
                            # index below must fail-stop rather than
                            # degrade.
                            curr_idx = (boundary + scheduled_tokens -
                                        1) // bs

                            # Fail-closed contract: an external HIT has already
                            # committed num_computed_tokens=boundary via
                            # get_num_new_matched_tokens; silently skipping a
                            # required slot
                            # would let forward read unrestored state and
                            # emit wrong tokens. Fail-stop (EngineCore
                            # fatal, same semantics as the TOCTOU gate)
                            # instead of producing a partial mamba load.
                            if scheduled_tokens <= 0 and not state.is_async:
                                # SYNCHRONOUS restore with no scheduled
                                # tokens means no forward step, so
                                # start_load_kv never runs and the slot
                                # stays unrestored while the core has
                                # already credited the tokens.
                                #
                                # An ASYNC restore is a different thing
                                # and is correct here. vLLM gives a
                                # parked request zero scheduled tokens,
                                # so curr_idx collapses to
                                # (boundary - 1) // bs -- which is
                                # exactly the index preprocess_mamba
                                # will read as prev_state_idx when the
                                # request is finally scheduled
                                # (num_computed_tokens == boundary by
                                # then). Its own prev -> curr copy then
                                # carries the snapshot into the slot
                                # forward reads. No hook is needed
                                # because no kernel runs until the
                                # release gate has already waited for
                                # the transfer.
                                raise RuntimeError(
                                    "kvshrink mamba external HIT with "
                                    "scheduled_tokens=0 "
                                    f"(req={req_id} boundary={boundary}): "
                                    "production hits must schedule >= 1 "
                                    "token; refusing to build load meta")
                            if not (0 <= curr_idx < len(ids)
                                    and ids[curr_idx] != 0):
                                raise RuntimeError(
                                    "kvshrink mamba load curr slot "
                                    f"invalid (req={req_id} "
                                    f"boundary={boundary} "
                                    f"sched={scheduled_tokens} "
                                    f"table_idx={curr_idx} "
                                    f"table={ids}): refusing to enter "
                                    "forward with unrestored state")
                            gpu_block = ids[curr_idx]
                            for layer_name in group.layer_names:
                                keys.append(self._page_key(
                                    key, layer_name))
                                gpu_ids.append(gpu_block)
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids),
                snapshot_boundary_tokens=boundary if group.kind == "mamba"
                else None))
        return ReqMeta(
            external_hit_tokens=boundary - state.num_computed_tokens,
            group_ops=tuple(group_ops),
            is_async=state.is_async,
            async_load_layers=state.async_load_layers,
        )

    def build_save_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> ReqMeta:
        """Production save: INCREMENTAL per-group page persistence.

        vLLM trigger: connector.build_connector_meta asks for a save
        plan for EVERY request scheduled this pass (new + cached); the
        worker executes it after forward (wait_for_save), so the GPU
        pages then hold state up to ``computed + scheduled`` tokens;
        boundaries are computed against THAT progress.

        The save path for NEWLY COMPUTED KV, end to end
        (pass N, request advances by S scheduled tokens)
        ---------------------------------------------------------------
        ::

          [scheduler, pass N]
            progress P = num_computed_tokens + S   (predictive:
                                  the plan is built BEFORE forward but
                                  describes the state AFTER forward)
            per group, emit ops for work not previously emitted:
              attention: blocks [next_stored_chunk_idx, P//block_size)
                         -- every newly COMPLETED block, per layer
              mamba:     snapshot of the running state block, ONLY if
                         P lands exactly on a block boundary
            next_stored_chunk_idx advances at EMIT time (the worker
            save is fail-stop, so indices cannot silently diverge)
                    |
                    |  ReqMeta pickled inside connector metadata
                    v
          [worker, pass N]
            forward          -> KV for the S new tokens is now in the
                                GPU blocks (block_ids recorded earlier)
            wait_for_save    -> per (layer, block) in the plan:
                                read GPU block -> compress -> stage
                                chunks under the content-hash keys
                                a block becomes visible to later
                                lookups the moment its write lands,
                                because that write is the commit
                    |
                    v
          [any later pass / any later request]
            get_num_new_matched_tokens can now HIT these pages:
            same content hash -> same keys -> restore instead of
            recompute.

        Each group tracks ``next_stored_chunk_idx``; a step emits save
        ops only for blocks/boundaries not previously emitted.

        - attention: every completed block hash in
          [next_stored, progress//gran) is saved (per-block pages are
          valid as soon as the block completes).
        - mamba: the running state block (last NON-NULL table slot) is
          saved only when progress lands EXACTLY on a block boundary --
          a partial tail is not a valid restore point, and snapshots are
          only addressable by boundary hashes.
        """
        state = self._req_states.get(req_id)
        if state is None:
            return ReqMeta()
        progress = state.num_computed_tokens + scheduled_tokens
        state.last_known_progress = max(state.last_known_progress,
                                        progress)
        group_ops = []
        for g_idx, group in enumerate(self._groups):
            gstate = state.groups[g_idx]
            ids = gstate.block_ids
            if not ids:
                continue
            keys: list[CacheKey] = []
            gpu_ids: list[int] = []
            snapshot_boundary: Optional[int] = None
            if group.kind == "attention":
                num_hash = min(progress // group.block_size, len(ids),
                               len(state.block_hashes))
                start = gstate.next_stored_chunk_idx
                for i in range(start, num_hash):
                    blk_hash = state.block_hashes[i]
                    for layer_name in group.layer_names:
                        keys.append(self._page_key(
                            self._boundary_key(group, blk_hash),
                            layer_name))
                        gpu_ids.append(ids[i])
                if num_hash > start:
                    gstate.next_stored_chunk_idx = num_hash
            elif group.kind == "mamba":
                # Save the running state block: the last NON-NULL block in
                # the group's table. Block tables vary by token count:
                # single-element [X] (545-token req), null-prefixed
                # [0,0,X], or [null, X] -- block 0 is the reserved null
                # block. Do NOT assume len(ids) > 1.
                if state.block_hashes:
                    block_pos = None
                    for pos in range(len(ids) - 1, -1, -1):
                        if ids[pos] != 0:  # 0 = null block placeholder
                            block_pos = pos
                            break
                    if block_pos is None:
                        if os.getenv("KVSHRINK_DEBUG_LOG"):
                            logger.info(
                                "save mamba g%d: no non-null block in "
                                "ids=%s", g_idx, ids)
                    elif (progress > 0
                          and progress % group.block_size == 0):
                        idx = progress // group.block_size - 1
                        if (idx >= gstate.next_stored_chunk_idx
                                and idx < len(state.block_hashes)):
                            blk_hash = state.block_hashes[idx]
                            snapshot_boundary = progress
                            for layer_name in group.layer_names:
                                keys.append(self._page_key(
                                    self._boundary_key(group, blk_hash),
                                    layer_name))
                                gpu_ids.append(ids[block_pos])
                            gstate.next_stored_chunk_idx = idx + 1
            group_ops.append(GroupTransferMeta(
                group_idx=g_idx,
                keys=tuple(keys), gpu_block_ids=tuple(gpu_ids),
                snapshot_boundary_tokens=snapshot_boundary))
        return ReqMeta(group_ops=tuple(group_ops))

    def _boundary_key(self, group: GroupInfo, block_hash) -> CacheKey:
        """This rank's boundary key for one group at one block hash."""
        return make_boundary_key(self._namespace, self._tp_size,
                                 self._rank, group.group_idx, block_hash)

    def _present(self, group_idx: int, block_hash) -> bool:
        """Store-presence predicate handed to the hit policy, which
        plans against boundary addresses without seeing store details."""
        return lookup_boundary(
            self._store,
            make_boundary_key(self._namespace, self._tp_size, self._rank,
                              group_idx, block_hash))

    @staticmethod
    def _page_key(boundary_key: CacheKey, layer_name: str) -> CacheKey:
        """Expand a boundary key to ONE layer's page key: same
        namespace/tp/rank/hash/group as the boundary, plus the layer
        name. This is the exact page address the worker must move."""
        return replace(boundary_key, layer_name=layer_name)

# ======================================================================
# longest-hit policy
# ======================================================================
class _StoreAsBlockPool:
    """The one thing vLLM's matching code needs that we must supply.

    ``find_longest_cache_hit`` asks a block pool "is this hash cached,
    for these groups?" and otherwise only compares the answers. Pointing
    that question at the external store is the whole adaptation; the
    matching rules stay upstream's.

    A hit returns a placeholder rather than a block: the callers only
    count and position the results, and the blocks the request will
    actually use are allocated by vLLM afterwards.
    """

    __slots__ = ("_present",)

    # Stands in for a skipped block. vLLM inserts it as padding and only
    # ever counts it, so it needs no identity beyond being a value.
    null_block = object()

    def __init__(self, present):
        self._present = present

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        blocks = []
        for group_id in kv_cache_group_ids:
            if not self._present(group_id, block_hash):
                return None
            blocks.append(_StoreAsBlockPool.null_block)
        return blocks


class HybridHitPolicy:
    """Fixed-point multi-group hit detection (pure function, testable)."""

    def __init__(
        self,
        groups: list[GroupInfo],
        present,
        hash_block_size: int,
        num_computed_tokens: int,
    ):
        """Configure the policy for one request: the groups, a
        store-presence predicate ``present(group_idx, block_hash)`` and
        the request's computed tokens. Orders groups attention-first
        (tighter initial bound) and takes the global mamba alignment as
        the minimum across mamba groups."""
        self._groups = groups
        self._present = present
        self._hash_block_size = hash_block_size
        self._num_computed = num_computed_tokens
        # full attention first (tighter initial bound)
        self._ordered = sorted(
            groups, key=lambda g: 0 if g.kind == "attention" else 1)
        self._mamba_align = None
        for g in groups:
            if g.kind == "mamba":
                a = g.mamba_align_size
                self._mamba_align = a if self._mamba_align is None \
                    else min(self._mamba_align, a)

    # ------------------------------------------------------------------
    def _lookup(self, group: GroupInfo, block_hashes,
                candidate: int) -> int:
        """How far this group alone is restorable, in tokens.

        The matching rules are vLLM's, called here rather than copied:
        full attention is a downward-closed prefix scan, a recurrent
        group is a right-to-left search for the nearest snapshot that
        also sits on an alignment boundary, and each has its own
        handling of EAGLE and of a block size that differs from the hash
        granularity. Reimplementing that would mean our hit length
        silently drifting from vLLM's whenever upstream refines it.

        The one substitution is where "cached" is looked up: vLLM asks
        its GPU block pool, we ask the external store.
        """
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
        manager_cls = KVCacheSpecRegistry.get_manager_class(group.spec)
        if manager_cls is None:
            raise RuntimeError(
                f"kvshrink: vLLM has no cache-hit rule for "
                f"{type(group.spec).__name__} (group {group.group_idx})")
        # vLLM indexes the hash list directly out to max_length, so the
        # caller owes it a length its own hashes cover. Its scheduler
        # gets this for free (the bound comes from the same request);
        # ours can be a boundary the request has not reached.
        max_length = min(candidate,
                         len(block_hashes) * group.block_size)
        blocks = manager_cls.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_length,
            kv_cache_group_ids=[group.group_idx],
            block_pool=_StoreAsBlockPool(self._present),
            kv_cache_spec=group.spec,
            drop_eagle_block=False,
            alignment_tokens=(self._mamba_align or group.block_size),
        )
        return len(blocks[0]) * group.block_size

    # ------------------------------------------------------------------
    def find_longest_cache_hit(
        self, block_hashes: list[int], max_length: int
    ) -> int:
        """Fixed-point convergence over all groups. Returns the
        restorable boundary in tokens; 0 = miss.

        Every group is looked up on its own: a hit on group A says
        nothing about group B, whose blocks live under a different
        label and may never have been written.
        """
        candidate = max_length
        if self._mamba_align is not None:
            # the last prompt token is always recomputed (logprobs + state)
            a = self._mamba_align
            candidate = min(candidate - 1, (candidate - 1) // a * a)

        while True:
            changed = False
            for group in self._ordered:
                hit = self._lookup(group, block_hashes, candidate)
                if hit < candidate:
                    candidate = hit
                    changed = True
                if candidate <= self._num_computed:
                    return 0
            if not changed:
                break
        return candidate if candidate > self._num_computed else 0
