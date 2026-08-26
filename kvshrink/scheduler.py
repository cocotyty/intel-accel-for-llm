# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Scheduler-side (EngineCore process) KV cache planning."""
from __future__ import annotations

import logging
import os
from typing import Optional

from iaxl import generate_block_hashs

from .kvshrink_connector import (CacheKey, GroupInfo, GroupTransferMeta, LookupResult,
                     LookupStatus, ReqMeta, ReqGroupState, ReqState,
                     align_down, make_boundary_key)

# ======================================================================
# request scheduler (EngineCore side)
# ======================================================================
# Scheduler-side request state for the hybrid connector.
#
# For each request we:
# 1. run the hit policy (find_longest_cache_hit) against the backend,
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
    """

    def __init__(
        self,
        groups: list[GroupInfo],
        backend,
        hash_block_size: int,
        namespace: str,
        tp_size: int,
        rank: int,
        async_load_config=None,
        block_hash_source: str = "vllm",
    ):
        """Record the per-group layout, hit-policy backend and TP
        identity, and own the per-request ReqState table plus the
        resume/cursor-rollback counter that spans a request's whole
        scheduling lifecycle.
        """
        self._groups = groups
        self._backend = backend
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
        self._req_states[req_id] = ReqState(
            request=request,
            block_hashes=list(block_hashes),
            num_computed_tokens=num_computed_tokens,
            groups=tuple(
                ReqGroupState() for _ in self._groups),
        )

    def take_async_load_plans(self, already_emitted: set) -> dict:
        """Load plans for requests vLLM parked, drained exactly once."""
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
        """This request's block identities, in block order."""
        if self._block_hash_source == "vllm":
            return list(getattr(request, "block_hashes", None) or ())
        tokens = getattr(request, "all_token_ids", None)
        if tokens is None:
            # Some registration paths only carry prompt ids; fall back
            # rather than register a request with no identity at all.
            return list(getattr(request, "block_hashes", None) or ())
        return [str(h) for h in generate_block_hashs(
            tokens[:-1], self._hash_block_size)]

    def _sync_block_hashes(self, state: ReqState) -> None:
        """Adopt block hashes vLLM has added since we registered."""
        if state.request is None:
            return
        live = self._request_block_hashes(state.request)
        if not live or len(live) <= len(state.block_hashes):
            return
        state.block_hashes.extend(live[len(state.block_hashes):])

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
        """
        state = self._req_states.get(req_id)
        if state is None:
            return
        self._sync_block_hashes(state)
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
            for g_idx in range(min(len(self._groups),
                                   len(new_block_ids))):
                ids = new_block_ids[g_idx]
                if resumed:
                    # upstream semantics: for resumed requests
                    # new_block_ids IS the table (replace), per group --
                    # including an EMPTY list, which clears stale blocks
                    state.groups[g_idx].block_ids = list(ids) if ids \
                        else []
                elif ids:
                    state.groups[g_idx].block_ids.extend(ids)

    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self, request, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        """External lookup; returns (hit_tokens, has_async_load)."""
        if num_computed_tokens >= request.num_tokens:
            return 0, False
        self.on_new_request(
            request.request_id, self._request_block_hashes(request),
            num_computed_tokens, request=request)
        policy = HybridHitPolicy(
            self._groups, self._backend, self._hash_block_size,
            num_computed_tokens, self._namespace, self._tp_size, self._rank)
        result, trace = policy.find_longest_cache_hit(
            self._request_block_hashes(request), request.num_tokens)
        if result.status == LookupStatus.HIT:
            # The policy HIT already gated on live chunk presence
            # (engine Record), so the boundary is complete by
            # construction; only record the snapshot point.
            state = self._req_states.get(request.request_id)
            if state is not None:
                if state.block_hashes:
                    state.snapshot_boundary = result.boundary_tokens
                else:
                    result = LookupResult(LookupStatus.MISS, 0)
        if os.getenv("KVSHRINK_DEBUG_LOG"):
            logger.info(
                "policy result req=%s status=%s boundary=%d hashes=%d "
                "trace=%s",
                request.request_id, result.status.value,
                result.boundary_tokens, len(request.block_hashes), trace)
        # Metrics recorded on the FINAL completeness result (not the
        # raw policy trace).
        external = result.boundary_tokens - num_computed_tokens
        if external < 0:
            external = 0
        use_async = self._decide_async(request.request_id, external)
        logger.debug(
            "req=%s external_hit=%d boundary=%d async=%s trace=%s",
            request.request_id, external, result.boundary_tokens,
            use_async, trace)
        return external, use_async

    # ------------------------------------------------------------------
    def _decide_async(self, req_id: str, external: int) -> bool:
        """Should this request's pages stream in while the GPU runs
        OTHER requests, instead of stalling a forward step on us?
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
        """Record the allocated block tables per group (after alloc)."""
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
        if hasattr(blocks, "get_block_ids"):
            all_block_ids = blocks.get_block_ids()
        else:
            all_block_ids = tuple(
                [b.block_id for b in group_blocks] if group_blocks else []
                for group_blocks in blocks)
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
        """Build the LOAD ReqMeta for a NewRequestData entry."""
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
            req_id, state, scheduled_tokens,
            num_tokens=getattr(new_req, "num_tokens", "?"))

    def build_resumed_load_meta(
        self, req_id: str, scheduled_tokens: int = 0
    ) -> Optional[ReqMeta]:
        """Build the LOAD ReqMeta for a PREEMPTION-RESUMED request."""
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
        num_tokens="?",
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
                "external=%d num_tokens=%s",
                req_id, state.snapshot_boundary,
                state.num_computed_tokens,
                state.snapshot_boundary - state.num_computed_tokens,
                num_tokens)
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
                    if self._backend.lookup_boundary(
                            key) != LookupStatus.HIT:
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
                        if self._backend.lookup_boundary(
                                key) == LookupStatus.HIT:
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

                            def _slot_ok(t):
                                return 0 <= t < len(ids) and ids[t] != 0

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
                            if not _slot_ok(curr_idx):
                                raise RuntimeError(
                                    "kvshrink mamba load curr slot "
                                    f"invalid (req={req_id} "
                                    f"boundary={boundary} "
                                    f"sched={scheduled_tokens} "
                                    f"table_idx={curr_idx} "
                                    f"table={ids}): refusing to enter "
                                    "forward with unrestored state")
                            targets = {curr_idx}
                            for table_idx in sorted(targets):
                                gpu_block = ids[table_idx]
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
        """Production save: INCREMENTAL per-group page persistence."""
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

    @staticmethod
    def _page_key(boundary_key: CacheKey, layer_name: str) -> CacheKey:
        """Expand a boundary key to ONE layer's page key: same
        namespace/tp/rank/hash/group as the boundary, plus the layer
        name. This is the exact page address the worker must move."""
        return CacheKey(
            namespace=boundary_key.namespace,
            tp_size=boundary_key.tp_size,
            rank=boundary_key.rank,
            block_hash=boundary_key.block_hash,
            group_idx=boundary_key.group_idx,
            layer_name=layer_name)

# ======================================================================
# longest-hit policy
# ======================================================================
class _StoreAsBlockPool:
    """The one thing vLLM's matching code needs that we must supply."""

    __slots__ = ("_backend", "_namespace", "_tp_size", "_rank")

    # Stands in for a skipped block. vLLM inserts it as padding and only
    # ever counts it, so it needs no identity beyond being a value.
    null_block = object()

    def __init__(self, backend, namespace: str, tp_size: int, rank: int):
        self._backend = backend
        self._namespace = namespace
        self._tp_size = tp_size
        self._rank = rank

    def get_cached_block(self, block_hash, kv_cache_group_ids):
        blocks = []
        for group_id in kv_cache_group_ids:
            key = make_boundary_key(self._namespace, self._tp_size,
                                    self._rank, group_id, block_hash)
            if self._backend.lookup_boundary(key) != LookupStatus.HIT:
                return None
            blocks.append(_StoreAsBlockPool.null_block)
        return blocks


class HybridHitPolicy:
    """Fixed-point multi-group hit detection (pure function, testable)."""

    def __init__(
        self,
        groups: list[GroupInfo],
        backend,
        hash_block_size: int,
        num_computed_tokens: int,
        namespace: str,
        tp_size: int,
        rank: int,
    ):
        """Configure the policy for one request: groups, backend, the
        request's computed tokens and its namespace/tp/rank identity.
        Orders groups attention-first (tighter initial bound) and takes
        the global mamba alignment as the minimum across mamba groups."""
        self._groups = groups
        self._backend = backend
        self._hash_block_size = hash_block_size
        self._num_computed = num_computed_tokens
        self._namespace = namespace
        self._tp_size = tp_size
        self._rank = rank
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
    def _boundary_key(self, group: GroupInfo, block_hash: int) -> CacheKey:
        """This rank's boundary key for one group at one block hash."""
        return make_boundary_key(self._namespace, self._tp_size,
                                 self._rank, group.group_idx, block_hash)

    def _lookup(self, group: GroupInfo, block_hashes,
                candidate: int) -> int:
        """How far this group alone is restorable, in tokens."""
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
            block_pool=_StoreAsBlockPool(
                self._backend, self._namespace, self._tp_size, self._rank),
            kv_cache_spec=group.spec,
            drop_eagle_block=False,
            alignment_tokens=(self._mamba_align or group.block_size),
        )
        return len(blocks[0]) * group.block_size

    # ------------------------------------------------------------------
    def find_longest_cache_hit(
        self, block_hashes: list[int], max_length: int
    ) -> tuple[LookupResult, dict]:
        """Fixed-point convergence over all groups."""
        candidate = max_length
        if self._mamba_align is not None:
            # the last prompt token is always recomputed (logprobs + state)
            candidate = min(candidate - 1,
                            align_down(candidate - 1, self._mamba_align))
        trace = {"iterations": [], "final": candidate}

        while True:
            changed = False
            iteration = {}
            for group in self._ordered:
                kind = group.kind
                # Every group is looked up on its own: a hit on group
                # A says nothing about group B, whose blocks live under
                # a different label and may never have been written.
                hit = self._lookup(group, block_hashes, candidate)
                iteration[group.group_idx] = {"kind": kind, "hit": hit}
                if hit < candidate:
                    candidate = hit
                    changed = True
                if candidate <= self._num_computed:
                    trace["iterations"].append(iteration)
                    trace["final"] = 0
                    return (LookupResult(LookupStatus.MISS, 0), trace)
            trace["iterations"].append(iteration)
            if not changed:
                break

        external = candidate - self._num_computed
        trace["final"] = candidate
        trace["external"] = external
        if external <= 0:
            return (LookupResult(LookupStatus.MISS, 0), trace)
        return (LookupResult(LookupStatus.HIT, candidate), trace)
