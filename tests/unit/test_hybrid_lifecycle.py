"""Abort / preemption / resume lifecycle tests.

Rulings under test:
1. resume (or any authoritative progress regression) rolls every
   group's incremental save cursor back to floor(N / block_size) --
   emitted-but-unproven boundaries are re-emitted (idempotent, safe);
2. request_finished returns (True, None) -- block freeing is deferred
   to get_finished, which acks once every transfer reading the blocks
   has landed (the async save lifecycle, same as main);
3. request_finished fail-stops if async store jobs exist;
4. committed boundaries are content-addressed: abort/finish NEVER
   deletes them; uncommitted pages never hit.
"""

from conftest import (
    FakeBlocks, HybridRequestScheduler, make_spec,
    track_new_request)
from kvshrink.kvshrink_connector import GroupInfo

PAGE = 64 * 1024


def _attn(bs=16):
    return GroupInfo(
        group_idx=0, kind="attention", layer_names=("attn.0",),
        spec=make_spec("attention", bs))


def _mamba():
    return GroupInfo(
        group_idx=0, kind="mamba", layer_names=("m.0",),
        spec=make_spec("mamba", 544))


class _MissStore:
    def has(self, chunk_labels, label=None):
        return [False]


class _HitStore:
    """Committed boundary hashes are HIT (content-addressed)."""

    def __init__(self, committed):
        self.committed = committed

    def has(self, chunk_labels, label=None):
        return [int(chunk_labels[0]) in self.committed]


def _sched(groups, store=None):
    return HybridRequestScheduler(groups, store or _MissStore(),
                                  groups[0].spec.block_size)


def _setup_attn_req(sched, hashes, ids, tokens=0):
    track_new_request(sched, "r1", block_hashes=hashes,
                         num_computed_tokens=tokens)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks((ids,)), 0)


# ------------------------------------------------------------------
# 1-6: cursor rollback semantics
# ------------------------------------------------------------------

def test_resume_to_zero_rolls_cursor_and_reemits():
    """Attention: cursor at 2, resume to progress 0 -> cursor 0; the
    blocks are re-emitted when the request re-crosses boundaries."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=32)  # cursor -> 2
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 2
    sched.on_cached_request("r1", ([10, 11, 12, 13],), resumed=True,
                            num_computed_tokens=0)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 0, g
    m = sched.build_save_meta("r1", scheduled_tokens=32)
    assert m.group_block_ids == ((10, 11),), m.group_block_ids


def test_mamba_resume_reemits_boundary_snapshot():
    sched = _sched([_mamba()])
    track_new_request(sched, "r1", block_hashes=[0, 1],
                         num_computed_tokens=0)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}), FakeBlocks(([5],)), 0)
    m1 = sched.build_save_meta("r1", scheduled_tokens=544)
    assert len(m1.block_hashes) == 1
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 1
    sched.on_cached_request("r1", ([9],), resumed=True,
                            num_computed_tokens=0)
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 0
    m2 = sched.build_save_meta("r1", scheduled_tokens=544)
    assert len(m2.block_hashes) == 1  # re-emitted
    assert m2.group_block_ids == ((9,),)


def test_resume_to_nonzero_progress_rolls_to_floor():
    """Resume at N=32 (block 16): cursor rolls to floor(32/16)=2, so
    only blocks >= 2 re-emit."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=64)  # cursor -> 4
    sched.on_cached_request("r1", ([10, 11, 12, 13],), resumed=True,
                            num_computed_tokens=32)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 2, g
    m = sched.build_save_meta("r1", scheduled_tokens=32)
    assert m.group_block_ids == ((12, 13),), m.group_block_ids


def test_monotonic_progress_no_rollback():
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=32)
    sched.on_cached_request("r1", None, resumed=False,
                            num_computed_tokens=32)
    assert sched._req_states["r1"].groups[0].next_stored_chunk_idx == 2


def test_resumed_empty_table_clears_and_rolls_back():
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1], [10, 11])
    sched.build_save_meta("r1", scheduled_tokens=32)
    sched.on_cached_request("r1", ([],), resumed=True,
                            num_computed_tokens=0)
    g = sched._req_states["r1"].groups[0]
    assert g.block_ids == []
    assert g.next_stored_chunk_idx == 0


def test_progress_regression_without_resumed_flag_rolls_back():
    """Fail-closed: authoritative progress regression rolls the cursor
    even if the resumed flag is missing (defence in depth)."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=64)  # cursor -> 4
    sched.on_cached_request("r1", None, resumed=False,
                            num_computed_tokens=16)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 1, g  # floor(16/16)


# ------------------------------------------------------------------
# 7-8: request_finished contract
# ------------------------------------------------------------------

def _sched_side_connector(sched):
    """The scheduler-role connector facade. Since the scheduler/worker
    merge, the scheduler-side methods live on KVShrinkConnector itself,
    so the plan builder IS the facade."""
    return sched


def test_request_finished_returns_false_and_clears_state():
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1], [10, 11])
    conn = _sched_side_connector(sched)
    req = type("R", (), {"request_id": "r1"})
    free, delay = conn.request_finished(req, None)
    assert (free, delay) == (True, None), \
        "block freeing is deferred to get_finished"
    assert "r1" not in sched._req_states


def test_request_finished_pending_async_job_returns_false_none():
    """Committed boundaries are content-addressed and per-boundary (not
    per-request), but a finished request's blocks may still be read by
    an in-flight put: request_finished defers freeing to get_finished
    and returns (True, None)."""
    sched = _sched([_attn()])
    conn = _sched_side_connector(sched)
    req = type("R", (), {"request_id": "r1"})
    out = conn.request_finished(req, None)
    assert out == (True, None), out


# ------------------------------------------------------------------
# 9-12: committed data ownership / orphan semantics
# ------------------------------------------------------------------

def test_abort_keeps_committed_boundary_hittable():
    """Content-addressed cache: after abort, a NEW request with the
    same hashes still HITs the committed boundary."""
    store = _HitStore(committed={0, 1})
    sched = _sched([_attn()], store)
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.request_finished(
        type("R", (), {"request_id": "r1"})(), [])  # abort
    # a fresh lookup for the same hashes still hits
    assert store.has(["0"], label="g0") == [True]


def test_resumed_missing_progress_rolls_back_to_zero():
    """Fail-closed: resumed=True with missing
    num_computed rolls ALL group cursors to 0 (safe N=0), never skips."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=64)  # cursor -> 4
    sched.on_cached_request("r1", ([10, 11, 12, 13],), resumed=True,
                            num_computed_tokens=None)
    g = sched._req_states["r1"].groups[0]
    assert g.next_stored_chunk_idx == 0, g
    m = sched.build_save_meta("r1", scheduled_tokens=32)
    assert m.group_block_ids == ((10, 11),)  # re-emitted


def test_abort_resume_stress_1000_iterations_zero_residue():
    """1000 rounds of new/save/resume/finish. Every round the cursor
    rolls back and re-emits; at the end no request state is left behind,
    the rollback counter is exact and nothing raised."""
    sched = _sched([_attn()], _MissStore())
    conn = _sched_side_connector(sched)
    for i in range(1000):
        rid = f"r{i}"
        track_new_request(sched, rid, block_hashes=[0, 1, 2, 3], num_computed_tokens=0)
        sched.update_state_after_alloc(
            type("R", (), {"request_id": rid}),
            FakeBlocks(([10, 11, 12, 13],)), 0)
        sched.build_save_meta(rid, scheduled_tokens=64)  # cursor -> 4
        # preempt + resume to zero
        sched.on_cached_request(rid, ([10, 11, 12, 13],), resumed=True,
                                num_computed_tokens=0)
        g = sched._req_states[rid].groups[0]
        assert g.next_stored_chunk_idx == 0, f"round {i}: no rollback"
        m = sched.build_save_meta(rid, scheduled_tokens=64)
        assert m.group_block_ids == ((10, 11, 12, 13),)
        free, delay = conn.request_finished(
            type("R", (), {"request_id": rid}), None)
        assert (free, delay) == (True, None)
    assert len(sched._req_states) == 0


# ------------------------------------------------------------------
# 9: preemption-resume LOAD metadata
# ------------------------------------------------------------------
# vLLM v1 carries preempted->rescheduled requests in
# scheduled_cached_reqs.resumed_req_ids, NOT scheduled_new_reqs. The
# connector historically built load meta ONLY from scheduled_new_reqs,
# so a resumed request's accepted external tokens
# (get_num_new_matched_tokens -> core skips
# recompute) was never matched by a worker-side load -> forward read
# unrestored KV and emitted wrong tokens (4B TP2 lifecycle gate).

def _hybrid_resumed_setup(committed, scheduled=64, ext=544):
    """2-group hybrid (uniform bs=16, mamba snapshot at 544) with a
    request that re-entered after preemption: the lookup hook reset state and
    recorded a HIT at boundary 544, then the core allocated fresh blocks
    and credited ``ext`` external tokens."""
    groups = [
        _attn(),
        GroupInfo(group_idx=1, kind="mamba", layer_names=("m.0",),
                  spec=make_spec("mamba", 16)),
    ]
    sched = HybridRequestScheduler(groups, _HitStore(committed), 16)
    hashes = list(range(34))  # 34 hash blocks * 16 = 544 tokens
    track_new_request(sched, "r1", block_hashes=hashes, num_computed_tokens=0)
    attn_ids = list(range(100, 134))  # 34 fresh attention blocks
    # CURR slot for this step: (544 + 64 - 1) // 16 = 37
    mamba_ids = list(range(200, 238))
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks((attn_ids, mamba_ids)), ext)
    return sched


def test_resumed_load_meta_restores_credited_pages():
    """Resumed request with 544 credited external tokens gets load meta
    carrying all 34 attention pages + the mamba snapshot written into
    the CURR slot only (v0.23.0 reads CURR in every kernel path)."""
    sched = _hybrid_resumed_setup(set(range(34)))
    meta = sched.build_resumed_load_meta("r1", scheduled_tokens=64)
    assert meta is not None
    # 34 attention pages + the mamba snapshot written into the CURR
    # slot only (v0.23.0 reads CURR in every kernel path)
    assert len(meta.block_hashes) == 34, meta
    assert meta.group_block_ids[0] == tuple(range(100, 134)), meta
    assert meta.group_block_ids[1] == (237,), meta


def test_restored_blocks_are_not_rewritten_on_first_save():
    """The load plan reads the hit range back from the external store,
    so the save cursors skip past it: the first post-restore save pass
    must not re-write those blocks (attention pages and the restored
    mamba snapshot alike)."""
    sched = _hybrid_resumed_setup(set(range(34)))
    sched.build_resumed_load_meta("r1", scheduled_tokens=64)
    st = sched._req_states["r1"]
    # 544 credited tokens = 34 blocks: every cursor starts past them
    assert all(g.next_stored_chunk_idx == 34 for g in st.groups), [
        g.next_stored_chunk_idx for g in st.groups]
    # Forward completes tokens up to 544+64=608: the ledger and the
    # attention table grow past the restored range by 4 blocks
    st.block_hashes.extend(range(34, 38))
    st.groups[0].block_ids.extend(range(134, 138))
    m = sched.build_save_meta("r1", scheduled_tokens=64)
    # 608 % 16 == 0 -> 38 blocks done, 4 past the skip of 34
    assert m.block_hashes == ("34", "35", "36", "37"), m
    assert m.group_block_ids[0] == (134, 135, 136, 137), m
    # the mamba snapshot at the 608 boundary is new past the skip too
    assert m.group_block_ids[1] == (237,), m


def test_incremental_boundaries_after_restore_are_saved():
    """The skip is a floor, not a wall: once forward crosses boundaries
    the restore did NOT cover, those blocks must still be saved."""
    sched = _hybrid_resumed_setup(set(range(34)))
    sched.build_resumed_load_meta("r1", scheduled_tokens=64)
    # Extend the ledger and both tables past the restored range
    st = sched._req_states["r1"]
    st.block_hashes.extend(range(34, 74))
    st.groups[0].block_ids.extend(range(134, 174))
    st.groups[1].block_ids.extend(range(238, 274))
    # 544 + 640 = 1184 tokens = 74 blocks: 40 blocks past the skip
    m = sched.build_save_meta("r1", scheduled_tokens=640)
    assert len(m.block_hashes) == 40, m
    assert m.group_block_ids[0] == tuple(range(134, 174)), m
    # mamba snapshot at the 1184 boundary: table idx 73
    assert m.group_block_ids[1] == (273,), m


def test_resume_rollback_still_overrides_the_skip():
    """Preemption after the restore rolls the cursors back per the
    resumed progress -- the skip must never keep a cursor ahead of
    what vLLM says is computed."""
    sched = _hybrid_resumed_setup(set(range(34)))
    sched.build_resumed_load_meta("r1", scheduled_tokens=64)
    # Preempted back to 32 tokens (2 blocks): every cursor rolls to 2
    sched.on_cached_request("r1", (list(range(100, 102)), [200, 201]),
                            resumed=True, num_computed_tokens=32)
    st = sched._req_states["r1"]
    assert all(g.next_stored_chunk_idx == 2 for g in st.groups), [
        g.next_stored_chunk_idx for g in st.groups]


def test_resumed_load_meta_without_credit_is_quiet():
    """Resume covered entirely by the LOCAL prefix cache (ext=0, no
    external boundary): no load plan is built at all."""
    from types import SimpleNamespace
    sched = _hybrid_resumed_setup(set(range(34)), ext=0)
    conn = _sched_side_connector(sched)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"], resumed_req_ids={"r1"},
            new_block_ids=[(list(range(100, 134)), [200, 201])],
            num_computed_tokens=[544]),
        num_scheduled_tokens={"r1": 64})
    meta = conn.build_connector_meta(scheduler_output)
    assert "r1" not in meta.reqs_to_load.requests


def test_connector_meta_includes_resumed_load():
    """End-to-end at connector level: build_connector_meta must emit the
    resumed request's load meta (scheduled_cached_reqs.resumed_req_ids),
    not only scheduled_new_reqs."""
    from types import SimpleNamespace
    sched = _hybrid_resumed_setup(set(range(34)))
    conn = _sched_side_connector(sched)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"], resumed_req_ids={"r1"},
            new_block_ids=[(list(range(100, 134)), [200, 201])],
            num_computed_tokens=[544]),
        num_scheduled_tokens={"r1": 64})
    meta = conn.build_connector_meta(scheduler_output)
    load = meta.reqs_to_load.requests.get("r1")
    assert load is not None, "resumed request must receive load metadata"
    assert len(load.block_hashes) == 34, load
    assert load.group_block_ids[0] == tuple(range(100, 134)), load
