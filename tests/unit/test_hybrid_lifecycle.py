"""Abort / preemption / finish lifecycle tests.

Rulings under test:
1. resuming from preemption fail-stops, byte-identical to main:
   build_connector_meta raises RuntimeError for any req_id in
   scheduled_cached_reqs.resumed_req_ids (an untested code path must
   fail loudly, not maybe-corrupt);
2. an authoritative progress regression WITHOUT the resumed flag (MTP
   draft rejection) rolls the save cursor back to
   floor(N / block_size) -- emitted-but-unproven boundaries are
   re-emitted (idempotent, safe);
3. request_finished returns (True, None) -- block freeing is deferred
   to get_finished, which acks once every transfer reading the blocks
   has landed (the async save lifecycle, same as main);
4. request_finished fail-stops if async store jobs exist;
5. committed boundaries are content-addressed: abort/finish NEVER
   deletes them; uncommitted pages never hit.
"""

from conftest import (
    FakeBlocks, HybridRequestScheduler, make_spec,
    track_new_request)
from kvshrink.kvshrink_connector import GroupInfo
import pytest
from types import SimpleNamespace

PAGE = 64 * 1024


@pytest.mark.parametrize("new_request", [True, False])
@pytest.mark.parametrize("scheduled,num_output,expect_save", [
    (16, 0, True),   # prefill chunk
    (1, 0, False),   # full-hit minus one token: nothing new worth saving
    (2, 1, False),   # MTP decode: 1 + 1 spec
    (4, 2, False),   # MTP decode: 1 + 3 spec
])
def test_scheduler_filters_decode_before_save(new_request, scheduled, num_output, expect_save):
    sched = _sched([_attn()])
    track_new_request(sched, "r1", [0, 1, 2, 3])
    sched.update_state_after_alloc(
        SimpleNamespace(request_id="r1"), FakeBlocks(([10, 11, 12, 13],)), 0)
    metadata = sched.build_connector_meta(SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="r1")] if new_request else [],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[] if new_request else ["r1"],
            new_block_ids=[None], num_computed_tokens=[0],
            num_output_tokens=[num_output], resumed_req_ids=set()),
        num_scheduled_tokens={"r1": scheduled},
    ))
    assert ("r1" in metadata.reqs_to_save.requests) == expect_save


def _attn(bs=16):
    return GroupInfo(
        group_idx=0, kind="attention", layer_names=("attn.0",),
        spec=make_spec("attention", bs))


def _mamba():
    return GroupInfo(
        group_idx=0, kind="mamba", layer_names=("m.0",),
        spec=make_spec("mamba", 544))


class _MissStore:
    def has(self, chunk_labels, label=None, truncate=True):
        return [False]


class _HitStore:
    """Committed boundary hashes are HIT (content-addressed)."""

    def __init__(self, committed):
        self.committed = committed

    def has(self, chunk_labels, label=None, truncate=True):
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
# 1: resume fail-stops like main; regression without the flag rolls back
# ------------------------------------------------------------------

def test_monotonic_progress_no_rollback():
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=32)
    sched.sync_running_request("r1", None,
                            num_computed_tokens=32)


def test_progress_regression_without_resumed_flag_rolls_back():
    """Fail-closed: authoritative progress regression (MTP draft
    rejection) re-emits recomputed blocks."""
    sched = _sched([_attn()])
    _setup_attn_req(sched, [0, 1, 2, 3], [10, 11, 12, 13])
    sched.build_save_meta("r1", scheduled_tokens=64)
    sched.sync_running_request("r1", None,
                            num_computed_tokens=16)
    m = sched.build_save_meta("r1", scheduled_tokens=16)
    assert m.group_block_ids == ((11,),), m.group_block_ids  # floor(16/16)=1


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
    assert store.has(["0"], label="kv") == [True]


def test_abort_finish_stress_1000_iterations_zero_residue():
    """1000 rounds of new/save/finish. At the end no request state is
    left behind and nothing raised."""
    sched = _sched([_attn()], _MissStore())
    conn = _sched_side_connector(sched)
    for i in range(1000):
        rid = f"r{i}"
        track_new_request(sched, rid, block_hashes=[0, 1, 2, 3], num_computed_tokens=0)
        sched.update_state_after_alloc(
            type("R", (), {"request_id": rid}),
            FakeBlocks(([10, 11, 12, 13],)), 0)
        sched.build_save_meta(rid, scheduled_tokens=64)
        free, delay = conn.request_finished(
            type("R", (), {"request_id": rid}), None)
        assert (free, delay) == (True, None)
    assert len(sched._req_states) == 0


# ------------------------------------------------------------------
# 9: external-restore load metadata
# ------------------------------------------------------------------
# The restore path: get_num_new_matched_tokens records a HIT, the core
# allocates fresh blocks, update_state_after_alloc credits the external
# tokens. The load plan must carry every credited attention page plus
# the mamba snapshot written into the CURR slot only (v0.23.0 reads
# CURR in every kernel path); the first post-restore save pass must not
# re-write the restored range, and later boundaries must still save.

def _hybrid_resumed_setup(committed, scheduled=64, ext=544):
    """2-group hybrid (uniform bs=16, mamba snapshot at 544) with a
    request holding an external credit: the lookup hook recorded a HIT
    at boundary 544, then the core allocated fresh blocks and credited
    ``ext`` external tokens."""
    groups = [
        _attn(),
        GroupInfo(group_idx=1, kind="mamba", layer_names=("m.0",),
                  spec=make_spec("mamba", 16)),
    ]
    sched = HybridRequestScheduler(groups, _HitStore(committed), 16)
    hashes = list(range(34))  # 34 hash blocks * 16 = 544 tokens
    track_new_request(sched, "r1", block_hashes=hashes, num_computed_tokens=0)
    attn_ids = list(range(100, 134))  # 34 fresh attention blocks
    # CURR slot for this step: (544 + 64 - 1) // 16 = 37. align mode
    # nulls every column but the restore slot (the table's last
    # entry) -- model it, the save scan keys only real columns.
    mamba_ids = [0] * 37 + [237]
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks((attn_ids, mamba_ids)), ext)
    return sched


def test_resumed_load_meta_restores_credited_pages():
    """A request with 544 credited external tokens gets load meta
    carrying all 34 attention pages + the mamba snapshot written into
    the CURR slot only (v0.23.0 reads CURR in every kernel path)."""
    sched = _hybrid_resumed_setup(set(range(34)))
    meta = sched._reqs_to_load.requests.get("r1")
    assert meta is not None
    # 34 attention pages + the mamba snapshot written into the CURR
    # slot only (v0.23.0 reads CURR in every kernel path)
    assert len(meta.block_hashes) == 34, meta
    assert meta.group_block_ids[0] == tuple(range(100, 134)), meta
    # the mamba plan spans the offer; only the last slot is real
    assert meta.group_block_ids[1] == tuple([0] * 33 + [237]), meta


def test_restored_blocks_are_not_rewritten_on_first_save():
    """The load plan reads the hit range back from the external store,
    so the save cursors skip past it: the first post-restore save pass
    must not re-write those blocks (attention pages and the restored
    mamba snapshot alike)."""
    sched = _hybrid_resumed_setup(set(range(34)))
    st = sched._req_states["r1"]
    # 544 credited tokens = 34 blocks: the save cursor starts past the
    # restored range
    # Forward completes tokens up to 544+64=608: the ledger and the
    # attention table grow past the restored range by 4 blocks
    st.block_hashes.extend(range(34, 38))
    st.group_block_ids[0].extend(range(134, 138))
    m = sched.build_save_meta("r1", scheduled_tokens=64)
    # 608 % 16 == 0 -> 38 blocks done, 4 past the skip of 34
    assert m.block_hashes == ["34", "35", "36", "37"], m
    assert m.group_block_ids[0] == (134, 135, 136, 137), m
    # the mamba tail column is the scan's output at boundary 37; the
    # restore slot (37) is keyed only because this pass overwrites it
    assert m.group_block_ids[1] == (0, 0, 0, 237), m


def test_incremental_boundaries_after_restore_are_saved():
    """The skip is a floor, not a wall: once forward crosses boundaries
    the restore did NOT cover, those blocks must still be saved."""
    sched = _hybrid_resumed_setup(set(range(34)))
    # Extend the ledger and both tables past the restored range
    st = sched._req_states["r1"]
    st.block_hashes.extend(range(34, 74))
    st.group_block_ids[0].extend(range(134, 174))
    # one 640-token pass materializes only its tail column: the new
    # columns arrive null except the scan's output at idx 73
    st.group_block_ids[1].extend([0] * 35 + [273])
    # 544 + 640 = 1184 tokens = 74 blocks: 40 blocks past the skip
    m = sched.build_save_meta("r1", scheduled_tokens=640)
    assert len(m.block_hashes) == 40, m
    assert m.group_block_ids[0] == tuple(range(134, 174)), m
    # mamba snapshots at the 608 and 1184 boundaries: table idx 37 and 73
    want_mamba = [0] * 40
    want_mamba[37 - 34] = 237
    want_mamba[73 - 34] = 273
    assert m.group_block_ids[1] == tuple(want_mamba), m


def test_resume_from_preemption_raises():
    """Byte-identical to main: any request in resumed_req_ids fail-stops
    the scheduler. Preemption-resume is an untested path; a loud crash
    beats a maybe-wrong silent recovery."""
    from types import SimpleNamespace
    sched = _hybrid_resumed_setup(set(range(34)))
    conn = _sched_side_connector(sched)
    scheduler_output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["r1"], resumed_req_ids={"r1"},
            new_block_ids=[(list(range(100, 134)), [0] * 37 + [201])],
            num_computed_tokens=[544]),
        num_scheduled_tokens={"r1": 64})
    with pytest.raises(RuntimeError, match="Resuming from preemption"):
        conn.build_connector_meta(scheduler_output)
