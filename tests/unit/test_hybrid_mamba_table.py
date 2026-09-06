"""Mamba block-table shape handling in build_save_meta/build_load_meta.

block tables vary by
token count -- single-element [X] (545-token req), null-prefixed
[0,0,X], or [null, X]. Block 0 is the reserved null block .
Old code assumed len(ids) > 1 and silently skipped mamba snapshots for
single-element tables; the fix picks the last NON-NULL block.

These tests import the real HybridRequestScheduler (needs vllm env,
so they run in the container test runner).
"""

from __future__ import annotations


from conftest import (
    FakeBlocks, HybridRequestScheduler, make_spec,
    track_new_request)
from kvshrink.kvshrink_connector import GroupInfo


class _Store:
    """Store stub: hash value N is HIT iff N in committed.

    committed_pairs (optional): set of (group_idx, hash) for group-aware
    manifests. Real stores commit a MAMBA group's manifest ONLY at the
    final progress boundary (debug_save stores hash[progress//bs-1]), so
    mixed-group tests must use pairs that reflect that -- a bare
    `committed` set would pretend mamba snapshots exist at every block.
    """

    def __init__(self, committed, committed_pairs=None):
        self.committed = committed
        self.committed_pairs = committed_pairs

    def has(self, chunk_labels, label=None):
        """Presence is per (group, block hash): a group's layers are all
        written in one call and recorded as one unit, so a single layer
        can never be the odd one out. What CAN differ is one group
        against another, which is what committed_pairs expresses."""
        h = int(chunk_labels[0])
        if self.committed_pairs is not None:
            g = int(label[len("g"):])
            return [(g, h) in self.committed_pairs]
        return [h in self.committed]


def _group(g_idx, kind, block_size):
    return GroupInfo(
        group_idx=g_idx, kind=kind,
        layer_names=(f"l{g_idx}.0", f"l{g_idx}.1"),
        spec=make_spec(kind, block_size))


def _hybrid_pairs(attn_hashes, mamba_hashes, attn_g=0, mamba_g=1):
    """Group-aware committed pairs for a 2-group hybrid model. The mamba
    group commits ONLY the final progress boundary (debug_save stores a
    single snapshot), so mamba_hashes is normally just [final_hash]."""
    return ({(attn_g, h) for h in attn_hashes}
            | {(mamba_g, h) for h in mamba_hashes})


def _make(groups, committed, block_ids_per_group, hashes=(0, 1)):
    sched = HybridRequestScheduler(groups, _Store(committed), 544)
    track_new_request(sched, "r1", block_hashes=list(hashes),
                         num_computed_tokens=0)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),  # request-like
        FakeBlocks(block_ids_per_group), 0)
    return sched


def test_save_meta_single_element_table():
    """545-token request: mamba table [X] (no null prefix). Snapshot at
    the 544 boundary must be saved (regression: old len(ids)>1 skip).
    New semantics: the engine has hashed exactly one completed block,
    so live holds one hash and the boundary state sits in ids[0]."""
    groups = [_group(0, "mamba", 544)]
    sched = _make(groups, {0}, [[5]], hashes=(0,))  # ids=[5] only
    meta = sched.build_save_meta("r1", scheduled_tokens=544)
    assert meta.block_hashes == ("0",), meta
    assert meta.group_block_ids == ((5,),), meta


def test_save_meta_null_prefixed_table():
    """Legal resumed shape [NULL, X]: the boundary column is the
    formula slot, not the last non-null block by accident."""
    groups = [_group(0, "mamba", 544)]
    sched = _make(groups, {0}, [[0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=1088)
    assert meta.block_hashes == ("1",), meta
    assert meta.group_block_ids == ((7,),), meta


def test_save_meta_partial_tail_still_saves_completed_blocks():
    """progress 1088+472 = 1560: not a boundary, but TWO full blocks
    completed and the engine hashed both. New semantics: block 1's
    snapshot is saved regardless of the partial tail -- waiting for a
    boundary-aligned step is exactly the decode-phase save loss bug.
    Block 2 is incomplete (no hash) and never enters the plan."""
    groups = [_group(0, "mamba", 544)]
    sched = _make(groups, {0, 1}, [[0, 7, 9]], hashes=(0, 1))
    meta = sched.build_save_meta("r1", scheduled_tokens=1560)
    assert meta.block_hashes == ("1",), meta
    assert meta.group_block_ids == ((7,),), meta


def test_save_meta_decode_boundary_late_hash_still_saved():
    """The decode-phase save loss bug, distilled. Block 4 completes at
    token 2640; the engine appends its hash right after that forward,
    so the NEXT scheduling step is the first that can see it. Old code
    required progress % bs == 0 on the very step the hash appears --
    by then progress is 2641 and the boundary window was missed
    forever. New code reads the engine's live list: once the hash is
    there, the save fires, and the prev column still holds the
    boundary state for the whole next block cycle."""
    groups = [_group(0, "mamba", 528)]
    sched = HybridRequestScheduler(groups, _Store({4}), 528)
    track_new_request(sched, "r1", block_hashes=[0, 1, 2, 3, 4],
                     num_computed_tokens=2640)
    st = sched._req_states["r1"]
    # real shape at progress 2641: cdiv(2641,528)=6 columns, the prev
    # column (idx 4) still holds the boundary-2640 state
    st.groups[0].block_ids = [60, 61, 62, 63, 70, 71]
    st.groups[0].next_block_to_save = 4
    # the step AFTER the crossing: progress is off the boundary now
    meta = sched.build_save_meta("r1", scheduled_tokens=1)
    assert meta.block_hashes == ("4",), meta
    assert meta.group_block_ids == ((70,),), meta
    assert st.groups[0].next_block_to_save == 5


def test_save_meta_resumed_short_table_waits():
    """Resume replaced the mamba table with a short one, but the
    engine's hash list survives preemption at full length. The slot
    for the newest hash does not physically exist yet: no save, no
    crash, cursor stays -- the save fires once the table grows back."""
    groups = [_group(0, "mamba", 544)]
    sched = HybridRequestScheduler(groups, _Store({0}), 544)
    track_new_request(sched, "r1", block_hashes=list(range(34)),
                     num_computed_tokens=0)
    st = sched._req_states["r1"]
    # resumed: vLLM replaced the table with just the curr slot
    st.groups[0].block_ids = [200, 201]
    sched.on_cached_request("r1", ([202],), resumed=True,
                            num_computed_tokens=0)
    meta = sched.build_save_meta("r1", scheduled_tokens=64)
    assert meta.block_hashes == () and meta.group_block_ids == ((),), meta
    assert st.groups[0].next_block_to_save == 0
    # the table grows back through forward; once the slot for hash 33
    # is real, the snapshot is saved under that hash
    st.groups[0].block_ids = list(range(200, 235))
    meta = sched.build_save_meta("r1", scheduled_tokens=64)
    assert meta.block_hashes == ("33",), meta
    assert meta.group_block_ids == ((233,),), meta


def test_save_meta_multi_block_boundary():
    """progress 1088 = 2 complete blocks -> hash idx=1 (hash1), block
    taken from the formula slot (same index the kernel wrote)."""
    groups = [_group(0, "mamba", 544)]
    sched = _make(groups, {1}, [[0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=1088)
    assert meta.block_hashes == ("1",), meta
    assert meta.group_block_ids == ((7,),), meta


def test_save_meta_prev_and_curr_blocks_save_curr():
    """Legal shape [NULL, prev, curr] at a 3-block boundary: the exit
    state is in the LAST column (block 47), never the surviving prev
    block 31 the old reverse scan could have preferred on a mis-sized
    table. The engine hashed all three completed blocks."""
    groups = [_group(0, "mamba", 544)]
    sched = _make(groups, {1, 2}, [[0, 31, 47]], hashes=(0, 1, 2))
    meta = sched.build_save_meta("r1", scheduled_tokens=1632)
    assert meta.block_hashes == ("2",), meta
    assert meta.group_block_ids == ((47,),), meta


def _load_plan(block_ids, committed, hashes, nc_before, ext):
    """Drive the real path: the plan is built when vLLM hands over the
    allocated blocks, so the test allocates and then collects."""
    groups = [_group(0, "mamba", 544)]
    sched = HybridRequestScheduler(groups, _Store(committed), 544)
    track_new_request(sched, "r1", block_hashes=list(hashes),
                      num_computed_tokens=nc_before)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks((block_ids,)), ext)
    return sched.build_load_meta(
        type("R", (), {"req_id": "r1"}))


def test_load_meta_targets_the_tables_state_slot():
    """The snapshot lands in the block the GDN kernel actually reads.

    In align mode the engine nulls every column but the last
    (MambaManager.allocate_new_blocks: the first
    num_required_blocks-1 entries are null blocks), and it sizes the
    table for the tokens already scheduled. So the state slot is the
    table's LAST entry, whatever shape the table has -- these four
    tables are a full block boundary, a chunk tail, a null-prefixed
    resume and a decode tail, and none of them needs the scheduled
    token count to resolve."""
    cases = [
        ([5, 6], 544, "0"),        # boundary 544, one block scheduled
        ([0, 1, 6], 1088, "1"),    # 1560-token prompt, chunk tail
        ([0, 7, 9], 1088, "1"),    # null-prefixed table
        ([0, 1, 6], 1088, "1"),    # decode tail
    ]
    for block_ids, nc, want_hash in cases:
        meta = _load_plan(block_ids, {int(want_hash)}, (0, 1),
                          nc_before=0, ext=nc)
        assert meta.block_hashes == (want_hash,), (block_ids, meta)
        assert meta.group_block_ids == ((block_ids[-1],),), \
            (block_ids, meta)


def test_load_meta_null_state_slot_fail_stop():
    """The table's state slot is a null block -> FAIL-STOP.

    There is no second slot to fall back on: the kernel reads exactly
    that column, so a null there means the state cannot be restored at
    all, while get_num_new_matched_tokens has already credited the
    external boundary."""
    raised = None
    try:
        _load_plan([0, 1, 0], {1}, (0, 1), nc_before=0, ext=1088)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "null state slot must raise"
    assert "unrestored state" in str(raised)


def test_load_meta_fail_closed_without_boundary():
    """No external credit (pending_load_tokens=0): the meta builder is
    never called for the request -- no plan, nothing to guess."""
    from types import SimpleNamespace
    groups = [_group(0, "mamba", 544)]
    sched = _make(groups, {0}, [[5]], hashes=(0,))
    meta = sched.build_connector_meta(SimpleNamespace(
        scheduled_new_reqs=[type("R", (), {"req_id": "r1"})],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], resumed_req_ids=set(),
            new_block_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={"r1": 545}))
    assert "r1" not in meta.reqs_to_load.requests, meta


def test_save_meta_all_null_table_fail_closed():
    """All-null table at a boundary -> FAIL-STOP, mirroring the load
    path: the kernel cannot have left the exit state anywhere, so
    saving any other block under the boundary hash would silently
    poison the store."""
    groups = [_group(0, "mamba", 544)]
    sched = _make(groups, {0}, [[0, 0]])
    raised = None
    try:
        sched.build_save_meta("r1", scheduled_tokens=1088)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "all-null table must raise"
    assert "boundary column is NULL" in str(raised)


def test_load_meta_all_null_table_fail_closed():
    """All-null table with boundary > 0 -> FAIL-STOP (state slot null)."""
    raised = None
    try:
        _load_plan([0, 0], {0}, (0,), nc_before=0, ext=544)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "all-null table must raise"
    assert "unrestored state" in str(raised)


def test_completeness_intact_boundary_unchanged():
    """Lookup: every group complete at the first boundary -> restore
    it whole. The last block boundary is never lent out (the last
    prompt token is always recomputed), so 1088 committed tokens
    restore 544."""
    groups = [_group(0, "attention", 544), _group(1, "mamba", 544)]
    backend = _Store(set(), committed_pairs=_hybrid_pairs([0, 1], [0, 1]))
    sched = HybridRequestScheduler(groups, backend, 544)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1], "num_tokens": 1088})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 544, ext
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks(([9], [9])), 544)
    # the credit landed num_computed_tokens exactly on the boundary
    assert sched._req_states["r1"].num_computed_tokens == 544


def test_mamba_partial_recovery_with_earlier_snapshot():
    """If an EARLIER mamba snapshot is committed and intact (future M3
    multi-snapshot save), partial recovery is legal: hash1 attention
    page missing -> 1088 unusable, but 544 has a complete mamba
    snapshot -> restore 544."""
    groups = [_group(0, "attention", 544), _group(1, "mamba", 544)]
    # attention has hash0 only; mamba has snapshots at both
    backend = _Store(set(), committed_pairs=_hybrid_pairs([0], [0, 1]))
    sched = HybridRequestScheduler(groups, backend, 544)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1], "num_tokens": 1088})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 544, ext  # earlier intact mamba snapshot -> recover


def test_partial_recovery_load_meta_targets_earlier_snapshot():
    """Scheduler-level chain: hash1 attention page missing + mamba hash0
    snapshot committed -> lookup recovers 544 -> build_load_meta targets
    the mamba hash0 pages in the slot the kernel reads this step
    ((544 + 544 - 1) // 544 = 1), attention loads only hash0's pages."""
    groups = [_group(0, "attention", 544), _group(1, "mamba", 544)]
    # attention has hash0 only, so 1088 is not reachable
    backend = _Store(set(), committed_pairs=_hybrid_pairs([0], [0, 1]))
    sched = HybridRequestScheduler(groups, backend, 544)
    req = type("R", (), {"request_id": "r1", "block_hashes": [0, 1],
                         "num_tokens": 1088})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 544, ext
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks(([3, 4], [5, 8])), 544)
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1088,
                       "block_ids": ([3, 4], [5, 8])}),
        scheduled_tokens=544)
    # attention: boundary 544 -> hash0's page, gpu block 3
    assert meta.block_hashes == ("0",), meta
    assert meta.group_block_ids[0] == (3,), meta
    # mamba: snapshot at hash0 -> curr table_idx 1 -> block 8
    assert meta.group_block_ids[1] == (8,), meta


