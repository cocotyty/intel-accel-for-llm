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


def _group(g_idx, kind, block_size, align=None):
    return GroupInfo(
        group_idx=g_idx, kind=kind,
        layer_names=(f"l{g_idx}.0", f"l{g_idx}.1"),
        block_size=block_size,
        mamba_align_size=align, spec=make_spec(kind, block_size))


def _hybrid_pairs(attn_hashes, mamba_hashes, attn_g=0, mamba_g=1):
    """Group-aware committed pairs for a 2-group hybrid model. The mamba
    group commits ONLY the final progress boundary (debug_save stores a
    single snapshot), so mamba_hashes is normally just [final_hash]."""
    return ({(attn_g, h) for h in attn_hashes}
            | {(mamba_g, h) for h in mamba_hashes})


def _make(groups, committed, block_ids_per_group):
    sched = HybridRequestScheduler(groups, _Store(committed), 16)
    track_new_request(sched, "r1", block_hashes=[0, 1],
                         num_computed_tokens=0)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),  # request-like
        FakeBlocks(block_ids_per_group), 0)
    return sched


def test_save_meta_single_element_table():
    """545-token request: mamba table [X] (no null prefix). Snapshot at
    the 544 boundary must be saved (regression: old len(ids)>1 skip)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5]])  # ids=[5] only
    meta = sched.build_save_meta("r1", scheduled_tokens=544)
    op = meta.group_ops[0]
    # progress = 0 + 544 = 544 -> boundary -> idx=0 -> hash0 committed
    assert len(op.keys) == 2, op  # 2 layers
    assert op.gpu_block_ids == (5, 5), op.gpu_block_ids


def test_save_meta_null_prefixed_table():
    """Null-prefixed [0,0,X]: last non-null block is used."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 2
    assert op.gpu_block_ids == (7, 7), op.gpu_block_ids


def test_save_meta_skips_partial_tail():
    """progress 1088+472 = 1560 (not a boundary) -> no mamba snapshot."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0, 1}, [[0, 0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=1560)
    op = meta.group_ops[0]
    assert len(op.keys) == 0, op  # partial tail never saved


def test_save_meta_multi_block_boundary():
    """progress 1088 = 2 complete blocks -> hash idx=1 (hash1)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 0, 7]])
    meta = sched.build_save_meta("r1", scheduled_tokens=1088)
    op = meta.group_ops[0]
    assert len(op.keys) == 2
    assert op.gpu_block_ids == (7, 7)


def test_load_meta_curr_slot_is_last_scheduled_block():
    """The snapshot lands in the block the GDN kernel actually reads.

    v0.23.0 mamba_get_block_table_tensor (align mode) gathers
    block_table[(seq_len - 1) // block_size] and the kernel uses column
    0 of that gather, where seq_len = computed + scheduled. Restoring
    boundary 544 with 544 scheduled tokens therefore targets table index
    (544 + 544 - 1) // 544 = 1 -> block 6."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5, 6]])
    sched._req_states["r1"].snapshot_boundary = 544
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1088,
                       "block_ids": ([5, 6],)}),
        scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op  # CURR slot x 2 layers
    assert op.gpu_block_ids == (6, 6), op.gpu_block_ids


def test_load_meta_chunk_tail_table_index():
    """Real shape [0,1,6], boundary 1088, sched 472 (1560-token prompt):
    (1088 + 472 - 1) // 544 = 2 -> block 6, the only slot the kernel
    reads this step."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 6]])
    sched._req_states["r1"].snapshot_boundary = 1088
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1560,
                       "block_ids": ([0, 1, 6],)}),
        scheduled_tokens=472)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op
    assert op.gpu_block_ids == (6, 6), op.gpu_block_ids


def test_load_meta_null_prefixed_table():
    """Null-prefixed table [0, 7, 9]: boundary 1088 + 544 scheduled ->
    index 2 -> block 9. Leading nulls are reserved placeholders, never
    load targets."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 7, 9]])
    sched._req_states["r1"].snapshot_boundary = 1088
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1632,
                       "block_ids": ([0, 7, 9],)}),
        scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op
    assert op.gpu_block_ids == (9, 9), op.gpu_block_ids


def test_load_meta_decode_tail():
    """Decode tail (sched == 1, e.g. a 1089-token prompt with boundary
    1088): (1088 + 1 - 1) // 544 = 2 -> block 6."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 6]])
    sched._req_states["r1"].snapshot_boundary = 1088
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 1089,
                       "block_ids": ([0, 1, 6],)}),
        scheduled_tokens=1)
    op = meta.group_ops[0]
    assert len(op.keys) == 2, op  # CURR slot x 2 layers
    assert op.gpu_block_ids == (6, 6), op.gpu_block_ids


def test_load_meta_curr_null_fail_stop():
    """Chunk path (sched >= 2) with a null CURR slot -> FAIL-STOP.

    There is no second slot to fall back on: the kernel reads exactly
    the gathered column, so a null there means the state cannot be
    restored at all."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 0]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1090,
                           "block_ids": ([0, 1, 0],)}),
            scheduled_tokens=2)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "null curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_curr_null_decode_fail_stop():
    """Decode tail (sched == 1) with a null CURR slot -> FAIL-STOP:
    never enter forward with unrestored state."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1, 0]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1089,
                           "block_ids": ([0, 1, 0],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "decode tail with null curr must raise"
    assert "unrestored state" in str(raised)


def test_load_meta_curr_out_of_range_fail_stop():
    """Chunk path (sched >= 2): the gathered index is beyond the table
    -> FAIL-STOP (the block the kernel will read was never allocated)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1090,
                           "block_ids": ([0, 1],)}),
            scheduled_tokens=2)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "out-of-range curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_curr_out_of_range_decode_fail_stop():
    """Decode tail (sched == 1): gathered index beyond table -> FAIL-STOP."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {1}, [[0, 1]])
    sched._req_states["r1"].snapshot_boundary = 1088
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1089,
                           "block_ids": ([0, 1],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "decode tail with out-of-range curr" \
        " must raise"
    assert "unrestored state" in str(raised)


def test_load_meta_table_idx_null_fail_closed():
    """Gathered table index resolves to a null block -> FAIL-STOP:
    get_num_new_matched_tokens already credited the external boundary,
    so proceeding would enter forward with unrestored state."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0, 6]])
    sched._req_states["r1"].snapshot_boundary = 544  # idx (544+1-1)//544 = 1
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 545,
                           "block_ids": ([0, 0, 6],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "null curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_table_idx_out_of_range_fail_closed():
    """Gathered table index beyond the table length -> FAIL-STOP."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = HybridRequestScheduler(groups, _Store({2}), 16)
    track_new_request(sched, "r1", block_hashes=[0, 1, 2],
                         num_computed_tokens=0)
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks(((5,),)), 0)
    sched._req_states["r1"].snapshot_boundary = 1632  # idx 1631//544 = 2
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 1633,
                           "block_ids": ([5],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "out-of-range curr slot must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_fail_closed_without_boundary():
    """No snapshot_boundary recorded -> fail closed (0 keys), never guess
    by recomputing."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5]])
    meta = sched.build_load_meta(
        type("R", (), {"req_id": "r1", "num_tokens": 545,
                       "block_ids": ([5],)}))
    op = meta.group_ops[0]
    assert len(op.keys) == 0, op  # boundary=0 -> idx=-1 -> no load


def test_save_meta_all_null_table_skipped():
    """All-null table [0,0] -> no mamba save keys."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0]])
    meta = sched.build_save_meta("r1", scheduled_tokens=544)
    op = meta.group_ops[0]
    assert len(op.keys) == 0, op


def test_load_meta_all_null_table_fail_closed():
    """All-null table with boundary > 0 -> FAIL-STOP (curr slot null)."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[0, 0]])
    sched._req_states["r1"].snapshot_boundary = 544
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 545,
                           "block_ids": ([0, 0],)}),
            scheduled_tokens=1)
    except RuntimeError as e:
        raised = e
    assert raised is not None, "all-null table must raise"
    assert "curr slot invalid" in str(raised)


def test_load_meta_hit_sched_zero_fail_stop():
    """External HIT with scheduled_tokens=0 -> FAIL-STOP. A production
    hit always schedules at least one token; sched=0 is a test-only
    path that must never drive a real mamba load."""
    groups = [_group(0, "mamba", 544, align=544)]
    sched = _make(groups, {0}, [[5, 9]])
    sched._req_states["r1"].snapshot_boundary = 544
    raised = None
    try:
        sched.build_load_meta(
            type("R", (), {"req_id": "r1", "num_tokens": 545,
                           "block_ids": ([5, 9],)}))
    except RuntimeError as e:
        raised = e
    assert raised is not None, "sched=0 external HIT must raise"
    assert "scheduled_tokens=0" in str(raised)


def test_completeness_intact_boundary_unchanged():
    """Lookup: all pages present -> boundary unchanged (1088)."""
    groups = [_group(0, "attention", 544), _group(1, "mamba", 544)]
    backend = _Store(set(), committed_pairs=_hybrid_pairs([0, 1], [1]))
    sched = HybridRequestScheduler(groups, backend, 544)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1], "num_tokens": 1088})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 1088, ext
    assert sched._req_states["r1"].snapshot_boundary == 1088


def test_mamba_align_granularity():
    """mamba_align_size > block_size (32 vs 16): final boundary 96 with
    a missing attention hash2 page -> mamba snapshot only at 96 -> full
    MISS (no intermediate boundary usable)."""
    groups = [
        _group(0, "attention", 16),
        _group(1, "mamba", 16, align=32),
    ]
    backend = _Store(
        set(),
        committed_pairs=_hybrid_pairs([0, 1, 3, 4, 5], [5]))
    sched = HybridRequestScheduler(groups, backend, 544)
    req = type("R", (), {
        "request_id": "r1", "block_hashes": [0, 1, 2, 3, 4, 5],
        "num_tokens": 96})
    ext, _ = sched.get_num_new_matched_tokens(req, 0)
    assert ext == 0, ext  # mamba snapshot unusable below 96 -> MISS


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
    attn_op = meta.group_ops[0]
    mamba_op = meta.group_ops[1]
    # attention: boundary 544 -> 1 hash (hash0) x 2 layers, gpu block 3
    assert len(attn_op.keys) == 2, attn_op
    assert all(k.block_hash == 0 for k in attn_op.keys)
    assert set(attn_op.gpu_block_ids) == {3}
    # mamba: snapshot at hash0 -> curr table_idx 1 -> block 8
    assert len(mamba_op.keys) == 2, mamba_op
    assert all(k.block_hash == 0 for k in mamba_op.keys)
    assert set(mamba_op.gpu_block_ids) == {8}


