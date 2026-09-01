"""Hit policy tests: multi-group convergence, GDN right-to-left, align/-1.

v0.21 verified semantics: request.block_hashes has ONE hash per COMPLETE
block (block_size granularity), e.g. 2135-token prompt with block_size=544
-> 3 hashes. Tests use this granularity.
"""
from conftest import make_spec
from kvshrink.kvshrink_connector import GroupInfo
from kvshrink.hybrid_hit import HybridHitPolicy


def _group(g_idx, kind, block_size):
    return GroupInfo(
        group_idx=g_idx, kind=kind,
        layer_names=(f"l{g_idx}.0", f"l{g_idx}.1"),
        spec=make_spec(kind, block_size))


def _hashes(committed: set[int], n_blocks: int):
    """committed: set of hash VALUES that are HIT; block_hashes[i] == i."""
    return lambda g, h: h in committed, list(range(n_blocks))


def test_attention_prefix_only():
    """[H,H,M,H] at block granularity: only first 2 blocks exist."""
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes({0, 1, 3}, 4)  # 4 blocks = 128 tokens
    policy = HybridHitPolicy(groups, b, 32, 0)
    # stops at hash 2 (block index 2)
    assert policy.find_longest_cache_hit(hashes, 128) == 64


def test_gdn_nearest_snapshot():
    """GDN finds the nearest snapshot walking right-to-left; no prefix
    contiguity required. block_size=16, 4 blocks (64 tokens). Only
    hash2 (48-token boundary) exists -> restore 48."""
    groups = [_group(0, "mamba", 16)]
    b, hashes = _hashes({2}, 4)
    policy = HybridHitPolicy(groups, b, 16, 0)
    assert policy.find_longest_cache_hit(hashes, 64) == 48


def test_gdn_walks_left_when_boundary_missing():
    """The snapshot nearest the candidate is missing; the scan keeps
    walking left. Only hash0 (16 tokens) exists -> restore 16."""
    groups = [_group(0, "mamba", 16)]
    b, hashes = _hashes({0}, 4)
    policy = HybridHitPolicy(groups, b, 16, 0)
    assert policy.find_longest_cache_hit(hashes, 64) == 16


def test_multi_group_convergence():
    """Attention and GDN groups, both fully present."""
    groups = [_group(0, "attention", 32), _group(1, "mamba", 32)]
    b, hashes = _hashes({0, 1, 2, 3}, 4)
    policy = HybridHitPolicy(groups, b, 32, 0)
    # the last prompt token is always recomputed: 128 -> 127, aligned
    # down to 96. Both groups reach 96, so the loop settles there.
    assert policy.find_longest_cache_hit(hashes, 128) == 96


def test_align_down_and_minus_one():
    """Non-aligned boundary rounds down; -1 applied once."""
    groups = [_group(0, "mamba", 16)]
    b, hashes = _hashes({0, 1, 2, 3, 4, 5}, 6)  # 96 tokens
    policy = HybridHitPolicy(groups, b, 16, 0)
    # candidate = min(99, 99//32*32) = 96 -> idx 6/5 ... hash5 HIT -> 96
    assert policy.find_longest_cache_hit(hashes, 100) == 96


def test_local_computed_tokens_reduce_external():
    """num_computed reduces the reported external hit."""
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes({0, 1}, 2)
    policy = HybridHitPolicy(groups, b, 32, 32)
    assert policy.find_longest_cache_hit(hashes, 64) == 64  # external = 64 - 32


def test_local_computed_at_boundary_is_miss():
    """num_computed == candidate boundary -> no external hit."""
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes({0, 1}, 2)
    policy = HybridHitPolicy(groups, b, 32, 64)
    assert policy.find_longest_cache_hit(hashes, 64) == 0


def test_no_hit():
    groups = [_group(0, "attention", 32)]
    b, hashes = _hashes(set(), 2)
    policy = HybridHitPolicy(groups, b, 32, 0)
    assert policy.find_longest_cache_hit(hashes, 64) == 0


def test_boundary_table():
    """Table-driven: prompt lengths around block boundaries."""
    groups = [_group(0, "attention", 16), _group(1, "mamba", 16)]
    for length in (31, 32, 33, 63, 64, 65, 95, 96, 97):
        n_blocks = length // 16 + 2
        b, hashes = _hashes(set(range(n_blocks)), n_blocks)
        policy = HybridHitPolicy(groups, b, 16, 0)
        boundary = policy.find_longest_cache_hit(hashes, length)
        expected = (length - 1) // 16 * 16
        if expected == 0:
            assert boundary == 0, f"len={length}"
        else:
            assert boundary == expected, (
                f"len={length} got={boundary} want={expected}")


def test_mamba_lookup_never_overshoots_the_candidate():
    """The mamba scan may never report a boundary past what the other
    groups already shrank the candidate to: attention misses hash4 ->
    candidate 64; mamba's own lookup then only sees hashes 0..3 and
    must settle at 64, not report a snapshot to the right of it."""
    # attention and mamba both block_size 16; 6 blocks = 96 tokens
    groups = [
        _group(0, "attention", 16),
        _group(1, "mamba", 16),
    ]
    # attention is missing hash4 (80-token boundary); mamba has all.
    b, hashes = _hashes({0, 1, 2, 3, 5}, 6)
    policy = HybridHitPolicy(groups, b, 16, 0)
    # candidate = 95//16*16 = 80; attention stops at hash3 -> 64;
    # mamba re-lookup under candidate 64: max_num_blocks=4 -> hash3
    assert policy.find_longest_cache_hit(hashes, 96) == 64


def test_mamba_snapshot_exactly_at_boundary():
    """Aligned boundary 64 with only hash3 committed -> hit 64."""
    groups = [_group(0, "mamba", 16)]
    b, hashes = _hashes({3}, 4)  # hash3 = 64-token boundary
    policy = HybridHitPolicy(groups, b, 16, 0)
    # candidate = min(63, 63//32*32) = 32 -> idx=32//16-1=1 ->
    # hash1 MISS -> hash0 MISS -> MISS (no snapshot below 32)
    assert policy.find_longest_cache_hit(hashes, 64) == 0


def test_mamba_snapshot_at_candidate_cannot_overshoot():
    """Even if hash right of candidate is committed, boundary never
    exceeds candidate (fail closed on overrun)."""
    groups = [_group(0, "mamba", 16)]
    b, hashes = _hashes({5}, 6)  # hash5 = 96-token boundary
    policy = HybridHitPolicy(groups, b, 16, 0)
    # 96 > candidate 32 -> miss
    assert policy.find_longest_cache_hit(hashes, 64) == 0


def test_mamba_empty_hashes_miss():
    """Empty hash list -> no scan -> MISS (no IndexError)."""
    groups = [_group(0, "mamba", 16)]
    policy = HybridHitPolicy(groups, lambda g, h: False, 16, 0)
    assert policy.find_longest_cache_hit([], 64) == 0


def test_mamba_candidate_below_gran_miss():
    """candidate < gran (and < align): aligned=0 -> max_idx=-1 -> MISS."""
    groups = [_group(0, "mamba", 16)]
    b, hashes = _hashes({0}, 2)  # hash0 (16 tokens) committed
    policy = HybridHitPolicy(groups, b, 16, 0)
    # candidate = min(15, 15//32*32=0) = 0 -> MISS
    assert policy.find_longest_cache_hit(hashes, 16) == 0


def _group_aware(committed_pairs: set, n_blocks: int):
    """committed_pairs: set of (group_idx, hash_value) that are HIT."""
    return lambda g, h: (g, h) in committed_pairs, list(range(n_blocks))


def test_multiple_attention_groups_all_validated():
    """Two attention groups, g1's manifest lacks hash2: the hit must
    shrink to 64 even though g0 has all four hashes. Pre-fix only the
    first attention group was looked up and later ones were blindly
    trimmed."""
    groups = [_group(0, "attention", 32), _group(1, "attention", 32)]
    b, hashes = _group_aware(
        {(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)}, 4)
    policy = HybridHitPolicy(groups, b, 32, 0)
    assert policy.find_longest_cache_hit(hashes, 128) == 64


def test_multiple_attention_groups_converge_to_zero():
    """g1 has NO committed hashes at all -> MISS, not a g0-only hit."""
    groups = [_group(0, "attention", 32), _group(1, "attention", 32)]
    b, hashes = _group_aware({(0, 0), (0, 1)}, 4)
    policy = HybridHitPolicy(groups, b, 32, 0)
    assert policy.find_longest_cache_hit(hashes, 128) == 0
