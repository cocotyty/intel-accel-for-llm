"""Prefix-match tests for `find_longest_prefix`.

`request.block_hashes` has ONE hash per COMPLETE block (block_size
granularity). Attention restores a leading run of blocks; a GDN snapshot
restores a prefix only at an aligned boundary, and the page for boundary
`b` is keyed by block `b - 1`'s hash. The last prompt token is always
recomputed.
"""

from kvshrink.kvshrink_connector import find_longest_prefix


def test_attention_leading_run_and_last_token_recompute():
    # 4 blocks = 128 tokens; a full hit still leaves the last block out.
    assert find_longest_prefix([True] * 4, None, 32, 128) == 96


def test_attention_stops_at_first_hole():
    assert find_longest_prefix([True, True, False, True], None, 32, 128) == 64


def test_no_hit():
    assert find_longest_prefix([False, False], None, 32, 64) == 0


def test_mamba_finds_nearest_boundary_left_of_candidate():
    # mamba boundary b uses flags[b - 1]; only the 48-token boundary exists.
    kv = [True] * 4
    assert find_longest_prefix(kv, [False, False, True, False], 16, 64) == 48


def test_mamba_walks_left_when_boundary_missing():
    assert find_longest_prefix([True] * 4, [True, False, False, False], 16, 64) == 16


def test_mamba_can_look_past_its_own_early_holes():
    # Raw flags (the store truncates in practice, this is the function's own
    # contract): boundary 4 is present even though blocks 0/1 are not.
    mamba = [False, False, True, True, False, False]
    assert find_longest_prefix([True] * 6, mamba, 16, 100) == 64


def test_mamba_cannot_overshoot_the_attention_candidate():
    # Attention's hole limits the candidate; a later mamba snapshot is dead.
    kv = [True, True, True, True, False, True]
    assert find_longest_prefix(kv, [True] * 6, 16, 96) == 64


def test_both_kinds_full():
    assert find_longest_prefix([True] * 4, [True] * 4, 32, 128) == 96


def test_unaligned_prompt_rounds_down():
    assert find_longest_prefix([True] * 6, [True] * 6, 16, 100) == 96


def test_candidate_below_one_block_is_miss():
    assert find_longest_prefix([True, False], [True, False], 16, 16) == 0


def test_boundary_table():
    for length in (31, 32, 33, 63, 64, 65, 95, 96, 97):
        n_blocks = length // 16 + 2
        expected = (length - 1) // 16 * 16
        assert find_longest_prefix(
            [True] * n_blocks, [True] * n_blocks, 16, length) == expected
