# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""End-to-end (pure logic) round trip: blocks a request generated during
DECODE are saved, and a SECOND request that replays them as its prompt
prefix gets an external hit that extends into the decode-produced range.

This is the multi-turn-conversation scenario: turn one generates tokens
past its prompt, turn two arrives carrying both the prompt and those
tokens. If the decode-phase hashes were never saved,
turn two's hit stops at turn one's prompt and the whole span is
recomputed with a warm cache sitting right there.

Pure logic: no GPU, no disk, no model.
"""

from __future__ import annotations

from conftest import make_spec
from kvshrink.kvshrink_connector import GroupInfo, ReqState
from conftest import HybridRequestScheduler


def _sched(block_size=16):
    groups = [GroupInfo(group_idx=0, kind="attention",
                        layer_names=("a0",),
                        spec=make_spec("attention", block_size))]
    return HybridRequestScheduler(
        groups, store=None, block_size=block_size)


class _LiveRequest:
    """Stands in for the live list vLLM's Request grows in place as the
    request produces tokens."""

    def __init__(self, hashes):
        self.block_hashes = list(hashes)

    # get_num_new_matched_tokens reads these two fields.
    request_id = "r1"
    num_tokens = 10 ** 9


class _Store:
    """Records what build_save_meta emits; has() answers from it."""

    def __init__(self):
        self.committed = set()

    def has(self, chunk_labels, label=None):
        return [int(c) in self.committed for c in chunk_labels]


def test_second_request_hits_decode_produced_blocks():
    # Turn one: prompt covered two blocks, decode produced two more.
    store = _Store()
    groups = [GroupInfo(group_idx=0, kind="attention",
                        layer_names=("a0",),
                        spec=make_spec("attention", 16))]
    sched = HybridRequestScheduler(groups, store, 16)

    # First turn prefill: save prompt blocks 1 and 2
    live = _LiveRequest([1, 2])
    st = ReqState(
        block_hashes=live.block_hashes,
        num_computed_tokens=0,
        num_prompt_tokens=32,
        group_block_ids=[[10, 11]])
    sched._req_states["r1"] = st
    meta_prefill = sched.build_save_meta("r1", scheduled_tokens=32)
    store.committed.update(int(h) for h in meta_prefill.block_hashes)
    assert store.committed == {1, 2}

    # Forward advances to 32 computed tokens (prefill complete)
    st.num_computed_tokens = 32
    st.group_block_ids[0].extend([12, 13])

    # Decode completed two more blocks; the engine appended their
    # hashes to the live list in place.
    live.block_hashes.extend([3, 4])
    sched.sync_running_request("r1", None, False, 64)

    # Under prefill-only policy, decode does NOT save blocks 3 and 4.
    meta_decode = sched.build_save_meta("r1", scheduled_tokens=0)
    assert meta_decode.block_hashes == []
    assert store.committed == {1, 2}

    # Turn two: prompt arrives with blocks [1, 2, 3, 4].
    # External cache matches only the prefill-saved blocks (32 tokens).
    turn_two = _LiveRequest([1, 2, 3, 4])
    hit, _ = sched.get_num_new_matched_tokens(turn_two, 0)
    assert hit == 32, f"expected prompt-only hit of 32 tokens, got {hit}"
