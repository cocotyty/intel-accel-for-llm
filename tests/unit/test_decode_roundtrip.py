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
from kvshrink.kvshrink_connector import GroupInfo, ReqGroupState, ReqState
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

    live = _LiveRequest([1, 2])
    st = ReqState(
        live_block_hashes=live.block_hashes,
        num_computed_tokens=32,
        groups=(ReqGroupState(block_ids=[10, 11, 12, 13]),))
    sched._req_states["r1"] = st

    # Decode completed two more blocks; the engine appended their
    # hashes to the live list in place.
    live.block_hashes.extend([3, 4])
    sched.on_cached_request("r1", None, False, 64)

    # The save plan reaches the new boundaries and the store
    # commits them.
    meta = sched.build_save_meta("r1", scheduled_tokens=0)
    store.committed.update(int(h) for h in meta.block_hashes)
    assert store.committed == {1, 2, 3, 4}

    # Turn two: same first blocks plus the two decode produced ones.
    # get_num_new_matched_tokens must see past the original prompt.
    turn_two = _LiveRequest([1, 2, 3, 4])
    hit, _ = sched.get_num_new_matched_tokens(turn_two, 0)
    assert hit == 64, (
        f"second turn restored only {hit} tokens; decode-produced "
        f"blocks did not round-trip through the external cache")
