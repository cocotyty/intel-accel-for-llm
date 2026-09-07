# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Decode-phase boundaries must reach the external cache.

vLLM appends a block hash whenever a request completes a block, during
DECODE as well as prefill. The save plan is bounded by the hashes we
hold, so a snapshot taken at registration silently caps external caching
at the prompt: everything a request generates is never offloaded. That
span is precisely what the next turn of a conversation replays, so it
would be recomputed every turn with a warm cache sitting right there.

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


def _state(sched, live, hashes):
    st = ReqState(
        live_block_hashes=(live.block_hashes
                           if live is not None else list(hashes)),
        groups=tuple(ReqGroupState() for _ in sched._groups))
    sched._req_states["r1"] = st
    return st


def test_missing_request_object_is_not_fatal():
    """Some registration paths only ever see NewRequestData, which
    carries no Request object; those requests simply keep the hashes
    they were registered with."""
    sched = _sched()
    st = _state(sched, None, [1, 2])

    sched.on_cached_request("r1", None, False, 32)
    assert st.live_block_hashes == [1, 2]


def test_save_plan_reaches_the_new_boundaries():
    """Blocks generated during decode become eligible for saving:
    the state holds the engine's own list, and vLLM appends each
    completed block's hash to it in place -- no reconciliation pass,
    the save plan just reads further."""
    sched = _sched()
    live = _LiveRequest([1, 2])
    st = _state(sched, live, [1, 2])
    st.groups[0].block_ids = [10, 11, 12, 13]

    live.block_hashes.extend([3, 4])       # two blocks produced by decode
    sched.on_cached_request("r1", None, False, 64)

    meta = sched.build_save_meta("r1", scheduled_tokens=0)
    assert list(meta.block_hashes) == ["1", "2", "3", "4"], (
        f"save plan stopped short of the decode boundaries: {meta}")


def test_finer_engine_hashes_still_name_our_blocks():
    """The engine may hash at a finer granularity than our block size.

    cache_config.hash_block_size lets prefix-caching keys be computed at
    the finest common granularity of a multi-group model and merged for
    larger physical blocks; the merge is the consumer's job. Here one of
    our 16-token blocks spans two engine hashes, so the plan must name
    each block by the LAST hash it covers -- the only one whose prefix
    reaches the block's end. Naming it by the first would let a request
    that shares just the first half restore the whole block.
    """
    sched = _sched()
    sched._hash_factor = 2
    live = _LiveRequest([1, 2, 3, 4])       # 4 engine hashes = 2 blocks
    st = _state(sched, live, [])
    st.groups[0].block_ids = [10, 11, 12, 13]

    meta = sched.build_save_meta("r1", scheduled_tokens=32)
    assert list(meta.block_hashes) == ["2", "4"], (
        f"blocks must be named by their last engine hash: {meta}")
    assert meta.group_block_ids == ((10, 11),), meta

    # Decode completes one more block: two more engine hashes arrive,
    # and exactly one new key appears.
    live.block_hashes.extend([5, 6])
    meta = sched.build_save_meta("r1", scheduled_tokens=48)
    assert list(meta.block_hashes) == ["6"], meta
    assert meta.group_block_ids == ((12,),), meta
