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
from kvshrink.scheduler import HybridRequestScheduler


def _sched(block_size=16):
    groups = [GroupInfo(group_idx=0, kind="attention",
                        layer_names=("a0",), block_size=block_size,
                        mamba_align_size=None,
                        spec=make_spec("attention", block_size))]
    return HybridRequestScheduler(
        groups, backend=None, hash_block_size=block_size,
        namespace="ns", tp_size=1, rank=0)


class _LiveRequest:
    """Stands in for vLLM's Request, whose block_hashes list grows in
    place as the request produces tokens."""

    def __init__(self, hashes):
        self.block_hashes = list(hashes)


def _state(sched, live, hashes):
    st = ReqState(
        request=live, block_hashes=list(hashes),
        groups=tuple(ReqGroupState() for _ in sched._groups))
    sched._req_states["r1"] = st
    return st


def test_hashes_added_during_decode_are_adopted():
    sched = _sched()
    live = _LiveRequest([1, 2, 3])
    st = _state(sched, live, [1, 2, 3])

    live.block_hashes.extend([4, 5])       # two blocks produced by decode
    sched.on_cached_request("r1", None, False, 80)

    assert st.block_hashes == [1, 2, 3, 4, 5], (
        "decode-phase boundaries stayed invisible to the save plan")


def test_sync_is_append_only():
    """A shorter live list is not a rollback we can act on: the save
    cursor may already have passed those indices."""
    sched = _sched()
    live = _LiveRequest([1])
    st = _state(sched, live, [1, 2, 3])

    sched.on_cached_request("r1", None, False, 48)
    assert st.block_hashes == [1, 2, 3]


def test_missing_request_object_is_not_fatal():
    """Some registration paths only ever see NewRequestData, which
    carries no Request object; those requests simply keep the hashes
    they were registered with."""
    sched = _sched()
    st = _state(sched, None, [1, 2])

    sched.on_cached_request("r1", None, False, 32)
    assert st.block_hashes == [1, 2]


def test_save_plan_reaches_the_new_boundaries():
    """The point of the sync: blocks generated during decode become
    eligible for saving."""
    sched = _sched()
    live = _LiveRequest([1, 2])
    st = _state(sched, live, [1, 2])
    st.groups[0].block_ids = [10, 11, 12, 13]

    live.block_hashes.extend([3, 4])
    sched.on_cached_request("r1", None, False, 64)

    meta = sched.build_save_meta("r1", scheduled_tokens=0)
    saved = [k.block_hash for op in meta.group_ops for k in op.keys]
    assert saved == [1, 2, 3, 4], (
        f"save plan stopped short of the decode boundaries: {saved}")
