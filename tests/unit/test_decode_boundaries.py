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


def _state(sched, live, hashes, num_prompt_tokens=32):
    st = ReqState(
        block_hashes=(live.block_hashes
                           if live is not None else list(hashes)),
        num_prompt_tokens=num_prompt_tokens,
        group_block_ids=[[] for _ in sched._groups])
    sched._req_states["r1"] = st
    return st


def test_missing_request_object_is_not_fatal():
    """Some registration paths only ever see NewRequestData, which
    carries no Request object; those requests simply keep the hashes
    they were registered with."""
    sched = _sched()
    st = _state(sched, None, [1, 2])

    sched.sync_running_request("r1", None, False, 32)
    assert st.block_hashes == [1, 2]


def test_save_plan_ignores_decode_boundaries():
    """Under prefill-only save, blocks generated during decode are ignored:
    build_save_meta emits empty plan."""
    sched = _sched()
    live = _LiveRequest([1, 2])
    st = _state(sched, live, [1, 2], num_prompt_tokens=32)
    st.group_block_ids[0] = [10, 11, 12, 13]

    live.block_hashes.extend([3, 4])       # two blocks produced by decode
    sched.sync_running_request("r1", None, False, 64)

    meta = sched.build_save_meta("r1", scheduled_tokens=0)
    assert meta.block_hashes == [] and meta.group_block_ids == ((),), meta
