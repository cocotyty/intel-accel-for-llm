# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Boundary presence check.

The check runs under the key's own rank label: each rank keeps its own
presence record, and the controller process shares one with the rank-0
worker only, so peer ledgers are not queryable at lookup time. TP ranks
save in lockstep, so rank 0 present stands for all; a rank that
diverged anyway fails loudly at load time (the native layer raises on
a missing key).

A store error reads as a MISS: a wrong hit silently corrupts output,
a wrong miss costs one recompute.

Pure logic: fake store, no GPU, no disk.
"""

from __future__ import annotations

from kvshrink.backend import group_label, lookup_boundary
from kvshrink.kvshrink_connector import CacheKey


def _key(group_idx=0, tp_size=2):
    return CacheKey(namespace="ns", tp_size=tp_size, rank=0,
                    block_hash=12345, group_idx=group_idx, layer_name="")


class _FakeStore:
    """Answers presence per namespace label."""

    def __init__(self, present_labels, blow_up=False):
        self.present = set(present_labels)
        self.blow_up = blow_up
        self.asked: list[str] = []

    def has(self, chunk_labels, label=None):
        if self.blow_up:
            raise RuntimeError("store is unwell")
        self.asked.append(label)
        return [label in self.present]


def test_present_is_hit():
    assert lookup_boundary(
        _FakeStore([group_label("ns", 0, 0)]), _key()) is True


def test_missing_is_miss():
    assert lookup_boundary(_FakeStore([]), _key()) is False


def test_queries_own_rank_label_only():
    """The controller can only see the ledger it shares with the
    rank-0 worker, so that is the only label it may ask about."""
    store = _FakeStore([group_label("ns", 0, 0)])
    assert lookup_boundary(store, _key(tp_size=2)) is True
    assert store.asked == [group_label("ns", 0, 0)]


def test_store_error_fails_closed_to_miss():
    """A wrong hit silently corrupts output; a wrong miss costs one
    recompute. Errors resolve to the cheap mistake."""
    assert lookup_boundary(
        _FakeStore([], blow_up=True), _key()) is False


def test_groups_do_not_alias_each_other():
    """The same prefix hash exists in every group, so the group must be
    part of the namespace or one group's data would answer for another.
    """
    store = _FakeStore([group_label("ns", 0, 0)])
    assert lookup_boundary(store, _key(group_idx=0)) is True
    assert lookup_boundary(store, _key(group_idx=1)) is False
