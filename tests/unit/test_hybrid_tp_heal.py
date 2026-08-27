# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Boundary presence check.

Each rank persists to its own store directory, and the controller
process opens only the rank0 one; presence is keyed by group label
within a directory. TP ranks save in lockstep, so rank 0 present
stands for all; a rank that diverged anyway fails loudly at load time
(the native layer raises on a missing key).

A store error reads as a MISS: a wrong hit silently corrupts output,
a wrong miss costs one recompute.

Pure logic: fake store, no GPU, no disk.
"""

from __future__ import annotations

from kvshrink.kvshrink_connector import group_label, lookup_boundary
from kvshrink.kvshrink_connector import CacheKey


def _key(group_idx=0):
    return CacheKey(block_hash=12345, group_idx=group_idx, layer_name="")


class _FakeStore:
    """Answers presence per group label."""

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
        _FakeStore([group_label(0)]), _key()) is True


def test_missing_is_miss():
    assert lookup_boundary(_FakeStore([]), _key()) is False


def test_queries_the_boundary_group_label_only():
    store = _FakeStore([group_label(0)])
    assert lookup_boundary(store, _key()) is True
    assert store.asked == [group_label(0)]


def test_store_error_fails_closed_to_miss():
    """A wrong hit silently corrupts output; a wrong miss costs one
    recompute. Errors resolve to the cheap mistake."""
    assert lookup_boundary(
        _FakeStore([], blow_up=True), _key()) is False


def test_groups_do_not_alias_each_other():
    """The same prefix hash exists in every group, so the label must
    carry the group or one group's data would answer for another.
    """
    store = _FakeStore([group_label(0)])
    assert lookup_boundary(store, _key(group_idx=0)) is True
    assert lookup_boundary(store, _key(group_idx=1)) is False
