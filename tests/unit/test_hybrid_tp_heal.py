# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""TP partial-commit guard.

Each rank writes its own shard under its own store namespace, and there
is no cross-rank transaction. A boundary present on one rank but missing
on another must therefore read as a MISS: restoring half of a
tensor-parallel state is worse than restoring none, because the output
is wrong rather than absent.

Nothing repairs the gap in the background. The request recomputes and
its own save writes every rank's shard again, which is safe because
writing a block twice is idempotent.

Pure logic: fake store, no GPU, no disk.
"""

from __future__ import annotations

from kvshrink.backend import KVStoreBackend, group_label
from kvshrink.kvshrink_connector import CacheKey


def _key(group_idx=0):
    return CacheKey(namespace="ns", tp_size=2, rank=0,
                    block_hash=12345, group_idx=group_idx, layer_name="")


class _FakeStore:
    """Answers presence per namespace, so a test can make one rank
    disagree with another."""

    def __init__(self, present_labels, blow_up=False):
        self.present = set(present_labels)
        self.blow_up = blow_up
        self.asked: list[str] = []

    def has(self, chunk_labels, label=None):
        if self.blow_up:
            raise RuntimeError("store is unwell")
        self.asked.append(label)
        return [label in self.present]


def _backend(present_labels, tp_size=2, blow_up=False):
    b = KVStoreBackend(_FakeStore(present_labels, blow_up))
    b.register_layout(namespace="ns", tp_size=tp_size, rank=0)
    return b


ALL = [group_label("ns", 0, 0), group_label("ns", 0, 1)]


def test_all_ranks_present_hit():
    assert _backend(ALL).lookup_boundary(_key()) is True


def test_other_rank_missing_is_miss():
    b = _backend([group_label("ns", 0, 0)])
    assert b.lookup_boundary(_key()) is False


def test_own_rank_missing_is_miss():
    b = _backend([group_label("ns", 0, 1)])
    assert b.lookup_boundary(_key()) is False


def test_single_rank_skips_cross_rank_check():
    b = _backend([group_label("ns", 0, 0)], tp_size=1)
    assert b.lookup_boundary(_key()) is True
    assert b._store.asked == [group_label("ns", 0, 0)]


def test_backend_error_fails_closed_to_miss():
    """A wrong hit silently corrupts output; a wrong miss costs one
    recompute. Errors resolve to the cheap mistake."""
    b = _backend(ALL, blow_up=True)
    assert b.lookup_boundary(_key()) is False


def test_groups_do_not_alias_each_other():
    """The same prefix hash exists in every group, so the group must be
    part of the namespace or one group's data would answer for another.
    """
    b = _backend([group_label("ns", 0, 0), group_label("ns", 0, 1)])
    assert b.lookup_boundary(_key(group_idx=0)) is True
    assert b.lookup_boundary(_key(group_idx=1)) is False
