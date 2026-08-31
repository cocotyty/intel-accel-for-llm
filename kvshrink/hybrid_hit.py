# SPDX-License-Identifier: Apache-2.0
"""Longest-hit detection across heterogeneous KV cache groups.

A hybrid model has one group per cache kind (full attention, GDN
recurrent state, ...). A prefix is restorable only when EVERY group
can restore it, and each group's own matcher constrains the candidate
differently (attention matches blocks; GDN restores at aligned
boundaries). The policy iterates the per-group lookups to a fixed
point: the largest prefix no group objects to.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from kvshrink.kvshrink_connector import GroupInfo


class _StoreAsBlockPool:
    """The one thing vLLM's matching code needs from us."""

    __slots__ = ("_present",)

    # Stands in for a skipped block. vLLM inserts it as padding and only
    # ever counts it, so it needs no identity beyond being a value.
    null_block = object()

    def __init__(self, present: Callable[[int, object], bool]):
        self._present: Callable[[int, object], bool] = present

    def get_cached_block(
        self, block_hash: object, kv_cache_group_ids: list[int],
    ) -> Optional[list[object]]:
        blocks = []
        for group_id in kv_cache_group_ids:
            if not self._present(group_id, block_hash):
                return None
            blocks.append(_StoreAsBlockPool.null_block)
        return blocks


class HybridHitPolicy:
    """Fixed-point multi-group hit detection (pure function, testable)."""

    def __init__(
        self,
        groups: list[GroupInfo],
        present: Callable[[int, object], bool],
        hash_block_size: int,
        num_computed_tokens: int,
    ):
        """Configure the policy for one request."""
        self._groups: list[GroupInfo] = groups
        self._present: Callable[[int, object], bool] = present
        self._hash_block_size: int = hash_block_size
        self._num_computed: int = num_computed_tokens
        # full attention first (tighter initial bound)
        self._ordered: list[GroupInfo] = sorted(
            groups, key=lambda g: 0 if g.kind == "attention" else 1)
        # A snapshot is restorable only on its group's own boundaries,
        # so the candidate must be floored to the coarsest granularity
        # any recurrent group enforces (they may differ per group).
        self._mamba_floor = min(
            (g.block_size for g in groups if g.kind == "mamba"),
            default=None)

    # ------------------------------------------------------------------
    def _lookup(self, group: GroupInfo, block_hashes: list[int],
                candidate: int) -> int:
        """How far this group alone is restorable, in tokens."""
        from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
        manager_cls = KVCacheSpecRegistry.get_manager_class(group.spec)
        # vLLM indexes the hash list directly out to max_length, so the
        # caller owes it a length its own hashes cover. Its scheduler
        # gets this for free (the bound comes from the same request);
        # ours can be a boundary the request has not reached.
        max_length = min(candidate,
                         len(block_hashes) * group.block_size)
        blocks = manager_cls.find_longest_cache_hit(
            block_hashes=block_hashes,
            max_length=max_length,
            kv_cache_group_ids=[group.group_idx],
            block_pool=_StoreAsBlockPool(self._present),
            kv_cache_spec=group.spec,
            drop_eagle_block=False,
            alignment_tokens=group.block_size,
        )
        return len(blocks[0]) * group.block_size

    # ------------------------------------------------------------------
    def find_longest_cache_hit(
        self, block_hashes: list[int], max_length: int
    ) -> int:
        """Fixed-point convergence over all groups; returns the
        restorable prefix in tokens (the snapshot boundary)."""
        candidate = max_length
        if self._mamba_floor is not None:
            # the last prompt token is always recomputed (logprobs + state)
            a = self._mamba_floor
            candidate = min(candidate - 1, (candidate - 1) // a * a)

        while True:
            changed = False
            for group in self._ordered:
                hit = self._lookup(group, block_hashes, candidate)
                if hit < candidate:
                    candidate = hit
                    changed = True
                if candidate <= self._num_computed:
                    return 0
            if not changed:
                break
        return candidate if candidate > self._num_computed else 0
