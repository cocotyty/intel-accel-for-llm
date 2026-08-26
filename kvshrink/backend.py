# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Store access for the connector: one layout, one path."""

from __future__ import annotations

import logging

logger = logging.getLogger("vllm." + __name__)


def group_label(namespace: str, group_idx: int, rank: int) -> str:
    """Store namespace for one group's block space on one rank."""
    return f"{namespace}_g{int(group_idx)}_r{int(rank)}"


def lookup_boundary(store, key) -> bool:
    """Is this boundary readable, on every rank?"""
    chunk_id = key.hash_str
    try:
        for r in range(key.tp_size):
            present = store.has([chunk_id],
                                label=group_label(key.namespace,
                                                  key.group_idx, r))
            if not present or not present[0]:
                if r != key.rank:
                    logger.info(
                        "boundary %s g%d present on rank %d but not on "
                        "rank %d; MISS (recompute re-saves all ranks)",
                        chunk_id[:12], key.group_idx, key.rank, r)
                return False
        return True
    except Exception:  # pragma: no cover - fail closed to MISS
        logger.exception("lookup error; treating as MISS")
        return False
