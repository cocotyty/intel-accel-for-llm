# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Store access for the connector: one layout, one path."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iaxl import KVStore

    from .kvshrink_connector import CacheKey

logger = logging.getLogger("vllm." + __name__)


def group_label(namespace: str, group_idx: int, rank: int) -> str:
    """Store namespace for one group's block space on one rank."""
    return f"{namespace}_g{int(group_idx)}_r{int(rank)}"


def lookup_boundary(store: "KVStore", key: "CacheKey") -> bool:
    """Is this boundary present in the store?"""
    try:
        present = store.has([key.hash_str],
                            label=group_label(key.namespace,
                                              key.group_idx, key.rank))
        return bool(present and present[0])
    except Exception:  # pragma: no cover - fail closed to MISS
        logger.exception("lookup error; treating as MISS")
        return False
