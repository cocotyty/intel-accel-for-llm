# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Store access for the connector: one layout, one path.

Every model is a list of KV cache groups, and every group is a block
space in ``iaxl.KVStore``. A pure-attention model has one group; a
GDN/Mamba model has two. Nothing here branches on the model kind.

How a group maps onto the store
-------------------------------
The store keys data as ``(label, chunk_id, tensor_key)``:

- ``tensor_key`` is the layer name, so layers never collide and a
  caller can wait for one layer while the others stream.
- ``chunk_id`` is the block's content hash.
- ``label`` is the namespace, and it is where everything the store does
  not otherwise know about must go. It carries the model namespace, the
  KV cache group and the TP rank: the store's own ``rank`` argument
  drives its management port and logs, NOT its keys, so two ranks
  sharing a label would overwrite each other's shards. Two groups
  sharing a label would be worse -- the same prefix hash exists in both
  groups, and the durability record is keyed by ``(label, chunk_id)``
  without the layer, so they would be tracked as one unit despite having
  different lifetimes.

Why the whole group goes in one call
------------------------------------
The engine requires every tensor in a call to share shape and dtype, and
a recurrent layer is two tensors of different shape (conv state, ssm
state) over one storage. Passing canonical int8 page views satisfies
that, and it also makes the call atomic: a block is finalized once, with
all of its layers, so presence IS the commit. There is no second phase
to publish and therefore nothing that can dangle.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iaxl import KVStore

    from .kvshrink_connector import CacheKey

logger = logging.getLogger("vllm." + __name__)


def group_label(namespace: str, group_idx: int, rank: int) -> str:
    """Store namespace for one group's block space on one rank.

    Underscore-separated because the store validates label components
    and rejects its own separator; see the module docstring for why all
    three parts must be present.
    """
    return f"{namespace}_g{int(group_idx)}_r{int(rank)}"


def lookup_boundary(store: "KVStore", key: "CacheKey") -> bool:
    """Is this boundary present in the store?

    The check runs under the key's own rank label: each rank keeps its
    own presence record, and the controller process shares one with the
    rank-0 worker only, so peer ledgers are not queryable here. TP
    ranks save in lockstep, so rank 0 present stands for all; a rank
    that diverged anyway fails loudly at load time (the native layer
    raises on a missing key).

    Any error is a MISS. A wrong hit silently corrupts output; a
    wrong miss costs one recompute.
    """
    try:
        present = store.has([key.hash_str],
                            label=group_label(key.namespace,
                                              key.group_idx, key.rank))
        return bool(present and present[0])
    except Exception:  # pragma: no cover - fail closed to MISS
        logger.exception("lookup error; treating as MISS")
        return False
