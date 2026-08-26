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

logger = logging.getLogger("vllm." + __name__)


def group_label(namespace: str, group_idx: int, rank: int) -> str:
    """Store namespace for one group's block space on one rank.

    Underscore-separated because the store validates label components
    and rejects its own separator; see the module docstring for why all
    three parts must be present.
    """
    return f"{namespace}_g{int(group_idx)}_r{int(rank)}"


def lookup_boundary(store, key) -> bool:
    """Is this boundary readable, on every rank?

    Under TP each rank writes its own shard with no cross-rank
    transaction, so a boundary present here but missing on a peer
    must be a MISS: half a restore is worse than none. The peer that
    is missing heals on the request's own re-save, because writing a
    block again is idempotent.

    Any error is a MISS. A wrong hit silently corrupts output; a
    wrong miss costs one recompute.
    """
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
