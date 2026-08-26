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
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger("vllm." + __name__)


def group_label(namespace: str, group_idx: int, rank: int) -> str:
    """Store namespace for one group's block space on one rank.

    Underscore-separated because the store validates label components
    and rejects its own separator; see the module docstring for why all
    three parts must be present.
    """
    return f"{namespace}_g{int(group_idx)}_r{int(rank)}"


class KVStoreBackend:
    """Group-oriented facade over ``iaxl.KVStore``.

    Thin by construction: it owns the label scheme and nothing else, so
    the scheduler and worker never build store keys themselves and the
    store never learns what a KV cache group is.
    """

    def __init__(self, kvstore: Any = None) -> None:
        # May be None at construction: on the worker the store cannot
        # exist until vLLM hands over kv_caches, which happens long
        # after the connector is built. bind_store closes that gap.
        self._store = kvstore
        self._namespace = ""
        self._tp_size = 1
        self._rank = 0

    # -- lifecycle ---------------------------------------------------
    def bind_store(self, kvstore: Any) -> None:
        if self._store is not None and kvstore is not self._store:
            raise RuntimeError(
                "backend already bound to a different KVStore; rebinding "
                "would orphan in-flight transfers")
        self._store = kvstore

    def register_layout(self, namespace: str, tp_size: int,
                        rank: int) -> None:
        """Record the identity that goes into every label. The layout
        itself is not copied here: page views arrive from the caller at
        transfer time, so a second copy would be a rival source of
        truth."""
        self._namespace = namespace
        self._tp_size = int(tp_size)
        self._rank = int(rank)

    def close(self) -> None:
        """The connector owns the store's lifecycle (it also serves the
        scheduler's presence-only role), so closing is not ours to do."""
        return None

    # -- helpers -----------------------------------------------------
    @property
    def _bound(self):
        if self._store is None:
            raise RuntimeError(
                "backend has no KVStore yet: register_kv_caches must run "
                "before any transfer")
        return self._store

    def _label(self, group_idx: int, rank: Optional[int] = None) -> str:
        return group_label(self._namespace, group_idx,
                           self._rank if rank is None else rank)

    @staticmethod
    def _ids(chunk_labels: Sequence[Any]) -> list[str]:
        return [str(label) for label in chunk_labels]

    # -- transfers ---------------------------------------------------
    # A layer contributes one page view, or two when its K and V are
    # separate tensors. The engine takes one flat tensor dict, but the
    # caller waits a LAYER at a time (vLLM's hook fires per layer), so
    # tasks are handed back regrouped under the layer name.
    @staticmethod
    def _flatten(layer_views) -> Dict[str, Any]:
        return {f"{ln}::{part}": view
                for ln, parts in layer_views.items()
                for part, view in parts.items()}

    @staticmethod
    def _by_layer(tasks) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, task in tasks.items():
            out.setdefault(key.rsplit("::", 1)[0], {})[key] = task
        return out

    def submit_group_loads(self, group_idx, layer_views, chunk_indices,
                           chunk_labels) -> Dict[str, Any]:
        """Enqueue store->GPU reads for one group; returns per-layer
        tasks so the caller can wait a layer at a time."""
        tensors = self._flatten(layer_views)
        return self._by_layer(self._bound.get(
            block_indices=list(chunk_indices),
            block_hashs=self._ids(chunk_labels),
            layer_names=list(tensors),
            tensors=tensors,
            chunk_dim=0,
            label=self._label(group_idx)))

    def wait_layer_loads(self, layer_tasks, wait: bool = True) -> bool:
        """Block until the reads land, or with ``wait=False`` report
        whether they have without consuming them -- the poll used to
        decide if an async request may be released, which must not stall
        the step doing the asking.

        Blocking and polling report failure differently on purpose. A
        blocking call is made when forward is about to read these
        blocks, so an incomplete transfer is fatal. A poll is a question
        and "not yet" is a legitimate answer.
        """
        if not layer_tasks:
            return True
        if not wait:
            return bool(self._bound.get_wait(get_results=layer_tasks,
                                             wait=False))
        if not self._bound.get_wait(get_results=layer_tasks, wait=True):
            raise RuntimeError(
                "kvshrink load failed: get_wait reported an incomplete "
                "transfer; forward would read unrestored blocks")
        return True

    def submit_group_stores(self, group_idx, layer_views, chunk_indices,
                            chunk_labels) -> Dict[str, Any]:
        """Enqueue GPU->store writes for one group. The call carries the
        group's whole layer set, so the block is finalized by this call
        alone."""
        tensors = self._flatten(layer_views)
        return self._by_layer(self._bound.put(
            block_indices=list(chunk_indices),
            block_hashs=self._ids(chunk_labels),
            layer_names=list(tensors),
            tensors=tensors,
            chunk_dim=0,
            label=self._label(group_idx)))

    def wait_group_stores(self, tasks) -> bool:
        if not tasks:
            return True
        flat = {k: t for per_layer in tasks.values()
                for k, t in per_layer.items()}
        return self._bound.put_wait(put_results=flat, wait=True)

    # -- presence ----------------------------------------------------
    def lookup_boundary(self, key):
        """Is this boundary readable, on every rank?

        Under TP each rank writes its own shard with no cross-rank
        transaction, so a boundary present here but missing on a peer
        must be a MISS: half a restore is worse than none. The peer that
        is missing heals on the request's own re-save, because writing a
        block again is idempotent.

        Any error is a MISS. A wrong hit silently corrupts output; a
        wrong miss costs one recompute.
        """
        from .kvshrink_connector import LookupStatus

        chunk_id = str(key.hash_str)
        try:
            for r in range(self._tp_size):
                present = self._bound.has([chunk_id],
                                          label=self._label(key.group_idx, r))
                if not present or not present[0]:
                    if r != self._rank:
                        logger.info(
                            "boundary %s g%d present on rank %d but not on "
                            "rank %d; MISS (recompute re-saves all ranks)",
                            chunk_id[:12], key.group_idx, self._rank, r)
                    return LookupStatus.MISS
            return LookupStatus.HIT
        except Exception:  # pragma: no cover - fail closed to MISS
            logger.exception("lookup error; treating as MISS")
            return LookupStatus.MISS


