# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""When each kind of layer is waited for.

vLLM calls ``wait_for_layer_load`` only at ATTENTION layers, never at
GDN/mamba ones. So the two are waited differently:

- attention pages stay pipelined -- each layer's hook waits its own
  pages, right before that layer reads them;
- every GDN page is waited in ``start_load``, in one barrier before
  forward begins, because no hook will ever come for it.

Anything left un-waited at the end of a step is a fail-stop: it would
mean forward read unrestored state.

Pure logic: fake store and canonicalizer, no GPU, no disk, no model.
"""

from __future__ import annotations

import pytest

from conftest import HybridWorker, make_spec
from kvshrink.kvshrink_connector import (
    RequestMetadata,
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)

PAGE = 4096


def _group(g_idx, kind, layers, block_size=16):
    return GroupInfo(
        group_idx=g_idx, kind=kind, layer_names=tuple(layers),
        block_size=block_size,
        mamba_align_size=block_size if kind == "mamba" else None,
        spec=make_spec(kind, block_size))


class _FakeStore:
    """Records submits and the ORDER in which tasks are waited."""

    def __init__(self, committed=True):
        self.submitted = []          # layer names, in submit order
        self.waited = []             # layer names, in wait order
        self.committed = committed

    def get(self, block_indices, block_hashs, layer_names,
            label=None):
        self.submitted.extend(layer_names)
        return {k: f"task:{k}" for k in layer_names}

    def get_wait(self, get_results, wait=True):
        for k in get_results:
            self.waited.append(k.rsplit("#", 1)[0])
        return True

    def has(self, chunk_labels, label=None):
        return [self.committed]


# Execution order: a leading GDN layer, then attention, more GDN, and a
# final attention layer with nothing after it.
ORDER = ["m0", "a1", "m2", "m3", "a4"]
ATTN = ["a1", "a4"]
GDN = ["m0", "m2", "m3"]


def _worker(store=None, order=ORDER, gdn=None):
    """Worker whose groups match ``order`` unless ``gdn`` overrides the
    mamba membership (used to test an unplaceable GDN layer)."""
    attn = [ln for ln in order if ln in ATTN]
    groups = [_group(0, "attention", attn),
              _group(1, "mamba", gdn if gdn is not None
                     else [ln for ln in order if ln in GDN])]
    layer_infos = {ln: None for ln in order}
    w = HybridWorker(groups, layer_infos, rank=0, tp_size=1)
    w.kvstore = store or _FakeStore()
    w._num_blocks = 2
    # Placeholder kv_caches: single-tensor attention pools; two-part
    # mamba pools so the raw-parts mapping sees both shapes.
    import torch
    kv_caches = {ln: (torch.zeros(2, 3), torch.zeros(2, 5))
                 if ln in (gdn or GDN) else torch.zeros(2, 8)
                 for ln in order}
    w.register(order)
    return w


def _load_meta(layers, group_idx, req_id="r1"):
    """One load op covering ``layers`` for a single block."""
    keys = tuple(CacheKey(block_hash=7, group_idx=group_idx,
                          layer_name=ln) for ln in layers)
    op = GroupTransferMeta(group_idx=group_idx, keys=keys,
                           gpu_block_ids=(5,) * len(layers))
    md = RequestMetadata()
    md.requests[req_id] = ReqMeta(group_ops=(op,))
    return type("M", (), {"reqs_to_load": md})


# ------------------------------------------------------------------
# registration
# ------------------------------------------------------------------

def test_recurrent_layers_are_recorded():
    w = _worker()
    assert w._mamba_layers == frozenset(GDN)


def test_attention_execution_order_is_recorded():
    """The async release gate holds a request until its first N layers
    have landed, which is a statement about position."""
    w = _worker(order=["m0", "a1", "m2", "m3", "a4"])
    assert w._attn_order == ("a1", "a4")


# ------------------------------------------------------------------
# load scheduling
# ------------------------------------------------------------------

def test_every_recurrent_layer_is_waited_before_forward():
    """GDN gets no per-layer hook from vLLM, so the whole recurrent set
    is waited in start_load. Nothing may be left pending: a GDN layer
    that reaches forward unrestored is silent output corruption."""
    be = _FakeStore()
    w = _worker(be)
    w.start_load(_load_meta(GDN, 1))
    assert sorted(be.submitted) == sorted(GDN)
    assert sorted(be.waited) == sorted(GDN), be.waited
    assert w._load_tasks == {}


def test_attention_pages_stay_pipelined():
    """Attention keeps its per-layer hook: its pages are waited when
    the layer is about to read them, not up front."""
    be = _FakeStore()
    w = _worker(be)
    meta = _load_meta(ATTN, 0)
    meta.reqs_to_load.requests.update(
        _load_meta(GDN, 1, req_id="r2").reqs_to_load.requests)
    w.start_load(meta)
    # GDN waited already; no attention layer has been waited yet
    assert sorted(be.waited) == sorted(GDN), be.waited

    w.wait_layer_load("a1")
    assert be.waited[-1] == "a1"
    w.wait_layer_load("a4")
    assert be.waited[-1] == "a4"


def test_failed_blocking_wait_raises():
    """An incomplete transfer at a blocking wait is fatal (EngineCore
    dies), same contract as the original path."""
    be = _FakeStore()

    def _boom(get_results, wait=True):
        raise RuntimeError("h2d failed")

    be.get_wait = _boom
    w = _worker(be)
    with pytest.raises(RuntimeError, match="h2d failed"):
        w.start_load(_load_meta(GDN, 1))


def test_empty_ops_are_skipped():
    """A group with nothing to load (empty keys) contributes no engine
    call -- regression: an empty op once produced a get with no
    tensors, tripping the engine's not-empty assert."""
    be = _FakeStore()
    w = _worker(be)
    meta = _load_meta(GDN, 1)
    req = next(iter(meta.reqs_to_load.requests.values()))
    req.group_ops = (GroupTransferMeta(group_idx=0),) + req.group_ops
    w.start_load(meta)
    assert sorted(be.submitted) == sorted(GDN), be.submitted
