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

Pure logic: fake backend and canonicalizer, no GPU, no disk, no model.
"""

from __future__ import annotations

import pytest

from conftest import make_spec
from kvshrink.kvshrink_connector import (
    RequestMetadata,
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)
from kvshrink.kvshrink_connector import LookupStatus
from kvshrink.worker import HybridWorker

PAGE = 4096


def _group(g_idx, kind, layers, block_size=16):
    return GroupInfo(
        group_idx=g_idx, kind=kind, layer_names=tuple(layers),
        block_size=block_size,
        mamba_align_size=block_size if kind == "mamba" else None,
        spec=make_spec(kind, block_size))


class _FakeBackend:
    """Records submits and the ORDER in which tasks are waited."""

    def __init__(self, committed=True):
        self.submitted = []          # layer names, in submit order
        self.waited = []             # layer names, in wait order
        self.committed = committed

    def submit_group_loads(self, g_idx, views, indices, labels):
        tasks = {}
        for ln in views:
            self.submitted.append(ln)
            tasks[ln] = {"layer": ln}
        return tasks

    def wait_layer_loads(self, task):
        self.waited.append(task["layer"])

    def lookup_boundary(self, key, expected_layers=None,
                        expected_boundary_tokens=None):
        return LookupStatus.HIT if self.committed else LookupStatus.MISS


class _FakeCanon:
    def register(self, kv_caches):
        pass

    def page_view_parts(self, layer_name):
        return {"page": layer_name}, 0


# Execution order: a leading GDN layer, then attention, more GDN, and a
# final attention layer with nothing after it.
ORDER = ["m0", "a1", "m2", "m3", "a4"]
ATTN = ["a1", "a4"]
GDN = ["m0", "m2", "m3"]


def _worker(backend=None, order=ORDER, gdn=None):
    """Worker whose groups match ``order`` unless ``gdn`` overrides the
    mamba membership (used to test an unplaceable GDN layer)."""
    attn = [ln for ln in order if ln in ATTN]
    groups = [_group(0, "attention", attn),
              _group(1, "mamba", gdn if gdn is not None
                     else [ln for ln in order if ln in GDN])]
    layer_infos = {ln: None for ln in order}
    w = HybridWorker(groups, layer_infos, backend or _FakeBackend(),
                     _FakeCanon(), rank=0, tp_size=1)
    w.register({ln: None for ln in order}, order)
    return w


def _load_meta(layers, group_idx, boundary=None, req_id="r1"):
    """One load op covering ``layers`` for a single block."""
    keys = tuple(CacheKey(namespace="ns", tp_size=1, rank=0,
                          block_hash=7, group_idx=group_idx,
                          layer_name=ln) for ln in layers)
    op = GroupTransferMeta(group_idx=group_idx, keys=keys,
                           gpu_block_ids=(5,) * len(layers),
                           snapshot_boundary_tokens=boundary)
    md = RequestMetadata()
    md.add_request(req_id, group_ops=(op,))
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


def test_model_without_attention_layers_fails_closed():
    """Nothing would ever wait for an attention page."""
    groups = [_group(0, "attention", []), _group(1, "mamba", GDN)]
    w = HybridWorker(groups, {ln: None for ln in GDN}, _FakeBackend(),
                     _FakeCanon(), rank=0, tp_size=1)
    with pytest.raises(RuntimeError, match="no attention layers"):
        w.register({ln: None for ln in GDN}, GDN)


# ------------------------------------------------------------------
# load scheduling
# ------------------------------------------------------------------

def test_every_recurrent_layer_is_waited_before_forward():
    """GDN gets no per-layer hook from vLLM, so the whole recurrent set
    is waited in start_load. Nothing may be left pending: a GDN layer
    that reaches forward unrestored is silent output corruption."""
    be = _FakeBackend()
    w = _worker(be)
    w.start_load(_load_meta(GDN, 1, boundary=16))
    assert sorted(be.submitted) == sorted(GDN), be.submitted
    assert sorted(be.waited) == sorted(GDN), be.waited
    assert w._load_tasks == {}


def test_attention_pages_stay_pipelined():
    """Attention keeps its per-layer hook: its pages are waited when
    the layer is about to read them, not up front."""
    be = _FakeBackend()
    w = _worker(be)
    meta = _load_meta(ATTN, 0)
    meta.reqs_to_load.requests.update(
        _load_meta(GDN, 1, boundary=16, req_id="r2").reqs_to_load.requests)
    w.start_load(meta)
    # GDN waited already; no attention layer has been waited yet
    assert sorted(be.waited) == sorted(GDN), be.waited

    w.wait_layer_load("a1")
    assert be.waited[-1] == "a1"
    w.wait_layer_load("a4")
    assert be.waited[-1] == "a4"
    w.loads_drained_check()  # nothing left un-waited


def test_unwaited_layer_at_step_end_fails_stop():
    be = _FakeBackend()
    w = _worker(be)
    w.start_load(_load_meta(ATTN, 0))
    # a1's hook never fired -> its pages were never restored
    with pytest.raises(RuntimeError, match="never ran"):
        w.loads_drained_check()


def test_stale_residue_from_previous_step_fails_stop():
    be = _FakeBackend()
    w = _worker(be)
    w._load_tasks = {"m2": [{"layer": "m2"}]}
    with pytest.raises(RuntimeError, match="stale step residue"):
        w.start_load(_load_meta(GDN, 1, boundary=16))


def test_load_poison_is_sticky_across_hooks():
    """A failed load must fail every later hook of the step, never
    degrade into a silent recompute."""
    be = _FakeBackend()

    def _boom(task):
        raise RuntimeError("h2d failed")

    be.wait_layer_loads = _boom
    w = _worker(be)
    with pytest.raises(RuntimeError, match="h2d failed"):
        w.start_load(_load_meta(GDN, 1, boundary=16))
    for call in (lambda: w.wait_layer_load("a1"),
                 lambda: w.raise_load_poison(),
                 lambda: w.start_load(_load_meta(GDN, 1, boundary=16))):
        with pytest.raises(RuntimeError, match="h2d failed"):
            call()


def test_mamba_toctou_change_fails_stop():
    """The committed boundary must still match the scheduler's HIT when
    the worker executes; otherwise the state we would restore is not the
    state the core credited."""
    be = _FakeBackend(committed=False)
    w = _worker(be)
    with pytest.raises(RuntimeError, match="boundary vanished"):
        w.start_load(_load_meta(GDN, 1, boundary=16))
    assert be.submitted == [], \
        "no transfer may be submitted once the boundary is gone"


def test_attention_load_needs_no_boundary_check():
    """Attention pages are per-block and content-addressed: they carry no
    snapshot boundary and are submitted without the mamba gate."""
    be = _FakeBackend(committed=False)  # would fail a boundary check
    w = _worker(be)
    w.start_load(_load_meta(ATTN, 0))
    assert sorted(be.submitted) == sorted(ATTN)
