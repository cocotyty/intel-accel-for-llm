# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""When each kind of layer is waited for.

vLLM calls ``wait_for_layer_load`` only at ATTENTION layers, never at
GDN/mamba ones. So the two are waited differently:

- attention pages stay pipelined -- each layer's hook waits its own
  pages, right before that layer reads them;
- leading GDN pages are waited in ``start_load``; later GDN runs are
  waited at the preceding attention post-hook, before they execute.

Anything left un-waited at the end of a step is a fail-stop: it would
mean forward read unrestored state.

Pure logic: fake store and canonicalizer, no GPU, no disk, no model.
"""

from __future__ import annotations

import pytest

from conftest import HybridWorker, drive_start_load, make_spec
from kvshrink.kvshrink_connector import (
    RequestMetadata, GroupInfo, KVShrinkConnectorMetadata, ReqMeta)

PAGE = 4096


def _group(g_idx, kind, layers):
    return GroupInfo(
        group_idx=g_idx, kind=kind, layer_names=tuple(layers),
        spec=make_spec(kind, 16))


class _FakeStore:
    """Records submits and the ORDER in which tasks are waited."""

    def __init__(self, committed=True):
        self.submitted = []          # layer names, in submit order
        self.waited = []             # layer names, in wait order
        self.committed = committed
        self.pending = set()

    def get(self, block_indices, block_hashs, layer_names,
            label=None, description=""):
        self.submitted.extend(layer_names)
        self.pending.update(layer_names)
        return {k: f"task:{k}" for k in layer_names}

    def get_wait(self, get_results, layer_names=None, wait=True):
        for k in (layer_names if layer_names is not None else get_results.keys()):
            if k in self.pending:
                self.waited.append(k)
                self.pending.remove(k)
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
    w = HybridWorker(groups, {ln: None for ln in order},
                     rank=0, tp_size=1)
    w.kvstore = store or _FakeStore()
    return w


def _load_meta(group_idx, req_id="r1"):
    """One load plan: a single block for group ``group_idx`` (the
    group's own config decides which layers it reaches)."""
    group_ids = [(), ()]
    group_ids[group_idx] = ("5",)
    md = RequestMetadata()
    md.requests[req_id] = ReqMeta(
        block_hashes=["7"], group_block_ids=tuple(group_ids))
    return KVShrinkConnectorMetadata(reqs_to_load=md, reqs_to_save=RequestMetadata())


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

def test_only_leading_recurrent_layers_are_waited_before_forward():
    be = _FakeStore()
    w = _worker(be)
    drive_start_load(w, _load_meta(1))
    assert sorted(be.submitted) == sorted(GDN)
    assert be.waited == ["m0"], be.waited
    # the batch stays open until the last attention hook (main's rule)
    assert set(w._current_get_tasks) == set(GDN)


def test_attention_pages_stay_pipelined():
    """Attention keeps its per-layer hook: its pages are waited when
    the layer is about to read them, not up front."""
    be = _FakeStore()
    w = _worker(be)
    meta = _load_meta(0)
    meta.reqs_to_load.requests.update(
        _load_meta(1, req_id="r2").reqs_to_load.requests)
    drive_start_load(w, meta)
    assert be.submitted == ORDER
    assert be.waited == ["m0"], be.waited

    w.wait_for_layer_load("a1")
    assert be.waited == ["m0", "a1"]
    assert be.pending == {"m2", "m3", "a4"}
    w.save_kv_layer("a1", None, None)
    assert be.waited == ["m0", "a1", "m2", "m3"]
    assert be.pending == {"a4"}
    w.wait_for_layer_load("a4")
    assert be.waited[-1] == "a4"
    w.save_kv_layer("a4", None, None)
    assert not be.pending
    assert w._current_get_tasks is None


@pytest.mark.parametrize("order,gdn,leading", [
    (["m0", "m2", "a1", "m3"], ["m0", "m2", "m3"], ["m0", "m2"]),
    (["a1", "m2", "m3"], ["m2", "m3"], []),
    (["m0", "m2", "m3"], ["m0", "m2", "m3"], ["m0", "m2", "m3"]),
])
def test_sync_leading_and_trailing_mamba_without_save_requests(order, gdn, leading):
    store = _FakeStore()
    worker = _worker(store, order=order, gdn=gdn)
    drive_start_load(worker, _load_meta(1))
    assert store.waited == leading
    if "a1" in order:
        worker.wait_for_layer_load("a1")
        assert store.waited == leading
        worker.save_kv_layer("a1", None, None)
    assert not store.pending
    assert worker._current_get_tasks is None


def test_later_recurrent_failure_raises_before_next_mamba_run():
    store = _FakeStore()
    worker = _worker(store)
    drive_start_load(worker, _load_meta(1))
    store.get_wait = lambda **kwargs: False
    with pytest.raises(RuntimeError, match="recurrent KV cache"):
        worker.save_kv_layer("a1", None, None)


def test_mismatched_load_metadata_is_rejected_before_submission():
    store = _FakeStore()
    worker = _worker(store)
    metadata = _load_meta(1)
    metadata.reqs_to_load.requests["r1"].block_hashes.append("8")
    with pytest.raises(ValueError, match="Mismatched block metadata"):
        drive_start_load(worker, metadata)
    assert store.submitted == []


def test_failed_blocking_wait_raises():
    """An incomplete transfer at a blocking wait is fatal (EngineCore
    dies), same contract as the original path."""
    be = _FakeStore()

    def _boom(get_results, layer_names=None, wait=True):
        raise RuntimeError("h2d failed")

    be.get_wait = _boom
    w = _worker(be)
    with pytest.raises(RuntimeError, match="h2d failed"):
        drive_start_load(w, _load_meta(1))


def test_attention_layers_of_an_idle_group_get_no_call():
    """A group with nothing to load (empty block ids) contributes no
    engine call for its layers -- regression: an empty op once produced
    a get with no tensors, tripping the engine's not-empty assert."""
    be = _FakeStore()
    w = _worker(be)
    drive_start_load(w, _load_meta(1))
    assert not (set(ATTN) & set(be.submitted)), be.submitted
