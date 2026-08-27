"""Pipelined attention save via save_kv_layer.

vLLM calls save_kv_layer on exit of every attention layer during
forward. HybridWorker submits that layer's async put immediately
(overlapping the remaining layers' compute); wait_save then only
waits. GDN groups always save in wait_save (their
state is final only post-forward). These tests use a fake store and
fake canonicalizer -- no GPU, no disk, no model.
"""

import os
from types import SimpleNamespace

from kvshrink.worker import HybridWorker
from conftest import make_spec
from kvshrink.kvshrink_connector import (
    RequestMetadata,
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)


def _group(g_idx, kind, layers):
    return GroupInfo(group_idx=g_idx, kind=kind,
                     layer_names=tuple(layers), block_size=16,
                     mamba_align_size=None, spec=make_spec(kind, 16))


def _key(layer_name, blk_hash=777, g_idx=0):
    return CacheKey(namespace="ns", tp_size=1, rank=0,
                    block_hash=blk_hash, group_idx=g_idx,
                    layer_name=layer_name)


class _FakeStore:
    """Records submit/wait calls."""

    def __init__(self):
        self.submits = []   # (label, sorted(layers), block_hashs)
        self.waits = 0

    def put(self, block_indices, block_hashs, layer_names, tensors,
            label=None):
        layers = sorted(k.rsplit("#", 1)[0] for k in tensors)
        self.submits.append((label, layers, list(block_hashs)))
        return {k: f"task:{k}" for k in tensors}

    def put_wait(self, put_results, wait=True):
        self.waits += 1
        return True


class _FakeCanon:
    def page_view_parts(self, layer_name):
        return {"page": layer_name}


def _save_meta():
    """One attention boundary (2 layers) + one mamba boundary."""
    attn_ops = GroupTransferMeta(
        group_idx=0,
        keys=tuple(_key(ln) for ln in ("a0", "a1")),
        gpu_block_ids=(10, 10))
    mamba_ops = GroupTransferMeta(
        group_idx=1,
        keys=(_key("m0", blk_hash=888, g_idx=1),),
        gpu_block_ids=(20,))
    saves = RequestMetadata()
    saves.add_request("r1", group_ops=(attn_ops, mamba_ops))
    return SimpleNamespace(reqs_to_save=saves)


def _worker():
    groups = [_group(0, "attention", ["a0", "a1"]),
              _group(1, "mamba", ["m0"])]
    w = HybridWorker(groups, {"a0": None, "a1": None, "m0": None},
                     "ns", _FakeCanon(), rank=0, tp_size=1)
    w.store = _FakeStore()
    w._kv_caches_ref = object()  # truthy: kv caches registered
    return w


def _env_off(monkeypatch_env=None):
    os.environ.pop("KVSHRINK_SAVE_PIPELINED", None)
    os.environ.pop("KVSHRINK_SAVE", None)
    os.environ.pop("KVSHRINK_DEBUG_AUTOSAVE", None)


def test_pipelined_attention_submits_during_forward():
    _env_off()
    c = _worker()
    # forward: vLLM calls save_kv_layer on exit of each attention layer
    c.save_kv_layer("a0", _save_meta())
    c.save_kv_layer("a1", _save_meta())
    submits_during_fwd = list(c.store.submits)
    assert len(submits_during_fwd) == 2
    assert submits_during_fwd[0][1] == ["a0"]  # one layer per call
    assert submits_during_fwd[1][1] == ["a1"]

    c.wait_save(_save_meta())
    # attention layers were NOT re-submitted; mamba submitted at wait
    submit_layers = [sorted(v) for _g, v, _l in c.store.submits]
    assert ["a0", "a1"] not in submit_layers  # no bulk re-submit
    assert ["m0"] in submit_layers
    # every group was written and waited for
    assert {l for l, _ls, _b in c.store.submits} == {
        "ns_g0_r0", "ns_g1_r0"}
    assert c.store.waits > 0


def test_fallback_when_hook_never_fired():
    """Older vLLM / decorator missing: attention submits at wait time,
    commits still correct (idempotent full coverage)."""
    _env_off()
    c = _worker()
    _pages, nbound = c.wait_save(_save_meta())  # no save_kv_layer first
    submit_layers = [sorted(v) for _g, v, _l in c.store.submits]
    assert ["a0"] in submit_layers and ["a1"] in submit_layers
    assert ["m0"] in submit_layers
    assert nbound == 2


def test_pipelined_disabled_by_env():
    _env_off()
    os.environ["KVSHRINK_SAVE_PIPELINED"] = "0"
    try:
        c = _worker()
        c.save_kv_layer("a0", _save_meta())
        assert c.store.submits == []  # nothing during forward
        assert c.wait_save(_save_meta())[1] == 2
    finally:
        os.environ.pop("KVSHRINK_SAVE_PIPELINED", None)


def test_save_kv_layer_ignores_mamba_and_unknown_layers():
    _env_off()
    c = _worker()
    c.save_kv_layer("m0", _save_meta())       # mamba layer: never served
    c.save_kv_layer("no.such.layer", _save_meta())
    assert c.store.submits == []
    assert c.wait_save(_save_meta())[1] == 2


def test_write_is_the_commit():
    """A block is finalized by its own write, with the group's whole
    layer set in one call. There is no separate publish step, so there
    is nothing that can become visible before the data it names -- the
    failure this file used to guard against cannot be expressed.
    """
    _env_off()
    w = _worker()
    pages, boundaries = w.wait_save(_save_meta())
    assert pages > 0 and boundaries > 0
    assert w.store.waits > 0, "the write was never waited for"
