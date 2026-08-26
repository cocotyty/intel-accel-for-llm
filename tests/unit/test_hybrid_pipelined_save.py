"""Pipelined attention save via save_kv_layer.

vLLM calls save_kv_layer on exit of every attention layer during
forward. HybridWorker submits that layer's async put immediately
(overlapping the remaining layers' compute); wait_save then only
waits. GDN groups always save in wait_save (their
state is final only post-forward). These tests use a fake backend and
fake canonicalizer -- no GPU, no disk, no model.
"""

import os
from types import SimpleNamespace

from kvshrink.worker import HybridWorker
from conftest import make_spec
from kvshrink.layout import (
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)


def _group(g_idx, kind, layers):
    return GroupInfo(group_idx=g_idx, kind=kind,
                     layer_names=tuple(layers), block_size=16,
                     page_size_bytes=1024,
                     mamba_cache_mode=None, mamba_align_size=None,
                     spec=make_spec(kind, 16))


def _key(layer_name, blk_hash=777, g_idx=0):
    return CacheKey(namespace="ns", tp_size=1, rank=0,
                    block_hash=blk_hash, group_idx=g_idx,
                    layer_name=layer_name)


class _FakeBackend:
    """Records submit/wait calls."""

    def __init__(self):
        self.submits = []   # (g_idx, sorted(layers), labels)
        self.waits = 0

    def submit_group_stores(self, g_idx, views, indices, labels):
        self.submits.append((g_idx, sorted(views), list(labels)))
        return {ln: {"layer": ln, "labels": list(labels)}
                for ln in views}

    def wait_group_stores(self, tasks):
        self.waits += 1
        return True


class _FakeCanon:
    def page_view_parts(self, layer_name):
        return [layer_name], 0


def _save_meta():
    """One attention boundary (2 layers) + one mamba boundary."""
    attn_ops = GroupTransferMeta(
        group_idx=0,
        keys=tuple(_key(ln) for ln in ("a0", "a1")),
        gpu_block_ids=(10, 10))
    mamba_ops = GroupTransferMeta(
        group_idx=1,
        keys=(_key("m0", blk_hash=888, g_idx=1),),
        gpu_block_ids=(20,),
        snapshot_boundary_tokens=544)
    return SimpleNamespace(reqs_to_save=[ReqMeta(
        req_id="r1", group_ops=(attn_ops, mamba_ops))])


def _worker():
    groups = [_group(0, "attention", ["a0", "a1"]),
              _group(1, "mamba", ["m0"])]
    w = HybridWorker(groups, {"a0": None, "a1": None, "m0": None},
                     num_blocks=64, backend=_FakeBackend(),
                     canonicalizer=_FakeCanon(), rank=0, tp_size=1)
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
    submits_during_fwd = list(c._backend.submits)
    assert len(submits_during_fwd) == 2
    assert submits_during_fwd[0][1] == ["a0"]  # one layer per call
    assert submits_during_fwd[1][1] == ["a1"]

    c.wait_save(_save_meta())
    # attention layers were NOT re-submitted; mamba submitted at wait
    submit_layers = [sorted(v) for _g, v, _l in c._backend.submits]
    assert ["a0", "a1"] not in submit_layers  # no bulk re-submit
    assert ["m0"] in submit_layers
    # every group was written and waited for
    assert {g for g, _l, _b in c._backend.submits} == {0, 1}
    assert c._backend.waits > 0


def test_fallback_when_hook_never_fired():
    """Older vLLM / decorator missing: attention submits at wait time,
    commits still correct (idempotent full coverage)."""
    _env_off()
    c = _worker()
    _pages, nbound = c.wait_save(_save_meta())  # no save_kv_layer first
    submit_layers = [sorted(v) for _g, v, _l in c._backend.submits]
    assert ["a0"] in submit_layers and ["a1"] in submit_layers
    assert ["m0"] in submit_layers
    assert nbound == 2


def test_pipelined_disabled_by_env():
    _env_off()
    os.environ["KVSHRINK_SAVE_PIPELINED"] = "0"
    try:
        c = _worker()
        c.save_kv_layer("a0", _save_meta())
        assert c._backend.submits == []  # nothing during forward
        assert c.wait_save(_save_meta())[1] == 2
    finally:
        os.environ.pop("KVSHRINK_SAVE_PIPELINED", None)


def test_save_kv_layer_ignores_mamba_and_unknown_layers():
    _env_off()
    c = _worker()
    c.save_kv_layer("m0", _save_meta())       # mamba layer: never served
    c.save_kv_layer("no.such.layer", _save_meta())
    assert c._backend.submits == []
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
    assert w._backend.waits > 0, "the write was never waited for"
