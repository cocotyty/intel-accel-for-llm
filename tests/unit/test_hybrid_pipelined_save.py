"""Pipelined save via save_kv_layer + async drain in get_finished.

vLLM calls save_kv_layer on exit of every attention layer during
forward. The connector submits that layer's async put immediately
(overlapping the remaining layers' compute); the mamba segment
preceding an attention layer rides the same hook, and the trailing
segment submits in submit_saves (post-forward). Nothing is waited
inside the step: writes drain in get_finished, which also releases
finished requests' blocks (finished_sending). These tests use a fake
store and fake canonicalizer -- no GPU, no disk, no model.
"""

import os
from conftest import HybridWorker, make_spec
from kvshrink.kvshrink_connector import (
    KVShrinkConnectorMetadata,
    RequestMetadata,
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta)


def _group(g_idx, kind, layers):
    return GroupInfo(group_idx=g_idx, kind=kind,
                     layer_names=tuple(layers), block_size=16,
                     mamba_align_size=None, spec=make_spec(kind, 16))


def _key(layer_name, blk_hash=777, g_idx=0):
    return CacheKey(namespace="ns", rank=0,
                    block_hash=blk_hash, group_idx=g_idx,
                    layer_name=layer_name)


class _FakeTask:
    """Engine Task stand-in: put_wait finalizes by clearing ctx."""

    def __init__(self):
        self.ctx = object()


class _FakeStore:
    """Records submit/wait calls."""

    def __init__(self):
        self.submits = []   # (label, sorted(layers), block_hashs)
        self.waits = 0

    def put(self, block_indices, block_hashs, layer_names, tensors,
            label=None):
        layers = sorted(k.rsplit("#", 1)[0] for k in tensors)
        self.submits.append((label, layers, list(block_hashs)))
        return {k: _FakeTask() for k in tensors}

    def put_wait(self, put_results, wait=True):
        self.waits += 1
        for t in put_results.values():
            t.ctx = None
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
    saves.requests["r1"] = ReqMeta(group_ops=(attn_ops, mamba_ops))
    return KVShrinkConnectorMetadata(reqs_to_save=saves)


def _worker():
    groups = [_group(0, "attention", ["a0", "a1"]),
              _group(1, "mamba", ["m0"])]
    w = HybridWorker(groups, {"a0": None, "a1": None, "m0": None},
                     "ns", _FakeCanon(), rank=0, tp_size=1)
    w.kvstore = _FakeStore()
    return w


def _env_off(monkeypatch_env=None):
    os.environ.pop("KVSHRINK_SAVE_PIPELINED", None)
    os.environ.pop("KVSHRINK_SAVE", None)
    os.environ.pop("KVSHRINK_DEBUG_AUTOSAVE", None)


def test_pipelined_attention_submits_during_forward():
    _env_off()
    c = _worker()
    # forward: vLLM binds the step metadata, then calls save_kv_layer
    # on exit of each attention layer
    meta = _save_meta()
    c.bind_connector_metadata(meta)
    c.save_kv_layer("a0", None, None)
    c.save_kv_layer("a1", None, None)
    submits_during_fwd = list(c.kvstore.submits)
    assert len(submits_during_fwd) == 2
    assert submits_during_fwd[0][1] == ["a0"]  # one layer per call
    assert submits_during_fwd[1][1] == ["a1"]

    c.submit_saves(meta)
    # attention layers were NOT re-submitted; mamba submitted post-forward
    submit_layers = [sorted(v) for _g, v, _l in c.kvstore.submits]
    assert ["a0", "a1"] not in submit_layers  # no bulk re-submit
    assert ["m0"] in submit_layers
    # every group was submitted; nothing waited inside the step
    assert {l for l, _ls, _b in c.kvstore.submits} == {
        "ns_g0_r0", "ns_g1_r0"}
    assert c.kvstore.waits == 0
    # the drain is what waits, and it releases the finished request
    sending, _ = c.get_finished({"r1"})
    assert sending == {"r1"}
    assert c.kvstore.waits > 0


def test_fallback_when_hook_never_fired():
    """Older vLLM / decorator missing: attention submits post-forward,
    commits still correct (idempotent full coverage)."""
    _env_off()
    c = _worker()
    _pages, nbound = c.submit_saves(_save_meta())  # no save_kv_layer first
    submit_layers = [sorted(v) for _g, v, _l in c.kvstore.submits]
    # the group's unhooked layers go out in ONE post-forward call
    assert ["a0", "a1"] in submit_layers
    assert ["m0"] in submit_layers
    assert nbound == 2


def test_pipelined_disabled_by_env():
    _env_off()
    os.environ["KVSHRINK_SAVE_PIPELINED"] = "0"
    try:
        c = _worker()
        c.bind_connector_metadata(_save_meta())
        c.save_kv_layer("a0", None, None)
        assert c.kvstore.submits == []  # nothing during forward
        assert c.submit_saves(_save_meta())[1] == 2
    finally:
        os.environ.pop("KVSHRINK_SAVE_PIPELINED", None)


def test_mamba_segment_rides_the_next_attention_hook():
    """Mamba layers before an attention layer are final when its save
    hook fires, so they submit there instead of post-forward."""
    _env_off()
    c = _worker()
    c._mamba_save_segments = {"a0": ("m0",)}
    meta = _save_meta()
    c.bind_connector_metadata(meta)
    c.save_kv_layer("a0", None, None)
    submit_layers = [sorted(v) for _g, v, _l in c.kvstore.submits]
    assert ["m0"] in submit_layers  # mamba segment piggybacked on the hook
    assert ["a0"] in submit_layers
    c.save_kv_layer("a1", None, None)
    c.submit_saves(meta)
    # everything submitted during forward; nothing left post-forward
    assert len(c.kvstore.submits) == 3


def test_mamba_segment_splits_by_group():
    """A segment between two attention layers interleaves the mamba
    groups; each group's layers must go under their own store label."""
    _env_off()
    groups = [_group(0, "attention", ["a0", "a1"]),
              _group(1, "mamba", ["m0"]),
              _group(2, "mamba", ["m1"])]
    c = HybridWorker(groups, {ln: None for ln in ("a0", "a1", "m0", "m1")},
                     "ns", _FakeCanon(), rank=0, tp_size=1)
    c.kvstore = _FakeStore()
    c._mamba_save_segments = {"a0": ("m0", "m1")}
    attn_ops = GroupTransferMeta(
        group_idx=0,
        keys=tuple(_key(ln) for ln in ("a0", "a1")),
        gpu_block_ids=(10, 10))
    m0_ops = GroupTransferMeta(
        group_idx=1, keys=(_key("m0", blk_hash=888, g_idx=1),),
        gpu_block_ids=(20,))
    m1_ops = GroupTransferMeta(
        group_idx=2, keys=(_key("m1", blk_hash=999, g_idx=2),),
        gpu_block_ids=(21,))
    saves = RequestMetadata()
    saves.requests["r1"] = ReqMeta(group_ops=(attn_ops, m0_ops, m1_ops))
    c.bind_connector_metadata(KVShrinkConnectorMetadata(reqs_to_save=saves))
    c.save_kv_layer("a0", None, None)
    by_label = {l: layers for l, layers, _h in c.kvstore.submits}
    assert by_label["ns_g1_r0"] == ["m0"]
    assert by_label["ns_g2_r0"] == ["m1"]
    assert by_label["ns_g0_r0"] == ["a0"]


def test_write_is_the_commit():
    """A block is finalized by its own write, with the group's whole
    layer set in one step's submissions. There is no separate publish
    step, so there is nothing that can become visible before the data
    it names -- the failure this file used to guard against cannot be
    expressed. The write is drained (and the request released) in
    get_finished.
    """
    _env_off()
    w = _worker()
    pages, boundaries = w.submit_saves(_save_meta())
    assert pages > 0 and boundaries > 0
    assert w.kvstore.waits == 0, "submission must not block the step"
    sending, _ = w.get_finished({"r1"})
    assert sending == {"r1"}
    assert w.kvstore.waits > 0, "the write was never drained"
