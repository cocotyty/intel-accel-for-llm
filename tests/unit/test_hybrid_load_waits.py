# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Mamba loads finish across steps; only pure attention uses layer waits."""

import pytest
from types import SimpleNamespace

from conftest import HybridWorker, drive_start_load, make_spec
from kvshrink.kvshrink_connector import (
    GroupInfo, KVShrinkConnectorMetadata, ReqMeta, RequestMetadata)


def _group(index, kind, layers):
    return GroupInfo(index, kind, tuple(layers), make_spec(kind, 16))


class _FakeStore:
    def __init__(self, layers):
        self.layers = layers
        self.submitted = []
        self.waited = []
        self.landed = set()

    def get(self, block_indices, block_hashs, layer_names=None,
            label=None, description=""):
        layers = self.layers if layer_names is None else layer_names
        self.submitted.append((label, list(layers), list(block_indices)))
        return {ln: ln for ln in layers}

    def get_wait(self, get_results, layer_names=None, wait=True):
        layers = list(get_results) if layer_names is None else layer_names
        assert set(layers) <= get_results.keys()
        if not wait:
            return set(layers) <= self.landed
        self.waited.extend(layers)
        return True


def _worker(groups, order):
    worker = HybridWorker(groups, order)
    worker.kvstore = _FakeStore(order)
    return worker


def _meta(group_ids, is_async=True):
    requests = RequestMetadata()
    requests.requests["r1"] = ReqMeta(
        group_block_ids=group_ids, block_hashes=["7"],
        is_async=is_async, async_load_layers=-1)
    return KVShrinkConnectorMetadata(requests, RequestMetadata())


def test_metadata_snapshots_hashes_without_mutating_request():
    hashes = [b"\x01\xff", 42]
    requests = RequestMetadata()
    requests.add_request("r1", ((5, 6),), hashes, is_async=True)
    hashes.append(43)
    assert requests.requests["r1"].block_hashes == ["01ff", "42"]
    assert hashes == [b"\x01\xff", 42, 43]


@pytest.mark.parametrize("order", [
    ["m0", "a1", "m2"], ["a1", "m0", "m2"], ["m0", "m2"],
])
def test_mamba_layers_finish_before_forward_regardless_of_order(order):
    groups = [_group(0, "mamba", ["m0", "m2"])]
    ids = ((5,),)
    if "a1" in order:
        groups.append(_group(1, "attention", ["a1"]))
        ids += ((6,),)
    worker = _worker(groups, order)
    store = worker.kvstore
    drive_start_load(worker, _meta(ids))
    assert store.waited == []
    assert worker._current_get_tasks is None
    store.landed = set(order) - {"m2"}
    assert worker.get_finished(set())[1] is None
    store.landed.add("m2")
    assert worker.get_finished(set())[1] == {"r1"}
    assert set(store.waited) == set(order)
    assert not worker._early_promoted_tasks
    assert not worker._pending_load_tasks


def test_attention_sync_pages_stay_pipelined():
    worker = _worker([_group(0, "attention", ["a0", "a1"])], ["a0", "a1"])
    drive_start_load(worker, _meta(((5,),), is_async=False))
    assert worker.kvstore.submitted == [(None, ["a0", "a1"], [5])]
    assert worker.kvstore.waited == []
    worker.wait_for_layer_load("a0")
    assert worker.kvstore.waited == ["a0"]
    worker.wait_for_layer_load("a1")
    assert worker.kvstore.waited == ["a0", "a1"]
    assert worker._current_get_tasks is None


def test_mismatched_load_metadata_is_rejected_before_submission():
    worker = _worker([_group(0, "mamba", ["m0"])], ["m0"])
    metadata = _meta(((5,),))
    metadata.reqs_to_load.requests["r1"].block_hashes.append("8")
    with pytest.raises(ValueError, match="Mismatched block metadata"):
        drive_start_load(worker, metadata)
    assert worker.kvstore.submitted == []


def test_failed_async_wait_raises():
    worker = _worker([_group(0, "mamba", ["m0"])], ["m0"])
    drive_start_load(worker, _meta(((5,),)))

    def fail(**kwargs):
        raise RuntimeError("h2d failed")

    worker.kvstore.get_wait = fail
    with pytest.raises(RuntimeError, match="h2d failed"):
        worker.get_finished(set())


def test_idle_group_gets_no_call():
    worker = _worker([
        _group(0, "attention", ["a0"]), _group(1, "mamba", ["m0"]),
    ], ["a0", "m0"])
    drive_start_load(worker, _meta(((), (5,))))
    assert worker.kvstore.submitted == [("mamba", ["m0"], [5])]


def test_registration_preserves_order_and_excludes_draft_layers(monkeypatch):
    import torch
    import kvshrink.kvshrink_connector as module

    attn, mamba, draft = "model.layers.1", "model.layers.0", "model.layers.2"
    worker = _worker([
        _group(0, "attention", [attn]), _group(1, "mamba", [mamba, draft]),
    ], [attn, mamba, draft])
    worker.num_layers = 2
    worker.model_config = SimpleNamespace(model="test-model")
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(
        static_forward_context={}))
    caches = {attn: torch.empty(2, 1), mamba: [torch.empty(2, 1)],
              draft: [torch.empty(2, 1)]}
    captured = {}

    def store(**kwargs):
        captured.update(kwargs)
        return _FakeStore(list(kwargs["kv_caches"]))

    monkeypatch.setattr(module, "KVStore", store)
    worker.register_kv_caches(caches)
    assert worker._layer_names == [attn, mamba]
    assert worker._mamba_layers == {mamba}
    assert list(captured["kv_caches"]) == [attn, mamba]
    drive_start_load(worker, _meta(((5,), (6,))))
    assert worker.kvstore.submitted == [
        ("kv", [attn], [5]), ("mamba", [mamba], [6])]
