# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by hybrid pages and upstream's local/remote store paths."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from iaxl.kvflow.scratch_pool import ScratchPool
from iaxl.kvstore import kvstore as module


@pytest.fixture
def store_factory(monkeypatch):
    monkeypatch.setattr(module, "KVFlow", Mock)
    monkeypatch.setattr(module, "start_profiling", lambda: None)
    monkeypatch.setattr(module, "mgmt_register", lambda *args: None)
    monkeypatch.setattr(module, "start_mgmt_server", lambda **kwargs: None)
    monkeypatch.setattr(module, "envs", SimpleNamespace(
        IAXL_DDR_POOL_SIZE_GB=0.01, IAXL_KVSTORE_SKIP_COMPRESSION_LAYERS=1))
    return module.KVStoreLocal


def test_dim1_benchmark_and_global_compression_skip(store_factory):
    tensors = {name: torch.zeros(2, 3, 4) for name in ("a0", "a1")}
    store = store_factory("test", block_dim=1, kv_caches=tensors)
    store.put([2], ["hash"], layer_names=["a1"])
    kwargs = store.tensorzip.put.call_args.kwargs
    assert kwargs["chunk_dim"] == 1
    assert kwargs["skip_compression_count"] == 0
    assert kwargs["tensors"]["a1"] is tensors["a1"]
    store.get([2], ["hash"])
    assert store.tensorzip.get.call_args.kwargs["chunk_dim"] == 1
    assert store.kvcache_shape == [2, 3, 4]


def test_hybrid_bound_pages_and_namespaces(store_factory):
    backing = torch.arange(3 * 64, dtype=torch.uint8).view(3, 64)
    conv = backing[:, :16]
    store = store_factory("test", block_dim=0, kv_caches={"m0": [conv]})
    pages = store.kv_caches["m0"]
    assert pages.data_ptr() == backing.data_ptr()
    assert torch.equal(pages, backing)
    store.put([1], ["hash"], label="mamba")
    store.tensorzip.put_finish.assert_called_once_with("mamba", ["hash"])
    assert store.tensorzip.put.call_args.kwargs["chunk_dim"] == 0
    store.get([1], ["hash"], label="mamba")
    assert store.tensorzip.get.call_args.kwargs["label"] == "mamba"


def test_controller_does_not_require_block_dim(store_factory):
    store = store_factory("test", layer_names=["a0"])
    assert store.has_only_mode
    assert store.kvcache_shape is None
    with pytest.raises(ValueError, match="block_dim is required"):
        store_factory("test", kv_caches={"a0": torch.zeros(2, 3)})


def test_attention_connector_calls_remote_store_without_label(monkeypatch):
    from conftest import HybridWorker, HybridRequestScheduler, make_spec, drive_start_load
    from iaxl.remote_pool.kvstore_remote import KVStoreRemote
    from kvshrink.kvshrink_connector import (
        GroupInfo, RequestMetadata, KVShrinkConnectorMetadata, ReqMeta)

    # Use the real remote API signatures; replace only the network/transfer.
    remote = object.__new__(KVStoreRemote)
    remote.has_only_mode = False
    remote._sync = lambda: None
    remote.rpc = SimpleNamespace(call=lambda *args: bytes([1]))
    calls = []
    remote._xfer = lambda *args: calls.append(args) or {"a0": None}
    groups = [GroupInfo(0, "attention", ("a0",), make_spec("attention", 16))]
    scheduler = HybridRequestScheduler(groups, remote, 16)
    scheduler.get_num_new_matched_tokens(SimpleNamespace(
        request_id="r", block_hashes=[b"h"], num_tokens=17), 0)
    worker = HybridWorker(groups, ["a0"])
    worker.kvstore = remote
    requests = RequestMetadata()
    requests.requests["r"] = ReqMeta(
        group_block_ids=((1,),), block_hashes=["h"], is_async=True,
        async_load_layers=-1)
    drive_start_load(worker, KVShrinkConnectorMetadata(requests, requests))
    worker.save_kv_layer("a0", None, None)
    assert [call[-1] for call in calls] == ["get", "put"]


def test_scratch_mixed_dtype_pages_keep_bytes_slots_and_lifetime(monkeypatch):
    import iaxl.kvflow.scratch_pool as scratch

    monkeypatch.setattr(scratch, "envs", SimpleNamespace(
        IAXL_SCRATCH_POOL_SIZE_GB=128 / 1024**3))
    pool = ScratchPool((2, 4), torch.float32, pin_memory=False)
    attn = pool.allocate(1)[0]
    attn.fill_(7)
    mamba = pool.allocate(1, (32,), torch.uint8)[0]
    mamba.fill_(19)
    assert mamba.shape == (32,) and mamba.dtype == torch.uint8
    assert mamba.data_ptr() == pool.pool.data_ptr() + mamba.block_idx * 32
    assert mamba.block_idx != attn.block_idx
    assert torch.all(attn == 7)
    before = pool.available_count()
    with pytest.raises(ValueError, match="block size"):
        pool.allocate(1, (33,), torch.uint8)
    assert pool.available_count() == before
    pool.release([mamba])
    reused = pool.allocate(1)[0]
    assert reused.data_ptr() == mamba.data_ptr()
    assert torch.all(reused.view(torch.uint8) == 19)
    pool.release([attn, reused])
    assert pool.available_count() == 4


@pytest.mark.parametrize("device,sync,expected", [
    ("cuda", False, [False]), ("cuda", True, [True]), (None, False, []),
])
def test_restore_orders_gpu_but_skips_cpu_daemon(monkeypatch, device, sync, expected):
    import iaxl.kvflow.flow as flow

    tensor = SimpleNamespace(
        dim=lambda: 2, shape=(3, 8), dtype=torch.float32, device="cuda",
        is_cuda=True, is_contiguous=lambda: True)
    ctx = Mock()
    obj = object.__new__(flow.KVFlow)
    obj.device_type = device
    obj.get_stream = None
    obj.mem = None
    obj._ensure_streams = lambda: None
    obj._ensure_pool = lambda *args: None
    obj._create_ctx = lambda *args, **kwargs: ctx
    obj.chunk_pool = SimpleNamespace(allocate=lambda *args: [torch.zeros(8)])
    monkeypatch.setattr(flow, "stream_sync_on_get", sync)
    obj.get("kv", {"a0": tensor}, 0, [1], ["h"])
    assert [c.kwargs["sync_cur_stream"] for c in ctx.xfer_wait_cur_stream.call_args_list] == expected
