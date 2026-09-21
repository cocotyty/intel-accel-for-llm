# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Contracts shared by hybrid pages and upstream's local/remote store paths."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from iaxl.kvstore import kvstore as module
from kvshrink.kv_cache_pages import PageLayout, bind_pages


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


def test_bind_mamba_is_one_opaque_page_per_block():
    backing = torch.arange(3 * 64, dtype=torch.uint8).view(3, 64)
    conv = backing[:, :16]
    pages = bind_pages({"m0": [conv]}, PageLayout(3, 64, {"m0": "mamba"}))["m0"]
    assert pages.data_ptr() == backing.data_ptr()
    assert torch.equal(pages, backing)


def test_bind_attention_reviews_along_the_logical_block():
    """A logical page spans `ratio` kernel blocks: binding re-views dim 0 so
    scheduler block IDs index it, keeping the trailing dims untouched."""
    num_blocks, ratio, kernel_tokens = 3, 4, 2
    attn = torch.arange(
        num_blocks * ratio * 2 * kernel_tokens * 2, dtype=torch.float32).reshape(
            num_blocks * ratio, 2, kernel_tokens, 2)
    page_elements = ratio * 2 * kernel_tokens * 2
    page_bytes = page_elements * 4  # float32
    mamba = [torch.zeros(num_blocks, page_elements)]  # same page bytes
    bound = bind_pages(
        {"a0": attn, "m0": mamba},
        PageLayout(num_blocks, page_bytes, {"a0": "attention", "m0": "mamba"}))
    pages = bound["a0"]
    assert pages.shape == (num_blocks, ratio, 2, kernel_tokens, 2)
    assert pages[1].data_ptr() == attn[ratio].data_ptr()
    assert pages.untyped_storage().data_ptr() == attn.untyped_storage().data_ptr()
    assert bound["m0"].shape == (num_blocks, page_elements * 4)


def test_store_label_namespaces(store_factory):
    """The store carries the kv/mamba namespace; pages are already bound."""
    backing = torch.zeros(3, 64, dtype=torch.uint8)
    store = store_factory("test", block_dim=0, kv_caches={"m0": backing})
    store.put([1], ["hash"], label="mamba")
    store.tensorzip.put_finish.assert_called_once_with("mamba", ["hash"])
    assert store.tensorzip.put.call_args.kwargs["label"] == "mamba"
    assert store.tensorzip.put.call_args.kwargs["chunk_dim"] == 0
    store.get([1], ["hash"], label="mamba")
    assert store.tensorzip.get.call_args.kwargs["label"] == "mamba"


def test_controller_does_not_require_block_dim(store_factory):
    store = store_factory("test", layer_names=["a0"])
    assert store.has_only_mode
    assert store.kvcache_shape is None
    with pytest.raises(ValueError, match="block_dim is required"):
        store_factory("test", kv_caches={"a0": torch.zeros(2, 3)})


def test_both_shells_share_the_block_dim_kwarg():
    """The dispatch picks one shell at import; both must accept the connector call."""
    import inspect
    from iaxl.remote_pool.kvstore_remote import KVStoreRemote

    for cls in (module.KVStoreLocal, KVStoreRemote):
        params = inspect.signature(cls.__init__).parameters
        assert "block_dim" in params and "kv_caches" in params, cls


def test_both_shells_share_data_path_signatures():
    """The connector calls put/get/has with namespace kwargs on either shell."""
    import inspect
    from iaxl.remote_pool.kvstore_remote import KVStoreRemote

    for name in ("put", "get", "has"):
        local = list(inspect.signature(getattr(module.KVStoreLocal, name)).parameters)
        remote = list(inspect.signature(getattr(KVStoreRemote, name)).parameters)
        assert local == remote, name


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
    remote._xfer = lambda *args, **kwargs: calls.append(
        (args[-1], kwargs.get("label"))) or {"a0": None}
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
    assert calls == [("get", None), ("put", None)]


@pytest.mark.parametrize("device,sync,expected_calls", [
    ("cuda", False, 0), ("cuda", True, 1), (None, False, 0),
])
def test_restore_syncs_gpu_only_when_enabled(monkeypatch, device, sync, expected_calls):
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
    assert ctx.xfer_wait_cur_stream.call_count == expected_calls
