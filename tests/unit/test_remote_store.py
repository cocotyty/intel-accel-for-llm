# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The RDMA client shell and the daemon must carry the hybrid namespace and
prefix semantics end to end (see `iaxl/remote_pool/rpc.py`)."""

from types import SimpleNamespace

from iaxl.remote_pool import rpc
from iaxl.remote_pool.kvstore_remote import KVStoreRemote


def test_blocks_codec_roundtrip_with_label():
    payload = rpc.pack_blocks([1, 2], ["h1", "h2"], [0, 3], "desc", "mamba")
    assert rpc.unpack_blocks(payload) == ([1, 2], ["h1", "h2"], [0, 3], "desc", "mamba")


def test_has_codec_roundtrip_with_label_and_truncate():
    assert rpc.unpack_has(rpc.pack_has(["h1"], "mamba", False)) == (["h1"], "mamba", False)
    assert rpc.unpack_has(rpc.pack_has([], "", True)) == ([], "", True)


def test_client_has_forwards_namespace_and_truncate():
    sent = {}
    remote = object.__new__(KVStoreRemote)
    remote.rpc = SimpleNamespace(
        call=lambda method, payload: sent.update(method=method, payload=payload) or [1, 0])
    assert remote.has(["h1", "h2"], label="mamba", truncate=False) == [True, False]
    assert sent["method"] == rpc.HAS
    assert rpc.unpack_has(sent["payload"]) == (["h1", "h2"], "mamba", False)


def test_client_put_and_get_forward_label():
    for name in ("put", "get"):
        sent = {}
        remote = object.__new__(KVStoreRemote)
        remote.has_only_mode = False
        remote._sync = lambda: None
        remote.layer_names = ["a0", "m0"]
        remote.layer_idx = {"a0": 0, "m0": 1}
        remote._pending = {}
        remote.rpc = SimpleNamespace(
            call=lambda method, payload: sent.update(method=method, payload=payload)
            or bytes([0, 0, 0, 0]))
        tasks = getattr(remote, name)([1], ["h"], layer_names=["m0"], label="mamba")
        assert set(tasks) == {"m0"}
        assert rpc.unpack_blocks(sent["payload"]) == ([1], ["h"], [1], "", "mamba")


def test_daemon_has_applies_namespace_and_truncate():
    service = object.__new__(rpc.KVStoreService)
    calls = []
    service.kvstore = SimpleNamespace(
        has=lambda hashes, label=None, truncate=True:
        calls.append((hashes, label, truncate)) or [1, 0])
    flags = service._has("peer", rpc.pack_has(["h1", "h2"], "mamba", False))
    assert calls == [(["h1", "h2"], "mamba", False)]
    assert list(flags) == [1, 0]


def test_daemon_transfer_applies_label():
    service = object.__new__(rpc.KVStoreService)
    service.layer_names = ["a0", "m0"]
    service.layer_idx = {"a0": 0, "m0": 1}
    service.jobs = {}
    service._next_job = 0
    got = {}
    service.kvstore = SimpleNamespace(
        get=lambda indices, hashes, names, desc, label=None:
        got.update(indices=indices, names=names, desc=desc, label=label) or {"m0": None})
    job_id = int.from_bytes(
        service._xfer("peer", rpc.pack_blocks([3], ["h"], [1], "req", "mamba"), False), "little")
    assert got == {"indices": [3], "names": ["m0"], "desc": "req", "label": "mamba"}
    assert job_id in service.jobs
