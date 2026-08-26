# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Block-identity source selection.

The two store layouts were written with different block-hash schemes.
Switching a deployment from one to the other does not migrate anything
-- it renames every key, so previously written entries become
unreachable until they are written again. That is a cold cache, not a
corrupt one, but it is not something an upgrade should do silently.

So the default preserves what each layout already wrote, and the
override exists for operators who accept one warm-up in exchange for a
single scheme.

Pure logic: no GPU, no disk, no model.
"""

from __future__ import annotations

import pytest

from kvshrink.kvshrink_connector import KVShrinkConnector
from conftest import make_spec
from kvshrink.kvshrink_connector import GroupInfo
from kvshrink.scheduler import HybridRequestScheduler

_pick = KVShrinkConnector._block_hash_source


def _group(kind):
    return GroupInfo(
        group_idx=0, kind=kind, layer_names=("l0",), block_size=16,
        mamba_align_size=16 if kind == "mamba" else None,
        spec=make_spec(kind, 16))


class _Req:
    """Carries both identities so a test can tell which one was used."""

    block_hashes = ["vllm-a", "vllm-b", "vllm-c"]
    all_token_ids = list(range(48 + 1))   # 3 full blocks at bs=16, +1


# ------------------------------------------------------------------
# defaults preserve existing caches
# ------------------------------------------------------------------
def test_default_keeps_each_layout_on_its_own_scheme(monkeypatch):
    monkeypatch.delenv("KVSHRINK_BLOCK_HASH_SOURCE", raising=False)
    assert _pick(recurrent=False) == "legacy", (
        "the block layout would stop finding caches written before the "
        "upgrade")
    assert _pick(recurrent=True) == "vllm", (
        "the boundary layout shipped on vLLM hashes")


@pytest.mark.parametrize("value,expected", [
    ("vllm", "vllm"), ("legacy", "legacy"),
    ("VLLM", "vllm"), ("auto", None),
])
def test_override_is_honoured(monkeypatch, value, expected):
    monkeypatch.setenv("KVSHRINK_BLOCK_HASH_SOURCE", value)
    if expected is None:
        assert _pick(recurrent=False) == "legacy"
        assert _pick(recurrent=True) == "vllm"
    else:
        assert _pick(recurrent=False) == expected
        assert _pick(recurrent=True) == expected


def test_unknown_value_is_refused(monkeypatch):
    """Fail loudly: a typo that silently fell back would quietly rename
    every key in the store."""
    monkeypatch.setenv("KVSHRINK_BLOCK_HASH_SOURCE", "vlm")
    with pytest.raises(ValueError, match="KVSHRINK_BLOCK_HASH_SOURCE"):
        _pick(recurrent=False)


# ------------------------------------------------------------------
# the sources really are different, and each is used as declared
# ------------------------------------------------------------------
def _sched(source):
    return HybridRequestScheduler(
        [_group("attention")], store=None, hash_block_size=16,
        namespace="ns", tp_size=1, rank=0, block_hash_source=source)


def test_vllm_source_uses_the_engine_hashes():
    assert _sched("vllm")._request_block_hashes(_Req()) == _Req.block_hashes


def test_legacy_source_recomputes_from_tokens():
    got = _sched("legacy")._request_block_hashes(_Req())
    assert got and got != _Req.block_hashes, (
        "legacy must not silently return the engine's hashes")
    assert all(isinstance(h, str) for h in got)


def test_scheduler_refuses_an_unknown_source():
    with pytest.raises(ValueError, match="unknown block hash source"):
        _sched("nonsense")
