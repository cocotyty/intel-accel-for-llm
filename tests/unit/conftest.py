# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the KVShrink hybrid unit tests.

These tests are pure logic: no GPU, no disk, no model, no machine
specifics. Storage and transfer engines are always faked, so the suite
runs anywhere vLLM and PyTorch import.
"""

from __future__ import annotations

import os
import sys

import pytest

# Import the package from the repository checkout without installing it.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Knobs that change connector behaviour. Cleared for every test so a
# developer's shell environment can never alter the results.
_KVSHRINK_ENV = (
    "KVSHRINK_SAVE",
    "KVSHRINK_SAVE_PIPELINED",
    "KVSHRINK_DEBUG_AUTOSAVE",
    "KVSHRINK_DEBUG_LOG",
    "KVSHRINK_DEBUG_DUMP",
    "KVSHRINK_PERSIST_DIR",
)


@pytest.fixture(autouse=True)
def _clean_kvshrink_env(monkeypatch):
    for name in _KVSHRINK_ENV:
        monkeypatch.delenv(name, raising=False)
    # The exporter binds a port; unit tests never need it.


class FakeBlocks:
    """KVCacheBlocks stand-in: production only calls get_block_ids()."""

    def __init__(self, ids_per_group):
        self._ids = tuple(tuple(ids) for ids in ids_per_group)

    def get_block_ids(self):
        return self._ids


def HybridRequestScheduler(groups, store, hash_block_size, namespace,
                           tp_size, rank, async_load_config=None,
                           block_hash_source="vllm"):
    """Scheduler-side connector instance without the vLLM config stack.

    The scheduler-side methods live on KVShrinkConnector; this factory
    builds one with just the fields they touch. Same signature the
    pre-merge HybridRequestScheduler class had.
    """
    from kvshrink.kvshrink_connector import KVShrinkConnector

    conn = object.__new__(KVShrinkConnector)
    conn._groups = list(groups)
    conn.kvstore = store
    conn._hash_block_size = hash_block_size
    conn._namespace = namespace
    conn.tp_size = tp_size
    conn._rank = rank
    conn._async_load_layer_config = async_load_config
    conn._block_hash_source = block_hash_source
    conn._attention_layers = tuple(
        ln for g in groups if g.kind != "mamba" for ln in g.layer_names)
    conn._req_states = {}
    conn._async_load_pending = set()
    return conn


def HybridWorker(groups, layer_infos, namespace, canonicalizer,
                 rank, tp_size):
    """Worker-side connector instance without the vLLM config stack.

    Same signature the pre-merge HybridWorker class had.
    """
    from kvshrink.kvshrink_connector import KVShrinkConnector, group_label

    conn = object.__new__(KVShrinkConnector)
    conn._groups = list(groups)
    conn._layer_infos = layer_infos
    conn._canon = canonicalizer
    conn._rank = rank
    conn.tp_size = tp_size
    conn._labels = [
        group_label(namespace, g.group_idx, rank) for g in groups]
    conn.kvstore = None
    conn._layer_names = []
    conn._load_tasks = {}
    conn._async_loads = {}
    conn._layer_group = {
        ln: g.group_idx for g in groups for ln in g.layer_names}
    conn._attn_layer_group = {
        ln: g.group_idx for g in groups if g.kind != "mamba"
        for ln in g.layer_names}
    conn._mamba_layers = frozenset()
    conn._attn_order = ()
    conn._mamba_save_segments = {}
    conn._saved_layers = set()
    conn._step_save_pages = 0
    conn._current_put_tasks = {}
    conn._deferred_finished_req_ids = set()
    conn._connector_metadata = None
    return conn


def make_spec(kind: str, block_size: int):
    """A real vLLM KVCacheSpec for one group.

    The hit policy hands the spec back to vLLM's own matching code, so a
    stand-in would not exercise the path the engine takes. These are the
    genuine spec classes with the smallest shape that is still valid.
    """
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    if kind == "mamba":
        return MambaSpec(
            block_size=block_size,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_cache_mode="align",
        )
    return FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.bfloat16,
    )
