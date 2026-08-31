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


def HybridRequestScheduler(groups, store, hash_block_size,
                           async_load_config=None):
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
    conn._async_load_layer_config = async_load_config
    conn._num_attn_layers = sum(
        len(g.layer_names) for g in groups if g.kind != "mamba")
    conn._req_states = {}
    conn._async_load_pending = set()
    return conn


def track_new_request(sched, req_id, block_hashes, num_computed_tokens=0):
    """Register a fresh ReqState (what get_num_new_matched_tokens does)."""
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    sched._req_states[req_id] = ReqState(
        live_source=list(block_hashes),
        block_hashes=list(block_hashes),
        num_computed_tokens=num_computed_tokens,
        groups=tuple(ReqGroupState() for _ in sched._groups),
    )


def HybridWorker(groups, layer_infos, rank=0, tp_size=1):
    """Worker-side connector instance without the vLLM config stack.

    ``layer_infos`` mirrors what the real register() receives as
    kv_caches; tests pass {name: None} placeholders (only the key set
    matters to the part mapping) plus the group descriptors.
    """
    from kvshrink.kvshrink_connector import KVShrinkConnector

    conn = object.__new__(KVShrinkConnector)
    conn._groups = list(groups)
    conn.rank = rank
    conn.tp_size = tp_size
    conn._labels = [f"g{g.group_idx}" for g in groups]
    conn.kvstore = None
    conn._layer_names = []
    conn._current_get_tasks = None
    conn._pending_load_tasks = {}
    conn._pending_load_layers = {}
    conn._early_promoted_tasks = {}
    conn._active_promoted_tasks = {}
    conn._layer_group = {
        ln: g.group_idx for g in groups for ln in g.layer_names}
    conn._mamba_layers = frozenset()
    conn._attn_order = ()
    conn._last_layer_name = None
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


def drive_start_load(w, metadata):
    """Submit one step's loads through the real entry point."""
    from types import SimpleNamespace
    w.bind_connector_metadata(metadata)
    w.start_load_kv(SimpleNamespace(attn_metadata=True))
