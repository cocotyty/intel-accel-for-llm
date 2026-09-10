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


class FakeBlock:
    """KVCacheBlock stand-in. vLLM's null block is the pool's first
    block (block_id 0) with is_null set, which is the convention these
    tests already used for an empty slot."""

    __slots__ = ("block_id",)

    def __init__(self, block_id):
        self.block_id = block_id

    @property
    def is_null(self):
        return self.block_id == 0


class FakeBlocks:
    """KVCacheBlocks stand-in: production reads get_block_ids() for the
    per-group tables and .blocks for the load plan's destinations."""

    def __init__(self, ids_per_group):
        self._ids = tuple(tuple(ids) for ids in ids_per_group)
        self.blocks = tuple(
            tuple(FakeBlock(i) for i in ids) for ids in self._ids)

    def get_block_ids(self):
        return self._ids


def HybridRequestScheduler(groups, store, block_size,
                           async_load_config=None):
    """Scheduler-side connector instance without the vLLM config stack.

    The scheduler-side methods live on KVShrinkConnector; this factory
    builds one with just the fields they touch. Same signature the
    pre-merge HybridRequestScheduler class had.
    """
    from kvshrink.kvshrink_connector import KVShrinkConnector, RequestMetadata

    conn = object.__new__(KVShrinkConnector)
    conn._groups = list(groups)
    conn.kvstore = store
    conn._block_size = block_size
    conn._async_load_layer_config = async_load_config
    conn._num_attn_layers = sum(
        len(g.layer_names) for g in groups if g.kind != "mamba")
    conn._req_states = {}
    conn._reqs_to_load = RequestMetadata()
    return conn


def track_new_request(sched, req_id, block_hashes, num_computed_tokens=0, num_prompt_tokens=0):
    """Register a fresh ReqState (what get_num_new_matched_tokens does)."""
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    if num_prompt_tokens == 0 and block_hashes:
        num_prompt_tokens = len(block_hashes) * sched._block_size
    sched._req_states[req_id] = ReqState(
        live_block_hashes=list(block_hashes),
        num_computed_tokens=num_computed_tokens,
        num_prompt_tokens=num_prompt_tokens,
        groups=tuple(ReqGroupState() for _ in sched._groups),
    )


def HybridWorker(groups, layer_infos, rank=0, tp_size=1):
    """Worker-side connector instance without the vLLM config stack."""
    from kvshrink.kvshrink_connector import KVShrinkConnector

    conn = object.__new__(KVShrinkConnector)
    conn._groups = list(groups)
    conn.rank = rank
    conn.tp_size = tp_size
    conn._labels = [f"g{g.group_idx}" for g in groups]
    conn.kvstore = None
    order = list(layer_infos.keys() if isinstance(layer_infos, dict) else layer_infos)
    conn._layer_names = order
    conn._current_get_tasks = None
    conn._pending_load_tasks = {}
    conn._pending_load_layers = {}
    conn._early_promoted_tasks = {}
    conn._active_promoted_tasks = {}
    conn._layer_group = {
        ln: g.group_idx for g in groups for ln in g.layer_names}
    conn._mamba_layers = frozenset(
        ln for g in groups if g.kind == "mamba" for ln in g.layer_names)
    conn._attn_order = tuple(
        ln for ln in order if ln not in conn._mamba_layers)
    segments = {}
    pending = []
    for ln in order:
        if ln in conn._mamba_layers:
            pending.append(ln)
        elif pending:
            segments[ln] = tuple(pending)
            pending = []
    conn._mamba_save_segments = segments
    conn._last_layer_name = conn._attn_order[-1] if conn._attn_order else None
    conn._saved_layers = set()
    conn._current_put_tasks = {}
    conn._deferred_finished_req_ids = set()
    conn._connector_metadata = None
    return conn


def make_spec(kind: str, block_size: int, num_speculative_blocks: int = 0):
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
            num_speculative_blocks=num_speculative_blocks,
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
