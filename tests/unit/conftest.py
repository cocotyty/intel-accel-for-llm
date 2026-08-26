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
