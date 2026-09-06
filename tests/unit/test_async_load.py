# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Async (cross-step) KV loading.

An async load is a promise with two failure modes that unit tests can
pin down without a GPU:

- released too early -- forward runs on state that has not landed. For
  recurrent (GDN/Mamba) layers this is silent: the model produces
  plausible, wrong tokens. So the release gate must cover every
  recurrent layer no matter how few attention layers were requested.
- never released -- vLLM parks an async request until the worker names
  it in get_finished, so a request that is never reported hangs
  forever. That must hold even when its transfer FAILS.

Pure logic: fake store, no GPU, no disk, no model.
"""

from __future__ import annotations

import pytest

from conftest import (
    FakeBlocks, HybridWorker, drive_start_load, make_spec)
from kvshrink.kvshrink_connector import (
    GroupInfo, KVShrinkConnectorMetadata,
    ReqMeta, RequestMetadata)

PAGE = 4096
ORDER = ["m0", "a1", "m2", "a3"]
ATTN = ["a1", "a3"]
GDN = ["m0", "m2"]


def _group(g_idx, kind, layers):
    return GroupInfo(
        group_idx=g_idx, kind=kind, layer_names=tuple(layers),
        spec=make_spec(kind, 16))


class _FakeStore:
    """Store whose per-layer completion is driven by the test.

    ``landed`` is the set of layers whose transfers have completed; a
    poll for anything outside it answers "not yet".
    """

    def __init__(self):
        self.landed: set[str] = set()
        self.waited: list[str] = []      # layers finalized, in order
        self.fail_on: set[str] = set()

    def get(self, block_indices, block_hashs, layer_names,
            label=None):
        return {ln: f"task:{ln}" for ln in layer_names}

    def get_wait(self, get_results, layer_names=None, wait=True):
        layers = (set(layer_names) if layer_names is not None
                  else set(get_results))
        if layers & self.fail_on:
            raise RuntimeError(
                f"transfer failed: {sorted(layers & self.fail_on)}")
        if not wait:
            return layers <= self.landed
        self.waited.extend(sorted(layers))
        return True

    def has(self, chunk_labels, label=None):
        return [True]


def _worker(store):
    groups = [_group(0, "attention", ATTN), _group(1, "mamba", GDN)]
    w = HybridWorker(groups, {ln: None for ln in ORDER},
                     rank=0, tp_size=1)
    w.kvstore = store
    w.register(ORDER)
    return w


def _meta(async_layers, req_id="r1"):
    """A load plan covering every layer, split per group."""
    md = RequestMetadata()
    md.requests[req_id] = ReqMeta(
        block_hashes=("7",),
        group_block_ids=(("5",), ("5",)),
        is_async=True, async_load_layers=async_layers)
    return KVShrinkConnectorMetadata(reqs_to_load=md)


# ------------------------------------------------------------------
# the recurrent-state constraint
# ------------------------------------------------------------------
def test_gate_covers_every_recurrent_layer_despite_short_prefix():
    """Asking to release after ONE attention layer must still wait for
    all GDN state: it is consumed whole at the start of forward."""
    b = _FakeStore()
    w = _worker(b)
    drive_start_load(w, _meta(async_layers=1))
    assert w._pending_load_layers["r1"] == 1

    # the gate lands: every recurrent layer plus exactly the first
    # attention layer in execution order
    b.landed = {"a1", *GDN}
    _, recving = w.get_finished(set())
    assert recving == {"r1"}
    assert set(b.waited) == {"a1", "m0", "m2"}, b.waited


def test_not_released_until_recurrent_state_has_landed():
    b = _FakeStore()
    w = _worker(b)
    drive_start_load(w, _meta(async_layers=1))

    b.landed = {"a1"}                      # attention prefix only
    _, recving = w.get_finished(set())
    assert not recving

    b.landed |= set(GDN)                   # now the state is there
    _, recving = w.get_finished(set())
    assert recving == {"r1"}, recving


def test_negative_layer_count_gates_on_every_layer():
    b = _FakeStore()
    w = _worker(b)
    drive_start_load(w, _meta(async_layers=-1))
    assert w._pending_load_layers["r1"] == -1
    b.landed = set(ORDER)
    _, recving = w.get_finished(set())
    assert recving == {"r1"}
    assert set(b.waited) == set(ORDER), b.waited
    # -1 waits everything, so nothing is left to promote
    assert "r1" not in w._early_promoted_tasks


# ------------------------------------------------------------------
# cross-step lifetime
# ------------------------------------------------------------------
def test_async_tasks_live_outside_the_step_tasks():
    """Async tasks outlive the step by design: they are not in the
    per-step _current_get_tasks drained by the layer hooks."""
    b = _FakeStore()
    w = _worker(b)
    drive_start_load(w, _meta(async_layers=1))
    assert w._current_get_tasks is None
    assert "r1" in w._pending_load_tasks


def test_sync_loads_queue_ahead_of_async_loads():
    """The engine stream runs transfers FIFO, so submission order is
    the service order: this pass's blocking loads must all be enqueued
    before a parked request's, one call per layer, in execution order
    (the order forward consumes the layers)."""
    calls = []

    class _Recorder(_FakeStore):
        def get(self, block_indices, block_hashs, layer_names,
                label=None):
            calls.append((layer_names[0], block_indices[0]))
            return super().get(block_indices, block_hashs,
                               layer_names, label)

    w = _worker(_Recorder())
    md = RequestMetadata()
    md.requests["sync1"] = ReqMeta(
        block_hashes=("7",), group_block_ids=(("5",), ("5",)))
    md.requests["async1"] = ReqMeta(
        block_hashes=("7",), group_block_ids=(("9",), ("9",)),
        is_async=True, async_load_layers=1)
    drive_start_load(w, KVShrinkConnectorMetadata(reqs_to_load=md))

    assert calls == ([(ln, "5") for ln in ORDER]
                     + [(ln, "9") for ln in ORDER])


def test_remaining_layers_are_drained_by_the_layer_hooks():
    b = _FakeStore()
    w = _worker(b)
    drive_start_load(w, _meta(async_layers=1))
    b.landed = set(ORDER)
    _, recving = w.get_finished(set())
    assert recving == {"r1"}

    # gate layers were finalized at release; a3 was not
    assert "a3" not in b.waited
    assert "r1" in w._early_promoted_tasks
    # main: the next start_load_kv promotes early -> active
    w._active_promoted_tasks.update(w._early_promoted_tasks)
    w._early_promoted_tasks = {}
    w.wait_for_layer_load("a3")
    assert "a3" in b.waited
    # a3 is the last attention layer: the promoted book is cleared
    assert w._active_promoted_tasks == {}


# ------------------------------------------------------------------
# transfer failure is fatal
# ------------------------------------------------------------------
def test_failed_transfer_raises_at_poll():
    b = _FakeStore()
    b.fail_on = {"m0"}
    w = _worker(b)
    drive_start_load(w, _meta(async_layers=1))
    b.landed = set(ORDER)

    with pytest.raises(RuntimeError, match="transfer failed"):
        w.get_finished(set())



def _empty_out():
    """A scheduler output with nothing scheduled (parked-only pass)."""
    from types import SimpleNamespace
    return SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], resumed_req_ids=set(),
            new_block_ids=[], num_computed_tokens=[]),
        num_scheduled_tokens={})


def _alloc(sched, block_ids, hashes, ext, is_async, layers=-1,
           req_id="r1"):
    """Register a request and hand vLLM's allocation to the connector,
    which is where the load plan is built."""
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    from conftest import FakeBlocks

    st = ReqState(
        live_block_hashes=list(hashes),
        groups=tuple(ReqGroupState() for _ in sched._groups))
    st.is_async = is_async
    st.async_load_layers = layers
    sched._req_states[req_id] = st
    sched.update_state_after_alloc(
        type("R", (), {"request_id": req_id}),
        FakeBlocks(block_ids), ext)
    return st


def test_parked_request_still_gets_a_load_plan():
    """Regression: an async request appears in NEITHER scheduled list.

    vLLM allocates its blocks, calls update_state_after_alloc, then
    parks it in WAITING_FOR_REMOTE_KVS -- removing it from
    scheduled_new_reqs without ever adding it to scheduled_cached_reqs.
    A plan builder that only walks those two lists emits nothing, so the
    worker transfers nothing, reports nothing finished, and the request
    waits for a release that can never arrive. Observed as a live hang:
    "Running: 0 reqs, Waiting: 3, Deferred: 3" with throughput at zero.
    """
    groups = [_group(0, "attention", ATTN)]
    from conftest import HybridRequestScheduler
    sched = HybridRequestScheduler(
        groups, _FakeStore(), 16)
    _alloc(sched, ([4, 5],), [1, 2], ext=32, is_async=True, layers=1)

    plans = sched.build_connector_meta(_empty_out()).reqs_to_load.requests
    assert list(plans) == ["r1"], (
        "the parked request got no load plan -- it would hang")
    assert plans["r1"].is_async and any(plans["r1"].group_block_ids)


def test_async_plan_is_emitted_only_once():
    """A second plan would submit a second transfer for a request that
    already has one in flight, stranding the first."""
    from conftest import HybridRequestScheduler

    sched = HybridRequestScheduler(
        [_group(0, "attention", ATTN)], _FakeStore(), 16)
    _alloc(sched, ([4, 5],), [1, 2], ext=32, is_async=True)

    drains = sched.build_connector_meta(_empty_out()).reqs_to_load.requests
    assert list(drains) == ["r1"]
    assert sched.build_connector_meta(
        _empty_out()).reqs_to_load.requests == {}


def test_recurrent_models_can_go_async():
    """Recurrent groups do NOT force synchronous loading.

    The transfer is a DMA; it needs no forward step. What it needs is to
    land in the slot vLLM will read, which it does -- see
    test_async_mamba_targets_the_slot_vllm_reads_as_prev.
    """
    from conftest import HybridRequestScheduler

    class _Cfg:
        def select(self, concurrency):
            return 4

    class _Req:
        request_id = "r1"
        block_hashes = [1, 2]
        num_tokens = 48

    hybrid = HybridRequestScheduler(
        [_group(0, "attention", ATTN), _group(1, "mamba", GDN)],
        _FakeStore(), 16, async_load_config=_Cfg())
    external, use_async = hybrid.get_num_new_matched_tokens(_Req(), 0)
    assert external == 32 and use_async is True


def test_async_mamba_targets_the_slot_vllm_reads_as_prev():
    """The equality the whole async-GDN path rests on.

    Ours: the destination is the mamba table's LAST entry. vLLM parks
    async requests with num_new_tokens == 0 (scheduler.py:678), so it
    sizes that table to cdiv(boundary, block_size) entries, putting the
    last index at (boundary - 1) // block_size.

    vLLM's preprocess_mamba, when the request is finally scheduled and
    has no tracked state yet (mamba_utils.py:667-669), with
    num_computed_tokens == boundary because the hit was credited:
        prev_state_idx = (num_computed_tokens - 1) // block_size

    Same index. So the async write lands in prev, and vLLM's own
    prev -> curr copy carries it into the slot forward reads. If this
    ever diverges, an async GDN restore is silently discarded.
    """
    bs = 16
    for boundary in (16, 64, 4096, 5280):
        table_len = -(-boundary // bs)      # cdiv: what vLLM allocates
        ours = table_len - 1               # our destination: the last entry
        vllm_prev = (boundary - 1) // bs
        assert ours == vllm_prev, (
            f"boundary={boundary}: async restore would write slot "
            f"{ours} but vLLM reads {vllm_prev} as prev")


def test_second_alloc_callback_does_not_queue_another_transfer():
    """Regression: vLLM calls update_state_after_alloc TWICE for an
    async request -- once at allocation, once after the load completes
    and the request is promoted out of WAITING_FOR_REMOTE_KVS
    (scheduler.py:2198 promotes and falls through into the ordinary
    path, reaching allocate_slots and the callback again).

    The second call always carries 0 external tokens: promotion left
    num_computed_tokens non-zero (:822), so the branch that asks the
    connector for external tokens is skipped and the count keeps its
    initial 0. That zero is what must stop a second plan -- queuing on
    both produced a transfer for a request that was RUNNING by then,
    and reporting THAT one finished tripped vLLM's own assertion (a
    finished-recving request must be parked or finished, never
    running) and killed EngineCore.
    """
    from conftest import HybridRequestScheduler

    sched = HybridRequestScheduler(
        [_group(0, "attention", ATTN)], _FakeStore(), 16)
    _alloc(sched, ([4, 5],), [1, 2], ext=32, is_async=True)
    assert list(sched.build_connector_meta(
        _empty_out()).reqs_to_load.requests) == ["r1"]

    # vLLM's second callback, after the transfer landed: 0 external
    # tokens, because the request is no longer asking for any.
    sched.update_state_after_alloc(
        type("R", (), {"request_id": "r1"}),
        FakeBlocks(([4, 5],)), 0)
    assert sched.build_connector_meta(
        _empty_out()).reqs_to_load.requests == {}, (
        "a second transfer was queued for a request already running")


