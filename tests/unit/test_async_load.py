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

from conftest import HybridWorker, make_spec
from kvshrink.kvshrink_connector import (
    CacheKey, GroupInfo, GroupTransferMeta, ReqMeta, RequestMetadata)

PAGE = 4096
ORDER = ["m0", "a1", "m2", "a3"]
ATTN = ["a1", "a3"]
GDN = ["m0", "m2"]


def _group(g_idx, kind, layers):
    return GroupInfo(
        group_idx=g_idx, kind=kind, layer_names=tuple(layers),
        block_size=16,
        mamba_align_size=16 if kind == "mamba" else None,
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

    def get(self, block_indices, block_hashs, layer_names, tensors,
            label=None):
        return {ln: f"task:{ln}" for ln in layer_names}

    def get_wait(self, get_results, wait=True):
        layers = {k.rsplit("#", 1)[0] for k in get_results}
        if layers & self.fail_on:
            raise RuntimeError(
                f"transfer failed: {sorted(layers & self.fail_on)}")
        if not wait:
            return layers <= self.landed
        self.waited.extend(sorted(layers))
        return True

    def has(self, chunk_labels, label=None):
        return [True]


class _FakeCanon:
    def register(self, kv_caches):
        pass

    def page_view_parts(self, layer_name):
        return {"page": layer_name}


def _worker(store):
    groups = [_group(0, "attention", ATTN), _group(1, "mamba", GDN)]
    w = HybridWorker(groups, {ln: None for ln in ORDER}, "ns",
                     _FakeCanon(), rank=0, tp_size=1)
    w.kvstore = store
    w.register({ln: None for ln in ORDER}, ORDER)
    return w


def _meta(async_layers, req_id="r1"):
    """A load plan covering every layer, split per group."""
    def op(gidx, layers):
        keys = tuple(CacheKey(namespace="ns", tp_size=1, rank=0,
                              block_hash=7, group_idx=gidx,
                              layer_name=ln) for ln in layers)
        return GroupTransferMeta(
            group_idx=gidx, keys=keys, gpu_block_ids=(5,) * len(layers))
    md = RequestMetadata()
    md.requests[req_id] = ReqMeta(
        group_ops=(op(0, ATTN), op(1, GDN)),
        is_async=True, async_load_layers=async_layers)
    return type("M", (), {"reqs_to_load": md})


# ------------------------------------------------------------------
# the recurrent-state constraint
# ------------------------------------------------------------------
def test_gate_covers_every_recurrent_layer_despite_short_prefix():
    """Asking to release after ONE attention layer must still wait for
    all GDN state: it is consumed whole at the start of forward."""
    b = _FakeStore()
    w = _worker(b)
    w.start_load(_meta(async_layers=1))

    entry = w._async_loads["r1"]
    assert set(GDN) <= entry.gate_layers, (
        "recurrent layers were left out of the release gate")
    # exactly one attention layer, the first in execution order
    assert entry.gate_layers & set(ATTN) == {"a1"}


def test_not_released_until_recurrent_state_has_landed():
    b = _FakeStore()
    w = _worker(b)
    w.start_load(_meta(async_layers=1))

    b.landed = {"a1"}                      # attention prefix only
    assert w.poll_finished_loads() == set()

    b.landed |= set(GDN)                   # now the state is there
    assert w.poll_finished_loads() == {"r1"}


def test_negative_layer_count_gates_on_every_layer():
    b = _FakeStore()
    w = _worker(b)
    w.start_load(_meta(async_layers=-1))
    assert w._async_loads["r1"].gate_layers == set(ORDER)


# ------------------------------------------------------------------
# cross-step lifetime
# ------------------------------------------------------------------
def test_async_tasks_live_outside_the_step_tasks():
    """Async tasks outlive the step by design: they are not in the
    per-step _load_tasks drained by the layer hooks."""
    b = _FakeStore()
    w = _worker(b)
    w.start_load(_meta(async_layers=1))
    assert w._load_tasks == {}


def test_remaining_layers_are_drained_by_the_layer_hooks():
    b = _FakeStore()
    w = _worker(b)
    w.start_load(_meta(async_layers=1))
    b.landed = set(ORDER)
    assert w.poll_finished_loads() == {"r1"}

    # gate layers were finalized at release; a3 was not
    assert "a3" not in b.waited
    w.wait_layer_load("a3")
    assert "a3" in b.waited
    assert "r1" not in w._async_loads      # fully drained, entry dropped


# ------------------------------------------------------------------
# transfer failure is fatal
# ------------------------------------------------------------------
def test_failed_transfer_raises_at_poll():
    b = _FakeStore()
    b.fail_on = {"m0"}
    w = _worker(b)
    w.start_load(_meta(async_layers=1))
    b.landed = set(ORDER)

    with pytest.raises(RuntimeError, match="transfer failed"):
        w.poll_finished_loads()


# ------------------------------------------------------------------
# the deadlock this file exists to prevent
# ------------------------------------------------------------------
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
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    from conftest import HybridRequestScheduler

    groups = [_group(0, "attention", ATTN)]
    sched = HybridRequestScheduler(
        groups, _FakeStore(), hash_block_size=16, namespace="ns",
        tp_size=1, rank=0)
    st = ReqState(
        block_hashes=[1, 2], snapshot_boundary=32,
        groups=(ReqGroupState(block_ids=[4, 5]),))
    st.is_async = True
    st.async_load_layers = 1
    sched._req_states["r1"] = st
    sched._async_load_pending.add("r1")

    plans = sched.take_async_load_plans(already_emitted=set())
    assert list(plans) == ["r1"], (
        "the parked request got no load plan -- it would hang")
    assert plans["r1"].is_async and plans["r1"].group_ops


def test_async_plan_is_emitted_only_once():
    """A second plan would submit a second transfer for a request that
    already has one in flight, stranding the first."""
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    from conftest import HybridRequestScheduler

    sched = HybridRequestScheduler(
        [_group(0, "attention", ATTN)], _FakeStore(),
        hash_block_size=16, namespace="ns", tp_size=1, rank=0)
    st = ReqState(
        block_hashes=[1, 2], snapshot_boundary=32,
        groups=(ReqGroupState(block_ids=[4, 5]),))
    st.is_async = True
    sched._req_states["r1"] = st
    sched._async_load_pending.add("r1")

    assert len(sched.take_async_load_plans(set())) == 1
    assert sched.take_async_load_plans(set()) == {}


def test_recurrent_models_can_go_async():
    """Recurrent groups do NOT force synchronous loading.

    The transfer is a DMA; it needs no forward step. What it needs is to
    land in the slot vLLM will read, which it does -- see
    test_async_mamba_targets_the_slot_vllm_reads_as_prev.
    """
    from kvshrink.kvshrink_connector import ReqState
    from conftest import HybridRequestScheduler

    class _Cfg:
        def select(self, concurrency):
            return 4

    hybrid = HybridRequestScheduler(
        [_group(0, "attention", ATTN), _group(1, "mamba", GDN)],
        _FakeStore(), hash_block_size=16, namespace="ns", tp_size=1,
        rank=0, async_load_config=_Cfg())
    hybrid._req_states["r1"] = ReqState()
    assert hybrid._decide_async("r1", external=64) is True


def test_async_mamba_targets_the_slot_vllm_reads_as_prev():
    """The equality the whole async-GDN path rests on.

    Ours (scheduled_tokens == 0 because vLLM parks async requests):
        curr_idx = (boundary + 0 - 1) // block_size

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
        ours = (boundary + 0 - 1) // bs
        vllm_prev = (boundary - 1) // bs
        assert ours == vllm_prev, (
            f"boundary={boundary}: async restore would write slot "
            f"{ours} but vLLM reads {vllm_prev} as prev")


def test_sync_still_refuses_zero_scheduled_tokens():
    """The guard must stay for the synchronous path: there the write
    lands in start_load_kv, and with no scheduled tokens no forward
    runs at all, so the slot would be left unrestored while the core
    has already credited the tokens."""
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    from conftest import HybridRequestScheduler

    groups = [_group(0, "attention", ATTN), _group(1, "mamba", GDN)]
    sched = HybridRequestScheduler(
        groups, _FakeStore(), hash_block_size=16, namespace="ns",
        tp_size=1, rank=0)
    st = ReqState(
        block_hashes=[1, 2, 3, 4], snapshot_boundary=64,
        groups=(ReqGroupState(block_ids=[1, 2, 3, 4]),
                ReqGroupState(block_ids=[7, 8, 9, 10])))
    st.is_async = False
    sched._req_states["r1"] = st
    with pytest.raises(RuntimeError, match="scheduled_tokens=0"):
        sched._build_load_meta_from_state("r1", st, scheduled_tokens=0)


def test_second_alloc_callback_does_not_queue_another_transfer():
    """Regression: vLLM calls update_state_after_alloc TWICE for an
    async request -- once at allocation, once after the load completes.

    Queuing on both produced a second transfer for a request that was
    RUNNING by then; reporting THAT one finished tripped vLLM's own
    assertion (scheduler.py:2243: a finished-recving request must be
    parked or finished, never running) and killed EngineCore.
    """
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    from conftest import HybridRequestScheduler

    sched = HybridRequestScheduler(
        [_group(0, "attention", ATTN)], _FakeStore(), hash_block_size=16,
        namespace="ns", tp_size=1, rank=0)
    st = ReqState(
        block_hashes=[1, 2], snapshot_boundary=32,
        groups=(ReqGroupState(block_ids=[4, 5]),))
    st.is_async = True
    sched._req_states["r1"] = st

    class _Req:
        request_id = "r1"
        block_hashes = [1, 2]

    class _Blocks:
        @staticmethod
        def get_block_ids():
            return ([4, 5],)

    sched.update_state_after_alloc(_Req(), _Blocks(), 32)
    assert len(sched.take_async_load_plans(set())) == 1

    # vLLM's second callback for the same request
    sched.update_state_after_alloc(_Req(), _Blocks(), 32)
    assert sched.take_async_load_plans(set()) == {}, (
        "a second transfer was queued for a request already running")


def test_request_is_synchronous_again_after_its_async_plan_is_emitted():
    """Regression: after release vLLM reschedules the request through
    the ordinary new-request path, which builds a plan again. If that
    plan is still marked async the worker opens a SECOND cross-step
    transfer and reports the request finished a second time -- by then
    it is RUNNING, and vLLM asserts a finished-recving request must be
    parked or done. Observed as EngineDeadError with 6/6 requests
    returning 500.
    """
    from kvshrink.kvshrink_connector import ReqGroupState, ReqState
    from conftest import HybridRequestScheduler

    sched = HybridRequestScheduler(
        [_group(0, "attention", ATTN)], _FakeStore(), hash_block_size=16,
        namespace="ns", tp_size=1, rank=0)
    st = ReqState(
        block_hashes=[1, 2], snapshot_boundary=32,
        groups=(ReqGroupState(block_ids=[4, 5]),))
    st.is_async = True
    sched._req_states["r1"] = st
    sched._async_load_pending.add("r1")

    plans = sched.take_async_load_plans(set())
    assert plans and plans["r1"].is_async

    assert st.is_async is False, (
        "a plan built after release would still be async")
    later = sched._build_load_meta_from_state("r1", st, scheduled_tokens=8)
    assert later.is_async is False
