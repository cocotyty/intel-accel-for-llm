#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Real GPU/RDMA regression for shared hybrid KV buffers; see README.md."""

import argparse
import multiprocessing
import time

import torch

from iaxl.remote_pool.kvstore_remote import KVStoreRemote


ATTENTION = ["attention.0", "attention.1", "attention.2"]
MAMBA = ["mamba.0", "mamba.1"]
HASHES = ["a" * 64, "b" * 64]
MISSING = "c" * 64


def wait(store, method, tasks, timeout):
    deadline = time.monotonic() + timeout
    while not getattr(store, method)(tasks, wait=False):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{method} did not complete within {timeout}s")
        time.sleep(0.001)


def worker(rank, args):
    torch.cuda.set_device(rank)
    caches = {
        name: torch.empty((4, 16384), dtype=torch.bfloat16, device=f"cuda:{rank}")
        for name in ATTENTION
    }
    for layer, name in enumerate(ATTENTION):
        for block in range(4):
            caches[name][block].fill_(rank * 32 + layer * 4 + block + 1)
    expected_attention = {name: tensor[[0, 2]].clone() for name, tensor in caches.items()}
    if args.layout == "shared":
        # Match the connector's Mamba-first order; the last attention layer
        # finishes the default-namespace PUT group.
        caches = {
            **{name: caches[attention].view_as(caches[attention])
               for name, attention in zip(MAMBA, ATTENTION)},
            **caches,
        }
    regions = len({t.data_ptr() for t in caches.values()})
    print(f"rank={rank} layout={args.layout} layers={len(caches)} regions={regions}", flush=True)
    store = KVStoreRemote(args.model_name, block_dim=0, kv_caches=caches,
                          rank=rank, tp_size=args.tp_size)

    wait(store, "put_wait", store.put([0, 2], HASHES, ATTENTION), args.timeout)
    if args.layout == "shared":
        for layer, name in enumerate(MAMBA):
            for block in range(4):
                caches[name][block].fill_(rank * 32 + layer * 4 + block + 128)
        expected_mamba = {name: caches[name][[0, 2]].clone() for name in MAMBA}
        wait(store, "put_wait", store.put([0, 2], HASHES, MAMBA, label="mamba"), args.timeout)

    for tensor in caches.values():
        tensor.zero_()
    torch.cuda.synchronize()
    wait(store, "get_wait", store.get([1, 3], HASHES, ATTENTION), args.timeout)
    torch.cuda.synchronize()
    for name in ATTENTION:
        torch.testing.assert_close(caches[name][[1, 3]], expected_attention[name], rtol=0, atol=0)
        assert torch.count_nonzero(caches[name][[0, 2]]).item() == 0, name

    if args.layout == "shared":
        wait(store, "get_wait", store.get([1, 3], HASHES, MAMBA, label="mamba"), args.timeout)
        torch.cuda.synchronize()
        for name in MAMBA:
            torch.testing.assert_close(caches[name][[1, 3]], expected_mamba[name], rtol=0, atol=0)
    print(f"PASS rank={rank}: registered, PUT/GET, exact restored bytes and destination blocks", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--layout", choices=("shared", "disjoint"), default="shared")
    parser.add_argument("--model-name", default="remote-shared-buffer-regression")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.tp_size <= torch.cuda.device_count():
        parser.error("--tp-size must fit the available GPUs")
    ctx = multiprocessing.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(rank, args)) for rank in range(args.tp_size)]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + args.timeout * 3
    try:
        for proc in procs:
            proc.join(max(0, deadline - time.monotonic()))
        assert all(proc.exitcode == 0 for proc in procs), [proc.exitcode for proc in procs]
    finally:
        for proc in procs:
            if proc.is_alive():
                proc.terminate()
        for proc in procs:
            proc.join(5)
            if proc.is_alive():
                proc.kill()
                proc.join()

    names = (MAMBA if args.layout == "shared" else []) + ATTENTION
    scheduler = KVStoreRemote(args.model_name, layer_names=names, tp_size=args.tp_size)
    for label in ([None, "mamba"] if args.layout == "shared" else [None]):
        assert scheduler.has([HASHES[0], MISSING, HASHES[1]], label=label,
                             truncate=False) == [True, False, True], label
        assert scheduler.has([HASHES[0], MISSING, HASHES[1]], label=label) == [True, False, False], label
    print("PASS scheduler: namespace lookup and sparse/prefix semantics", flush=True)
    print("RESULT: ALL PASS", flush=True)


if __name__ == "__main__":
    main()
