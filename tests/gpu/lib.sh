#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the GPU gates. Everything that varies per machine
# comes from an environment variable with a default, so the gates run
# unchanged anywhere.

set -euo pipefail

GATE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "$GATE_DIR/../.." && pwd)

: "${MODEL:?set MODEL to a model path or HF id}"

# The connector requires the full runtime environment and refuses to
# start without it, so the gates go through the same entry point an
# operator would. Every variable in setvars.sh honours a value already
# present in the environment, so the caller stays in control; a machine
# without the Intel accelerators just turns them off, e.g.
#   IAXL_QAT_ZIP_ENABLE=0 IAXL_DSA_GD_ENABLE=0 tests/gpu/run_gates.sh
source "$REPO_DIR/setvars.sh" >/dev/null

GATE_PORT="${GATE_PORT:-8000}"
GATE_TP="${TP_SIZE:-1}"
GATE_LOG_DIR="${GATE_LOG_DIR:-$REPO_DIR/_data/gate-logs}"
GATE_CACHE_DIR="${GATE_CACHE_DIR:-$REPO_DIR/_data/gate-cache}"
GATE_KEEP_CACHE="${GATE_KEEP_CACHE:-0}"
GATE_STARTUP_TIMEOUT="${GATE_STARTUP_TIMEOUT:-600}"
GATE_MAX_MODEL_LEN="${GATE_MAX_MODEL_LEN:-8192}"
GATE_GPU_UTIL="${GATE_GPU_UTIL:-0.85}"
# Reproducible block hashes: vLLM seeds its first-block hash from
# PYTHONHASHSEED and randomizes it when unset, which makes every
# restart miss.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

_SERVER_PID=""
FAILURES=0

log()  { printf '[gate] %s\n' "$*"; }
pass() { printf 'PASS  %s\n' "$*"; }
fail() { printf 'FAIL  %s\n' "$*"; FAILURES=$((FAILURES + 1)); }

# check <description> <condition-exit-code>
check() {
    local desc="$1"; shift
    if "$@"; then pass "$desc"; else fail "$desc"; fi
}

gate_reset_cache() {
    # A reused engine already opened the cache; wiping it underneath
    # would be a lie about what the run then proves.
    if [[ "${GATE_REUSE_SERVER:-0}" == "1" ]]; then
        log "reusing an engine: leaving the existing cache in place"
        mkdir -p "$GATE_CACHE_DIR" "$GATE_LOG_DIR"
        return 0
    fi
    rm -rf "$GATE_CACHE_DIR"
    mkdir -p "$GATE_CACHE_DIR" "$GATE_LOG_DIR"
}

gate_cleanup() {
    if [[ -n "$_SERVER_PID" ]] && kill -0 "$_SERVER_PID" 2>/dev/null; then
        kill "$_SERVER_PID" 2>/dev/null || true
        wait "$_SERVER_PID" 2>/dev/null || true
        # The next gate in a suite run starts right after this one
        # exits, so release the GPUs before returning.
        gate_wait_gpu_free
    fi
    _SERVER_PID=""
    if [[ "$GATE_KEEP_CACHE" != "1" && "${GATE_REUSE_SERVER:-0}" != "1" ]]; then
        rm -rf "$GATE_CACHE_DIR"
    fi
}
trap gate_cleanup EXIT

# gate_serve <log-name> [extra vllm args...]
# Starts an engine with the KVShrink connector and waits until it is
# ready. Hybrid-specific flags are added only when GATE_HYBRID=1.
#
# During development, engine startup dominates the loop. Set
# GATE_REUSE_SERVER=1 to run the checks against an engine that is
# already listening on GATE_PORT (see tests/gpu/dev_server.sh); then
# point GATE_SERVER_LOG at that engine's log so the log assertions have
# something to read.
gate_serve() {
    local log_name="$1"; shift
    local log_file="$GATE_LOG_DIR/$log_name.log"
    mkdir -p "$GATE_LOG_DIR"

    if [[ "${GATE_REUSE_SERVER:-0}" == "1" ]]; then
        GATE_LAST_LOG="${GATE_SERVER_LOG:-$log_file}"
        if [[ ! -r "$GATE_LAST_LOG" ]]; then
            log "GATE_REUSE_SERVER=1 needs a readable GATE_SERVER_LOG" \
                "(got '${GATE_SERVER_LOG:-unset}')"
            return 1
        fi
        if ! curl -sf "http://127.0.0.1:$GATE_PORT/health" >/dev/null 2>&1; then
            log "no engine answering on port $GATE_PORT"
            return 1
        fi
        log "reusing the engine on port $GATE_PORT (log: $GATE_LAST_LOG)"
        return 0
    fi

    local -a args=(
        serve "$MODEL"
        --kv-transfer-config
        '{"kv_connector":"KVShrinkConnector","kv_connector_module_path":"kvshrink.kvshrink_connector","kv_role":"kv_both"}'
        --trust-remote-code
        --tensor-parallel-size "$GATE_TP"
        --max-model-len "$GATE_MAX_MODEL_LEN"
        --gpu-memory-utilization "$GATE_GPU_UTIL"
        --port "$GATE_PORT"
        --enforce-eager
    )
    if [[ "${GATE_HYBRID:-0}" == "1" ]]; then
        # GDN snapshots are only addressable on aligned boundaries, and
        # the hybrid memory allocator must stay enabled. Prefix caching
        # must be requested explicitly: vLLM defaults it OFF for hybrid
        # models and then silently rewrites the cache mode to 'none'.
        args+=(--enable-prefix-caching
               --mamba-cache-mode align
               --no-disable-hybrid-kv-cache-manager)
    fi
    args+=("$@")

    log "starting engine -> $log_file"
    GATE_LAST_LOG="$log_file"
    IAXL_CACHE_DIR="$GATE_CACHE_DIR" vllm "${args[@]}" >"$log_file" 2>&1 &
    _SERVER_PID=$!

    local waited=0
    until curl -sf "http://127.0.0.1:$GATE_PORT/health" >/dev/null 2>&1; do
        if ! kill -0 "$_SERVER_PID" 2>/dev/null; then
            log "engine exited during startup; last lines:"
            tail -30 "$log_file" >&2
            return 1
        fi
        sleep 2
        waited=$((waited + 2))
        if (( waited >= GATE_STARTUP_TIMEOUT )); then
            log "engine did not become ready in ${GATE_STARTUP_TIMEOUT}s"
            tail -30 "$log_file" >&2
            return 1
        fi
    done
    log "engine ready after ${waited}s"
}

gate_stop() {
    # Never kill an engine we did not start.
    if [[ -n "$_SERVER_PID" ]]; then
        kill "$_SERVER_PID" 2>/dev/null || true
        wait "$_SERVER_PID" 2>/dev/null || true
        _SERVER_PID=""
        gate_wait_gpu_free
    fi
}

# Flush every worker rank's DDR-resident chunks to disk. Restart gates
# must call this before gate_stop: saves land in the DDR pool with
# persisted=0, and a fresh worker's Record startup deletes unpersisted
# rows, so without an explicit persist the next engine sees an empty
# cache. Persistence is operator-triggered by design (the engine
# exposes POST /v1/cache/persist per rank).
gate_persist_cache() {
    local base="${IAXL_API_WORKER_BASE_PORT:-18800}"
    local r
    for ((r = 0; r < GATE_TP; r++)); do
        curl -sf -X POST "http://127.0.0.1:$((base + r))/v1/cache/persist" \
            -H 'Content-Type: application/json' \
            -d '{"count": 100000000}' >/dev/null || {
            log "persist failed on rank $r"
            return 1
        }
    done
}

# Block until the GPUs are actually released.
#
# Killing the `vllm` process does not immediately reclaim anything: the
# engine core and one worker per rank are separate processes, and vLLM
# escalates to SIGKILL only after its own SIGTERM grace period. Starting
# the next engine while the previous one still owns GPU memory and
# pinned host buffers produced a worker that died mid-forward several
# gates into a suite run, while the same gate passed on its own. A gate
# that only fails when run alongside others is worthless, so the
# harness waits instead of racing.
gate_wait_gpu_free() {
    command -v nvidia-smi >/dev/null 2>&1 || { sleep 5; return 0; }
    local budget="${GATE_GPU_FREE_TIMEOUT:-60}"
    local waited=0 used
    while (( waited < budget )); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               2>/dev/null | sort -rn | head -1)
        [[ -z "$used" ]] && { sleep 5; return 0; }
        # A few hundred MiB is normal residue from other tenants; we
        # only need the previous engine's multi-GB footprint gone.
        (( used < ${GATE_GPU_FREE_MIB:-1024} )) && break
        sleep 2
        waited=$((waited + 2))
    done
    if (( waited >= budget )); then
        log "GPUs still hold ${used}MiB after ${budget}s; continuing anyway"
    fi
    # Even once memory is back, the driver needs a moment to settle.
    sleep 3
}

# gate_completion <prompt> [max_tokens]
# Greedy completion (temperature 0) so cold and hot runs are comparable.
gate_completion() {
    local prompt="$1"
    local max_tokens="${2:-64}"
    curl -sf "http://127.0.0.1:$GATE_PORT/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "$(python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "prompt": sys.argv[2],
                  "max_tokens": int(sys.argv[3]), "temperature": 0,
                  "seed": 0}))' "$MODEL" "$prompt" "$max_tokens")" \
        | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])'
}

# gate_completions_concurrent <outdir> <max_tokens> <prompt>...
# Fire every prompt at once and wait for all of them, so the engine has
# to batch them into shared steps. Writes reply N to <outdir>/N.txt and
# returns non-zero if any request failed.
#
# This exists because gate_completion is strictly sequential: with one
# request in flight a step commits at most one boundary, the partial
# boundary path never triggers, and the zip/unzip queues are empty --
# precisely the conditions under which the concurrency-sensitive parts
# of the save path are NOT exercised.
gate_completions_concurrent() {
    local outdir="$1"; shift
    local max_tokens="$1"; shift
    mkdir -p "$outdir"
    local pids=() i=0
    for prompt in "$@"; do
        (
            curl -sf "http://127.0.0.1:$GATE_PORT/v1/completions" \
                -H 'Content-Type: application/json' \
                -d "$(python3 -c '
import json, sys
print(json.dumps({"model": sys.argv[1], "prompt": sys.argv[2],
                  "max_tokens": int(sys.argv[3]), "temperature": 0,
                  "seed": 0}))' "$MODEL" "$prompt" "$max_tokens")" \
                | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["text"])' \
                > "$outdir/$i.txt"
        ) &
        pids+=($!)
        i=$((i + 1))
    done
    local rc=0
    for pid in "${pids[@]}"; do
        wait "$pid" || rc=1
    done
    return $rc
}

# gate_branching_prompts <count> [shared_segments] [tail_segments]
# Prompts that share a long common prefix and then diverge. The shared
# head crosses several GDN alignment boundaries (so all of them contend
# for the same cached prefix), while the distinct tails force each
# sequence to reach its own boundaries at a different step.
gate_branching_prompts() {
    python3 -c '
import sys
n = int(sys.argv[1]); head = int(sys.argv[2]); tail = int(sys.argv[3])
shared = " ".join("segment %d carries token payload alpha beta gamma." % i
                  for i in range(head))
for k in range(n):
    extra = " ".join("branch %d segment %d delta epsilon zeta." % (k, j)
                     for j in range(tail + k * 7))
    print(shared + " " + extra)
' "${1:-4}" "${2:-400}" "${3:-40}"
}

# A prompt long enough to cross several GDN alignment boundaries.
gate_long_prompt() {
    local repeats="${1:-400}"
    python3 -c '
import sys
n = int(sys.argv[1])
print(" ".join("segment %d carries token payload alpha beta gamma." % i
               for i in range(n)))' "$repeats"
}

gate_summary() {
    echo
    if (( FAILURES == 0 )); then
        echo "RESULT: ALL PASS"
    else
        echo "RESULT: $FAILURES FAILED"
    fi
    return $(( FAILURES > 0 ))
}
