#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Batched (concurrent) correctness gate for hybrid (GDN/Mamba) models.
#
# Why this gate exists, separately from probe_cold_hot.sh: every other
# hybrid gate drives the engine with one request at a time. With a
# single sequence in flight, a step commits at most one boundary, the
# partial-boundary path never fires, and the zip/unzip work queues are
# essentially empty. The parts of the save path that only exist because
# of batching therefore went unexercised:
#
#   * several boundaries committed in one step (per_group carries more
#     than one entry in gslot["bnds"])
#   * the partial-boundary skip path, reached when concurrent sequences
#     cross their alignment boundaries on different steps
#   * unzip racing against zip work still queued -- the documented
#     trigger for a missing cache entry on restore
#
# What is asserted, and what deliberately is not: byte-exact output is
# NOT asserted here, because under concurrency it is not a property the
# engine has. Measured on this gate (Qwen3.5-4B, TP=2, 6 concurrent
# requests): running the batch twice with NO external cache involved at
# all -- GATE_BATCH_CONTROL=1, both phases cold -- already produced
# different text for 2 of 6 requests. Concurrent arrivals do not batch
# identically from run to run, vLLM does not promise batch-invariant
# numerics, and greedy decoding turns the first differing logit into a
# completely different continuation. An equality check here would fail
# for reasons that have nothing to do with the cache.
#
# Bit-exact restore is therefore proven where it IS deterministic, by
# the serial gate (probe_cold_hot.sh). This gate proves the other half:
# that batching does not break the save/restore machinery -- boundaries
# still commit, the restore still hits, and nothing degrades into a
# poisoned load or a dead worker. Per-request differences are printed
# for information only.
#
# Run with GATE_BATCH_CONTROL=1 to re-measure that baseline.
#
# Usage:
#   MODEL=/path/to/hybrid-model TP_SIZE=2 tests/gpu/probe_batch_hybrid.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1

# Like cold_hot, this gate is defined by a restart: a reused engine
# would answer the hot phase out of memory and prove nothing.
if [[ "${GATE_REUSE_SERVER:-0}" == "1" ]]; then
    echo "probe_batch_hybrid.sh cannot reuse a running engine; it must" \
         "restart one to prove batched restore really came off disk." >&2
    exit 2
fi

NREQ="${GATE_BATCH_REQUESTS:-6}"
MAX_TOKENS="${GATE_MAX_TOKENS:-64}"

mapfile -t PROMPTS < <(gate_branching_prompts \
    "$NREQ" "${GATE_PROMPT_SEGMENTS:-400}" "${GATE_BATCH_TAIL:-40}")

if [[ "${#PROMPTS[@]}" -ne "$NREQ" ]]; then
    echo "prompt generation produced ${#PROMPTS[@]} prompts, expected $NREQ" >&2
    exit 2
fi
log "batched gate: $NREQ concurrent requests sharing a long prefix"

COLD_DIR="$GATE_LOG_DIR/batch_cold"
HOT_DIR="$GATE_LOG_DIR/batch_hot"
rm -rf "$COLD_DIR" "$HOT_DIR"

gate_reset_cache

# ---------------------------------------------------------------- cold
log "cold batched run: empty cache, all requests in flight together"
gate_serve batch_cold || { fail "cold engine startup"; gate_summary; exit 1; }
if gate_completions_concurrent "$COLD_DIR" "$MAX_TOKENS" "${PROMPTS[@]}"; then
    pass "cold batched run: all $NREQ requests answered"
else
    fail "cold batched run: at least one request failed"
fi
gate_persist_cache || { fail "cold run persist"; gate_summary; exit 1; }
gate_stop
COLD_LOG="$GATE_LAST_LOG"

check "hybrid path active" \
    grep -q "kvshrink hybrid path enabled" "$COLD_LOG"
check "cold batched run saved boundaries" \
    grep -qE "chunk_save: [1-9][0-9]* pages submitted, [1-9][0-9]* boundaries" "$COLD_LOG"

# The point of the gate: a step that commits more than one boundary.
# Without this the run degenerated to sequential and proves nothing.
if grep -qE "chunk_save: [1-9][0-9]* pages submitted, ([2-9]|[1-9][0-9]+) boundaries" "$COLD_LOG"; then
    pass "a single step committed multiple boundaries (batching reached the save path)"
else
    fail "no step committed more than one boundary: requests did not actually batch"
fi

# ----------------------------------------------------------------- hot
# Control mode (GATE_BATCH_CONTROL=1): wipe the cache so the second
# phase is ALSO cold. Both phases then differ in nothing but scheduling,
# so any output difference is the engine's own run-to-run variation --
# concurrent requests do not batch identically every time, and vLLM does
# not promise batch-invariant numerics. Use this to tell "the cache
# restored something wrong" apart from "this comparison was never
# deterministic to begin with".
if [[ "${GATE_BATCH_CONTROL:-0}" == "1" ]]; then
    log "CONTROL MODE: second phase runs cold too (no restore)"
    gate_reset_cache
    log "control batched run: empty cache again, same requests"
else
    log "hot batched run: fresh engine, same requests, cache on disk"
fi
gate_serve batch_hot || { fail "hot engine startup"; gate_summary; exit 1; }
if gate_completions_concurrent "$HOT_DIR" "$MAX_TOKENS" "${PROMPTS[@]}"; then
    pass "hot batched run: all $NREQ requests answered"
else
    fail "hot batched run: at least one request failed"
fi
gate_stop
HOT_LOG="$GATE_LAST_LOG"

if [[ "${GATE_BATCH_CONTROL:-0}" == "1" ]]; then
    log "control mode: skipping the external-cache-hit check (cache wiped)"
else
    check "hot batched run hit the external cache" \
        grep -qE "start_load_kv: [1-9][0-9]* pages loaded" "$HOT_LOG"
fi

# Per-request comparison, reported but NOT asserted: see the header.
# Two cold batched runs already disagree on some requests, so a
# difference here is not evidence of a cache fault. What would be
# evidence is a missing or empty answer, which is asserted.
ndiff=0
nmissing=0
for ((i = 0; i < NREQ; i++)); do
    if [[ ! -s "$COLD_DIR/$i.txt" || ! -s "$HOT_DIR/$i.txt" ]]; then
        nmissing=$((nmissing + 1))
        printf '  request %d: missing or empty output\n' "$i"
        continue
    fi
    if ! cmp -s "$COLD_DIR/$i.txt" "$HOT_DIR/$i.txt"; then
        ndiff=$((ndiff + 1))
        printf '  request %d diverges (informational):\n    first : %s\n    second: %s\n' "$i" \
            "$(head -c 160 "$COLD_DIR/$i.txt")" \
            "$(head -c 160 "$HOT_DIR/$i.txt")"
    fi
done
log "output divergence: $ndiff/$NREQ requests (expected under concurrency, not asserted)"
if [[ "$nmissing" -eq 0 ]]; then
    pass "every request produced a non-empty answer in both phases"
else
    fail "$nmissing request(s) produced no output"
fi

# Fail-closed guarantees, under batching this time.
for phase_log in "$COLD_LOG" "$HOT_LOG"; do
    phase="$(basename "$phase_log")"
    check "no unrestored-state errors ($phase)" \
        bash -c '! grep -q "refusing to enter forward with unrestored state" "$1"' _ "$phase_log"
    check "no load poison ($phase)" \
        bash -c '! grep -q "kvshrink load poison" "$1"' _ "$phase_log"
    # A missing cache entry must degrade to a miss, never abort the
    # worker: an unreadable cache is recoverable, a dead worker is not.
    check "no native abort ($phase)" \
        bash -c '! grep -q "IAXL check failed" "$1"' _ "$phase_log"
    check "no worker died ($phase)" \
        bash -c '! grep -q "died unexpectedly" "$1"' _ "$phase_log"
done

gate_summary
