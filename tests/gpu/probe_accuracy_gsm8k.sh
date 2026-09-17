#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Accuracy gate: does restoring KV from the external cache cost the
# model anything on a real workload?
#
# The other gates prove one prompt at a time that restored output is
# byte-identical to computed output. This gate asks the same question
# on GSM8K: the engine answers a slice of the test split twice --
# once from an empty cache (later questions can reuse earlier prefixes)
# and once warm (vLLM's own prefix cache dropped, so every answer
# starts from restored KV). Warm accuracy must stay within a small
# tolerance of cold, and the cold pass must clear a sanity floor so a
# broken engine cannot pass by being equally bad twice.
#
# Flips are reported, not asserted: restoring a GDN/Mamba prefix and
# recomputing the tail is mathematically exact but numerically
# different from scanning the whole prompt in one chunk (the chunked
# scan accumulates in a different order), and greedy decoding
# amplifies any numeric difference into divergent text. A corrupted
# restore still fails: garbage KV collapses accuracy far below the
# tolerance. Thinking mode is disabled so answers are short, direct
# CoT that fits the generation cap.
#
# Prompts are 8-shot CoT: a bare GSM8K question is ~100 tokens, shorter
# than one hybrid block (528 tokens on the gate model), so without the
# shared few-shot prefix nothing would ever be saved or restored.
#
# Unlike the customer repro (test/accuracy-gsm8k in KVCacheClip, which
# drives evalscope), this gate uses tests/gpu/gsm8k_eval.py -- stdlib
# only, same dataset, same greedy generation config -- so it needs no
# extra install inside the regression container.
#
# Usage (GATE_HYBRID tells lib.sh whether to add the GDN flags;
# run_gates.sh sets it per model):
#   GATE_HYBRID=1 MODEL=/path/to/Qwen3.5-4B tests/gpu/probe_accuracy_gsm8k.sh
#   GATE_HYBRID=0 MODEL=/path/to/Qwen3-14B  tests/gpu/probe_accuracy_gsm8k.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID="${GATE_HYBRID:?set GATE_HYBRID=1 (GDN model) or 0 (attention)}"
# Needed for POST /reset_prefix_cache.
export VLLM_SERVER_DEV_MODE=1

LIMIT="${GATE_GSM8K_LIMIT:-50}"
PARALLEL="${GATE_GSM8K_PARALLEL:-8}"
MAX_TOKENS="${GATE_GSM8K_MAX_TOKENS:-1024}"
FLOOR="${GATE_GSM8K_ACC_FLOOR:-0.5}"
TOLERANCE="${GATE_GSM8K_ACC_TOLERANCE:-0.05}"
DATA="${GATE_GSM8K_DATA:-$REPO_DIR/_data/gsm8k/test.jsonl}"
DATA_URL="${GATE_GSM8K_URL:-https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl}"

gate_reset_cache

if [[ ! -s "$DATA" ]]; then
    log "downloading the GSM8K test split -> $DATA"
    mkdir -p "$(dirname "$DATA")"
    curl -sfL --retry 3 -o "$DATA" "$DATA_URL" || {
        fail "GSM8K dataset download"; gate_summary; exit 1; }
fi

gate_serve gsm8k || { fail "engine startup"; gate_summary; exit 1; }

COLD="$GATE_LOG_DIR/gsm8k.cold.json"
WARM="$GATE_LOG_DIR/gsm8k.warm.json"

log "cold pass: $LIMIT questions, $PARALLEL-way concurrent (populates the cache)"
python3 "$GATE_DIR/gsm8k_eval.py" run \
    --model "$MODEL" --port "$GATE_PORT" --data "$DATA" \
    --limit "$LIMIT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" \
    --out "$COLD" || { fail "cold pass"; gate_summary; exit 1; }

# Saves are async (main's lifecycle): let the backlog drain, then drop
# vLLM's own prefix cache so the warm pass must come to the connector.
log "waiting for async saves to land"
sleep "${GATE_SAVE_SETTLE:-15}"
if curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" >/dev/null; then
    pass "internal prefix cache reset"
else
    fail "internal prefix cache reset (is VLLM_SERVER_DEV_MODE=1 set?)"
fi

log "warm pass: same questions, answered from restored KV"
python3 "$GATE_DIR/gsm8k_eval.py" run \
    --model "$MODEL" --port "$GATE_PORT" --data "$DATA" \
    --limit "$LIMIT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS" \
    --out "$WARM" || { fail "warm pass"; gate_summary; exit 1; }

# Without a real external hit the accuracy comparison would prove
# nothing: a warm pass that never touched the cache is just a second
# cold pass. The compare also enforces a >=90% per-question hit rate.
WARM_CACHED=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cached_tokens"])' "$WARM")
check "warm pass hit the external cache" test "$WARM_CACHED" -gt 0

if python3 "$GATE_DIR/gsm8k_eval.py" compare \
    --cold "$COLD" --warm "$WARM" --floor "$FLOOR" --tolerance "$TOLERANCE"; then
    pass "restoring KV costs no GSM8K accuracy"
else
    fail "restoring KV costs GSM8K accuracy"
fi

gate_summary
