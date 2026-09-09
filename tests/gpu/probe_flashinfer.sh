#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Regression gate: verifies that KVShrink works with the FlashInfer
# attention backend without errors, saves boundaries, and reuses them.
#
# Usage:
#   MODEL=/path/to/Qwen3-14B TP_SIZE=2 tests/gpu/probe_flashinfer.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=0
PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-200}")"
MAX_TOKENS="${GATE_MAX_TOKENS:-64}"

gate_reset_cache

log "first run: engine with FLASHINFER backend populates the cache"
gate_serve flashinfer_first --attention-backend FLASHINFER || {
    fail "engine startup with FlashInfer"; gate_summary; exit 1; }
FIRST_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_persist_cache || { fail "first run persist"; gate_summary; exit 1; }
gate_stop

FIRST_LOG="$GATE_LOG_DIR/flashinfer_first.log"
check "engine initialized with FlashInfer backend" \
    grep -qi "flashinfer" "$FIRST_LOG"
check "KV store registered" \
    grep -q "Registered .* KV cache layers" "$FIRST_LOG"
check "first run produced output" test -n "$FIRST_OUT"

log "second run: fresh engine reuses the cache with FlashInfer"
gate_serve flashinfer_second --attention-backend FLASHINFER || {
    fail "engine startup with FlashInfer"; gate_summary; exit 1; }
SECOND_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_stop

SECOND_LOG="$GATE_LOG_DIR/flashinfer_second.log"
check "second run hit the external cache" \
    grep -qE "start_load_kv: [1-9][0-9]* pages loaded" "$SECOND_LOG"
if [[ "$FIRST_OUT" == "$SECOND_OUT" ]]; then
    pass "output identical across FlashInfer runs"
else
    fail "output changed across FlashInfer runs"
    printf '  first : %s\n  second: %s\n' \
        "${FIRST_OUT:0:200}" "${SECOND_OUT:0:200}"
fi
check "no connector errors" \
    bash -c '! grep -qE "Failed to load KV cache|kvshrink load poison" "$1"' \
    _ "$SECOND_LOG"

gate_summary
