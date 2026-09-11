#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Regression gate: verifies that KVShrink works with MTP speculative decoding
# on hybrid models (e.g. Qwen3.5), saves prefill boundaries, and restores
# them properly when num_speculative_blocks > 0.
#
# Usage:
#   MODEL=/path/to/Qwen3.5-4B TP_SIZE=2 tests/gpu/probe_speculative_mtp.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1

if [[ "${GATE_REUSE_SERVER:-0}" == "1" ]]; then
    echo "probe_speculative_mtp.sh cannot reuse a running engine." >&2
    exit 2
fi

PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-400}")"
MAX_TOKENS="${GATE_MAX_TOKENS:-64}"

gate_reset_cache

SPEC_ARGS=(--speculative-config '{"method": "mtp", "num_speculative_tokens": 1}')

# ---------------------------------------------------------------- cold
log "cold run: empty cache, MTP speculative decoding enabled"
gate_serve spec_mtp_cold "${SPEC_ARGS[@]}" || { fail "cold MTP engine startup"; gate_summary; exit 1; }
COLD_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_persist_cache || { fail "cold run persist"; gate_summary; exit 1; }
gate_stop

COLD_LOG="$GATE_LAST_LOG"
check "hybrid path active" \
    grep -q "kvshrink hybrid path enabled" "$COLD_LOG"
check "MTP speculative decoding detected by engine" \
    grep -qi "Qwen3_5MTP" "$COLD_LOG"
check "cold run produced output" test -n "$COLD_OUT"

# ----------------------------------------------------------------- hot
log "hot run: fresh engine with MTP, same prompt, cache on disk"
gate_serve spec_mtp_hot "${SPEC_ARGS[@]}" || { fail "hot MTP engine startup"; gate_summary; exit 1; }
HOT_OUT="$(gate_completion "$PROMPT" "$MAX_TOKENS")"
gate_stop

HOT_LOG="$GATE_LAST_LOG"

check "hot run hit the external cache" \
    test "$(gate_cached_tokens)" -gt 0

if [[ "$COLD_OUT" == "$HOT_OUT" ]]; then
    pass "restored output is byte-identical to the recomputed output under MTP"
else
    fail "restored output differs from the recomputed output under MTP"
    printf '  cold: %s\n  hot : %s\n' "${COLD_OUT:0:200}" "${HOT_OUT:0:200}"
fi

check "no unrestored-state errors" \
    bash -c '! grep -q "refusing to enter forward with unrestored state" "$1"' _ "$HOT_LOG"
check "no connector errors" \
    bash -c '! grep -qE "Failed to load KV cache|kvshrink load poison" "$1"' _ "$HOT_LOG"

gate_summary
