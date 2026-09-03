#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Decode-reuse gate for PURE-ATTENTION models -- the control arm of
# probe_decode_reuse.sh.
#
# Why this gate exists: the hybrid decode-reuse gate found that a
# restored continuation diverges from a cold recompute after a few
# hundred tokens. One candidate explanation is GDN-inherent (the decode
# kernel and the prefill kernel reach the same state through different
# floating-point orders, so a snapshot taken from one and replayed as
# the other differs in the last bits). Attention KV has no such split:
# pages are exact key/value tensors, saved and restored bit-for-bit. So
# this gate answers the question the hybrid gate cannot: is the
# save/restore PIPELINE itself bit-exact?
#
# Method (same shape as the hybrid gate, scaled to a 16-token block):
#   1. R1 = completion(PROMPT, 512): decode crosses many blocks.
#   2. R2 = completion(PROMPT ids, 1): baseline hit, prompt pages only.
#   3. R3 = completion(PROMPT+GENERATED ids, 512): must restore MORE
#      pages than R2.
#   4. R4 = same ids with a cache_salt (genuine cold): R3 and R4 must
#      agree byte for byte. For attention KV there is no float-order
#      argument to excuse a mismatch -- a difference here is a bug.
#
# Usage:
#   MODEL=/path/to/Qwen3-14B tests/gpu/probe_decode_reuse_attn.sh

source "$(dirname "$0")/lib.sh"

# Pure-attention model: no hybrid flags, plain vLLM serving.
# Needed for POST /reset_prefix_cache.
export VLLM_SERVER_DEV_MODE=1
PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-200}")"
MAX_TOKENS="${GATE_MAX_TOKENS:-512}"

pages_loaded_in() {
    grep -oE "start_load_kv: [1-9][0-9]* pages loaded" "$1" \
        | tail -1 | grep -oE "[1-9][0-9]*"
}

gate_completion_ids() {
    # gate_completion_ids <prompt_ids> <max_tokens> <ids_file> [salt]
    local prompt_ids="$1" max_tokens="$2" ids_file="$3" salt="${4:-}"
    python3 - "$MODEL" "$prompt_ids" "$max_tokens" \
        "$GATE_PORT" "$ids_file" "$salt" <<'PYEOF'
import json, sys, urllib.request
model, prompt_ids, max_tokens, port, ids_file, salt = sys.argv[1:7]
prompt = json.loads(prompt_ids)
payload = {
    "model": model, "prompt": prompt,
    "max_tokens": int(max_tokens),
    "temperature": 0, "seed": 0,
    "return_token_ids": True,
}
if salt:
    payload["cache_salt"] = salt
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    body = json.load(r)
choice = body["choices"][0]
gen_ids = choice.get("token_ids") or []
open(ids_file, "w").write(json.dumps(
    {"prompt_ids": prompt, "generated_ids": gen_ids}))
print(choice["text"], end="")
PYEOF
}

gate_reset_cache

gate_serve decode_reuse_attn || { fail "engine startup"; gate_summary; exit 1; }
LOG="$GATE_LAST_LOG"

# ------------------------------------------------------- first request
log "R1: compute and generate; decode-produced blocks get saved"
R1_IDS="$GATE_LOG_DIR/decode_reuse_attn.r1_ids.json"
R1_TEXT="$(python3 - "$MODEL" "$PROMPT" "$MAX_TOKENS" "$GATE_PORT" "$R1_IDS" <<'PYEOF'
import json, sys, urllib.request
model, prompt, max_tokens, port, ids_file = sys.argv[1:6]
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": int(max_tokens),
        "temperature": 0, "seed": 0,
        "return_token_ids": True,
    }).encode(),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=600) as r:
    body = json.load(r)
c = body["choices"][0]
open(ids_file, "w").write(json.dumps(
    {"prompt_ids": c["prompt_token_ids"],
     "generated_ids": c["token_ids"]}))
print(c["text"], end="")
PYEOF
)"
PROMPT_IDS="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["prompt_ids"]))' "$R1_IDS")"
GEN_IDS="$(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))["generated_ids"]))' "$R1_IDS")"
check "R1 generated tokens recorded" \
    python3 -c 'import json,sys; exit(0 if len(json.load(open(sys.argv[1]))["generated_ids"]) > 100 else 1)' "$R1_IDS"
check "R1 prompt ids recorded" \
    python3 -c 'import json,sys; exit(0 if len(json.load(open(sys.argv[1]))["prompt_ids"]) > 100 else 1)' "$R1_IDS"
check "R1 saved boundaries" \
    grep -qE "chunk_save: [1-9][0-9]* pages submitted, [1-9][0-9]* boundaries" "$LOG"

log "waiting for async saves to land"
sleep "${GATE_SAVE_SETTLE:-15}"

# ------------------------------------------------------ second request
MARK1=$(wc -l < "$LOG")
if ! curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" \
        >/dev/null; then
    fail "internal prefix cache reset"
fi
R2_LOG="$GATE_LOG_DIR/decode_reuse_attn.r2.log"
R2_TEXT="$(gate_completion_ids "$PROMPT_IDS" 1 "$R1_IDS.r2")"
tail -n "+$((MARK1 + 1))" "$LOG" >"$R2_LOG"
N2="$(pages_loaded_in "$R2_LOG")"
log "R2 (prompt only): ${N2:-0} pages restored"
check "R2 restored from the external cache" test -n "$N2"

# The follow-up turn: prompt plus everything R1 generated.
MARK2=$(wc -l < "$LOG")
curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" >/dev/null
TURN2_IDS="$(python3 -c '
import json, sys
print(json.dumps(json.loads(sys.argv[1]) + json.loads(sys.argv[2])))' \
    "$PROMPT_IDS" "$GEN_IDS")"
R3_LOG="$GATE_LOG_DIR/decode_reuse_attn.r3.log"
R3_TEXT="$(gate_completion_ids "$TURN2_IDS" "$MAX_TOKENS" "$R1_IDS.r3")"
tail -n "+$((MARK2 + 1))" "$LOG" >"$R3_LOG"
N3="$(pages_loaded_in "$R3_LOG")"
log "R3 (prompt + generated): ${N3:-0} pages restored"
check "R3 restored from the external cache" test -n "$N3"

if [[ -n "${N2:-}" && -n "${N3:-}" && "${N3}" -gt "${N2}" ]]; then
    pass "decode-produced blocks were restored (${N3} > ${N2} pages)"
else
    fail "decode-produced blocks were NOT restored " \
         "(R3=${N3:-none} pages vs R2=${N2:-none} pages)"
fi

# ------------------------------------------------ cold recompute control
# Attention KV is exact: R3 must equal R4 byte for byte, no excuses.
MARK3=$(wc -l < "$LOG")
curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" >/dev/null
R4_LOG="$GATE_LOG_DIR/decode_reuse_attn.r4.log"
R4_TEXT="$(gate_completion_ids "$TURN2_IDS" "$MAX_TOKENS" "$R1_IDS.r4" cold)"
tail -n "+$((MARK3 + 1))" "$LOG" >"$R4_LOG"
if ! grep -qE "start_load_kv:" "$R4_LOG"; then
    pass "R4 was a genuine cold recompute (no external load)"
else
    fail "R4 unexpectedly hit the external cache"
fi

if [[ "$R3_TEXT" == "$R4_TEXT" ]]; then
    pass "restored continuation is byte-identical to recomputed"
else
    fail "restored continuation differs from recomputed"
    printf '  restored: %s\n  computed: %s\n' \
        "${R3_TEXT:0:200}" "${R4_TEXT:0:200}"
fi

check "no unrestored-state errors" \
    bash -c '! grep -q "refusing to enter forward with unrestored state" "$1"' _ "$LOG"
check "no load poison" \
    bash -c '! grep -q "kvshrink load poison" "$1"' _ "$LOG"

gate_stop
gate_summary
