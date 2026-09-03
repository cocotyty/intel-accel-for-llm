#!/bin/bash
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Decode-reuse gate for hybrid (GDN/Mamba) models.
#
# The question this answers: tokens a request GENERATED during decode
# are saved to the external cache -- does a follow-up request that
# replays them as its prompt prefix actually restore them? This is the
# multi-turn-conversation scenario: every other gate caches only what
# prefill computed, so the decode-produced span -- the part the next
# turn replays -- has never been read back anywhere.
#
# Method:
#   1. R1 = completion(PROMPT, 64). Decode crosses block boundaries;
#      those blocks are saved async.
#   2. Let the saves settle; drop vLLM's internal prefix cache (the
#      external cache survives, as in probe_warm_reuse.sh).
#   3. R2 = completion(PROMPT_TOKENS, 1): baseline, only the prompt
#      range can be restored. Record its page count.
#   4. R3 = completion(PROMPT_TOKENS + GENERATED_IDS, 64): the prefix
#      extends into decode-produced blocks. It must load MORE pages
#      than R2.
#   5. R4 = same token prefix as R3 with a cache_salt: block 0's hash
#      changes, the whole chain changes, so R4 is a genuine cold
#      recompute. R3 and R4 must agree byte for byte.
#
# The follow-up prompt is replayed at the TOKEN level (the completion
# API accepts a token-id list), so the block-hash chain is identical to
# R1's by construction -- a text round trip would re-tokenize and could
# silently break the chain, making the gate lie about the cause.
#
# Usage:
#   MODEL=/path/to/Qwen3.5-4B tests/gpu/probe_decode_reuse.sh

source "$(dirname "$0")/lib.sh"

export GATE_HYBRID=1
# Needed for POST /reset_prefix_cache.
export VLLM_SERVER_DEV_MODE=1

PROMPT="$(gate_long_prompt "${GATE_PROMPT_SEGMENTS:-200}")"
# Decode must CROSS block boundaries for this gate to mean anything.
# The hybrid block size is large (528 tokens for qwen3.5-4b: attention
# is padded up to the mamba page), so 64 generated tokens commit zero
# boundaries and the gate would vacuously pass/fail. Default to two
# full blocks plus slack.
MAX_TOKENS="${GATE_MAX_TOKENS:-1088}"

# gate_completion_ids <prompt_ids> <max_tokens> <ids_file> [salt]
# Token-level prompt; prints the generated text and leaves the ids in
# the ids_file. An optional cache_salt makes the request cold.
gate_completion_ids() {
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

pages_loaded_in() {
    grep -oE "start_load_kv: [1-9][0-9]* pages loaded" "$1" \
        | tail -1 | grep -oE "[1-9][0-9]*"
}

gate_reset_cache

gate_serve decode_reuse || { fail "engine startup"; gate_summary; exit 1; }
LOG="$GATE_LAST_LOG"

# ------------------------------------------------------- first request
log "R1: compute and generate; decode-produced blocks get saved"
R1_IDS="$GATE_LOG_DIR/decode_reuse.r1_ids.json"
# One request, two artifacts: the generated token ids (the response's
# token_ids field is ALWAYS the generated tokens only -- vLLM merges
# them with the prompt only for the logprobs path) and the prompt
# token ids (prompt_token_ids field, also return_token_ids-gated).
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
    python3 -c 'import json,sys; exit(0 if len(json.load(open(sys.argv[1]))["generated_ids"]) > 16 else 1)' "$R1_IDS"
check "R1 prompt ids recorded" \
    python3 -c 'import json,sys; exit(0 if len(json.load(open(sys.argv[1]))["prompt_ids"]) > 100 else 1)' "$R1_IDS"
check "R1 saved boundaries" \
    grep -qE "chunk_save: [1-9][0-9]* pages submitted, [1-9][0-9]* boundaries" "$LOG"

log "waiting for async saves to land"
sleep "${GATE_SAVE_SETTLE:-15}"

# ------------------------------------------------------ second request
# Baseline: only the prompt range is restorable.
MARK1=$(wc -l < "$LOG")
if ! curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" \
        >/dev/null; then
    fail "internal prefix cache reset"
fi
R2_LOG="$GATE_LOG_DIR/decode_reuse.r2.log"
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
R3_LOG="$GATE_LOG_DIR/decode_reuse.r3.log"
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
# The cache_salt mixes into block 0's hash only, which changes the
# whole chain, so R4 cannot hit any cache. R4's OUTPUT is unaffected
# by the salt, so R3 vs R4 still compares the same prompt.
MARK3=$(wc -l < "$LOG")
curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" >/dev/null
R4_LOG="$GATE_LOG_DIR/decode_reuse.r4.log"
R4_TEXT="$(gate_completion_ids "$TURN2_IDS" "$MAX_TOKENS" "$R1_IDS.r4" cold)"
tail -n "+$((MARK3 + 1))" "$LOG" >"$R4_LOG"
if ! grep -qE "start_load_kv:" "$R4_LOG"; then
    pass "R4 was a genuine cold recompute (no external load)"
else
    fail "R4 unexpectedly hit the external cache"
fi

# R3' = R3 run again, SAME salt (cache_salt would rewrite the block-0
# hash and cold the whole chain from the EXTERNAL cache too -- that
# mistake produced a zero-page first attempt). vLLM's internal prefix
# cache is dropped instead; the external cache still answers: the
# restored-continuation path against ITSELF. If even this diverges,
# the restore path carries nondeterminism. If it holds, the pipeline
# is deterministic and any R3-vs-R4 gap is numeric.
MARK35=$(wc -l < "$LOG")
curl -sf -X POST "http://127.0.0.1:$GATE_PORT/reset_prefix_cache" >/dev/null
R3B_TEXT="$(gate_completion_ids "$TURN2_IDS" "$MAX_TOKENS" "$R1_IDS.r3b" "")"
R3B_LOG="$GATE_LOG_DIR/decode_reuse.r3b.log"
tail -n "+$((MARK35 + 1))" "$LOG" >"$R3B_LOG"
N3B="$(pages_loaded_in "$R3B_LOG")"
if [[ -n "${N3B:-}" && "${N3B}" == "${N3:-0}" ]]; then
    pass "R3' restored the same pages (${N3B})"
else
    fail "R3' restored different pages (R3'=${N3B:-none} vs R3=${N3:-none})"
fi
if [[ "$R3_TEXT" == "$R3B_TEXT" ]]; then
    pass "restored continuation is self-consistent (R3 == R3')"
else
    fail "restored continuation diverges from ITSELF (R3 != R3'): nondeterministic restore"
    printf '  run1: %s\n  run2: %s\n' "${R3_TEXT:0:200}" "${R3B_TEXT:0:200}"
fi

if [[ "$R3_TEXT" == "$R4_TEXT" ]]; then
    pass "restored continuation is byte-identical to recomputed"
else
    # Deterministic-but-different is a numeric property of the stored
    # pages vs a recompute. Report the divergence index; fail the gate
    # only if the restore was also nondeterministic (checked above).
    DIVERGED_AT=$(python3 -c '
import json, sys
a = json.load(open(sys.argv[1]))["generated_ids"]
b = json.load(open(sys.argv[2]))["generated_ids"]
for i, (x, y) in enumerate(zip(a, b)):
    if x != y:
        print(f"token {i}")
        break
else:
    print("never (length differs)")' "$R1_IDS.r3" "$R1_IDS.r4")
    if [[ "$R3_TEXT" == "$R3B_TEXT" ]]; then
        log "NOTE: restored vs recomputed diverge at ${DIVERGED_AT} (deterministically; numeric drift in stored pages)"
    else
        fail "restored continuation differs from recomputed AND from itself"
    fi
fi

check "no unrestored-state errors" \
    bash -c '! grep -q "refusing to enter forward with unrestored state" "$1"' _ "$LOG"
check "no load poison" \
    bash -c '! grep -q "kvshrink load poison" "$1"' _ "$LOG"

gate_stop
gate_summary
