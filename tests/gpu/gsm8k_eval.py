#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""GSM8K accuracy probe for the GPU gates.

Stdlib-only runner, so the gate needs no evalscope/datasets install.
The dataset is the official GSM8K test split (one JSON per line:
{"question": ..., "answer": "... #### 42"}).

Prompts are 8-shot chain-of-thought. The shots are not pedagogy: a
hybrid (GDN/Mamba) model saves and restores KV in blocks of several
hundred tokens (528 on the gate model), and a bare GSM8K question is
~100 tokens -- shorter than one block, so nothing would ever be saved
or restored. The shared few-shot prefix makes every prompt long enough
to cross block boundaries, and cross-request reuse of a long shared
prefix is exactly the scenario the external cache exists for.

Subcommands:

  run      Ask the engine the first --limit questions with greedy
           decoding and write per-question results to --out.
  compare  Compare a cold and a warm results file: the warm accuracy
           must stay within --tolerance of the cold accuracy, >=90% of
           the warm questions must have hit the external cache, and the
           cold pass must clear --floor. Answer flips are REPORTED but
           not asserted: restoring a GDN/Mamba prefix and recomputing
           the tail is mathematically exact but numerically different
           from scanning the whole prompt in one chunk (the chunked
           scan accumulates in a different order), and greedy decoding
           amplifies any numeric difference into divergent text. A
           corrupted restore still fails: garbage KV collapses accuracy
           far below any tolerance.

Thinking mode is disabled (chat_template_kwargs enable_thinking=false):
the gate measures cache correctness, not reasoning endurance, and
multi-thousand-token thinking traces both truncate at the generation
cap and maximise chaos-driven divergence.
"""

import argparse
import concurrent.futures as cf
import json
import re
import sys
import urllib.request

GOLD_SEP = "####"
NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# Fixed CoT examples used as a shared prefix for the cache probe.
SHOTS = """\
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
Answer: Janet sells 16 - 3 - 4 = 9 duck eggs a day. She makes 9 * 2 = $18 every day at the farmers' market. The answer is 18.

Question: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Answer: It takes 2 / 2 = 1 bolt of white fiber. So the total number of bolts is 2 + 1 = 3. The answer is 3.

Question: Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?
Answer: The value of the house increased by 150%, so the new value is 80,000 * 2.5 = $200,000. His profit is 200,000 - 80,000 - 50,000 = $70,000. The answer is 70000.

Question: James decides to run 3 sprints 3 times a week. He runs 60 meters each sprint. How many total meters does he run a week?
Answer: He runs 3 * 3 = 9 sprints a week. So he runs 9 * 60 = 540 meters a week. The answer is 540.

Question: Every day, Wendi feeds each of her chickens three cups of mixed chicken feed. She gives the chickens their feed in three separate meals. In the morning, she gives her flock of chickens 15 cups of feed. In the afternoon, she gives her chickens another 25 cups of feed. How many cups of feed does she need to give her chickens in the final meal of the day if the size of Wendi's flock is 20 chickens?
Answer: The flock needs 20 * 3 = 60 cups of feed per day. She already gave 15 + 25 = 40 cups. So the final meal needs 60 - 40 = 20 cups. The answer is 20.

Question: Kylar went to the store to buy glasses for his new apartment. One glass costs $5, but every second glass costs only 60% of the price. Kylar wants to buy 16 glasses. How much does he need to pay for them?
Answer: Every second glass costs 5 * 0.6 = $3. So a pair of glasses costs 5 + 3 = $8. Kylar wants 16 / 2 = 8 pairs. So he needs to pay 8 * 8 = $64. The answer is 64.

Question: Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?
Answer: Charleston has 20 * 4 = 80 sheep. Toulouse has 80 * 2 = 160 sheep. Together they have 20 + 80 + 160 = 260 sheep. The answer is 260.

Question: Carla is downloading a 200 GB file. Normally she can download 2 GB/minute, but 40% of the way through the download, Windows forces a restart to install updates, which takes 20 minutes. Then Carla has to restart the download from the beginning. How long does it take to download the file?
Answer: 40% of the file is 200 * 0.4 = 80 GB, which takes 80 / 2 = 40 minutes. After the restart she downloads the whole file in 200 / 2 = 100 minutes. The total time is 40 + 20 + 100 = 160 minutes. The answer is 160."""

PROMPT = SHOTS + "\n\nQuestion: {question}\nAnswer:"


def parse_number(text):
    """The last number in the text, or None. Commas are ignored."""
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    value = float(matches[-1].replace(",", ""))
    return int(value) if value.is_integer() else value


def parse_gold(answer):
    return parse_number(answer.split(GOLD_SEP)[-1])


def ask(port, model, question, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT.format(question=question)}],
        "temperature": 0,
        "seed": 0,
        "max_tokens": max_tokens,
        # Short direct CoT: thinking traces truncate at the cap and
        # amplify divergence; the gate measures the cache, not reasoning.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = json.load(r)
    choice = body["choices"][0]
    text = choice["message"]["content"] or ""
    details = body.get("usage", {}).get("prompt_tokens_details", {}) or {}
    return text, details.get("cached_tokens", 0) or 0, choice.get("finish_reason")


def cmd_run(args):
    questions = []
    with open(args.data) as f:
        for line in f:
            row = json.loads(line)
            questions.append((row["question"], parse_gold(row["answer"])))
            if len(questions) >= args.limit:
                break

    def one(item):
        i, (question, gold) = item
        try:
            text, cached, finish = ask(args.port, args.model, question,
                                       args.max_tokens)
            pred = parse_number(text)
        except Exception as exc:  # a failed request is a wrong answer
            print(f"[gsm8k] question {i} failed: {exc}", file=sys.stderr)
            text, cached, pred, finish = "", 0, None, "error"
        return {"i": i, "gold": gold, "pred": pred, "ok": pred == gold,
                "cached_tokens": cached, "finish_reason": finish,
                "text": text}

    with cf.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(one, enumerate(questions)))

    n_ok = sum(r["ok"] for r in results)
    truncated = sum(1 for r in results if r["finish_reason"] == "length")
    report = {
        "model": args.model,
        "n": len(results),
        "acc": n_ok / len(results) if results else 0.0,
        "hits": sum(1 for r in results if r["cached_tokens"] > 0),
        "cached_tokens": sum(r["cached_tokens"] for r in results),
        "truncated": truncated,
        "results": results,
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[gsm8k] {args.out}: n={report['n']} acc={report['acc']:.3f} "
          f"cache_hits={report['hits']}/{report['n']} "
          f"cached_tokens={report['cached_tokens']} truncated={truncated}")


def cmd_compare(args):
    with open(args.cold) as f:
        cold = json.load(f)
    with open(args.warm) as f:
        warm = json.load(f)
    if cold["n"] != warm["n"]:
        print(f"[gsm8k] question count differs: cold={cold['n']} warm={warm['n']}")
        return 1

    flips = sum(1 for c, w in zip(cold["results"], warm["results"])
                if c["pred"] != w["pred"])
    print(f"[gsm8k] cold acc={cold['acc']:.3f}  warm acc={warm['acc']:.3f}  "
          f"answer flips={flips}/{cold['n']}  warm hits={warm['hits']}/{warm['n']}  "
          f"floor={args.floor} tolerance={args.tolerance}")

    rc = 0
    if cold["acc"] < args.floor:
        print(f"[gsm8k] COLD accuracy {cold['acc']:.3f} below floor {args.floor} "
              "-- engine/model setup is broken, not the cache")
        rc = 1
    if warm["acc"] < cold["acc"] - args.tolerance:
        print(f"[gsm8k] WARM accuracy {warm['acc']:.3f} dropped more than "
              f"{args.tolerance} below cold {cold['acc']:.3f} "
              "-- restoring KV costs accuracy")
        rc = 1
    if warm["hits"] < 0.9 * warm["n"]:
        print(f"[gsm8k] only {warm['hits']}/{warm['n']} warm questions hit "
              "the external cache -- the comparison proves nothing")
        rc = 1
    return rc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--port", type=int, required=True)
    run.add_argument("--data", required=True)
    run.add_argument("--limit", type=int, default=50)
    run.add_argument("--parallel", type=int, default=1)
    run.add_argument("--max-tokens", type=int, default=1024)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run)

    comp = sub.add_parser("compare")
    comp.add_argument("--cold", required=True)
    comp.add_argument("--warm", required=True)
    comp.add_argument("--floor", type=float, default=0.5)
    comp.add_argument("--tolerance", type=float, default=0.05)
    comp.set_defaults(func=cmd_compare)

    args = parser.parse_args()
    return args.func(args) or 0


if __name__ == "__main__":
    sys.exit(main())
