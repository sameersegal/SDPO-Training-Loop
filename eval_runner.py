#!/usr/bin/env python3
"""Generic eval harness for gemma-4-E2B-it served behind an OpenAI-compatible
vLLM endpoint. Reports quality (accuracy) and throughput (tokens/sec).

Start with: python eval_runner.py --dataset gsm8k
Designed to extend to: gsm8k, math500, aime, humaneval, mbpp, livecodebench,
arc_challenge, zebralogic, strategyqa.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

# --------------------------------------------------------------------------
# Answer extraction helpers
# --------------------------------------------------------------------------
_NUM_RE = re.compile(r"-?\$?\d[\d,]*\.?\d*")


def _norm_num(s):
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        f = float(s)
        return int(f) if f.is_integer() else f
    except ValueError:
        return None


def extract_final_number(text):
    """Pull a final numeric answer. Prefer an explicit 'answer is X' / boxed,
    else fall back to the last number in the text."""
    if not text:
        return None
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        n = _norm_num(_NUM_RE.search(m[-1]).group()) if _NUM_RE.search(m[-1]) else None
        if n is not None:
            return n
    m = re.search(r"answer\s*(?:is|:)\s*\*{0,2}\$?(-?[\d,]+\.?\d*)", text, re.I)
    if m:
        return _norm_num(m.group(1))
    nums = _NUM_RE.findall(text)
    return _norm_num(nums[-1]) if nums else None


# --------------------------------------------------------------------------
# Dataset adapters: each returns (list_of_items, scorer_fn)
# item = {"id", "messages", "gold"}
# scorer_fn(prediction_text, gold) -> bool
# --------------------------------------------------------------------------
def load_gsm8k():
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    sys_prompt = (
        "You are a careful math problem solver. Reason step by step, then give "
        "the final numeric answer on the last line in the form 'The answer is X'."
    )
    items = []
    for i, row in enumerate(ds):
        gold = _norm_num(row["answer"].split("####")[-1])
        items.append(
            {
                "id": f"gsm8k-{i}",
                "messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": row["question"]},
                ],
                "gold": gold,
            }
        )

    def scorer(pred_text, gold):
        return extract_final_number(pred_text) == gold

    return items, scorer


ADAPTERS = {
    "gsm8k": load_gsm8k,
}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def run(args):
    items, scorer = ADAPTERS[args.dataset]()
    n_total = len(items)

    # deterministic 2% sample
    import random

    rng = random.Random(args.seed)
    k = max(1, round(n_total * args.sample_frac))
    if args.limit:
        k = min(k, args.limit)
    idxs = sorted(rng.sample(range(n_total), k))
    sample = [items[i] for i in idxs]
    print(
        f"[{args.dataset}] total={n_total}  sample={len(sample)} "
        f"({args.sample_frac*100:.1f}%)  concurrency={args.concurrency}  "
        f"max_tokens={args.max_tokens}"
    )

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")

    def one(item):
        t0 = time.perf_counter()
        kwargs = dict(
            model=args.model,
            messages=item["messages"],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
        resp = client.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0
        text = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason
        ct = resp.usage.completion_tokens
        pt = resp.usage.prompt_tokens
        ok = scorer(text, item["gold"])
        return {
            "id": item["id"],
            "ok": ok,
            "finish_reason": finish,
            "completion_tokens": ct,
            "prompt_tokens": pt,
            "latency": dt,
            "pred_tail": text[-160:].replace("\n", " "),
            "gold": item["gold"],
        }

    results = []
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(one, it): it for it in sample}
        for j, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if j % 5 == 0 or j == len(sample):
                acc = sum(x["ok"] for x in results) / len(results)
                print(f"  {j}/{len(sample)} done | running acc={acc:.3f}", flush=True)
    wall = time.perf_counter() - wall0

    n = len(results)
    correct = sum(r["ok"] for r in results)
    tot_completion = sum(r["completion_tokens"] for r in results)
    tot_prompt = sum(r["prompt_tokens"] for r in results)
    # aggregate decode throughput (concurrent serving)
    agg_tps = tot_completion / wall
    # mean single-request decode speed
    per_req_tps = sum(r["completion_tokens"] / r["latency"] for r in results) / n

    truncated = sum(1 for r in results if r["finish_reason"] == "length")
    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "n": n,
        "accuracy": correct / n,
        "correct": correct,
        "truncated_hit_max_tokens": truncated,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "wall_s": round(wall, 1),
        "total_completion_tokens": tot_completion,
        "total_prompt_tokens": tot_prompt,
        "aggregate_decode_tok_per_s": round(agg_tps, 1),
        "mean_per_request_decode_tok_per_s": round(per_req_tps, 1),
        "concurrency": args.concurrency,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(ADAPTERS))
    p.add_argument("--sample-frac", type=float, default=0.02)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="google/gemma-4-E2B-it")
    p.add_argument("--out", default="")
    args = p.parse_args()
    if not args.out:
        args.out = f"results_{args.dataset}.json"
    run(args)


if __name__ == "__main__":
    main()
