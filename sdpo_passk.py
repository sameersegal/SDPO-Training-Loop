#!/usr/bin/env python3
"""pass@k on the OJBench held-out set via a vLLM endpoint.

Samples n completions per (problem, language) at temperature>0, judges each on
PRIVATE test cases, and reports the unbiased pass@k (Chen et al., 2021) for
k in {1,2,4,8} from a single generation pass — by language x difficulty.

Why: greedy pass@1 wobbles +-2/25 (vLLM batch nondeterminism); pass@k is far
more stable AND it is the metric SDPO's `use_successful_as_teacher` exploits
(the gap between pass@1 and pass@k = solvable-with-more-tries = what successful
rollouts teach the failing ones).

Serve the model first (base, or adapter via --enable-lora), then:
  python sdpo_passk.py --served-model google/gemma-4-E2B-it --tag base
  python sdpo_passk.py --served-model sdpo --tag sdpo   # adapter via --enable-lora
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from math import comb
from pathlib import Path

from openai import OpenAI

from sdpo_ojbench import (SPLITS, PROMPT_BY_ID, CPP_PROMPT_BY_ID, DIFF_BY_ID,
                          judge_completion)

ROOT = Path(__file__).parent

# load .env (WANDB_API_KEY)
_envf = ROOT / ".env"
if _envf.exists():
    for _line in _envf.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))


def pass_at_k(n, c, k):
    """Unbiased estimator: prob that >=1 of k draws (without replacement from n,
    of which c are correct) is correct."""
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=8, help="samples per problem")
    ap.add_argument("--ks", default="1,2,4,8", help="comma list of k to report")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--languages", default="python,cpp")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=2400)
    languages = args.languages.split(",")
    tasks = []
    for lang in languages:
        pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
        for pid in SPLITS["heldout"]:
            if pid in pmap:
                tasks.append((lang, pid, pmap[pid]))

    def one(t):
        lang, pid, prompt = t
        # n samples in a single request (vLLM shares the prefill across them)
        resp = client.chat.completions.create(
            model=args.served_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens, n=args.n,
        )
        verdicts = []
        n_ac = 0
        for ch in resp.choices:
            text = ch.message.content or ""
            _, verdict, _ = judge_completion(text, int(pid), which="private", language=lang)
            verdicts.append(verdict)
            n_ac += int(verdict == "AC")
        return {"id": pid, "language": lang, "difficulty": DIFF_BY_ID[pid],
                "n": len(resp.choices), "n_ac": n_ac, "verdicts": verdicts}

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"  [{r['language']:<6}|{r['difficulty']:<6}] loj-{r['id']}: "
                  f"{r['n_ac']}/{r['n']} AC", flush=True)

    # aggregate: mean pass@k over problems, by language x difficulty and overall
    difficulties = ["easy", "medium", "hard"]
    present = [d for d in difficulties if any(r["difficulty"] == d for r in results)]

    def agg(subset):
        if not subset:
            return {f"pass@{k}": None for k in ks} | {"n_problems": 0}
        row = {"n_problems": len(subset)}
        for k in ks:
            row[f"pass@{k}"] = round(
                sum(pass_at_k(r["n"], r["n_ac"], k) for r in subset) / len(subset), 4)
        return row

    summary = {"tag": args.tag, "served_model": args.served_model,
               "n_samples": args.n, "temperature": args.temperature,
               "max_tokens": args.max_tokens, "by_language": {}}
    flat = {}
    for lang in languages:
        lang_rows = [r for r in results if r["language"] == lang]
        summary["by_language"][lang] = {
            "overall": agg(lang_rows),
            **{d: agg([r for r in lang_rows if r["difficulty"] == d]) for d in present},
        }
        for k in ks:
            v = summary["by_language"][lang]["overall"][f"pass@{k}"]
            flat[f"passk/{args.tag}/{lang}/pass@{k}"] = v if v is not None else 0.0
    summary["overall"] = agg(results)

    out = ROOT / f"sdpo_passk_{args.tag}.json"
    json.dump({"summary": summary, "results": sorted(
        results, key=lambda r: (r["language"], r["difficulty"], r["id"]))},
        open(out, "w"), indent=2)
    print("\n=== pass@k SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")

    if args.wandb and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "sdpo-gemma-ojbench"),
                   name=f"passk-{args.tag}", job_type="eval", reinit=True)
        wandb.log(flat)
        wandb.finish()
        print("logged pass@k to wandb")


if __name__ == "__main__":
    main()
