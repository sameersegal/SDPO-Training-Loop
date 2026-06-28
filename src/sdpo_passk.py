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
from _paths import load_env

ROOT = Path.cwd()  # outputs land in the CWD (run from runs/iteration-XX/)
load_env()  # WANDB_API_KEY etc. from repo-root .env (no-op in Modal)


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
    ap.add_argument("--limit", type=int, default=0,
                    help="cap to first N tasks (easy-first) for a cheap smoke; 0 = all")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--single-sample", action="store_true",
                    help="issue n separate n=1 requests instead of one n-sample request "
                         "(the GB10 hangs on high-concurrency multi-sample n>1 inference)")
    ap.add_argument("--system", default="none", choices=["cp_method", "expert", "none"],
                    help="system prompt for eval (match training: iteration-03 used cp_method)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    import sdpo_ojbench as _S
    SYSTEM = _S.SYSTEM_PROMPTS[args.system]
    ks = [int(x) for x in args.ks.split(",")]

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=2400)
    languages = args.languages.split(",")
    tasks = []
    for lang in languages:
        pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
        for pid in SPLITS["heldout"]:
            if pid in pmap:
                tasks.append((lang, pid, pmap[pid]))

    if args.limit:
        # easy-first so a small smoke still exercises the AC (not just NO_CODE) path
        rank = {"easy": 0, "medium": 1, "hard": 2}
        tasks.sort(key=lambda t: rank.get(DIFF_BY_ID[t[1]], 9))
        tasks = tasks[:args.limit]

    sys_msg = [{"role": "system", "content": SYSTEM}] if SYSTEM else []

    def _judge(text, pid, lang):
        # Eval needs only the VERDICT, so use early-exit (binary): a failing solution
        # stops at the smallest failing case and never reads the huge cases (some
        # held-out problems ship 100s of MB). Verdict is identical to fraction mode.
        _, verdict, _ = judge_completion(text, int(pid), which="private", language=lang,
                                         reward_mode="binary")
        return verdict

    def one(t):
        lang, pid, prompt = t
        msgs = sys_msg + [{"role": "user", "content": prompt}]
        verdicts = []
        if args.single_sample:
            # n separate n=1 requests — avoids the GB10 multi-sample (n>1) hang.
            for _ in range(args.n):
                r = client.chat.completions.create(
                    model=args.served_model, messages=msgs, temperature=args.temperature,
                    top_p=args.top_p, max_tokens=args.max_tokens, n=1)
                verdicts.append(_judge(r.choices[0].message.content or "", pid, lang))
        else:
            resp = client.chat.completions.create(
                model=args.served_model, messages=msgs, temperature=args.temperature,
                top_p=args.top_p, max_tokens=args.max_tokens, n=args.n)
            verdicts = [_judge(ch.message.content or "", pid, lang) for ch in resp.choices]
        n_ac = sum(v == "AC" for v in verdicts)
        return {"id": pid, "language": lang, "difficulty": DIFF_BY_ID[pid],
                "n": len(verdicts), "n_ac": n_ac, "verdicts": verdicts}

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
