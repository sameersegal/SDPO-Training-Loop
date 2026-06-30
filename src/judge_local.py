#!/usr/bin/env python3
"""Offline judge for a generate-only pass@k run.

Reads the per-sample completions JSONL that `sdpo_passk.py --no-judge` streamed to disk
(`sdpo_passk_<tag>_samples.jsonl`), judges every completion locally on the PRIVATE test
cases (binary verdict, the same `judge_completion` the cloud path uses), and writes the
canonical `sdpo_passk_<tag>.json` (summary + per-problem n/n_ac/verdicts) so downstream
analysis (`iters30_analysis.py`, `generate_slides.py`) is unchanged.

WHY this exists: judging is pure CPU (stdout-diff for python, g++ for cpp). Doing it on the
Modal H200 leaves a $/s GPU idle while it diffs test cases. Generate on the cloud GPU
(`sdpo_passk.py --no-judge`), pull the small samples JSONL, judge here on the GB10/local box
for free. Same verdicts, no idle GPU.

  # cloud (GPU busy generating):  python sdpo_passk.py ... --no-judge --tag iters30_base
  # local (free CPU):             python src/judge_local.py --tag iters30_base
"""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import comb
from pathlib import Path

from sdpo_ojbench import DIFF_BY_ID, judge_completion
from _paths import load_env

ROOT = Path.cwd()
load_env()


def pass_at_k(n, c, k):
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _judge_one(args):
    pid, lang, text = args
    try:
        _, verdict, _ = judge_completion(text, int(pid), which="private",
                                         language=lang, reward_mode="binary")
        return verdict
    except Exception as e:  # one bad completion must not sink the batch
        return f"ERR:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="matches sdpo_passk_<tag>_samples.jsonl")
    ap.add_argument("--samples", default="", help="explicit JSONL path (overrides --tag)")
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("JUDGE_WORKERS", "8")),
                    help="parallel judge processes (cpp g++ is CPU-bound; ~= cores)")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    samples_path = Path(args.samples) if args.samples else ROOT / f"sdpo_passk_{args.tag}_samples.jsonl"
    if not samples_path.exists():
        raise SystemExit(f"no samples file: {samples_path} (run sdpo_passk.py --no-judge first)")

    # group completions by (pid, language) — the JSONL has one row per sample
    by_key = {}
    langs = set()
    with open(samples_path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["id"], r["language"])
            by_key.setdefault(key, []).append(r.get("completion", ""))
            langs.add(r["language"])

    # flatten to (pid, lang, text) so judging parallelizes across ALL completions, not just problems
    flat_jobs, owner = [], []
    for (pid, lang), texts in by_key.items():
        for t in texts:
            flat_jobs.append((pid, lang, t)); owner.append((pid, lang))
    print(f"[judge_local] {len(by_key)} problems, {len(flat_jobs)} completions, "
          f"{args.workers} workers", flush=True)

    verdicts_by_key = {k: [] for k in by_key}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_judge_one, job): i for i, job in enumerate(flat_jobs)}
        for fut in as_completed(futs):
            i = futs[fut]
            verdicts_by_key[owner[i]].append(fut.result())
            done += 1
            if done % 50 == 0:
                print(f"  judged {done}/{len(flat_jobs)}", flush=True)

    results = []
    for (pid, lang), verdicts in verdicts_by_key.items():
        n_ac = sum(v == "AC" for v in verdicts)
        results.append({"id": pid, "language": lang, "difficulty": DIFF_BY_ID[pid],
                        "n": len(verdicts), "n_ac": n_ac, "verdicts": verdicts})

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

    summary = {"tag": args.tag, "judged_offline": True, "by_language": {}}
    flat = {}
    for lang in sorted(langs):
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
    print("\n=== pass@k SUMMARY (offline-judged) ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")

    if args.wandb and os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "sdpo-gemma-ojbench"),
                   name=f"passk-{args.tag}", job_type="eval", reinit=True)
        wandb.log(flat)
        wandb.finish()


if __name__ == "__main__":
    main()
