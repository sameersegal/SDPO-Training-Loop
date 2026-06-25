#!/usr/bin/env python3
"""Fast held-out pass@1 eval via a vLLM OpenAI endpoint, reported per language.

Serve the model first (base or merged adapter) on :8000, then:
  python sdpo_eval_vllm.py --served-model google/gemma-4-E2B-it --tag base
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from sdpo_ojbench import (SPLITS, PROMPT_BY_ID, CPP_PROMPT_BY_ID, DIFF_BY_ID,
                          judge_completion)

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--served-model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--languages", default="python,cpp")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1200)
    languages = args.languages.split(",")
    tasks = []
    for lang in languages:
        pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
        for pid in SPLITS["heldout"]:
            if pid in pmap:
                tasks.append((lang, pid, pmap[pid]))

    def one(t):
        lang, pid, prompt = t
        resp = client.chat.completions.create(
            model=args.served_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature, max_tokens=args.max_tokens,
        )
        text = resp.choices[0].message.content or ""
        reward, verdict, _ = judge_completion(text, int(pid), which="private", language=lang)
        return {"id": pid, "language": lang, "difficulty": DIFF_BY_ID[pid],
                "verdict": verdict, "is_passed": verdict == "AC", "private_reward": reward,
                "completion_tokens": resp.usage.completion_tokens,
                "finish_reason": resp.choices[0].finish_reason}

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result(); results.append(r)
            print(f"  [{r['language']:<6}|{r['difficulty']:<6}] loj-{r['id']}: {r['verdict']} "
                  f"(reward={r['private_reward']:.2f}, ct={r['completion_tokens']})", flush=True)

    def rate(lang, d=None):
        sub = [r for r in results if r["language"] == lang and (d is None or r["difficulty"] == d)]
        return sum(r["is_passed"] for r in sub), len(sub)

    summary = {"tag": args.tag, "served_model": args.served_model, "by_language": {}}
    for lang in languages:
        e, m, o = rate(lang, "easy"), rate(lang, "medium"), rate(lang)
        summary["by_language"][lang] = {"easy_pass": f"{e[0]}/{e[1]}",
                                        "medium_pass": f"{m[0]}/{m[1]}", "overall_pass": f"{o[0]}/{o[1]}"}
    out = ROOT / f"sdpo_eval_{args.tag}.json"
    json.dump({"summary": summary, "results": sorted(results, key=lambda r: (r["language"], r["difficulty"], r["id"]))},
              open(out, "w"), indent=2)
    print("\n=== HELD-OUT SUMMARY (by language) ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
