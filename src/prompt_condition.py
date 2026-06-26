"""Test whether prompt conditioning lifts base gemma on *reachable* hard problems
(the high-partial-credit WA ones identified by diagnose_hard.py).

Diagnosis (loj-2442): base writes essays hunting a closed-form instead of (a)
simulating the rule the problem states and (b) tiering by the Data Range table
(direct O(n) for small n, matrix-exp / cycle detection only for the 1e18 tail).
So we test system prompts that push "implement the stated rule, tier by input
size, output code" against base.

Serve base on :8000 first. Then:
  PYTHONPATH=src OJB_SPLITS=ojb_splits_full.json python src/prompt_condition.py \
    --pids 2442,900011 --n 6 --out reports/iteration-03/data/prompt_condition.md
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import sdpo_ojbench as S

# Variant A: the iteration-02 expert prompt (reason about complexity, then solve).
EXPERT_SYS = ("You are an expert competitive programmer. First reason briefly about the "
              "algorithm and its time complexity given the input limits, then write a single "
              "correct, efficient solution that reads from stdin and writes to stdout in the "
              "exact required format.")

# Variant B: targets the observed over-theorizing failure directly — implement the
# stated rule, tier by the Data Range table, stop searching for a closed-form.
CP_METHOD_SYS = (
    "You are an expert competitive programmer. Follow this method, then output only the final code:\n"
    "1. Restate the exact rule, recurrence, or process the problem defines — do NOT try to guess a "
    "closed-form pattern when the problem already states the rule. Implement the stated rule.\n"
    "2. Read the Data Range / Constraints table. Decide per input size which method is needed: a "
    "direct O(n) simulation is correct and sufficient for small n; only the largest limits "
    "(e.g. n up to 1e18) require a faster technique (matrix exponentiation, cycle/period detection, "
    "or a closed form).\n"
    "3. Write ONE solution that simulates directly for small n and switches to the faster method "
    "only when n is too large to loop. Apply the modulus throughout. Read stdin, write stdout in the "
    "exact format. Output only the code.")

VARIANTS = {"base": None, "expert_sys": EXPERT_SYS, "cp_method": CP_METHOD_SYS}


def clip(s, n=2200):
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n...<+{len(s)-n} chars clipped>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--pids", default="2442,900011")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=10240)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--language", default="python", choices=["python", "cpp"])
    ap.add_argument("--which", default="private", choices=["public", "private"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out", default="reports/iteration-03/data/prompt_condition.md")
    args = ap.parse_args()

    pids = [int(x) for x in args.pids.split(",")]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    print(f"prompt-conditioning A/B on {pids}  variants={list(VARIANTS)}  n={args.n}")

    def run(pid, vname):
        sys = VARIANTS[vname]
        msgs = ([{"role": "system", "content": sys}] if sys else []) + \
               [{"role": "user", "content": S.PROMPT_BY_ID[pid]}]
        resp = client.chat.completions.create(model=args.model, messages=msgs,
                                              temperature=args.temperature, max_tokens=args.max_tokens, n=args.n)
        rolls = []
        for ch in resp.choices:
            txt = ch.message.content or ""
            reward, verdict, feedback = S.judge_completion(
                txt, pid, which=args.which, language=args.language, reward_mode="fraction")
            rolls.append({"reward": reward, "verdict": verdict, "feedback": feedback, "text": txt})
        return pid, vname, rolls

    jobs = [(p, v) for p in pids for v in VARIANTS]
    out = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(run, p, v) for p, v in jobs]):
            pid, vname, rolls = fut.result()
            out[(pid, vname)] = rolls
            best = max(r["reward"] for r in rolls)
            mean = sum(r["reward"] for r in rolls) / len(rolls)
            ac = sum(r["verdict"] == "AC" for r in rolls)
            vd = dict(Counter(r["verdict"] for r in rolls))
            print(f"  loj-{pid} {vname:<11} best={best:.2f} mean={mean:.2f} AC={ac}/{len(rolls)} {vd}", flush=True)

    # markdown
    lines = ["# Prompt conditioning on reachable hard problems", "",
             f"n={args.n} · temp={args.temperature} · judged on **{args.which}** · dense reward", "",
             "| problem | variant | best | mean | AC | verdicts |", "|---|---|---|---|---|---|"]
    for pid in pids:
        for v in VARIANTS:
            rolls = out[(pid, v)]
            best = max(r["reward"] for r in rolls)
            mean = sum(r["reward"] for r in rolls) / len(rolls)
            ac = sum(r["verdict"] == "AC" for r in rolls)
            vd = dict(Counter(r["verdict"] for r in rolls))
            lines.append(f"| loj-{pid} | {v} | {best:.2f} | {mean:.2f} | {ac}/{len(rolls)} | {vd} |")
    # include the best rollout per (pid, best variant) for reading
    for pid in pids:
        bestv = max(VARIANTS, key=lambda v: max(r["reward"] for r in out[(pid, v)]))
        br = max(out[(pid, bestv)], key=lambda r: r["reward"])
        lines += ["", f"## loj-{pid} — best rollout (variant `{bestv}`, reward {br['reward']:.2f}, {br['verdict']})",
                  "", "judge feedback:", "", "```", clip(br["feedback"], 600), "```", "",
                  "model output:", "", "````", clip(br["text"], 2200), "````"]
    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines))
    json.dump({f"{p}|{v}": out[(p, v)] for p, v in jobs}, open(args.out.replace(".md", ".json"), "w"), indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
