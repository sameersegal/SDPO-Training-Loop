"""Diagnose HOW base gemma fails on hard OJBench problems.

Generates a few base rollouts per hard problem, judges each with the DENSE
(fraction) reward so we see partial credit + the first failing case, and writes a
human-readable markdown report (raw model output + extracted code + verdict +
judge feedback). Use it to (a) find which hards are *reachable* (high partial
credit / occasional AC) vs hopeless, and (b) read where the model errs so we can
condition the prompt.

Serve base first (bump context for hard problems):
  vllm serve google/gemma-4-E2B-it --port 8000 --dtype bfloat16 \
    --max-model-len 16384 --gpu-memory-utilization 0.85 --max-num-seqs 64
Then:
  PYTHONPATH=src python src/diagnose_hard.py --n 4 --out reports/iteration-03/data/hard_rollouts.md
  PYTHONPATH=src python src/diagnose_hard.py --pids 2356,2085,2442 --n 6
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import sdpo_ojbench as S
import sdpo_prompts as P  # noqa: F401  (kept for optional teacher-reprompt rendering)

EXPERT_SYS = ("You are an expert competitive programmer. First reason briefly about the "
              "algorithm and its time complexity given the input limits, then write a single "
              "correct, efficient solution that reads from stdin and writes to stdout in the "
              "exact required format.")


def clip(s, n=2400):
    s = s or ""
    return s if len(s) <= n else s[:n] + f"\n...<+{len(s)-n} chars clipped>"


def select_hard_pids(k):
    """Smallest-test-case train-hard problems first (cheap to read + judge)."""
    full = json.load(open("data/ojb_splits_full.json"))
    by, part = full["by_id"], full["part_by_id"]
    tr = [int(x) for x in full["train"]]
    hard = [p for p in tr if by[str(p)] == "hard"]
    sized = []
    for p in hard:
        try:
            pub, _ = S.public_private_cases(p)
            smin = min(c[0].stat().st_size for c in pub) if pub else 1 << 30
            sized.append((p, smin, part[str(p)]))
        except Exception:
            continue
    sized.sort(key=lambda r: r[1])
    return [p for p, _, _ in sized[:k]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--pids", default=None, help="comma list; else auto-select small-case train-hard")
    ap.add_argument("--k-problems", type=int, default=6, help="how many hard problems if auto-selecting")
    ap.add_argument("--n", type=int, default=4, help="rollouts per problem")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--concurrency", type=int, default=2, help="keep low: GB10 hangs on high-concurrency n>1")
    ap.add_argument("--expert-sys", action="store_true", help="prepend the expert system prompt")
    ap.add_argument("--language", default="python", choices=["python", "cpp"])
    ap.add_argument("--which", default="private", choices=["public", "private"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out", default="reports/iteration-03/data/hard_rollouts.md")
    args = ap.parse_args()

    pids = [int(x) for x in args.pids.split(",")] if args.pids else select_hard_pids(args.k_problems)
    pmap = S.CPP_PROMPT_BY_ID if args.language == "cpp" else S.PROMPT_BY_ID
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=3600)
    print(f"diagnosing {len(pids)} hard problems {pids}  n={args.n} which={args.which} expert_sys={args.expert_sys}")

    def gen(pid):
        sys = [{"role": "system", "content": EXPERT_SYS}] if args.expert_sys else []
        msgs = sys + [{"role": "user", "content": pmap[pid]}]
        resp = client.chat.completions.create(model=args.model, messages=msgs,
                                              temperature=args.temperature, max_tokens=args.max_tokens, n=args.n)
        rolls = []
        for ch in resp.choices:
            txt = ch.message.content or ""
            reward, verdict, feedback = S.judge_completion(
                txt, pid, which=args.which, language=args.language, reward_mode="fraction")
            rolls.append({"text": txt, "reward": reward, "verdict": verdict, "feedback": feedback,
                          "finish": ch.finish_reason})
        return pid, rolls

    out = {}
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for fut in as_completed([ex.submit(gen, p) for p in pids]):
            pid, rolls = fut.result()
            out[pid] = rolls
            best = max(r["reward"] for r in rolls)
            vd = Counter(r["verdict"] for r in rolls)
            print(f"  loj-{pid:<7} best_reward={best:.2f}  verdicts={dict(vd)}", flush=True)

    # ---- markdown report ----
    lines = ["# Base gemma on hard OJBench — rollout diagnosis", "",
             f"model `{args.model}` · n={args.n} · temp={args.temperature} · judged on **{args.which}** cases "
             f"· dense (fraction) reward · expert_sys={args.expert_sys}", "",
             "`reward` = fraction of test cases passed (1.00 = AC). `best` per problem tells you how "
             "*reachable* it is: high partial credit or occasional AC = a prompt/curriculum can plausibly "
             "flip it; flat 0.00 with a wrong algorithm = base can't get there.", "",
             "## Summary (sorted by reachability)", "",
             "| pid | difficulty | part | best reward | verdicts |", "|---|---|---|---|---|"]
    rows = []
    for pid in pids:
        rolls = out[pid]
        best = max(r["reward"] for r in rolls)
        vd = dict(Counter(r["verdict"] for r in rolls))
        rows.append((best, pid, vd))
    for best, pid, vd in sorted(rows, reverse=True):
        rows_part = S.PART_BY_ID.get(pid, "?") if hasattr(S, "PART_BY_ID") else "?"
        lines.append(f"| loj-{pid} | {S.DIFF_BY_ID[pid]} | {rows_part} | {best:.2f} | {vd} |")
    lines.append("")

    for best, pid, vd in sorted(rows, reverse=True):
        rolls = out[pid]
        lines += [f"## loj-{pid}  (best reward {best:.2f}, verdicts {vd})", "",
                  "<details><summary>task prompt</summary>", "", "```", clip(pmap[pid], 2000), "```", "", "</details>", ""]
        for i, r in enumerate(sorted(rolls, key=lambda r: -r["reward"])):
            lines += [f"### rollout {i} — verdict **{r['verdict']}**  reward **{r['reward']:.2f}**  "
                      f"(finish={r['finish']})", "",
                      "judge feedback:", "", "```", clip(r["feedback"], 700), "```", "",
                      "model output:", "", "````", clip(r["text"], 2600), "````", ""]

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w").write("\n".join(lines))
    json.dump({str(p): out[p] for p in pids}, open(args.out.replace(".md", ".json"), "w"), indent=2)
    print(f"\nwrote {args.out}  (+ .json)")


if __name__ == "__main__":
    main()
