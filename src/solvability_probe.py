"""Solvability probe -> learnability frontier band (VARIANCE-based).

Samples the base/current model n times on each TRAIN problem under the iteration-03
system prompt, judges with the dense (fraction) reward, and classifies each problem by
the **intra-group reward variance** — the exact quantity that gives GRPO/SDPO a
non-zero advantage:

  - frontier (KEEP): reward std > --min-std  → the rollouts disagree → live advantage.
  - saturated (drop): std ~ 0 and mean ~ 1   → all AC → no signal (and already solved).
  - hopeless  (drop): std ~ 0 and mean ~ 0   → passes ~no cases → no signal.
  - flat      (drop): std ~ 0 and 0<mean<1   → every attempt gets the SAME partial score
                       (e.g. always 5/10) → still zero variance → no signal.

This is broader than the old `0<pass@1<1` rule: it KEEPS never-AC-but-partial problems
(rewards like .2/.5/.8) that pass@1 would call hopeless, and it DROPS uniform-partial
problems pass@1 missed. See docs/EXPERIMENT.md §6a. Re-run each curriculum round (the
band MOVES as the policy improves — saturation is policy-relative).

Serve the base model on :8000 first (set OJB_SPLITS to choose NOI vs full).
  OJB_SPLITS=ojb_splits_full.json PYTHONPATH=src python src/solvability_probe.py --n 8
  ... --limit 6   # smoke: probe only the first 6 train problems
"""
import argparse
import json
import os
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import sdpo_ojbench as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--language", default="python")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=10240)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--system", default="cp_method", choices=["cp_method", "expert", "none"],
                    help="system prompt — MUST match training so solvability is measured under the same prompt")
    ap.add_argument("--min-std", type=float, default=1e-6,
                    help="reward std above this => frontier (keep). Default ~0 = keep any non-degenerate group; "
                         "raise (e.g. 0.05) to drop near-degenerate weak-signal problems.")
    ap.add_argument("--limit", type=int, default=None, help="probe only the first N train problems (smoke)")
    ap.add_argument("--pids", default=None, help="comma list of pids to probe instead of the split")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out", default="data/frontier_band.json")
    args = ap.parse_args()

    pmap = S.CPP_PROMPT_BY_ID if args.language == "cpp" else S.PROMPT_BY_ID
    if args.pids:
        pids = [int(x) for x in args.pids.split(",") if int(x) in pmap]
    else:
        pids = [p for p in S.SPLITS["train"] if p in pmap]
        if args.limit:
            pids = pids[:args.limit]
    system = S.SYSTEM_PROMPTS[args.system]
    print(f"probing {len(pids)} train problems (OJB_SPLITS={os.environ.get('OJB_SPLITS','ojb_splits.json')}), "
          f"n={args.n}, {args.language}, system={args.system}, min_std={args.min_std}")
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=2400)

    def probe(pid):
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": pmap[pid]}]
        resp = client.chat.completions.create(
            model=args.model, messages=msgs,
            temperature=args.temperature, max_tokens=args.max_tokens, n=args.n)
        rewards, ac = [], 0
        for ch in resp.choices:
            r, v, _ = S.judge_completion(ch.message.content or "", pid, which="public",
                                         language=args.language, reward_mode="fraction")
            rewards.append(r)
            ac += int(v == "AC")
        return pid, rewards, ac / len(rewards)

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for f in as_completed([ex.submit(probe, p) for p in pids]):
            pid, rewards, p1 = f.result()
            std = statistics.pstdev(rewards)
            mean = statistics.fmean(rewards)
            rows.append((pid, rewards, p1, mean, std))
            print(f"  loj-{pid} ({S.DIFF_BY_ID[pid]}): pass@1={p1:.2f} mean={mean:.2f} std={std:.3f}", flush=True)

    band, sat, hop, flat = [], [], [], []
    for pid, rewards, p1, mean, std in rows:
        if std > args.min_std:
            band.append(pid)
        elif mean >= 1.0 - 1e-6:
            sat.append(pid)
        elif mean <= 1e-6:
            hop.append(pid)
        else:
            flat.append(pid)  # uniform partial -> zero variance -> no signal

    json.dump({"frontier_band": sorted(band), "saturated": sorted(sat),
               "hopeless": sorted(hop), "flat": sorted(flat),
               "probe": {"n": args.n, "language": args.language, "system": args.system,
                         "min_std": args.min_std},
               "per_problem": {str(p): {"pass@1": round(p1, 3), "mean": round(m, 3),
                                        "std": round(s, 3), "rewards": [round(x, 3) for x in rw]}
                               for p, rw, p1, m, s in rows}},
              open(args.out, "w"), indent=2)
    from collections import Counter
    bc = Counter(S.DIFF_BY_ID[p] for p in band)
    print(f"\nFRONTIER BAND: {len(band)} kept  (by difficulty: {dict(bc)})")
    print(f"dropped: saturated {len(sat)}, hopeless {len(hop)}, flat-partial {len(flat)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
