"""Solvability probe -> learnability frontier band.

Samples the base model n times on each TRAIN problem, measures pass@1 and mean dense
reward (fraction of public cases passed), and classifies each problem:
  - saturated: pass@1 >= --sat (already solved -> nothing to teach)        -> drop
  - hopeless:  mean dense reward <= --floor AND pass@1 == 0 (model never even
               partially solves -> feedback can't be acted on)             -> drop
  - frontier:  everything in between (partial progress or occasional AC)   -> KEEP

Writes data/frontier_band.json = the kept pids (the iteration-02 training set).
Serve the base model on :8000 first (set OJB_SPLITS to choose NOI vs full).

  OJB_SPLITS=ojb_splits_full.json PYTHONPATH=src python src/solvability_probe.py --n 6
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import sdpo_ojbench as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--language", default="python")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--sat", type=float, default=0.875, help="pass@1 >= this => saturated (drop)")
    ap.add_argument("--floor", type=float, default=0.05, help="mean dense reward <= this & 0 AC => hopeless (drop)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out", default="data/frontier_band.json")
    args = ap.parse_args()

    pmap = S.CPP_PROMPT_BY_ID if args.language == "cpp" else S.PROMPT_BY_ID
    pids = [p for p in S.SPLITS["train"] if p in pmap]
    print(f"probing {len(pids)} train problems (OJB_SPLITS={os.environ.get('OJB_SPLITS','ojb_splits.json')}), "
          f"n={args.n}, {args.language}")
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=2400)

    def probe(pid):
        resp = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": pmap[pid]}],
            temperature=args.temperature, max_tokens=args.max_tokens, n=args.n)
        ac, rsum = 0, 0.0
        for ch in resp.choices:
            r, v, _ = S.judge_completion(ch.message.content or "", pid, which="public",
                                         language=args.language, reward_mode="fraction")
            ac += int(v == "AC")
            rsum += r
        return pid, ac / args.n, rsum / args.n  # pass@1, mean dense reward

    rows = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for f in as_completed([ex.submit(probe, p) for p in pids]):
            rows.append(f.result())
            pid, p1, dr = rows[-1]
            print(f"  loj-{pid} ({S.DIFF_BY_ID[pid]}): pass@1={p1:.2f} dense={dr:.2f}", flush=True)

    band, sat, hop = [], [], []
    for pid, p1, dr in rows:
        if p1 >= args.sat:
            sat.append(pid)
        elif dr <= args.floor and p1 == 0:
            hop.append(pid)
        else:
            band.append(pid)
    json.dump({"frontier_band": sorted(band), "saturated": sorted(sat), "hopeless": sorted(hop),
               "probe": {"n": args.n, "language": args.language, "sat": args.sat, "floor": args.floor},
               "per_problem": {str(p): {"pass@1": round(a, 3), "dense": round(d, 3)} for p, a, d in rows}},
              open(args.out, "w"), indent=2)
    from collections import Counter
    bc = Counter(S.DIFF_BY_ID[p] for p in band)
    print(f"\nFRONTIER BAND: {len(band)} kept  (by difficulty: {dict(bc)})")
    print(f"dropped: saturated {len(sat)}, hopeless {len(hop)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
