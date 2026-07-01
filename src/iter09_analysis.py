"""iteration-09 DOSE-RESPONSE analysis: mean SAMPLED pass@1 (primary; pass@8 saturates — see
docs/FINDINGS) across base + dose checkpoints, split train_eval vs heldout (data/eval_iter09.json),
with bootstrap 95% CI. Optionally overlays per-checkpoint ||ΔW|| (src/adapter_delta.py output) as the
dose axis. Writes the figure + prints a table.

Reads sdpo_passk_iter09dose_*.json from cwd or reports/iteration-09/data.
  python src/iter09_analysis.py                       # steps from filenames
  ITER=iteration-09 python src/iter09_analysis.py
"""
import glob
import json
import os
import re
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEARCH = (".", "reports/iteration-09/data")
EVALSET = None
for d in SEARCH:
    p = os.path.join(d, "..", "..", "data", "eval_iter09.json") if d != "." else "data/eval_iter09.json"
    if os.path.exists(p):
        EVALSET = json.load(open(p)); break
if EVALSET is None:
    EVALSET = json.load(open("data/eval_iter09.json"))
TRAIN = set(EVALSET["train_eval"])
HELD = set(EVALSET["heldout"])


def pak(n, c, k):
    k = min(k, n)
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def find(tag):
    # tag is the short form (base|ckptN); files are sdpo_passk_iter09dose_<tag>.json
    for d in SEARCH:
        fs = glob.glob(f"{d}/sdpo_passk_iter09dose_{tag}.json")
        if fs:
            return fs[0]
    return None


def ci(results, k=1, B=2000):
    if not results:
        return None
    per = np.array([pak(r["n"], r["n_ac"], k) for r in results])
    rng = np.random.default_rng(0)
    boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return per.mean(), lo, hi


def load(tag):
    f = find(tag)
    if not f: return None
    d = json.load(open(f))
    return d["results"] if "results" in d else None


def main():
    # discover checkpoints present (base + ckptN)
    tags = []
    for d in SEARCH:
        for f in glob.glob(f"{d}/sdpo_passk_iter09dose_*.json"):
            m = re.search(r"iter09dose_(base|ckpt\d+)\.json$", f)
            if m:
                tags.append(m.group(1))
    tags = sorted(set(tags), key=lambda t: -1 if t == "base" else int(t[4:]))
    if not tags:
        print("no sdpo_passk_iter09dose_*.json found (run eval_dose + judge_local first)"); return

    def step_of(t):
        return 0 if t == "base" else int(t[4:])

    rows = {"train_eval": [], "heldout": []}
    print(f"{'model':>10} {'subset':>10} {'pass@1':>14} {'pass@4':>8} {'pass@8':>8} {'n':>3}")
    for t in tags:
        res = load(t)
        if res is None:
            continue
        for name, idset in (("train_eval", TRAIN), ("heldout", HELD)):
            sub = [r for r in res if r["id"] in idset]
            c1 = ci(sub, 1)
            if c1 is None:
                continue
            p4 = ci(sub, 4)[0] if sub else None
            p8 = ci(sub, 8)[0] if sub else None
            rows[name].append((step_of(t), c1[0], c1[1], c1[2]))
            print(f"{t:>10} {name:>10} {c1[0]:>6.3f} [{c1[1]:.2f},{c1[2]:.2f}] "
                  f"{p4:>8.3f} {p8:>8.3f} {len(sub):>3}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    colors = {"train_eval": "navy", "heldout": "darkorange"}
    for name, pts in rows.items():
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        lo = [p[1] - p[2] for p in pts]; hi = [p[3] - p[1] for p in pts]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=4, lw=2,
                    color=colors[name], label=f"{name} (n_problems varies)")
        if xs and xs[0] == 0:  # base reference line for this subset
            ax.axhline(ys[0], ls="--", color=colors[name], alpha=0.35, lw=1)
    ax.set_xlabel("dose (training step; 0 = base)")
    ax.set_ylabel("mean sampled pass@1  (bootstrap 95% CI)")
    ax.set_title("iter-09 dose–response: does an applied frontier-band gradient raise pass@1?\n"
                 "(pass@1 = the resolvable, on-objective metric; pass@8 is saturated here)",
                 fontweight="bold", fontsize=11)
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3); ax.legend()
    os.makedirs("reports/iteration-09/figures", exist_ok=True)
    out = "reports/iteration-09/figures/iter09_dose_response.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
