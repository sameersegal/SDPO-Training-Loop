"""iteration-08 DEFINITIVE eval analysis: pass@8 (with bootstrap 95% CI) for base + iter
05/06/07/08 ckpts on the SAME 30 train==eval problems (n=12 -> graded pass@8). Resolves the
12-probe noise and gives the matched collapse->fix story. Writes the figure + prints a table.

Reads sdpo_passk_iters30_*.json (from eval_iterations) in the cwd or reports/iteration-08/data.
"""
import glob
import json
import os
from math import comb

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# tag substring -> (label, color, order)
MODELS = [
    ("iters30_base", "base", "slategray"),
    ("iteration05", "iter-05\n(collapse)", "crimson"),
    ("iter06fast", "iter-06\n(lr+warmup)", "darkorange"),
    ("iter07frontier", "iter-07\n(band n=4)", "seagreen"),
    ("iter08frontierv2", "iter-08\n(band v2)", "navy"),
]


def pak(n, c, k):
    k = min(k, n)
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def find(sub):
    for d in (".", "reports/iteration-08/data"):
        fs = [f for f in glob.glob(f"{d}/sdpo_passk_*.json") if sub in f]
        if fs:
            return fs[0]
    return None


def passk_ci(results, k=8, B=2000):
    per = np.array([pak(r["n"], r["n_ac"], k) for r in results])
    point = per.mean()
    rng = np.random.default_rng(0)
    boot = [per[rng.integers(0, len(per), len(per))].mean() for _ in range(B)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, lo, hi


def main():
    labels, points, los, his, cols = [], [], [], [], []
    print(f"{'model':>22} {'pass@1':>7} {'pass@8':>7} {'95% CI':>16} {'n':>4}")
    for sub, label, col in MODELS:
        f = find(sub)
        if not f:
            print(f"  {label!r}: MISSING ({sub})"); continue
        res = json.load(open(f))["results"]
        p8, lo, hi = passk_ci(res, 8)
        p1, _, _ = passk_ci(res, 1)
        labels.append(label); points.append(p8); los.append(lo); his.append(hi); cols.append(col)
        print(f"{label.replace(chr(10),' '):>22} {p1:>7.3f} {p8:>7.3f}  [{lo:.2f},{hi:.2f}]  {len(res):>4}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = range(len(labels))
    yerr = [[p - l for p, l in zip(points, los)], [h - p for p, h in zip(points, his)]]
    ax.bar(x, points, color=cols, alpha=0.85)
    ax.errorbar(x, points, yerr=yerr, fmt="none", ecolor="black", capsize=5, lw=1.5)
    for i, p in enumerate(points):
        ax.text(i, his[i] + 0.015, f"{p:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("train==eval pass@8  (30 problems, n=12)"); ax.set_ylim(0, 1.02)
    ax.set_title("Definitive pass@8 across iterations (95% bootstrap CI)\n"
                 "collapse → attenuate → revive policy gradient", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    os.makedirs("reports/iteration-08/figures", exist_ok=True)
    out = "reports/iteration-08/figures/iters30_passk_definitive.png"
    plt.tight_layout(); plt.savefig(out, dpi=120)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
