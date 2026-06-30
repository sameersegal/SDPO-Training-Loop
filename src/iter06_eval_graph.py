"""iteration-06 pass@8 verdict: base vs checkpoint-8 on the 12-problem train==eval probe,
split into 6 seen + 6 unseen. Reads the two sdpo_passk_*_iter06probe.json files, computes
the unbiased pass@k per subset, and writes the verdict bar chart.

Usage: python src/iter06_eval_graph.py
"""
import glob
import json
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEN = {2132, 2361, 2420, 2423, 2595, 3896}
UNSEEN = {2297, 2415, 2594, 2602, 3387, 4001}


def pass_at_k(n, c, k):
    if k > n:
        k = n
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def subset_passk(results, ids, k=8):
    rs = [r for r in results if r["id"] in ids]
    if not rs:
        return None
    return sum(pass_at_k(r["n"], r["n_ac"], k) for r in rs) / len(rs)


def load(tag_glob):
    fs = glob.glob(tag_glob)
    if not fs:
        return None, None
    d = json.load(open(fs[0]))
    return d["results"], fs[0]


def main():
    all_js = glob.glob("sdpo_passk_*iter06probe*.json")
    base_f = [f for f in all_js if "base" in f]
    sdpo_f = [f for f in all_js if "base" not in f]
    if not base_f or not sdpo_f:
        print("missing result json(s); have:", all_js)
        return
    base_res = json.load(open(base_f[0]))["results"]
    sdpo_res = json.load(open(sdpo_f[0]))["results"]
    print(f"base={base_f[0]}  sdpo={sdpo_f[0]}")

    rows = []
    for label, ids in [("overall (12)", SEEN | UNSEEN), ("seen (6)", SEEN), ("unseen (6)", UNSEEN)]:
        b = subset_passk(base_res, ids)
        s = subset_passk(sdpo_res, ids)
        rows.append((label, b, s))
        print(f"{label:14s}  base pass@8={b:.3f}  ckpt-8 pass@8={s:.3f}  Δ={s-b:+.3f}")

    # also per-k overall for the table
    print("\nper-k overall:")
    for k in (1, 2, 4, 8):
        b = subset_passk(base_res, SEEN | UNSEEN, k)
        s = subset_passk(sdpo_res, SEEN | UNSEEN, k)
        print(f"  pass@{k}: base {b:.3f} -> ckpt-8 {s:.3f}  (Δ {s-b:+.3f})")

    labels = [r[0] for r in rows]
    base_v = [r[1] for r in rows]
    sdpo_v = [r[2] for r in rows]
    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.38
    ax.bar([i - w / 2 for i in x], base_v, w, label="base (Qwen3-8B)", color="slategray")
    ax.bar([i + w / 2 for i in x], sdpo_v, w, label="iter-06 ckpt-8", color="seagreen")
    ax.axhline(0.50, ls="--", color="crimson", alpha=0.6, label="iter-05 ckpt-20 (collapsed)")
    for i, (b, s) in enumerate(zip(base_v, sdpo_v)):
        ax.text(i - w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
        ax.text(i + w / 2, s + 0.01, f"{s:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("pass@8"); ax.set_ylim(0, 1.05)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_title("iter-06 train==eval pass@8: base vs ckpt-8 (the collapse test)\n"
                 "iter-05 ckpt-20 fell to 0.50 here; did iter-06 hold?", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = "reports/iteration-06/figures/iter06_passk_verdict.png"
    plt.savefig(out, dpi=120)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
