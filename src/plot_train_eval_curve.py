"""train==eval 3-point generalization curve: seen-train / unseen-train / held-out.

Buckets the per-problem results of a base+adapter train eval into seen vs unseen
(per reports/iteration-05/data/train_eval_split.json), recomputes pass@k per bucket,
and pairs with the held-out easy+medium eval -> a 3-point generalization gradient.
"""
import argparse
import json
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [1, 2, 4, 8]


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def agg(results, ids):
    rows = [r for r in results if r["id"] in ids]
    out = {"n_problems": len(rows)}
    for k in KS:
        out[f"pass@{k}"] = sum(pass_at_k(r["n"], r["n_ac"], k) for r in rows) / len(rows) if rows else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-train", required=True)
    ap.add_argument("--adapter-train", required=True)
    ap.add_argument("--base-heldout", required=True)
    ap.add_argument("--adapter-heldout", required=True)
    ap.add_argument("--split", default="reports/iteration-05/data/train_eval_split.json")
    ap.add_argument("--seen-key", default="fast_seen6")
    ap.add_argument("--unseen-key", default="fast_unseen6")
    ap.add_argument("--adapter-label", default="SDPO+critic ckpt-20")
    ap.add_argument("--title", default="train==eval generalization — Qwen3-8B, 20 steps (Python, easy+medium)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sp = json.load(open(args.split))
    seen, unseen = set(sp[args.seen_key]), set(sp[args.unseen_key])
    bt = json.load(open(args.base_train))["results"]
    at = json.load(open(args.adapter_train))["results"]
    bh = json.load(open(args.base_heldout))["summary"]["by_language"]["python"]["overall"]
    ah = json.load(open(args.adapter_heldout))["summary"]["by_language"]["python"]["overall"]

    points = [
        ("seen-train", agg(bt, seen), agg(at, seen)),
        ("unseen-train", agg(bt, unseen), agg(at, unseen)),
        ("held-out", bh, ah),
    ]
    # print table
    print(f"\n{'bucket':14s} {'n':>3s}  " + "  ".join(f"base@{k}/sdpo@{k}" for k in KS))
    for name, b, a in points:
        print(f"{name:14s} {b.get('n_problems','?'):>3}  " +
              "  ".join(f"{b[f'pass@{k}']:.2f}/{a[f'pass@{k}']:.2f}" for k in KS))

    # Figure: generalization gradient — pass@1 and pass@8 across the 3 buckets, base vs sdpo
    xs = list(range(len(points)))
    labels = [p[0] for p in points]
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    series = [
        ("base pass@1", [p[1]["pass@1"] for p in points], "#999999", "o-"),
        ("base pass@8", [p[1]["pass@8"] for p in points], "#555555", "o-"),
        (f"{args.adapter_label} pass@1", [p[2]["pass@1"] for p in points], "#7fb1d8", "s--"),
        (f"{args.adapter_label} pass@8", [p[2]["pass@8"] for p in points], "#1f77b4", "s--"),
    ]
    for lbl, ys, c, st in series:
        ax.plot(xs, ys, st, color=c, lw=2.4, ms=9, label=lbl)
        for xi, yv in zip(xs, ys):
            ax.annotate(f"{yv:.2f}", (xi, yv), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8.5, color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{l}\n(n={p[1].get('n_problems','?')})" for l, p in zip(labels, points)],
                       fontsize=11)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("pass@k  (private-test AC rate)")
    ax.set_title(args.title, fontsize=12.5, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9, ncol=2)
    ax.annotate("directly optimized  →  in-distribution  →  out-of-distribution",
                xy=(0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=9, color="#666666")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
