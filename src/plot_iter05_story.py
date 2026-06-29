"""Iteration-05 "story" figure: the held-out null next to the train==eval regression.

Left  : held-out pass@k overall, base vs ckpt-20 — looks like "no change".
Right : pass@8 across seen-train / unseen-train / held-out — reveals a clear
        train-distribution regression (mode collapse) the held-out eval missed.
"""
import argparse
import json
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [1, 2, 4, 8]
GREY, BLUE = "#555555", "#1f77b4"


def pass_at_k(n, c, k):
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def agg8(results, ids):
    rows = [r for r in results if r["id"] in ids]
    return sum(pass_at_k(r["n"], r["n_ac"], 8) for r in rows) / len(rows) if rows else None


def agg1(results, ids):
    rows = [r for r in results if r["id"] in ids]
    return sum(pass_at_k(r["n"], r["n_ac"], 1) for r in rows) / len(rows) if rows else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-train", required=True)
    ap.add_argument("--adapter-train", required=True)
    ap.add_argument("--base-heldout", required=True)
    ap.add_argument("--adapter-heldout", required=True)
    ap.add_argument("--split", default="reports/iteration-05/data/train_eval_split.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sp = json.load(open(args.split))
    seen, unseen = set(sp["fast_seen6"]), set(sp["fast_unseen6"])
    bt = json.load(open(args.base_train))["results"]
    at = json.load(open(args.adapter_train))["results"]
    bh = json.load(open(args.base_heldout))["summary"]["by_language"]["python"]["overall"]
    ah = json.load(open(args.adapter_heldout))["summary"]["by_language"]["python"]["overall"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # LEFT: held-out pass@k, base vs ckpt-20
    x = list(range(len(KS)))
    b = [bh[f"pass@{k}"] for k in KS]
    a = [ah[f"pass@{k}"] for k in KS]
    axL.plot(x, b, "o-", color=GREY, lw=2.6, ms=9, label="base")
    axL.plot(x, a, "s--", color=BLUE, lw=2.6, ms=9, label="SDPO+critic ckpt-20")
    for xi, (bv, av) in enumerate(zip(b, a)):
        axL.annotate(f"{bv:.2f}", (xi, bv), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=GREY)
        axL.annotate(f"{av:.2f}", (xi, av), textcoords="offset points", xytext=(0, -16), ha="center", fontsize=9, color=BLUE)
    axL.set_xticks(x); axL.set_xticklabels([f"pass@{k}" for k in KS])
    axL.set_ylim(-0.03, 1.03); axL.grid(True, alpha=0.3)
    axL.set_ylabel("pass@k  (private-test AC rate)")
    axL.set_title("Held-out (n=10):  looks like no change", fontsize=12.5, fontweight="bold")
    axL.legend(loc="lower right", fontsize=10, framealpha=0.9)

    # RIGHT: pass@8 across the 3 buckets
    pts = [("seen-train", seen, 6), ("unseen-train", unseen, 6), ("held-out", None, 10)]
    xs = list(range(3))
    base8 = [agg8(bt, seen), agg8(bt, unseen), bh["pass@8"]]
    sdpo8 = [agg8(at, seen), agg8(at, unseen), ah["pass@8"]]
    base1 = [agg1(bt, seen), agg1(bt, unseen), bh["pass@1"]]
    sdpo1 = [agg1(at, seen), agg1(at, unseen), ah["pass@1"]]
    axR.plot(xs, base8, "o-", color=GREY, lw=2.8, ms=10, label="base pass@8")
    axR.plot(xs, sdpo8, "s--", color=BLUE, lw=2.8, ms=10, label="ckpt-20 pass@8")
    axR.plot(xs, base1, "o-", color="#aaaaaa", lw=1.8, ms=7, label="base pass@1")
    axR.plot(xs, sdpo1, "s--", color="#7fb1d8", lw=1.8, ms=7, label="ckpt-20 pass@1")
    for xi, (bv, av) in enumerate(zip(base8, sdpo8)):
        axR.annotate(f"{bv:.2f}", (xi, bv), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9.5, color=GREY, fontweight="bold")
        axR.annotate(f"{av:.2f}", (xi, av), textcoords="offset points", xytext=(0, -17), ha="center", fontsize=9.5, color=BLUE, fontweight="bold")
    # shade the train-distribution regression gap at seen/unseen
    for xi in (0, 1):
        axR.annotate("", xy=(xi, base8[xi]), xytext=(xi, sdpo8[xi]),
                     arrowprops=dict(arrowstyle="<->", color="#cc3333", lw=1.6))
    axR.text(0.5, 0.66, "−0.33 pass@8\n(diversity collapse)", color="#cc3333",
             fontsize=10, ha="center", fontweight="bold")
    axR.set_xticks(xs)
    axR.set_xticklabels([f"{n}\n(n={c})" for n, _, c in pts], fontsize=11)
    axR.set_ylim(-0.03, 1.03); axR.grid(True, alpha=0.3, axis="y")
    axR.set_title("train==eval probe:  regression on the trained distribution", fontsize=12.5, fontweight="bold")
    axR.legend(loc="lower left", fontsize=9, framealpha=0.9, ncol=2)
    axR.annotate("directly optimized  →  in-distribution  →  out-of-distribution",
                 xy=(0.5, -0.16), xycoords="axes fraction", ha="center", fontsize=9, color="#666666")

    fig.suptitle("Qwen3-8B, SDPO+critic, 20 steps (Python, easy+medium): held-out missed a real regression",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
