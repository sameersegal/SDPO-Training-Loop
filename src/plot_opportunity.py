#!/usr/bin/env python3
"""Opportunity graph(s) from a sdpo_passk_*.json — model-agnostic.

The "opportunity graph" is the pass@1 -> pass@8 jump by difficulty: it shows the
headroom SDPO's `use_successful_as_teacher` exploits (problems unsolved at the
first try but solvable within k tries become teacher signal). generate_slides.py
hardcodes the iteration-01 Gemma frontier; this plots the SAME story for ANY
model straight from a pass@k run's JSON.

  ITER=iteration-05 python src/plot_opportunity.py \
      --passk reports/iteration-05/data/sdpo_passk_qwen3_base.json --label "Qwen3-8B (base)"

Writes opportunity_gap_<lang>.png + frontier_<lang>.png to reports/<ITER>/figures/.
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import repo_root

ROOT = repo_root()
KS = [1, 2, 4, 8]
COLORS = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728",
          "overall": "#1f77b4"}
DIFFS = ["easy", "medium", "hard"]


def load_passk(path):
    """{lang: {diff: [pass@1, pass@2, pass@4, pass@8]}} from a pass@k JSON."""
    blob = json.load(open(path))
    summ = blob["summary"] if "summary" in blob else blob
    by_lang = summ["by_language"]
    out = {}
    for lang, cells in by_lang.items():
        out[lang] = {}
        for diff in list(cells.keys()):
            row = cells[diff]
            # only difficulties actually present (n_problems > 0) and with all ks
            if all(row.get(f"pass@{k}") is not None for k in KS):
                out[lang][diff] = [row[f"pass@{k}"] for k in KS]
    return out, summ


def fig_gap(data, lang, label, out_dir):
    """pass@1 vs pass@8 grouped bars by difficulty — THE opportunity graph."""
    diffs = [d for d in DIFFS if d in data[lang]]
    p1 = [100 * data[lang][d][0] for d in diffs]
    p8 = [100 * data[lang][d][3] for d in diffs]
    x = range(len(diffs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], p1, w, label="pass@1", color="#aec7e8")
    b8 = ax.bar([i + w / 2 for i in x], p8, w, label="pass@8", color="#1f77b4")
    for i in x:
        gap = p8[i] - p1[i]
        if gap > 0.5:
            ax.annotate(f"+{gap:.0f}", (i, p8[i] + 1.5), ha="center",
                        fontsize=10, color="#1f77b4", weight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(diffs)
    ax.set_ylabel("pass rate (%)")
    ax.set_title(f"The SDPO opportunity: pass@1 vs pass@8 — {label}, {lang}\n"
                 "gap = problems solvable with more tries = teacher signal")
    ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=9)
    ax.bar_label(b8, fmt="%.0f", padding=2, fontsize=9)
    ax.legend()
    # match the iteration-01 Gemma chart: a clean fixed ceiling with ~10pp headroom
    # over the tallest bar (there, 60 -> 70), not auto-scaled to ~100.
    import math
    ax.set_ylim(0, math.ceil(max(p8 + [10]) / 10) * 10 + 10)
    fig.tight_layout()
    p = out_dir / f"opportunity_gap_{lang}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_frontier(data, lang, label, out_dir):
    """pass@k vs k by difficulty — the frontier curve."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for diff in DIFFS + ["overall"]:
        if diff in data[lang]:
            ax.plot(KS, [100 * v for v in data[lang][diff]], "o-",
                    color=COLORS[diff], lw=2.2, ms=7, label=diff)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_xlabel("k (attempts)")
    ax.set_ylabel("pass@k  (%)")
    ax.set_title(f"pass@k frontier — {label}, held-out {lang}\n"
                 "the pass@1→pass@8 gap is what SDPO exploits")
    ax.grid(True, alpha=0.3)
    ax.legend(title="difficulty")
    fig.tight_layout()
    p = out_dir / f"frontier_{lang}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passk", required=True, help="path to sdpo_passk_*.json")
    ap.add_argument("--label", default="base model", help="model label for titles")
    args = ap.parse_args()
    iter_name = os.environ.get("ITER", "iteration-05")
    out_dir = ROOT / "reports" / iter_name / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data, summ = load_passk(ROOT / args.passk if not os.path.isabs(args.passk) else args.passk)
    made = []
    for lang in data:
        if not data[lang]:
            continue
        made.append(fig_gap(data, lang, args.label, out_dir))
        made.append(fig_frontier(data, lang, args.label, out_dir))
    # persist the tidy {lang:{diff:[passk]}} alongside the figures
    json.dump({"label": args.label, "served_model": summ.get("served_model"),
               "n_samples": summ.get("n_samples"), "max_tokens": summ.get("max_tokens"),
               "by_language": data},
              open(out_dir / "passk_opportunity.json", "w"), indent=2)
    for p in made:
        print("  wrote", os.path.relpath(p, ROOT))
    if not made:
        print("[plot] no complete difficulty rows found in", args.passk)


if __name__ == "__main__":
    main()
