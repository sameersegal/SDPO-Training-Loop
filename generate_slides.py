#!/usr/bin/env python3
"""Generate slide-ready graphs + data tables from the SDPO experiment results.

Re-runnable: picks up sdpo_passk_sdpo_python.json (the 100-step pass@k) if present
and adds the base-vs-100-step comparison panel. Writes PNGs + a data JSON to slides/.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
# Figures are tracked per iteration under reports/<ITER>/figures so progress is
# diffable iteration-by-iteration. Bump ITER for the next run.
ITER = os.environ.get("ITER", "iteration-01")
OUT = ROOT / "reports" / ITER / "figures"
OUT.mkdir(parents=True, exist_ok=True)
KS = [1, 2, 4, 8]

# Base model pass@k frontier (Modal H100, held-out, n=8, temp 0.8). Source of truth
# also saved to slides/passk_base.json for reproducibility.
BASE = {
    "python": {
        "easy":    [0.325, 0.4571, 0.5714, 0.60],
        "medium":  [0.15, 0.25, 0.3543, 0.40],
        "hard":    [0.0, 0.0, 0.0, 0.0],
        "overall": [0.095, 0.1414, 0.1851, 0.20],
    },
    "cpp": {
        "easy":    [0.30, 0.3786, 0.4857, 0.60],
        "medium":  [0.20, 0.3071, 0.3857, 0.40],
        "hard":    [0.0, 0.0, 0.0, 0.0],
        "overall": [0.10, 0.1371, 0.1743, 0.20],
    },
    "overall": [0.0975, 0.1393, 0.1797, 0.20],
}

COLORS = {"easy": "#2ca02c", "medium": "#ff7f0e", "hard": "#d62728",
          "overall": "#1f77b4"}

# Per-step training metrics live in reports/<ITER>/data/train_metrics_*.csv
# (committed). loss/length graphs rebuild from THESE, not the raw logs.
DATA = ROOT / "reports" / ITER / "data"


def load_metrics():
    """{label: {col: [values]}} from every train_metrics_*.csv in the data dir."""
    import csv
    runs = {}
    for p in sorted(DATA.glob("train_metrics_*.csv")):
        label = p.stem.replace("train_metrics_", "")
        cols = {}
        for row in csv.DictReader(open(p)):
            for k, v in row.items():
                cols.setdefault(k, []).append(float(v) if v not in ("", None) else 0.0)
        runs[label] = cols
    return runs


def fig_loss(runs):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
    for label, c in runs.items():
        a1.plot(c["step"], c["loss"], "-o", ms=3, lw=1.5, label=label)
        a2.plot(c["step"], c["grad_norm"], "-o", ms=3, lw=1.5, label=label)
    a1.set_ylabel("SDPO loss (top-100 KL)")
    a1.set_title("SDPO loss over training\nbounces with batch composition — NOT a convergence/quality signal")
    a1.grid(True, alpha=0.3); a1.legend(fontsize=8)
    a2.set_ylabel("grad norm"); a2.set_xlabel("training step"); a2.grid(True, alpha=0.3)
    fig.tight_layout(); p = OUT / "loss_curve.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def fig_length(runs):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, c in runs.items():
        ax.plot(c["step"], c["mean_length"], "-o", ms=4, lw=2, label=label)
    ax.set_xlabel("training step"); ax.set_ylabel("mean completion length (tokens)")
    ax.set_title("Completion length over training\n(iteration 01: collapsed ~3,500 -> ~900 tokens — overfitting)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout(); p = OUT / "length_collapse.png"; fig.savefig(p, dpi=150); plt.close(fig)
    return p


def load_sdpo_passk():
    """Return {lang: {diff: [pass@k...]}} for the 100-step adapter, or None."""
    for f in ["sdpo_passk_sdpo_python.json", "sdpo_passk_sdpo100.json"]:
        p = ROOT / f
        if p.exists():
            d = json.load(open(p))["summary"]["by_language"]
            out = {}
            for lang, cells in d.items():
                out[lang] = {diff: [cells[diff][f"pass@{k}"] for k in KS]
                             for diff in cells if diff != "overall"}
                out[lang]["overall"] = [cells["overall"][f"pass@{k}"] for k in KS]
            return out
    return None


def fig_frontier(lang="python"):
    """pass@k vs k by difficulty — the frontier + SDPO-opportunity story."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for diff in ["easy", "medium", "hard", "overall"]:
        ax.plot(KS, [100 * v for v in BASE[lang][diff]], "o-", color=COLORS[diff],
                lw=2.2, ms=7, label=diff)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_xlabel("k (attempts)")
    ax.set_ylabel("pass@k  (%)")
    ax.set_title(f"Base model pass@k frontier — held-out, {lang}\n"
                 "the pass@1→pass@8 gap is what SDPO exploits")
    ax.grid(True, alpha=0.3)
    ax.legend(title="difficulty")
    ax.set_ylim(-3, 70)
    fig.tight_layout()
    p = OUT / f"frontier_{lang}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_gap(lang="python"):
    """pass@1 vs pass@8 grouped bars — the opportunity gap."""
    diffs = ["easy", "medium", "hard"]
    p1 = [100 * BASE[lang][d][0] for d in diffs]
    p8 = [100 * BASE[lang][d][3] for d in diffs]
    x = range(len(diffs))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    b1 = ax.bar([i - w / 2 for i in x], p1, w, label="pass@1", color="#aec7e8")
    b8 = ax.bar([i + w / 2 for i in x], p8, w, label="pass@8", color="#1f77b4")
    for i, d in enumerate(diffs):
        gap = p8[i] - p1[i]
        if gap > 0:
            ax.annotate(f"+{gap:.0f}", (i, p8[i] + 1.5), ha="center",
                        fontsize=10, color="#1f77b4", weight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(diffs)
    ax.set_ylabel("pass rate (%)")
    ax.set_title(f"The SDPO opportunity: pass@1 vs pass@8 — held-out, {lang}\n"
                 "gap = problems solvable with more tries = teacher signal")
    ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=9)
    ax.bar_label(b8, fmt="%.0f", padding=2, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 70)
    fig.tight_layout()
    p = OUT / f"opportunity_gap_{lang}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def fig_compare(sdpo, lang="python"):
    """Base vs 100-step pass@k, by difficulty (only if sdpo data present)."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for diff, c in [("easy", COLORS["easy"]), ("medium", COLORS["medium"]),
                    ("overall", COLORS["overall"])]:
        ax.plot(KS, [100 * v for v in BASE[lang][diff]], "o--", color=c, lw=1.8,
                ms=6, alpha=0.6, label=f"base · {diff}")
        if lang in sdpo and diff in sdpo[lang]:
            ax.plot(KS, [100 * v for v in sdpo[lang][diff]], "s-", color=c, lw=2.4,
                    ms=7, label=f"100-step · {diff}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS)
    ax.set_xticklabels(KS)
    ax.set_xlabel("k (attempts)")
    ax.set_ylabel("pass@k  (%)")
    ax.set_title(f"Base vs 100-step SDPO — pass@k frontier, held-out {lang}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-3, 70)
    fig.tight_layout()
    p = OUT / f"compare_{lang}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def main():
    json.dump(BASE, open(OUT / "passk_base.json", "w"), indent=2)
    made = [fig_frontier("python"), fig_frontier("cpp"),
            fig_gap("python"), fig_gap("cpp")]
    runs = load_metrics()
    if runs:
        made += [fig_loss(runs), fig_length(runs)]
    else:
        print(f"[slides] no train_metrics_*.csv in {DATA} — skipping loss/length")
    sdpo = load_sdpo_passk()
    if sdpo:
        json.dump(sdpo, open(OUT / "passk_sdpo.json", "w"), indent=2)
        made.append(fig_compare("python", "python") if False else fig_compare(sdpo, "python"))
        if "cpp" in sdpo:
            made.append(fig_compare(sdpo, "cpp"))
        print("[slides] included 100-step comparison")
    else:
        print("[slides] 100-step pass@k not found yet — base-only graphs")
    for p in made:
        print("  wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
