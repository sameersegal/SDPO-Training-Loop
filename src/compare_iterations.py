#!/usr/bin/env python3
"""Overlay training metrics ACROSS iterations from their committed CSVs.

Reads reports/iteration-*/data/train_metrics_*.csv (per iteration, the run with the
most steps is used as that iteration's representative) and overlays loss, completion
length, reward, and success-group-fraction so progress is comparable iteration by
iteration. Writes to reports/comparison/figures/. Depends only on committed data —
no raw logs needed.

  python compare_iterations.py
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _paths import repo_root

ROOT = repo_root()
REPORTS = ROOT / "reports"
OUT = REPORTS / "comparison" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = [
    ("loss", "SDPO loss (top-100 KL)"),
    ("mean_length", "mean completion length (tokens)"),
    ("reward_mean", "reward mean"),
    ("success_group_fraction", "success group fraction"),
]


def representative_csv(itdir):
    """The longest train_metrics_*.csv in an iteration = its headline run."""
    csvs = list((itdir / "data").glob("train_metrics_*.csv"))
    if not csvs:
        return None, None
    p = max(csvs, key=lambda f: sum(1 for _ in open(f)))
    cols = {}
    for row in csv.DictReader(open(p)):
        for k, v in row.items():
            cols.setdefault(k, []).append(float(v) if v not in ("", None) else 0.0)
    return p.stem.replace("train_metrics_", ""), cols


def main():
    iters = sorted(d for d in REPORTS.glob("iteration-*") if (d / "data").is_dir())
    runs = {}
    for d in iters:
        run_label, cols = representative_csv(d)
        if cols:
            runs[f"{d.name} ({run_label})"] = cols
    if not runs:
        print("no iteration train_metrics_*.csv found")
        return
    print(f"comparing {len(runs)} iteration(s): {', '.join(runs)}")

    # individual overlays
    for col, ylabel in METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for label, c in runs.items():
            if col in c:
                ax.plot(c["step"], c[col], "-o", ms=3, lw=1.6, label=label)
        ax.set_xlabel("training step"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — across iterations")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        p = OUT / f"compare_{col}.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        print("  wrote", p.relative_to(ROOT))

    # combined 2x2 dashboard
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (col, ylabel) in zip(axes.flat, METRICS):
        for label, c in runs.items():
            if col in c:
                ax.plot(c["step"], c[col], "-o", ms=2.5, lw=1.4, label=label)
        ax.set_xlabel("step"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
        ax.set_title(ylabel)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("Training metrics across iterations", fontsize=13)
    p = OUT / "compare_dashboard.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    print("  wrote", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
