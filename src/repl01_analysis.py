#!/usr/bin/env python3
"""Replication-01 analysis: SDPO (w=1.0 + feedback) vs GRPO (w=0.0) on OJBench, in-domain.

Inputs (CWD = runs/replication-01/evaldata):
  repl01sdpo/sdpo_passk_repl01sdpo_{base,ckpt8..40}.json   judged in-domain evals (63 pids, n=8)
  repl01grpo/sdpo_passk_repl01grpo_{ckpt8..40}.json        (base shared from the sdpo prefix)
  rollouts_sdpo.jsonl                                       per-rollout training capture (SDPO arm)

Outputs (CWD): repl01_summary.json, figures repl01_passk_trajectory.png + repl01_lengths.png,
and a printed paired-bootstrap table. The paired bootstrap resamples PROBLEMS (n=63) and, for
each draw, compares per-problem sampled pass@1 (c/n) between two eval sets — the same-problems
pairing our iter-09/10 analyses used.
"""
import json
import sys
from pathlib import Path

import numpy as np

RNG = np.random.default_rng(0)
B = 10_000
STEPS = [8, 16, 24, 32, 40]


def per_problem_rate(path):
    d = json.load(open(path))
    out = {}
    for r in d["results"]:
        # results rows: {id, language, difficulty, n, n_ac, verdicts} (judged sdpo_passk schema)
        if r.get("language", "python") != "python":
            continue
        out[int(r["id"])] = r["n_ac"] / r["n"]
    return out


def paired_bootstrap(a, b):
    """mean(a-b) with a 95% CI over problem resamples. a,b: {pid: rate} on the same pool."""
    pids = sorted(set(a) & set(b))
    da = np.array([a[p] for p in pids])
    db = np.array([b[p] for p in pids])
    diff = da - db
    idx = RNG.integers(0, len(pids), size=(B, len(pids)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), len(pids)


def main():
    cwd = Path(".")
    base = per_problem_rate(cwd / "repl01sdpo/sdpo_passk_repl01sdpo_base.json")
    arms = {}
    for arm in ("sdpo", "grpo"):
        arms[arm] = {}
        for s in STEPS:
            p = cwd / f"repl01{arm}/sdpo_passk_repl01{arm}_ckpt{s}.json"
            if p.exists():
                arms[arm][s] = per_problem_rate(p)

    summary = {"base_pass1": float(np.mean(list(base.values()))), "steps": {}}
    print(f"base in-domain sampled pass@1 (n=8, 63 problems): {summary['base_pass1']:.3f}\n")
    print(f"{'step':>4} | {'SDPO':>6} {'Δvs base [95% CI]':>22} | {'GRPO':>6} {'Δvs base [95% CI]':>22} | {'SDPO−GRPO [95% CI]':>24}")
    for s in STEPS:
        row = {}
        line = f"{s:>4} |"
        for arm in ("sdpo", "grpo"):
            if s in arms[arm]:
                m = float(np.mean(list(arms[arm][s].values())))
                d, lo, hi, _ = paired_bootstrap(arms[arm][s], base)
                row[arm] = {"pass1": m, "delta_base": [d, lo, hi]}
                line += f" {m:6.3f} {d:+.3f} [{lo:+.3f},{hi:+.3f}] |"
            else:
                line += " " * 31 + "|"
        if s in arms["sdpo"] and s in arms["grpo"]:
            d, lo, hi, _ = paired_bootstrap(arms["sdpo"][s], arms["grpo"][s])
            row["sdpo_minus_grpo"] = [d, lo, hi]
            sig = " *" if lo > 0 or hi < 0 else ""
            line += f" {d:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}"
        summary["steps"][s] = row
        print(line)

    # SDPO training length trajectory (per-step mean from the rollout capture)
    lens = {}
    ro = cwd / "rollouts_sdpo.jsonl"
    if ro.exists():
        for linej in open(ro):
            r = json.loads(linej)
            lens.setdefault(r["step"], []).append(r["n_tokens"])
        summary["sdpo_len_by_step"] = {s: float(np.mean(v)) for s, v in sorted(lens.items())}

    json.dump(summary, open("repl01_summary.json", "w"), indent=1)
    print("\nwrote repl01_summary.json")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        for arm, c in (("sdpo", "tab:red"), ("grpo", "tab:blue")):
            xs = [s for s in STEPS if s in arms[arm]]
            ys = [float(np.mean(list(arms[arm][s].values()))) for s in xs]
            ax[0].plot(xs, ys, "o-", color=c, label=arm.upper())
        ax[0].axhline(summary["base_pass1"], ls="--", c="gray", label="base")
        ax[0].set(xlabel="training step", ylabel="in-domain sampled pass@1 (n=8)",
                  title="Replication-01: SDPO vs GRPO (63-problem train pool)")
        ax[0].legend(); ax[0].grid(alpha=0.3)
        if lens:
            xs = sorted(lens); ax[1].plot(xs, [np.mean(lens[s]) / 1000 for s in xs], "o-", c="tab:red")
            ax[1].set(xlabel="training step", ylabel="mean rollout length (k tokens)",
                      title="SDPO arm training lengths (no collapse = flat)")
            ax[1].grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("repl01_passk_trajectory.png", dpi=120)
        print("wrote repl01_passk_trajectory.png")
    except Exception as e:  # noqa: BLE001
        print(f"(figures skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
