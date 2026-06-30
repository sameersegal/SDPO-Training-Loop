"""Pull a W&B run's full step history to CSV (for offline canary/metric analysis).

Usage:
  PYTHONPATH=src python src/fetch_wandb_history.py --run 8281dbd7 \
      --out reports/iteration-05/data/train_history_8281dbd7.csv
"""
import argparse
import os
import sys

from _paths import load_env

load_env()

import wandb


def resolve_run(api, entity, project, run_id):
    # Try direct id first; fall back to scanning by id-prefix / display name.
    try:
        return api.run(f"{entity}/{project}/{run_id}")
    except Exception:
        pass
    for r in api.runs(f"{entity}/{project}"):
        if r.id == run_id or r.id.startswith(run_id) or r.name == run_id:
            return r
    raise SystemExit(f"run {run_id!r} not found in {entity}/{project}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="8281dbd7")
    ap.add_argument("--entity", default="sameersegal-personal")
    ap.add_argument("--project", default="sdpo-gemma-ojbench")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    api = wandb.Api()
    run = resolve_run(api, args.entity, args.project, args.run)
    print(f"[fetch] run={run.id} name={run.name!r} state={run.state}")

    # Full, unsampled history; scan_history streams every logged step.
    rows = list(run.scan_history())
    if not rows:
        print("[fetch] no history rows", file=sys.stderr)
        return
    # Union of all keys, stable order: _step first, then sorted.
    keys = set()
    for row in rows:
        keys.update(row.keys())
    cols = ["_step"] + sorted(k for k in keys if k != "_step")

    import csv
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[fetch] wrote {len(rows)} rows x {len(cols)} cols -> {args.out}")
    print(f"[fetch] columns: {cols}")


if __name__ == "__main__":
    main()
