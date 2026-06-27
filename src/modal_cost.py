"""Actual $ cost of Modal runs — from `modal billing report` (real billing, not napkin math).

Modal bills per-second for GPU + CPU + memory. This wraps `modal billing report
--show-resources`, parses the per-(app, resource) rows, and aggregates cost per app so
you can see what a run actually cost (and the GPU/CPU/Memory split).

  python src/modal_cost.py                         # per-app totals for today
  python src/modal_cost.py --for "this month"      # different window
  python src/modal_cost.py --this-run              # just the app in runs/iteration-NN/RUNNING_APP_ID.txt
  python src/modal_cost.py --app ap-GxyGoDs        # a specific app (full id or prefix)

Notes: billing reports "full intervals only", so an in-flight run undercounts the
current partial interval — re-run after it finishes for the final number. Range is
[start, end) like the underlying API.
"""
import argparse
import os
import subprocess
import sys
from collections import defaultdict


def _modal_bin():
    # prefer the modal next to the running interpreter (the venv), else PATH
    cand = os.path.join(os.path.dirname(sys.executable), "modal")
    return cand if os.path.exists(cand) else "modal"


def fetch_rows(for_range=None, start=None, end=None):
    cmd = [_modal_bin(), "billing", "report", "--show-resources"]
    if start:
        cmd += ["--start", start]
    if end:
        cmd += ["--end", end]
    if for_range and not (start or end):
        cmd += ["--for", for_range]
    env = dict(os.environ, COLUMNS="260")  # untruncated app ids
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if out.returncode != 0:
        sys.exit(f"modal billing report failed:\n{out.stderr or out.stdout}")
    rows = []
    for line in out.stdout.splitlines():
        if not line.lstrip().startswith("│"):
            continue
        cells = [c.strip() for c in line.strip().strip("│").split("│")]
        if len(cells) < 6 or not cells[0].startswith("ap-"):
            continue  # skip header / non-data rows
        app, desc, env_, interval, resource, cost = cells[:6]
        try:
            cost = float(cost)
        except ValueError:
            continue
        rows.append({"app": app, "desc": desc, "env": env_, "interval": interval,
                     "resource": resource, "cost": cost})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--for", dest="for_range", default="today",
                    help="report window: 'today', 'yesterday', 'this month', 'last month' (default today)")
    ap.add_argument("--start", default=None, help="inclusive start date YYYY-MM-DD (overrides --for)")
    ap.add_argument("--end", default=None, help="exclusive end date YYYY-MM-DD")
    ap.add_argument("--app", default=None, help="show one app only (full id or prefix)")
    ap.add_argument("--this-run", action="store_true",
                    help="use the app id in runs/iteration-*/RUNNING_APP_ID.txt")
    args = ap.parse_args()

    app_filter = args.app
    if args.this_run and not app_filter:
        import glob
        cands = sorted(glob.glob("runs/iteration-*/RUNNING_APP_ID.txt"))
        if not cands:
            sys.exit("no runs/iteration-*/RUNNING_APP_ID.txt found")
        app_filter = open(cands[-1]).readline().strip()
        print(f"[modal_cost] this-run app = {app_filter} (from {cands[-1]})")

    rows = fetch_rows(args.for_range, args.start, args.end)
    if app_filter:
        rows = [r for r in rows if r["app"].startswith(app_filter) or app_filter.startswith(r["app"])]
        if not rows:
            sys.exit(f"no billing rows for {app_filter} in this window (still running? billing lags partial intervals)")

    # aggregate per app, and per (app, resource)
    by_app = defaultdict(float)
    by_app_res = defaultdict(lambda: defaultdict(float))
    by_res = defaultdict(float)
    for r in rows:
        by_app[r["app"]] += r["cost"]
        by_app_res[r["app"]][r["resource"]] += r["cost"]
        by_res[r["resource"]] += r["cost"]

    window = f"--start {args.start} --end {args.end}" if args.start else f"--for '{args.for_range}'"
    print(f"\n=== Modal actual cost ({window}) ===")
    for app in sorted(by_app, key=by_app.get, reverse=True):
        res = by_app_res[app]
        split = "  ".join(f"{k} ${v:.2f}" for k, v in sorted(res.items(), key=lambda kv: -kv[1]) if v > 0)
        print(f"  {app}  ${by_app[app]:.2f}   [{split}]")
    total = sum(by_app.values())
    res_split = "  ".join(f"{k} ${v:.2f}" for k, v in sorted(by_res.items(), key=lambda kv: -kv[1]) if v > 0)
    print(f"\n  TOTAL ${total:.2f}   ({res_split})")
    print(f"  ({len(by_app)} app(s))")


if __name__ == "__main__":
    main()
