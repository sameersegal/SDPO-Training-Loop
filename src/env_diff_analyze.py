#!/usr/bin/env python3
"""ENV-DIFF analysis: compute the quantitative stats for each probe.

P1a (OJBench-side, FREE, no GPU): per-problem solve-rate histogram, verdict mix,
completion-length distribution, and SDPO-signal availability, all from the reusable
repl-01 base rollouts (63 problems x n=8, judged).

P1b (LCB-side): same stats from our own LCB generation (env_diff_lcb_base.jsonl).
P2 (teacher-edge): per-env/arm teacher one-shot solve rate vs base per-problem rate.
P3 (prompt-style): OJBench solve rate + length + NO_CODE for our style vs their CODE_PROMPT.

Reads reusable data via absolute paths under runs/. Writes tables to stdout and, if
--out given, a JSON summary. Analysis prose is assembled by the caller into ANALYSIS.md.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

ROOT = Path("/home/sameersegal/Code/SparkyCoder")
REPL01_DIR = ROOT / "runs/replication-01/evaldata/repl01sdpo"
BASE_JSON = REPL01_DIR / "sdpo_passk_repl01sdpo_base.json"
BASE_SAMPLES = REPL01_DIR / "sdpo_passk_repl01sdpo_base_samples.jsonl"
ENVDIFF = ROOT / "runs/env-diff"


def pctiles(xs, ps=(10, 50, 90)):
    if not xs:
        return {p: None for p in ps}
    xs = sorted(xs)
    out = {}
    for p in ps:
        # nearest-rank percentile
        k = max(0, min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1)))))
        out[p] = xs[k]
    return out


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def p1a_ojbench():
    """OJBench-side stats from the reusable repl-01 base data (curated easy+medium python)."""
    base = json.load(open(BASE_JSON))
    results = base["results"]  # per problem: id, n, n_ac, verdicts (len n)
    samples = load_jsonl(BASE_SAMPLES)  # per sample: id, sample_k, completion, n_chars

    # per-problem solve rate at n=8
    solve_rates = [r["n_ac"] / r["n"] for r in results]
    hist = collections.Counter(r["n_ac"] for r in results)  # n_ac in 0..8

    # verdict mix over all 504 samples (from the judged per-problem verdicts)
    all_verdicts = [v for r in results for v in r["verdicts"]]
    vmix = collections.Counter(all_verdicts)
    total = len(all_verdicts)

    # completion length distribution (chars, from samples jsonl)
    lens = [s["n_chars"] for s in samples]
    lp = pctiles(lens)

    # NO_CODE breakdown: cap-clip (thinking hit the eval cap, >80k chars) vs genuine format miss.
    # Pair samples (id, sample_k) with verdicts to classify each NO_CODE.
    samp_by = {(s["id"], s["sample_k"]): s for s in samples}
    nocode_clip = nocode_format = 0
    for r in results:
        for k, v in enumerate(r["verdicts"]):
            if v == "NO_CODE":
                s = samp_by.get((r["id"], k))
                nch = s["n_chars"] if s else 0
                if nch > 80000:
                    nocode_clip += 1
                else:
                    nocode_format += 1

    # SDPO-signal availability = fraction of problems with 0 < n_ac < n
    n = results[0]["n"] if results else 8
    frontier = [r for r in results if 0 < r["n_ac"] < r["n"]]
    all_pass = [r for r in results if r["n_ac"] == r["n"]]
    all_fail = [r for r in results if r["n_ac"] == 0]

    return {
        "env": "ojbench",
        "pool": "curated easy+medium python, 63 problems x n=8 (repl-01 base rollouts)",
        "n_problems": len(results),
        "n_samples": total,
        "solve_rate_mean": round(statistics.mean(solve_rates), 4),
        "solve_rate_hist_by_n_ac": {int(k): hist.get(k, 0) for k in range(n + 1)},
        "verdict_mix": {k: round(vmix[k] / total, 4) for k in sorted(vmix)},
        "verdict_counts": dict(vmix),
        "nocode_breakdown": {
            "cap_clip_>80k_chars": nocode_clip,
            "genuine_format_miss": nocode_format,
            "cap_casualty_frac_of_all_samples": round(nocode_clip / total, 4),
        },
        "completion_len_chars_p10_50_90": lp,
        "sdpo_signal_frontier_frac": round(len(frontier) / len(results), 4),
        "n_frontier_0<nac<n": len(frontier),
        "n_all_pass": len(all_pass),
        "n_all_fail": len(all_fail),
    }


def p1b_lcb(path=None):
    """LCB-side stats from our own generation (env_diff_lcb_base.jsonl)."""
    path = path or (ENVDIFF / "env_diff_lcb_base.jsonl")
    rows = load_jsonl(path)
    if not rows:
        return {"env": "lcb", "status": "no data yet", "path": str(path)}

    # group by problem_id
    by_pid = collections.defaultdict(list)
    for r in rows:
        by_pid[r["problem_id"]].append(r)

    solve_rates, hist_nac, per_prob_n = [], collections.Counter(), []
    n_frontier = n_all_pass = n_all_fail = 0
    for pid, rr in by_pid.items():
        n = len(rr)
        n_ac = sum(1 for r in rr if r.get("verdict") == "AC")
        solve_rates.append(n_ac / n)
        hist_nac[n_ac] += 1
        per_prob_n.append(n)
        if n_ac == 0:
            n_all_fail += 1
        elif n_ac == n:
            n_all_pass += 1
        else:
            n_frontier += 1

    vmix = collections.Counter(r.get("verdict") for r in rows)
    total = len(rows)
    lens = [r.get("n_tokens") for r in rows if r.get("n_tokens") is not None]
    lp = pctiles(lens)

    return {
        "env": "lcb",
        "pool": f"LCBv6 seed-0 subset, {len(by_pid)} problems (our own generation)",
        "n_problems": len(by_pid),
        "n_samples": total,
        "n_per_problem_mode": collections.Counter(per_prob_n).most_common(1)[0][0] if per_prob_n else None,
        "solve_rate_mean": round(statistics.mean(solve_rates), 4) if solve_rates else None,
        "solve_rate_hist_by_n_ac": dict(sorted(hist_nac.items())),
        "verdict_mix": {k: round(vmix[k] / total, 4) for k in sorted(vmix, key=str)},
        "verdict_counts": dict(vmix),
        "completion_len_tokens_p10_50_90": lp,
        "sdpo_signal_frontier_frac": round(n_frontier / len(by_pid), 4) if by_pid else None,
        "n_frontier": n_frontier,
        "n_all_pass": n_all_pass,
        "n_all_fail": n_all_fail,
    }


def p2_teacher_edge(path=None):
    """Teacher-edge: per env/arm one-shot solve rate vs student base per-problem rate."""
    path = path or (ENVDIFF / "env_diff_p2_teacher.jsonl")
    rows = load_jsonl(path)
    if not rows:
        return {"probe": "P2-teacher-edge", "status": "no data yet", "path": str(path)}

    # arm buckets: env x arm -> list of (solved bool). Each row is one teacher attempt.
    buckets = collections.defaultdict(lambda: {"solved": 0, "n": 0, "problems": set()})
    # student base rate on the SAME problems attempted per env (from meta on rows)
    base_rate = collections.defaultdict(dict)  # env -> pid -> base_rate
    for r in rows:
        env = r["env"]
        arm = r["arm"]
        key = (env, arm)
        buckets[key]["n"] += 1
        buckets[key]["problems"].add(r["problem_id"])
        if r.get("verdict") == "AC":
            buckets[key]["solved"] += 1
        if "base_rate" in r:
            base_rate[env][r["problem_id"]] = r["base_rate"]

    out = {}
    for (env, arm), b in sorted(buckets.items()):
        probs = b["problems"]
        brs = [base_rate[env].get(p) for p in probs if base_rate[env].get(p) is not None]
        out[f"{env}/{arm}"] = {
            "env": env, "arm": arm,
            "n_attempts": b["n"],
            "n_problems": len(probs),
            "teacher_oneshot_solve_rate": round(b["solved"] / b["n"], 4) if b["n"] else None,
            "student_base_rate_on_these": round(statistics.mean(brs), 4) if brs else None,
            "teacher_edge": round(b["solved"] / b["n"] - statistics.mean(brs), 4) if (b["n"] and brs) else None,
        }
    return {"probe": "P2-teacher-edge", "arms": out}


def p3_prompt_style(path=None):
    """P3 prompt-style ablation on OJBench: our style vs their CODE_PROMPT."""
    path = path or (ENVDIFF / "env_diff_p3_prompt.jsonl")
    rows = load_jsonl(path)
    if not rows:
        return {"probe": "P3-prompt-style", "status": "no data yet", "path": str(path)}
    out = {}
    for arm in sorted(set(r["arm"] for r in rows)):
        rr = [r for r in rows if r["arm"] == arm]
        by_pid = collections.defaultdict(list)
        for r in rr:
            by_pid[r["problem_id"]].append(r)
        solve_rates = [sum(1 for x in v if x.get("verdict") == "AC") / len(v) for v in by_pid.values()]
        lens = [r.get("n_tokens") for r in rr if r.get("n_tokens") is not None]
        no_code = sum(1 for r in rr if r.get("verdict") == "NO_CODE")
        out[arm] = {
            "n_problems": len(by_pid),
            "n_samples": len(rr),
            "solve_rate_mean": round(statistics.mean(solve_rates), 4) if solve_rates else None,
            "completion_len_tokens_p10_50_90": pctiles(lens),
            "no_code_rate": round(no_code / len(rr), 4) if rr else None,
        }
    return {"probe": "P3-prompt-style", "arms": out}


def markdown_tables(result):
    """Emit ready-to-paste markdown tables for whatever probes have data."""
    out = []
    p1b = result.get("P1b", {})
    if "status" not in p1b:
        out.append("### A.2 LCB (P1b)")
        out.append(f"- pool: {p1b['pool']}; {p1b['n_samples']} samples")
        vm = p1b["verdict_mix"]
        out.append("| verdict | " + " | ".join(vm) + " |")
        out.append("|" + "---|" * (len(vm) + 1))
        out.append("| frac | " + " | ".join(str(v) for v in vm.values()) + " |")
        out.append(f"- solve_rate_mean={p1b['solve_rate_mean']}; "
                   f"SDPO-signal frontier frac={p1b['sdpo_signal_frontier_frac']} "
                   f"({p1b['n_frontier']} frontier / {p1b['n_all_pass']} all-pass / {p1b['n_all_fail']} all-fail)")
        out.append(f"- completion len tokens p10/50/90: {p1b['completion_len_tokens_p10_50_90']}")
        out.append(f"- solve-rate histogram (n_ac): {p1b['solve_rate_hist_by_n_ac']}")
        out.append("")
    p2 = result.get("P2", {})
    if "arms" in p2:
        out.append("### B — teacher-edge (env/arm)")
        out.append("| env/arm | n_att | n_prob | teacher_solve | student_base | edge |")
        out.append("|---|---|---|---|---|---|")
        for k, a in p2["arms"].items():
            out.append(f"| {k} | {a['n_attempts']} | {a['n_problems']} | "
                       f"{a['teacher_oneshot_solve_rate']} | {a['student_base_rate_on_these']} | "
                       f"{a['teacher_edge']} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=["p1a", "p1b", "p2", "p3", "all"], default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tables", action="store_true", help="also print ready-to-paste markdown tables")
    args = ap.parse_args()

    result = {}
    if args.probe in ("p1a", "all"):
        result["P1a"] = p1a_ojbench()
    if args.probe in ("p1b", "all"):
        result["P1b"] = p1b_lcb()
    if args.probe in ("p2", "all"):
        result["P2"] = p2_teacher_edge()
    if args.probe in ("p3", "all"):
        result["P3"] = p3_prompt_style()

    print(json.dumps(result, indent=2, default=str))
    if args.tables:
        print("\n" + "=" * 60 + "\nMARKDOWN TABLES\n" + "=" * 60)
        print(markdown_tables(result))
    if args.out:
        json.dump(result, open(args.out, "w"), indent=2, default=str)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
