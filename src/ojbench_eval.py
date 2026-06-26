#!/usr/bin/env python3
"""OJBench comparison harness (lightweight judge).

Generates Python solutions from a served model for the selected NOI problems,
then judges them by running each solution against the problem's test cases
(stdin -> stdout, whitespace-normalized diff). Reports pass@1 by difficulty
and tokens/sec.

Two phases:
  gen   : python ojbench_eval.py gen   --model <id> --tag <tag> [sampling opts]
  judge : python ojbench_eval.py judge --tag <tag>

Judging caveat: this is NOT the official DMOJ sandbox. It runs solutions under
a wall-clock timeout + memory cap and does a normalized line diff. All 15
selected problems use standard (non-special-judge) checkers, so this matches
DMOJ's diff verdict for them; it does not enforce DMOJ's exact time/memory
limits (we use a generous CPython timeout).
"""
import argparse
import json
import os
import re
import resource
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tempfile

# Serializes lazy test extraction. The reward function now judges a group's
# completions concurrently (threads), so multiple threads can hit the SAME
# problem's tests.zip at once — without this lock one thread sees a freshly
# mkdir'd _extracted/ and reads a file another thread hasn't written yet.
_EXTRACT_LOCK = threading.Lock()

import yaml
from openai import OpenAI

from _paths import find_file, ojbench_dir

ROOT = Path(tempfile.gettempdir())  # scratch dir for transient _sol_* artifacts
DATA = ojbench_dir() / "NOI"
SELECTED = json.load(open(find_file("ojbench_selected.json")))

# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def gen(args):
    # long timeout: reasoning traces to 20k tokens can take >15 min when batched
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.req_timeout)
    jsonl_path = ROOT / f"ojb_responses_{args.tag}.jsonl"
    jsonl_lock = __import__("threading").Lock()
    open(jsonl_path, "w").close()  # truncate

    def one(p):
        t0 = time.perf_counter()
        kwargs = dict(
            model=args.model,
            messages=[{"role": "user", "content": p["prompt"]}],
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.top_p is not None:
            kwargs["top_p"] = args.top_p
        rec = {"id": p["id"], "difficulty": p["difficulty"], "dataset": p["dataset"],
               "language": p["language"]}
        try:
            resp = client.chat.completions.create(**kwargs)
            rec.update(
                content=resp.choices[0].message.content or "",
                completion_tokens=resp.usage.completion_tokens,
                prompt_tokens=resp.usage.prompt_tokens,
                finish_reason=resp.choices[0].finish_reason,
                latency=time.perf_counter() - t0,
            )
        except Exception as e:  # noqa: BLE001 - tolerate per-request failures
            rec.update(content="", completion_tokens=0, prompt_tokens=0,
                       finish_reason=f"error:{type(e).__name__}",
                       latency=time.perf_counter() - t0)
        # incremental save so progress is never lost
        with jsonl_lock:
            with open(jsonl_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        return rec

    results = []
    wall0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one, p) for p in SELECTED]
        for j, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            print(f"  generated {j}/{len(SELECTED)}  (loj-{r['id']} {r['difficulty']} "
                  f"ct={r['completion_tokens']} {r['finish_reason']})", flush=True)
    wall = time.perf_counter() - wall0
    results.sort(key=lambda r: (r["difficulty"], r["id"]))

    tot_ct = sum(r["completion_tokens"] for r in results)
    out = ROOT / f"ojb_responses_{args.tag}.json"
    json.dump(
        {
            "model": args.model,
            "tag": args.tag,
            "wall_s": round(wall, 1),
            "total_completion_tokens": tot_ct,
            "aggregate_decode_tok_per_s": round(tot_ct / wall, 1),
            "mean_per_request_decode_tok_per_s": round(
                sum(r["completion_tokens"] / r["latency"] for r in results) / len(results), 1
            ),
            "sampling": {"temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens},
            "responses": results,
        },
        open(out, "w"),
        indent=2,
    )
    print(f"wrote {out}  (gen tok/s aggregate={round(tot_ct/wall,1)})")


# --------------------------------------------------------------------------
# Judging
# --------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text):
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()  # last fenced block = final answer
    return None


def normalize(s):
    return "\n".join(line.rstrip() for line in s.splitlines()).rstrip("\n")


def _limit():
    # cap address space ~2GB, cpu time as backstop
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))


# Optional registry id -> test-data dir (populated from a splits file's testdir_by_id).
# Lets extract_tests resolve BOTH NOI (loj-<int>) and ICPC (string) problems. Falls back
# to NOI/loj-<pid> when unset, so NOI-only splits work unchanged.
_TESTDIR_BY_ID = {}


def register_testdirs(mapping):
    _TESTDIR_BY_ID.update(mapping)


def extract_tests(pid):
    pdir = _TESTDIR_BY_ID.get(pid, DATA / f"loj-{pid}")
    init = yaml.safe_load(open(pdir / "init.yml"))
    ex = pdir / "_extracted"
    cases = [(ex / c["in"], ex / c["out"]) for c in init["test_cases"]]

    def complete():
        return all(i.exists() and o.exists() for i, o in cases)

    # Re-extract if missing OR incomplete (a partial _extracted/ can come from an
    # interrupted run or a half-uploaded volume dir). Extract to a temp dir then
    # atomically rename, so _extracted/ only ever appears fully populated — and
    # serialize under a lock so concurrent judge threads don't race on it.
    if not complete():
        with _EXTRACT_LOCK:
            if not complete():
                tmp = pdir / f"_extracted.tmp.{os.getpid()}"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                with zipfile.ZipFile(pdir / init["archive"]) as z:
                    z.extractall(tmp)
                shutil.rmtree(ex, ignore_errors=True)
                os.replace(tmp, ex)  # atomic: _extracted appears only when complete
    return cases


def _clip(s, n=600):
    s = s if isinstance(s, str) else s.decode("utf-8", "replace")
    return s if len(s) <= n else s[:n] + f"...<+{len(s)-n} chars>"


def judge_solution(code, cases, timeout, count_all=False):
    """Run code against all cases, SMALLEST INPUT FIRST.
    Returns (verdict, passed_n, total, detail).

    count_all=False (default): early-exit on the first failure. `passed_n` = cases
      passed before that failure (a cheap prefix count). Fastest.
    count_all=True: run EVERY case; `passed_n` = the TRUE number passed (for a dense
      passed/total reward). `detail`/`verdict` still describe the SMALLEST failing case
      (good teacher feedback). Costs more (no early-exit; pays the timeout per TLE case).

    Order never changes the AC verdict (must pass all); the public/private split is
    decided upstream, so re-ordering here does not move cases between splits."""
    if not code:
        return "NO_CODE", 0, len(cases), {"reason": "no code block found in response"}
    cases = sorted(cases, key=lambda c: c[0].stat().st_size)
    sol = ROOT / f"_sol_{os.getpid()}_{time.perf_counter_ns()}.py"
    sol.write_text(code)
    passed = 0
    first_fail = None  # (verdict, detail) of the smallest failing case
    try:
        for infile, outfile in cases:
            try:
                inp = infile.read_bytes()
                proc = subprocess.run(
                    [sys.executable, str(sol)],
                    input=inp, capture_output=True, timeout=timeout, preexec_fn=_limit,
                )
            except subprocess.TimeoutExpired:
                fail = ("TLE", {"failing_case": infile.name,
                                "input": _clip(inp.decode("utf-8", "replace"))})
            else:
                exp = normalize(outfile.read_text(encoding="utf-8", errors="replace"))
                if proc.returncode != 0:
                    fail = ("RE", {"failing_case": infile.name,
                                   "input": _clip(inp.decode("utf-8", "replace")),
                                   "stderr": _clip(proc.stderr.decode("utf-8", "replace"))})
                elif normalize(proc.stdout.decode("utf-8", "replace")) != exp:
                    fail = ("WA", {"failing_case": infile.name,
                                   "input": _clip(inp.decode("utf-8", "replace")),
                                   "expected": _clip(exp),
                                   "got": _clip(normalize(proc.stdout.decode("utf-8", "replace")))})
                else:
                    fail = None
            if fail is None:
                passed += 1
            else:
                if first_fail is None:
                    first_fail = fail
                if not count_all:
                    return fail[0], passed, len(cases), fail[1]
        if first_fail is None:
            return "AC", len(cases), len(cases), {}
        return first_fail[0], passed, len(cases), first_fail[1]
    finally:
        sol.unlink(missing_ok=True)


def _load_responses(tag):
    """Prefer the final summary json; fall back to the incremental jsonl."""
    jp = ROOT / f"ojb_responses_{tag}.json"
    if jp.exists():
        data = json.load(open(jp))
        return data["model"], data, data["responses"]
    jl = ROOT / f"ojb_responses_{tag}.jsonl"
    responses = [json.loads(l) for l in open(jl)]
    model = next((r.get("model") for r in responses if r.get("model")), tag)
    return model, None, responses


def judge(args):
    model, data, responses = _load_responses(args.tag)
    resp_by_id = {r["id"]: r for r in responses}
    prompt_by_id = {p["id"]: p for p in SELECTED}

    def do(p):
        r = resp_by_id.get(p["id"], {"content": "", "completion_tokens": 0,
                                     "finish_reason": "missing"})
        code = extract_code(r.get("content", ""))
        cases = extract_tests(p["id"])
        verdict, passed, total, detail = judge_solution(code, cases, args.timeout)
        return {
            "id": p["id"],
            "difficulty": p["difficulty"],
            "dataset": p["dataset"],
            "language": p["language"],
            "verdict": verdict,
            "is_passed": verdict == "AC",
            "cases_passed": passed,
            "cases_total": total,
            "completion_tokens": r.get("completion_tokens", 0),
            "finish_reason": r.get("finish_reason"),
            "had_code": code is not None,
            "fail_detail": detail,
            "_code": code,
            "_response": r.get("content", ""),
        }

    judged = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(do, p) for p in SELECTED]
        for f in as_completed(futs):
            j = f.result()
            judged.append(j)
            print(f"  [{j['difficulty']:<6}] loj-{j['id']}: {j['verdict']} "
                  f"({j['cases_passed']}/{j['cases_total']})", flush=True)
    judged.sort(key=lambda r: (r["difficulty"], r["id"]))

    def rate(diff):
        sub = [j for j in judged if j["difficulty"] == diff]
        return sum(j["is_passed"] for j in sub), len(sub)

    e_ac, e_n = rate("easy")
    m_ac, m_n = rate("medium")
    failed_ids = {
        "easy": [j["id"] for j in judged if j["difficulty"] == "easy" and not j["is_passed"]],
        "medium": [j["id"] for j in judged if j["difficulty"] == "medium" and not j["is_passed"]],
    }
    summary = {
        "model": model,
        "tag": args.tag,
        "easy_pass": f"{e_ac}/{e_n}",
        "medium_pass": f"{m_ac}/{m_n}",
        "overall_pass": f"{e_ac+m_ac}/{e_n+m_n}",
        "failed_ids": failed_ids,
        "verdicts": {j["id"]: j["verdict"] for j in judged},
    }
    if data:
        summary.update(
            gen_aggregate_tok_per_s=data["aggregate_decode_tok_per_s"],
            gen_mean_per_req_tok_per_s=data["mean_per_request_decode_tok_per_s"],
            total_completion_tokens=data["total_completion_tokens"],
            gen_wall_s=data["wall_s"],
            sampling=data["sampling"],
        )

    out = ROOT / f"ojb_judged_{args.tag}.json"
    json.dump({"summary": summary, "judged": judged}, open(out, "w"), indent=2)

    # Failure dataset: one record per FAILED problem, with everything needed to
    # study/improve on it (prompt, model output, extracted code, failing case).
    fail_path = ROOT / f"ojb_failures_{args.tag}.jsonl"
    n_fail = 0
    with open(fail_path, "w") as fh:
        for j in judged:
            if j["is_passed"]:
                continue
            n_fail += 1
            p = prompt_by_id[j["id"]]
            fh.write(json.dumps({
                "id": j["id"],
                "problem_dir": f"NOI/loj-{j['id']}",
                "difficulty": j["difficulty"],
                "dataset": j["dataset"],
                "language": j["language"],
                "model": model,
                "verdict": j["verdict"],
                "cases_passed": j["cases_passed"],
                "cases_total": j["cases_total"],
                "fail_detail": j["fail_detail"],
                "prompt": p["prompt"],
                "model_response": j["_response"],
                "extracted_code": j["_code"],
            }) + "\n")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out}")
    print(f"wrote failure dataset: {fail_path}  ({n_fail} failed problems)")


def compare(args):
    judged = {}
    for tag in args.tags:
        d = json.load(open(ROOT / f"ojb_judged_{tag}.json"))
        judged[tag] = d
    prompt_by_id = {p["id"]: p for p in SELECTED}

    # --- comparison table ---
    print("\n=== OJBench comparison (10 easy + 5 medium NOI problems) ===")
    hdr = f"{'metric':<26}" + "".join(f"{t:>16}" for t in args.tags)
    print(hdr)
    for metric in ["easy_pass", "medium_pass", "overall_pass",
                   "gen_aggregate_tok_per_s", "gen_mean_per_req_tok_per_s",
                   "total_completion_tokens", "gen_wall_s"]:
        row = f"{metric:<26}"
        for t in args.tags:
            row += f"{str(judged[t]['summary'].get(metric, '-')):>16}"
        print(row)

    # --- per-problem verdict matrix ---
    print("\n=== per-problem verdicts ===")
    print(f"{'problem':<16}{'difficulty':<10}" + "".join(f"{t:>14}" for t in args.tags))
    by_id = {}
    for t in args.tags:
        for j in judged[t]["judged"]:
            by_id.setdefault(j["id"], {})[t] = j
    for pid in sorted(by_id, key=lambda i: (by_id[i][args.tags[0]]["difficulty"], i)):
        row = by_id[pid]
        diff = row[args.tags[0]]["difficulty"]
        line = f"loj-{pid:<11}{diff:<10}"
        for t in args.tags:
            v = row[t]["verdict"]
            mark = "AC" if v == "AC" else v
            line += f"{mark:>14}"
        print(line)

    # --- union failure dataset (problems failed by ANY model) ---
    union_path = ROOT / "ojb_failures_union.jsonl"
    n = 0
    with open(union_path, "w") as fh:
        for pid in sorted(by_id, key=lambda i: (by_id[i][args.tags[0]]["difficulty"], i)):
            row = by_id[pid]
            failed_by = [t for t in args.tags if not row[t]["is_passed"]]
            if not failed_by:
                continue
            n += 1
            p = prompt_by_id[pid]
            fh.write(json.dumps({
                "id": pid,
                "problem_dir": f"NOI/loj-{pid}",
                "difficulty": row[args.tags[0]]["difficulty"],
                "failed_by": failed_by,
                "passed_by": [t for t in args.tags if row[t]["is_passed"]],
                "verdicts": {t: row[t]["verdict"] for t in args.tags},
                "prompt": p["prompt"],
            }) + "\n")
    print(f"\nwrote union failure dataset: {union_path}  ({n} problems failed by >=1 model)")
    print("per-model failure datasets: " + ", ".join(f"ojb_failures_{t}.jsonl" for t in args.tags))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--tag", required=True)
    g.add_argument("--temperature", type=float, default=0.0)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--concurrency", type=int, default=8)
    g.add_argument("--req-timeout", type=float, default=3600.0)
    g.add_argument("--base-url", default="http://localhost:8000/v1")
    g.set_defaults(func=gen)

    j = sub.add_parser("judge")
    j.add_argument("--tag", required=True)
    j.add_argument("--timeout", type=float, default=8.0)
    j.add_argument("--workers", type=int, default=8)
    j.set_defaults(func=judge)

    c = sub.add_parser("compare")
    c.add_argument("--tags", nargs="+", required=True)
    c.set_defaults(func=compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
