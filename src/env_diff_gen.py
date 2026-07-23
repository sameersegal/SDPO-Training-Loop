#!/usr/bin/env python3
"""ENV-DIFF generation harness (GB10, vLLM OpenAI endpoint on :8001).

Runs the GPU probes P1b / P2 / P3. Streams EVERY prompt+completion to a JSONL under
runs/env-diff/ with the mandated schema, one line per completion, written the instant it
finishes. Supports --resume (skips (problem_id, arm, n_sample) keys already present).

Schema per line:
  {"probe", "env", "problem_id", "arm", "n_sample", "messages", "completion",
   "n_tokens", "verdict", "reward", "feedback", "ts"}

GB10 rules (CLAUDE.md): n=1 per request, modest client concurrency (<=4), enforce_eager
(set server-side), stream-to-disk per completion. Do NOT pkill -f vllm.

Probes:
  p1b   LCB base: seed-0 subset of the 131 LCBv6 problems, n=8, temp 1.0 top_p 1.0,
        max_tokens 8192, THEIR exact CODE_PROMPT prompt (parquet messages verbatim),
        judged with THEIR code.py.
  p2    Teacher-edge: reprompt template (verbatim) over failed base attempts. Arms:
        (a) solution-only, (b) feedback-only, (c) solution+feedback. OJBench arm (b) is
        run TWICE (native feedback vs their LeetCode-shape feedback) = P4 fold-in. Env
        selectable (--env ojbench|lcb). 1 teacher sample per (failed_attempt, arm), temp 1.0.

P3 (prompt-style ablation) was DESCOPED per the overnight scope change — freed budget went
to larger P2 problem counts (n_mixed=18, n_allfail=8) and a bigger P1b subset. The run_p3
code is retained but is NOT invoked by the overnight launch.
"""
import argparse
import collections
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/home/sameersegal/Code/SparkyCoder/src")

ROOT = Path("/home/sameersegal/Code/SparkyCoder")
ENVDIFF = ROOT / "runs/env-diff"
REPL01_DIR = ROOT / "runs/replication-01/evaldata/repl01sdpo"
BASE_JSON = REPL01_DIR / "sdpo_passk_repl01sdpo_base.json"
BASE_SAMPLES = REPL01_DIR / "sdpo_passk_repl01sdpo_base_samples.jsonl"

# --- OpenAI client (vLLM endpoint) ---------------------------------------
def make_client(base_url):
    """OpenAI client with a LONG timeout: a 32k-cap gen at ~13 tok/s/stream takes ~42 min,
    far past the default 600s — the default silently times out EVERY long request (cost us
    the first launch: 0 records, all 'Request timed out'). 90 min covers the worst case."""
    from openai import OpenAI
    return OpenAI(base_url=base_url, api_key="EMPTY", timeout=5400.0, max_retries=1)


def chat_once(client, model, messages, max_tokens, temperature=1.0, top_p=1.0,
              enable_thinking=True):
    """One completion (n=1). Returns (text, n_completion_tokens).

    enable_thinking=False matters for the LCB arms: the paper's verl pipeline runs
    Qwen3-8B effectively NON-thinking (~700-tok responses at an 8192 cap — measured in
    repl-02), while our chat endpoint defaults thinking ON, which floods the 8192 cap
    with <think> and produced 67% cap-clipped NO_CODE in the first P1b attempt
    (preserved as *_thinkon.jsonl — itself an env-diff datapoint)."""
    extra = {"chat_template_kwargs": {"enable_thinking": enable_thinking}} if not enable_thinking else {}
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p, n=1, extra_body=extra,
    )
    ch = resp.choices[0]
    text = ch.message.content or ""
    ntok = resp.usage.completion_tokens if resp.usage else None
    return text, ntok


# --- streaming persistence + resume --------------------------------------
def load_done_keys(out_path):
    """Return set of (problem_id, arm, n_sample) already in the output file."""
    done = set()
    if Path(out_path).exists():
        for l in open(out_path):
            if not l.strip():
                continue
            try:
                d = json.loads(l)
                done.add((d["problem_id"], d["arm"], d["n_sample"]))
            except Exception:
                continue
    return done


def append_record(out_path, rec):
    with open(out_path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
        f.flush()


def _base_verdicts():
    """repl-01 base per-problem verdicts (len n=8), plus the paired sample completions."""
    base = json.load(open(BASE_JSON))
    verd = {r["id"]: r["verdicts"] for r in base["results"]}
    n_ac = {r["id"]: r["n_ac"] for r in base["results"]}
    diff = {r["id"]: r["difficulty"] for r in base["results"]}
    samples = collections.defaultdict(dict)  # id -> sample_k -> completion
    for l in open(BASE_SAMPLES):
        if not l.strip():
            continue
        d = json.loads(l)
        samples[d["id"]][d["sample_k"]] = d["completion"]
    return verd, n_ac, diff, samples


# =========================================================================
# P1b — LCB base generation
# =========================================================================
def run_p1b(client, model, out_path, n_subset=40, n_sample=8, max_tokens=8192,
            seed=0, resume=True):
    from env_diff_lcb_judge import load_lcb_rows, judge_lcb
    rows = load_lcb_rows("test")
    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:n_subset])
    print(f"[P1b] {len(chosen)} LCB problems x n={n_sample}, max_tokens={max_tokens}", flush=True)

    done = load_done_keys(out_path) if resume else set()
    tasks = []
    for ci in chosen:
        row = rows[ci]
        pid = row["index"]
        for k in range(n_sample):
            if (pid, "base", k) in done:
                continue
            tasks.append((row, pid, k))
    print(f"[P1b] {len(tasks)} tasks remaining after resume", flush=True)

    def do(task):
        row, pid, k = task
        messages = row["prompt"]
        text, ntok = chat_once(client, model, messages, max_tokens, enable_thinking=False)
        verdict, reward, feedback = judge_lcb(text, row["ground_truth"], row["extra_info"])
        return {
            "probe": "P1-lcb-base", "env": "lcb", "problem_id": pid, "arm": "base",
            "n_sample": k, "messages": messages, "completion": text,
            "n_tokens": ntok, "verdict": verdict, "reward": reward,
            "feedback": feedback, "ts": time.time(),
        }
    _run_pool(tasks, do, out_path, concurrency=8, label="P1b")


# =========================================================================
# P3 — prompt-style ablation (OJBench, their CODE_PROMPT around OJB statement)
# =========================================================================
LCB_CODE_PROMPT_HEADER = (
    "You are a coding expert. You will be given a coding problem, and you need to write a "
    "correct Python program that matches the specification and passes all tests. The time "
    "limit is 1 second. You may start by outlining your thought process. In the end, please "
    "provide the complete code in a code block enclosed with ``` ```."
)


def their_style_ojbench_prompt(problem_statement):
    """Their CODE_PROMPT shape wrapped around the OJBench problem statement. NO system msg,
    plain single user turn. OJBench problems are stdin/stdout; the header already says
    'passes all tests' and the OJBench statement carries its own Input/Output format."""
    return LCB_CODE_PROMPT_HEADER + "\n\n" + problem_statement


def run_p3(client, model, out_path, n_problems=20, n_sample=4, max_tokens=8192,
           seed=0, resume=True):
    import sdpo_ojbench as ojb
    verd, n_ac, diff, _samples = _base_verdicts()
    # 63-pool, stratify by base solve rate into 4 buckets, sample 20 seed-0
    ids = sorted(verd.keys())
    rng = random.Random(seed)
    # stratify by n_ac bucket
    buckets = collections.defaultdict(list)
    for pid in ids:
        r = n_ac[pid] / 8.0
        b = 0 if r == 0 else (3 if r == 1.0 else (1 if r <= 0.5 else 2))
        buckets[b].append(pid)
    chosen = []
    per = max(1, n_problems // 4)
    for b in sorted(buckets):
        pool = buckets[b][:]
        rng.shuffle(pool)
        chosen += pool[:per]
    # top up to n_problems from remaining
    rest = [p for p in ids if p not in chosen]
    rng.shuffle(rest)
    chosen = sorted((chosen + rest)[:n_problems])
    print(f"[P3] {len(chosen)} OJBench problems x n={n_sample} (their CODE_PROMPT arm 'their')", flush=True)

    done = load_done_keys(out_path) if resume else set()
    tasks = []
    for pid in chosen:
        stmt = ojb.PROMPT_BY_ID[pid]
        for k in range(n_sample):
            if (pid, "their", k) in done:
                continue
            tasks.append((pid, stmt, k))
    print(f"[P3] {len(tasks)} tasks remaining after resume", flush=True)

    def do(task):
        pid, stmt, k = task
        messages = [{"role": "user", "content": their_style_ojbench_prompt(stmt)}]
        text, ntok = chat_once(client, model, messages, max_tokens)
        reward, verdict, feedback = ojb.judge_completion(text, pid, which="public", language="python")
        return {
            "probe": "P3-prompt-style", "env": "ojbench", "problem_id": pid, "arm": "their",
            "n_sample": k, "messages": messages, "completion": text,
            "n_tokens": ntok, "verdict": verdict, "reward": reward,
            "feedback": feedback, "ts": time.time(),
        }
    _run_pool(tasks, do, out_path, concurrency=8, label="P3")


# =========================================================================
# P2 — teacher-edge probe
# =========================================================================
def _their_leetcode_feedback(verdict, detail):
    """Render an OJBench failure in THEIR LeetCode feedback shape (P4 fold-in).

    Mimics feedback/code.py::format_test_feedback output for a single failing case:
      Wrong Answer: "Test Case N: Wrong Answer\n\nInput\n...\n\nOutput\n...\n\nExpected\n..."
      TLE: "Time Limit Exceeded\n..." ; RE: "Runtime Error\n..."
    250-char clips, matching their max_input/expected/actual_chars.
    """
    def clip(s, n=250):
        s = str(s)
        return s[:n] + "..." if len(s) > n else s
    case = detail.get("failing_case", "1")
    if verdict == "WA":
        return (f"Test Case {case}: Wrong Answer\n\n"
                f"Input\n{clip(detail.get('input', detail.get('failing_case','')))}\n\n"
                f"Output\n{clip(detail.get('got',''))}\n\n"
                f"Expected\n{clip(detail.get('expected',''))}")
    if verdict == "TLE":
        return (f"Time Limit Exceeded\n\n"
                f"Last Executed Input\n{clip(detail.get('failing_case',''))}")
    if verdict == "RE":
        return (f"Runtime Error\n{clip(detail.get('stderr',''))}\n\n"
                f"Last Executed Input\n{clip(detail.get('failing_case',''))}")
    return f"Test Case {case}: {verdict}"


# Real judged failures only. NO_CODE is EXCLUDED as a P2 seed: 24/504 base NO_CODE samples
# are cap-clipped (>80k chars, thinking hit the 24,576-tok eval cap) — reprompting those
# measures the cap, not the teacher. The other 13 NO_CODE are genuine format misses but still
# carry no informative judge feedback (no failing test), so we exclude all NO_CODE and seed
# reprompts ONLY from {WA, RE, TLE}. Documented in ANALYSIS.md (4.8% cap-casualty caveat).
REAL_FAIL_VERDICTS = {"WA", "RE", "TLE"}


def _real_fail_ks(verdicts, cap=2):
    """Indices of real judged failures (WA/RE/TLE), excluding AC and NO_CODE (cap-clip/format)."""
    return [k for k, v in enumerate(verdicts) if v in REAL_FAIL_VERDICTS][:cap]


# A think-stripped demo should be a small solution (few k chars). If <think> was never closed
# (clipped/malformed) strip_thinking is a no-op and the "demo" is the full ~100k-char rollout —
# using it blows the context and corrupts the sol arm. Require a clean, small demo.
_MAX_DEMO_CHARS = 12000  # ~3k tokens; a real easy/medium solution is well under this


def _clean_demo(ac_texts, strip_fn):
    """First AC sibling that strips cleanly (has a closed </think>) and is small enough to use
    as a teacher demo. Returns None if none qualify (then the sol/sol_fb arms are skipped)."""
    for raw in ac_texts:
        if "<think>" in raw and "</think>" not in raw:
            continue  # unterminated think -> strip is a no-op -> giant demo
        d = strip_fn(raw)
        if 0 < len(d) <= _MAX_DEMO_CHARS:
            return d
    return None


def _select_p2_ojbench_problems(verd, n_ac, samples, n_mixed=12, n_allfail=6, seed=0):
    """Pick problems that HAVE at least one real judged failure (WA/RE/TLE):
    mixed = has both an AC and a real failure; allfail = 0 AC but >=1 real failure."""
    rng = random.Random(seed)
    mixed = [pid for pid in verd if 0 < n_ac[pid] < 8 and _real_fail_ks(verd[pid])]
    allfail = [pid for pid in verd if n_ac[pid] == 0 and _real_fail_ks(verd[pid])]
    rng.shuffle(mixed)
    rng.shuffle(allfail)
    return sorted(mixed[:n_mixed]), sorted(allfail[:n_allfail])


def run_p2_ojbench(client, model, out_path, max_tokens=32768, seed=0, resume=True,
                   n_mixed=18, n_allfail=8):
    """Teacher-edge on OJBench using repl-01 base failures + sibling ACs.

    For each selected problem, take up to a few FAILED base attempts. For each failed
    attempt build teacher reprompts:
      arm 'sol'        : solution-only demo (a sibling AC, think-stripped)
      arm 'fb_native'  : feedback-only, our native _format_feedback
      arm 'fb_their'   : feedback-only, THEIR LeetCode shape (P4 fold-in)
      arm 'sol_fb'     : solution + native feedback
    All-fail problems: only the feedback arms (no sibling AC available).
    1 teacher sample per (failed_attempt, arm), temp 1.0. Judge with OJBench judge.
    """
    import sdpo_ojbench as ojb
    from sdpo_prompts import build_teacher_messages, strip_thinking
    verd, n_ac, diff, samples = _base_verdicts()
    mixed, allfail = _select_p2_ojbench_problems(verd, n_ac, samples, n_mixed, n_allfail, seed)
    print(f"[P2-ojbench] mixed={len(mixed)} allfail={len(allfail)} problems", flush=True)
    done = load_done_keys(out_path) if resume else set()
    MAX_FAILS_PER_PROBLEM = 2  # cap teacher attempts per problem to bound cost

    # --- Phase 1: build the (pid, failed_attempt) feedback CONCURRENTLY -------
    # We judge on PUBLIC tests (the reward the teacher would have seen at train time);
    # "failure" is defined by the repl-01 stored verdict (public+private). Some public
    # judges of a private-failure come back AC — we still treat it as a failed attempt
    # (its public feedback is 'All public tests passed', a degenerate but faithful case)
    # but PREFER attempts that also fail public so the feedback is informative.
    fail_units = []  # (pid, fk)
    for pid in mixed + allfail:
        # real judged failures only (WA/RE/TLE); NO_CODE excluded (cap-clip/format)
        fail_units += [(pid, fk) for fk in _real_fail_ks(verd[pid], MAX_FAILS_PER_PROBLEM)]

    def _judge_fail(unit):
        pid, fk = unit
        ft = samples[pid][fk]
        reward_f, verdict_f, native_fb = ojb.judge_completion(ft, pid, which="public", language="python")
        _v, _p, detail = _rejudge_detail(ojb, ft, pid)
        their_fb = _their_leetcode_feedback(verdict_f, detail)
        return (pid, fk), {"native_fb": native_fb, "their_fb": their_fb, "public_verdict": verdict_f}

    print(f"[P2-ojbench] judging {len(fail_units)} failed attempts (concurrent)…", flush=True)
    fb_cache = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for key, val in ex.map(_judge_fail, fail_units):
            fb_cache[key] = val
    print(f"[P2-ojbench] feedback built for {len(fb_cache)} attempts", flush=True)

    # --- Phase 2: build teacher tasks (no judging here) ----------------------
    tasks = []
    for pid in mixed + allfail:
        stmt = ojb.PROMPT_BY_ID[pid]
        orig_msgs = [{"role": "user", "content": stmt}]
        base_rate = n_ac[pid] / 8.0
        verdicts = verd[pid]
        ac_ks = [k for k, v in enumerate(verdicts) if v == "AC"]
        demo = _clean_demo([samples[pid][k] for k in ac_ks], strip_thinking)
        fail_ks = _real_fail_ks(verdicts, MAX_FAILS_PER_PROBLEM)
        for fk in fail_ks:
            fb = fb_cache[(pid, fk)]
            arms = []
            if demo is not None:
                arms.append(("sol", dict(demo_text=demo, feedback_raw=None)))
                arms.append(("sol_fb", dict(demo_text=demo, feedback_raw=fb["native_fb"])))
            arms.append(("fb_native", dict(demo_text=None, feedback_raw=fb["native_fb"])))
            arms.append(("fb_their", dict(demo_text=None, feedback_raw=fb["their_fb"])))
            for arm, kw in arms:
                arm_key = f"{arm}#f{fk}"
                if (pid, arm_key, fk) in done:
                    continue
                teacher_msgs = build_teacher_messages(orig_msgs, **kw)
                tasks.append((pid, arm, arm_key, fk, teacher_msgs, base_rate))

    print(f"[P2-ojbench] {len(tasks)} teacher tasks remaining after resume", flush=True)

    def do(task):
        pid, arm, arm_key, nk, teacher_msgs, base_rate = task
        msgs = teacher_msgs if isinstance(teacher_msgs, list) else [{"role": "user", "content": teacher_msgs}]
        text, ntok = chat_once(client, model, msgs, max_tokens)
        reward, verdict, feedback = ojb.judge_completion(text, pid, which="public", language="python")
        return {
            "probe": "P2-teacher-edge", "env": "ojbench", "problem_id": pid, "arm": arm_key,
            "n_sample": nk, "messages": msgs, "completion": text,
            "n_tokens": ntok, "verdict": verdict, "reward": reward,
            "feedback": feedback, "ts": time.time(),
            "base_rate": base_rate, "arm_family": arm,
        }
    _run_pool(tasks, do, out_path, concurrency=8, label="P2-ojbench")


def _rejudge_detail(ojb, text, pid):
    """Re-run the OJBench judge to extract the raw `detail` dict (for their-shape feedback).
    judge_completion hides detail, so replicate the inner call."""
    from ojbench_eval import extract_code, judge_solution
    pub, prv = ojb.public_private_cases(pid)
    mb, mc = ojb.reward_case_caps()
    cases = sorted(pub, key=lambda c: c[0].stat().st_size)
    if mb:
        small = [c for c in cases if c[0].stat().st_size <= mb]
        cases = small or cases[:1]
    if mc:
        cases = cases[:mc]
    verdict, passed, total, detail = judge_solution(extract_code(text), cases, 6.0, count_all=True)
    # attach failing input text if available
    if "failing_case" in detail:
        for infile, outfile in pub:
            if infile.name == detail["failing_case"]:
                try:
                    detail["input"] = infile.read_text(errors="replace")[:250]
                except Exception:
                    pass
                break
    return verdict, passed, detail


def run_p2_lcb(client, model, out_path, max_tokens=8192, seed=0, resume=True,
               n_mixed=18, n_allfail=8, lcb_base_path=None):
    """Teacher-edge on LCB using our P1b base failures + sibling ACs.
    Same arm structure but feedback is the LCB judge's OWN LeetCode feedback (native to
    that env), so there is no fb_their split here (their env's feedback IS the LeetCode shape)."""
    from env_diff_lcb_judge import load_lcb_rows, judge_lcb
    from sdpo_prompts import build_teacher_messages, strip_thinking
    lcb_base_path = lcb_base_path or (ENVDIFF / "env_diff_lcb_base.jsonl")
    if not Path(lcb_base_path).exists():
        print(f"[P2-lcb] no LCB base data at {lcb_base_path} yet — run P1b first", flush=True)
        return
    base_rows = [json.loads(l) for l in open(lcb_base_path) if l.strip()]
    lcb_rows = {r["index"]: r for r in load_lcb_rows("test")}

    by_pid = collections.defaultdict(list)
    for r in base_rows:
        by_pid[r["problem_id"]].append(r)

    # only fully-generated problems (n==8) for stable base_rate + demos
    ready = {p: rr for p, rr in by_pid.items() if len(rr) >= 8}
    rng = random.Random(seed)
    mixed, allfail = [], []
    for p, rr in ready.items():
        nac = sum(1 for r in rr if r["verdict"] == "AC")
        has_real_fail = any(r["verdict"] in REAL_FAIL_VERDICTS for r in rr)
        if not has_real_fail:
            continue  # need at least one WA/RE/TLE to seed a teacher reprompt
        if nac == 0:
            allfail.append(p)
        elif nac < len(rr):
            mixed.append(p)
    rng.shuffle(mixed); rng.shuffle(allfail)
    mixed, allfail = sorted(mixed[:n_mixed]), sorted(allfail[:n_allfail])
    print(f"[P2-lcb] mixed={len(mixed)} allfail={len(allfail)} (from {len(ready)} ready problems)", flush=True)

    done = load_done_keys(out_path) if resume else set()
    tasks = []
    MAX_FAILS = 2
    for pid in mixed + allfail:
        rr = ready[pid]
        row = lcb_rows[pid]
        orig_msgs = row["prompt"]
        base_rate = sum(1 for r in rr if r["verdict"] == "AC") / len(rr)
        ac = [r for r in rr if r["verdict"] == "AC"]
        demo = _clean_demo([r["completion"] for r in ac], strip_thinking)
        # real judged failures only (WA/RE/TLE); exclude NO_CODE cap-clip/format, same as OJBench
        fails = [r for r in rr if r["verdict"] in REAL_FAIL_VERDICTS][:MAX_FAILS]
        for r in fails:
            fk = r["n_sample"]
            native_fb = r.get("feedback", "") or ""
            arms = []
            if demo is not None:
                arms.append(("sol", dict(demo_text=demo, feedback_raw=None)))
                arms.append(("sol_fb", dict(demo_text=demo, feedback_raw=native_fb)))
            arms.append(("fb_native", dict(demo_text=None, feedback_raw=native_fb)))
            for arm, kw in arms:
                arm_key = f"{arm}#f{fk}"
                if (pid, arm_key, fk) in done:
                    continue
                teacher_msgs = build_teacher_messages(orig_msgs, **kw)
                tasks.append((pid, arm, arm_key, fk, teacher_msgs, base_rate, row))
    print(f"[P2-lcb] {len(tasks)} teacher tasks remaining after resume", flush=True)

    def do(task):
        pid, arm, arm_key, fk, teacher_msgs, base_rate, row = task
        msgs = teacher_msgs if isinstance(teacher_msgs, list) else [{"role": "user", "content": teacher_msgs}]
        text, ntok = chat_once(client, model, msgs, max_tokens, enable_thinking=False)
        verdict, reward, feedback = judge_lcb(text, row["ground_truth"], row["extra_info"])
        return {
            "probe": "P2-teacher-edge", "env": "lcb", "problem_id": pid, "arm": arm_key,
            "n_sample": fk, "messages": msgs, "completion": text,
            "n_tokens": ntok, "verdict": verdict, "reward": reward,
            "feedback": feedback, "ts": time.time(),
            "base_rate": base_rate, "arm_family": arm,
        }
    _run_pool(tasks, do, out_path, concurrency=8, label="P2-lcb")


# --- shared concurrent pool with per-completion write --------------------
def _run_pool(tasks, do_fn, out_path, concurrency=4, label=""):
    if not tasks:
        print(f"[{label}] nothing to do (all resumed).", flush=True)
        return
    lock_written = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(do_fn, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
            except Exception as e:
                t = futs[fut]
                print(f"[{label}] TASK FAILED {t[:3] if isinstance(t,tuple) else t}: {e}", flush=True)
                continue
            append_record(out_path, rec)
            lock_written += 1
            el = time.time() - t0
            print(f"[{label}] {lock_written}/{len(tasks)} pid={rec['problem_id']} "
                  f"arm={rec['arm']} verdict={rec['verdict']} ntok={rec['n_tokens']} "
                  f"({el:.0f}s elapsed)", flush=True)
    print(f"[{label}] DONE {lock_written}/{len(tasks)} in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, choices=["p1b", "p2-ojbench", "p2-lcb", "p3"])
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--n-subset", type=int, default=40)
    ap.add_argument("--n-sample", type=int, default=8)
    args = ap.parse_args()

    client = make_client(args.base_url)
    resume = not args.no_resume
    ENVDIFF.mkdir(parents=True, exist_ok=True)

    if args.probe == "p1b":
        out = args.out or (ENVDIFF / "env_diff_lcb_base.jsonl")
        run_p1b(client, args.model, out, n_subset=args.n_subset, n_sample=args.n_sample,
                max_tokens=args.max_tokens or 8192, resume=resume)
    elif args.probe == "p2-ojbench":
        out = args.out or (ENVDIFF / "env_diff_p2_teacher.jsonl")
        run_p2_ojbench(client, args.model, out, max_tokens=args.max_tokens or 20480, resume=resume)
    elif args.probe == "p2-lcb":
        out = args.out or (ENVDIFF / "env_diff_p2_teacher.jsonl")
        run_p2_lcb(client, args.model, out, max_tokens=args.max_tokens or 8192, resume=resume)
    elif args.probe == "p3":
        out = args.out or (ENVDIFF / "env_diff_p3_prompt.jsonl")
        run_p3(client, args.model, out, n_sample=args.n_sample,
               max_tokens=args.max_tokens or 8192, resume=resume)


if __name__ == "__main__":
    main()
