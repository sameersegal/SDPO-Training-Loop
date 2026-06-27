#!/usr/bin/env python3
"""OJBench environment adapter for SDPO.

Provides:
  - problem loading from ojb_splits.json (train / heldout)
  - per-problem public/private test split (paper protocol: public = 50% subset
    used as training feedback; private = held back for validation)
  - judge_completion(): run a model completion's code against a problem's PUBLIC
    (or private) cases, returning (reward, verdict, feedback_text)
  - build_dataset(): HF Dataset of {prompt(messages), id} for SDPOTrainer
  - make_reward_func(): a TRL-compatible reward function closure

Reuses the judging primitives from ojbench_eval.py.
"""
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import tempfile

from ojbench_eval import (extract_code, extract_tests, judge_solution, normalize,
                          register_testdirs, _clip, _limit, _run_capped)
from _paths import find_file, ojbench_dir

ROOT = Path(tempfile.gettempdir())  # scratch dir for transient _cpp_* compile artifacts
# OJB_SPLITS selects the dataset: "ojb_splits.json" (NOI-only, default) or
# "ojb_splits_full.json" (NOI+ICPC, diff-checkable; built by build_splits_full.py).
SPLITS = json.load(open(find_file(os.environ.get("OJB_SPLITS", "ojb_splits.json"))))
# part-aware test-data resolution when the splits carry a registry (full dataset).
if "testdir_by_id" in SPLITS:
    register_testdirs({int(k): ojbench_dir() / v for k, v in SPLITS["testdir_by_id"].items()})
PROMPT_BY_ID = {int(k): v for k, v in SPLITS["py_prompt_by_id"].items()}
CPP_PROMPT_BY_ID = {int(k): v for k, v in SPLITS["cpp_prompt_by_id"].items()}
DIFF_BY_ID = {int(k): v for k, v in SPLITS["by_id"].items()}

_CPP_FENCE = re.compile(r"```(?:cpp|c\+\+|cc|c)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_cpp(text):
    blocks = _CPP_FENCE.findall(text or "")
    return blocks[-1].strip() if blocks else None


def judge_cpp(code, cases, timeout, count_all=False):
    """Compile with g++17 then run the binary against cases, smallest input first.
    count_all=False: early-exit (prefix passed). count_all=True: run all (true passed,
    smallest failing case in detail). See judge_solution for the count_all rationale."""
    if not code:
        return "NO_CODE", 0, len(cases), {"reason": "no code block found"}
    base = ROOT / f"_cpp_{os.getpid()}_{time.perf_counter_ns()}"
    src, binp = base.with_suffix(".cpp"), base
    src.write_text(code)
    try:
        cp = _run_capped(["g++", "-O2", "-std=c++17", "-o", str(binp), str(src)], None, 60)
        if cp.returncode != 0:
            return "CE", 0, len(cases), {"stderr": _clip(cp.stderr.decode("utf-8", "replace"))}
        cases = sorted(cases, key=lambda c: c[0].stat().st_size)  # smallest input first
        passed = 0
        first_fail = None
        for infile, outfile in cases:
            try:
                proc = _run_capped([str(binp)], infile.read_bytes(), timeout)
            except subprocess.TimeoutExpired:
                fail = ("TLE", {"failing_case": infile.name})
            else:
                exp = normalize(outfile.read_text(encoding="utf-8", errors="replace"))
                if proc.returncode != 0:
                    fail = ("RE", {"failing_case": infile.name,
                                   "stderr": _clip(proc.stderr.decode("utf-8", "replace"))})
                elif normalize(proc.stdout.decode("utf-8", "replace")) != exp:
                    fail = ("WA", {"failing_case": infile.name, "expected": _clip(exp),
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
        src.unlink(missing_ok=True)
        binp.unlink(missing_ok=True)

# --- per-problem public/private split (deterministic) ---------------------
_split_cache = {}


def public_private_cases(pid, public_frac=0.5, seed=0):
    if pid in _split_cache:
        return _split_cache[pid]
    cases = extract_tests(pid)            # list[(in_path, out_path)]
    idx = list(range(len(cases)))
    random.Random(seed + pid).shuffle(idx)
    n_pub = max(1, int(len(cases) * public_frac))
    pub = [cases[i] for i in sorted(idx[:n_pub])]
    prv = [cases[i] for i in sorted(idx[n_pub:])] or pub
    _split_cache[pid] = (pub, prv)
    return pub, prv


# --- worked-example hint: fill an EMPTY "### Example" section -----------------
# ~17% of NOI problems have an Example header but no content. Injecting the
# smallest PUBLIC case as a worked example is format grounding (low risk: one
# small case is not a lookup table; private cases still require a real solution).
_EXAMPLE_RE = re.compile(r"(#+[ \t]*Examples?\b[^\n]*\n)(.*?)(?=\n#+[ \t]|\Z)", re.DOTALL | re.IGNORECASE)


def example_section_is_empty(prompt):
    m = _EXAMPLE_RE.search(prompt)
    return bool(m) and m.group(2).strip() == ""


def inject_example(prompt, input_text, output_text):
    """Fill an empty Example section with one worked (input, output) case. No-op if
    there is no empty Example header. Uses re.sub with a function (no str.format),
    so braces/% in the case content are safe."""
    block = f"\nInput:\n```\n{input_text.rstrip()}\n```\n\nOutput:\n```\n{output_text.rstrip()}\n```\n"
    if not example_section_is_empty(prompt):
        return prompt
    return _EXAMPLE_RE.sub(lambda m: m.group(1) + block, prompt, count=1)


def augment_prompt_with_example(prompt, pid, language="python", which="public", max_input_bytes=400):
    """If the Example section is empty, inject the SMALLEST public case as a worked
    example. No-op otherwise, or if even the smallest case is too large to be a hint.
    Only PUBLIC cases are leaked."""
    if not example_section_is_empty(prompt):
        return prompt
    pub, prv = public_private_cases(pid)
    cases = pub if which == "public" else prv
    smallest = min(cases, key=lambda c: c[0].stat().st_size)
    if smallest[0].stat().st_size > max_input_bytes:
        return prompt
    inp = smallest[0].read_text(encoding="utf-8", errors="replace")
    out = smallest[1].read_text(encoding="utf-8", errors="replace")
    return inject_example(prompt, inp, out)


def judge_completion(text, pid, which="public", timeout=6.0, language="python",
                     reward_mode="fraction", max_case_bytes=None, max_cases=None):
    """Returns (reward, verdict, feedback_text). verdict in AC/WA/RE/TLE/CE/NO_CODE.

    reward_mode:
      "fraction" (default): dense reward = TRUE fraction of cases passed (runs all
        cases). Distinguishes "passes 8/10" from "1/10" -> non-zero intra-group
        variance even when no rollout is AC, which gives GRPO/SDPO an advantage signal
        on all-fail groups. Costs more (no early-exit).
      "binary": reward = 1.0 iff AC (all cases pass) else 0.0. Cheapest/strictest;
        but all-fail groups have zero variance -> no policy-gradient signal.
    AC always yields reward 1.0. Eval (pass@k / pass@1) keys on the VERDICT, not this
    reward, so eval is unaffected by reward_mode.

    max_case_bytes / max_cases bound the IN-LOOP reward to small, smallest-first cases.
    Some OJBench problems ship multi-hundred-MB to 1.4 GB test sets (inputs up to 74 MB);
    dense reward (count_all) runs EVERY case, so judging those across the per-step batch
    grew unboundedly slow and eventually hung a step (binary reward early-exited before
    reaching them, which is why this only bit with fraction reward). Capping to small cases
    keeps the dense signal (fraction of small cases passed) while making judging cheap and
    hang-proof. Eval keeps the FULL set (no caps) — it's not in the hot loop."""
    pub, prv = public_private_cases(pid)
    cases = pub if which == "public" else prv
    if max_case_bytes or max_cases:
        cases = sorted(cases, key=lambda c: c[0].stat().st_size)  # smallest first
        if max_case_bytes:
            small = [c for c in cases if c[0].stat().st_size <= max_case_bytes]
            cases = small or cases[:1]  # always keep >=1 (the smallest) so total>0
        if max_cases:
            cases = cases[:max_cases]
    count_all = reward_mode == "fraction"
    if language == "cpp":
        verdict, passed, total, detail = judge_cpp(extract_code_cpp(text), cases, timeout, count_all=count_all)
    else:
        verdict, passed, total, detail = judge_solution(extract_code(text), cases, timeout, count_all=count_all)
    if verdict == "AC":
        reward = 1.0
    elif reward_mode == "binary":
        reward = 0.0
    else:  # fraction
        reward = passed / total if total else 0.0
    # Pass-rate line is only honest in fraction mode (count_all ran EVERY case, so
    # `passed` is the TRUE count). In binary mode `passed` is a cheap prefix count
    # (cases cleared before the early-exit failure) -> omit it to avoid misleading
    # the teacher with a fake percentage.
    pr = (passed, total) if count_all else (None, None)
    return reward, verdict, _format_feedback(verdict, detail, passed=pr[0], total=pr[1])


def _format_feedback(verdict, detail, passed=None, total=None):
    """Textual environment output for the SDPO self-teacher reprompt.

    When passed/total are given (dense/fraction mode only — see judge_completion),
    a pass-rate line is added so the teacher sees HOW CLOSE the attempt was: a
    "passed 16/20 (80%)" rollout is a near-miss worth nudging, vs "1/20" which is
    fundamentally off. This proximity signal is what dense reward exposes; surfacing
    it in the text gives the self-teacher the same information the advantage does.
    """
    if verdict == "AC":
        return "All public tests passed."
    if verdict == "NO_CODE":
        return ("Your response did not contain a Python code block in the "
                "required ```python ...``` format, so it could not be run.")
    parts = [f"Verdict: {verdict}."]
    if passed is not None and total:
        parts.append(f"Passed {passed}/{total} tests ({100 * passed // total}%).")
    if "failing_case" in detail:
        parts.append(f"Failing test '{detail['failing_case']}'.")
    if "input" in detail:
        parts.append(f"Input:\n{detail['input']}")
    if verdict == "WA":
        parts.append(f"Expected output:\n{detail.get('expected','')}")
        parts.append(f"Your output:\n{detail.get('got','')}")
    elif verdict == "RE":
        parts.append(f"Runtime error:\n{detail.get('stderr','')}")
    elif verdict == "TLE":
        parts.append("Your program exceeded the time limit on this test.")
    return "\n".join(parts)


# --- canonical system prompts (shared by training, eval, probe) -----------
# Keep these in ONE place so the prompt the model is trained on matches the one it
# is probed + evaluated on. iteration-02 used EXPERT_SYS; iteration-03 defaults to
# CP_METHOD_SYS (targets base's over-theorizing on hard: implement the *stated* rule,
# tier by the Data Range table, output code — see reports/iteration-03/REPORT.md §Pre-flight).
EXPERT_SYS = ("You are an expert competitive programmer. First reason briefly about the "
              "algorithm and its time complexity given the input limits, then write a single "
              "correct, efficient solution that reads from stdin and writes to stdout in the "
              "exact required format.")

CP_METHOD_SYS = (
    "You are an expert competitive programmer. Follow this method, then output only the final code:\n"
    "1. Restate the exact rule, recurrence, or process the problem defines — do NOT try to guess a "
    "closed-form pattern when the problem already states the rule. Implement the stated rule.\n"
    "2. Read the Data Range / Constraints table. Decide per input size which method is needed: a "
    "direct O(n) simulation is correct and sufficient for small n; only the largest limits "
    "(e.g. n up to 1e18) require a faster technique (matrix exponentiation, cycle/period detection, "
    "or a closed form).\n"
    "3. Write ONE solution that simulates directly for small n and switches to the faster method "
    "only when n is too large to loop. Apply the modulus throughout. Read stdin, write stdout in the "
    "exact format. Output only the code.")

SYSTEM_PROMPTS = {"cp_method": CP_METHOD_SYS, "expert": EXPERT_SYS, "none": None}


# --- dataset + reward func for TRL ---------------------------------------
def build_dataset(split="train", difficulties=None, languages=("python",), system=None, ids=None):
    """Build a dataset over (problem, language) pairs. Each row carries a
    'language' column so the reward function judges in the right language.

    system: optional system-message text prepended to every prompt (keeps the TRAIN
      prompt identical to the eval/probe prompt — iteration-03 trains under CP_METHOD_SYS).
    ids: optional explicit pid list (e.g. the frontier band) overriding the split's ids.
    """
    from datasets import Dataset
    pool = ids if ids is not None else SPLITS[split]
    ids = [i for i in pool if difficulties is None or DIFF_BY_ID[i] in difficulties]
    sys_msg = [{"role": "system", "content": system}] if system else []
    rows = {"prompt": [], "id": [], "difficulty": [], "language": []}
    for lang in languages:
        pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
        for i in ids:
            if i not in pmap:
                continue
            rows["prompt"].append(sys_msg + [{"role": "user", "content": pmap[i]}])
            rows["id"].append(i)
            rows["difficulty"].append(DIFF_BY_ID[i])
            rows["language"].append(lang)
    return Dataset.from_dict(rows)


def reward_case_caps():
    """In-loop reward case caps from env (keep judging cheap + hang-proof on huge test
    sets). Defaults: cases ≤ 1 MB, at most 20 (smallest-first). 0/empty disables a cap."""
    import os
    mb = int(os.environ.get("SDPO_MAX_CASE_BYTES", "1000000")) or None
    mc = int(os.environ.get("SDPO_MAX_CASES", "20")) or None
    return mb, mc


def make_reward_func(which="public", timeout=6.0, reward_mode="fraction"):
    """TRL reward_func(completions, **kwargs) -> list[float]. The 'id' and
    'language' dataset columns are forwarded by TRL via kwargs (per rollout).
    reward_mode: "fraction" (dense passed/total, default) or "binary" (AC=1 else 0).

    Judges the group's completions concurrently (subprocess judge releases the GIL);
    dense judging is the per-step bottleneck on hard problems. SDPO_JUDGE_WORKERS env
    caps the pool (default 16). ex.map preserves order. Cases are size/count-capped
    (reward_case_caps) so multi-hundred-MB test sets can't stall a step."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = int(os.environ.get("SDPO_JUDGE_WORKERS", "16"))
    mb, mc = reward_case_caps()

    def reward_func(completions, id=None, language=None, **kwargs):
        def judge_k(k):
            comp = completions[k]
            lang = language[k] if language is not None else "python"
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            r, _, _ = judge_completion(text, int(id[k]), which=which, timeout=timeout,
                                       language=lang, reward_mode=reward_mode,
                                       max_case_bytes=mb, max_cases=mc)
            return float(r)

        n = len(completions)
        with ThreadPoolExecutor(max_workers=min(workers, max(1, n))) as ex:
            return list(ex.map(judge_k, range(n)))
    reward_func.__name__ = f"ojbench_{which}_reward"
    return reward_func


if __name__ == "__main__":
    # smoke check
    print("train:", len(SPLITS["train"]), "heldout:", len(SPLITS["heldout"]))
    pid = SPLITS["train"][0]
    pub, prv = public_private_cases(pid)
    print(f"problem {pid}: {len(pub)} public / {len(prv)} private cases")
