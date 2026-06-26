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

from ojbench_eval import extract_code, extract_tests, judge_solution, normalize, _clip, _limit
from _paths import find_file

ROOT = Path(tempfile.gettempdir())  # scratch dir for transient _cpp_* compile artifacts
SPLITS = json.load(open(find_file("ojb_splits.json")))
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
        cp = subprocess.run(["g++", "-O2", "-std=c++17", "-o", str(binp), str(src)],
                            capture_output=True, timeout=60)
        if cp.returncode != 0:
            return "CE", 0, len(cases), {"stderr": _clip(cp.stderr.decode("utf-8", "replace"))}
        cases = sorted(cases, key=lambda c: c[0].stat().st_size)  # smallest input first
        passed = 0
        first_fail = None
        for infile, outfile in cases:
            try:
                proc = subprocess.run([str(binp)], input=infile.read_bytes(),
                                      capture_output=True, timeout=timeout, preexec_fn=_limit)
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
                     reward_mode="fraction"):
    """Returns (reward, verdict, feedback_text). verdict in AC/WA/RE/TLE/CE/NO_CODE.

    reward_mode:
      "fraction" (default): dense reward = TRUE fraction of cases passed (runs all
        cases). Distinguishes "passes 8/10" from "1/10" -> non-zero intra-group
        variance even when no rollout is AC, which gives GRPO/SDPO an advantage signal
        on all-fail groups. Costs more (no early-exit).
      "binary": reward = 1.0 iff AC (all cases pass) else 0.0. Cheapest/strictest;
        but all-fail groups have zero variance -> no policy-gradient signal.
    AC always yields reward 1.0. Eval (pass@k / pass@1) keys on the VERDICT, not this
    reward, so eval is unaffected by reward_mode."""
    pub, prv = public_private_cases(pid)
    cases = pub if which == "public" else prv
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
    return reward, verdict, _format_feedback(verdict, detail)


def _format_feedback(verdict, detail):
    """Textual environment output for the SDPO self-teacher reprompt."""
    if verdict == "AC":
        return "All public tests passed."
    if verdict == "NO_CODE":
        return ("Your response did not contain a Python code block in the "
                "required ```python ...``` format, so it could not be run.")
    parts = [f"Verdict: {verdict}."]
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


# --- dataset + reward func for TRL ---------------------------------------
def build_dataset(split="train", difficulties=None, languages=("python",)):
    """Build a dataset over (problem, language) pairs. Each row carries a
    'language' column so the reward function judges in the right language."""
    from datasets import Dataset
    ids = [i for i in SPLITS[split]
           if difficulties is None or DIFF_BY_ID[i] in difficulties]
    rows = {"prompt": [], "id": [], "difficulty": [], "language": []}
    for lang in languages:
        pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
        for i in ids:
            if i not in pmap:
                continue
            rows["prompt"].append([{"role": "user", "content": pmap[i]}])
            rows["id"].append(i)
            rows["difficulty"].append(DIFF_BY_ID[i])
            rows["language"].append(lang)
    return Dataset.from_dict(rows)


def make_reward_func(which="public", timeout=6.0, reward_mode="fraction"):
    """TRL reward_func(completions, **kwargs) -> list[float]. The 'id' and
    'language' dataset columns are forwarded by TRL via kwargs (per rollout).
    reward_mode: "fraction" (dense passed/total, default) or "binary" (AC=1 else 0)."""
    def reward_func(completions, id=None, language=None, **kwargs):
        out = []
        for k, (comp, pid) in enumerate(zip(completions, id)):
            lang = language[k] if language is not None else "python"
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            r, _, _ = judge_completion(text, int(pid), which=which, timeout=timeout,
                                       language=lang, reward_mode=reward_mode)
            out.append(float(r))
        return out
    reward_func.__name__ = f"ojbench_{which}_reward"
    return reward_func


if __name__ == "__main__":
    # smoke check
    print("train:", len(SPLITS["train"]), "heldout:", len(SPLITS["heldout"]))
    pid = SPLITS["train"][0]
    pub, prv = public_private_cases(pid)
    print(f"problem {pid}: {len(pub)} public / {len(prv)} private cases")
