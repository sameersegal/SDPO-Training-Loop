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


def judge_cpp(code, cases, timeout):
    """Compile with g++17 then run binary against cases. Returns (verdict,passed,total,detail)."""
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
        cases = sorted(cases, key=lambda c: c[0].stat().st_size)  # smallest input first (see judge_solution)
        for k, (infile, outfile) in enumerate(cases):
            try:
                proc = subprocess.run([str(binp)], input=infile.read_bytes(),
                                      capture_output=True, timeout=timeout, preexec_fn=_limit)
            except subprocess.TimeoutExpired:
                return "TLE", k, len(cases), {"failing_case": infile.name}
            if proc.returncode != 0:
                return "RE", k, len(cases), {"failing_case": infile.name,
                                             "stderr": _clip(proc.stderr.decode("utf-8", "replace"))}
            got = normalize(proc.stdout.decode("utf-8", "replace"))
            exp = normalize(outfile.read_text(encoding="utf-8", errors="replace"))
            if got != exp:
                return "WA", k, len(cases), {"failing_case": infile.name,
                                             "expected": _clip(exp), "got": _clip(got)}
        return "AC", len(cases), len(cases), {}
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


def judge_completion(text, pid, which="public", timeout=6.0, language="python"):
    """Returns (reward, verdict, feedback_text).
    reward = fraction of cases passed (dense); verdict in AC/WA/RE/TLE/CE/NO_CODE."""
    pub, prv = public_private_cases(pid)
    cases = pub if which == "public" else prv
    if language == "cpp":
        code = extract_code_cpp(text)
        verdict, passed, total, detail = judge_cpp(code, cases, timeout)
    else:
        code = extract_code(text)
        verdict, passed, total, detail = judge_solution(code, cases, timeout)
    reward = passed / total if total else 0.0
    if verdict == "AC":
        reward = 1.0
    feedback = _format_feedback(verdict, detail)
    return reward, verdict, feedback


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


def make_reward_func(which="public", timeout=6.0):
    """TRL reward_func(completions, **kwargs) -> list[float]. The 'id' and
    'language' dataset columns are forwarded by TRL via kwargs (per rollout)."""
    def reward_func(completions, id=None, language=None, **kwargs):
        out = []
        for k, (comp, pid) in enumerate(zip(completions, id)):
            lang = language[k] if language is not None else "python"
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            r, _, _ = judge_completion(text, int(pid), which=which, timeout=timeout, language=lang)
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
