#!/usr/bin/env python3
"""Thin wrapper around the SDPO paper's own LCB judge (feedback/code.py).

Imports their compute_score by FILE PATH to bypass verl/__init__ (which pulls in ray),
exactly as runs/replication-02/DATA_VERIFICATION.md proved works locally. Also loads the
LCBv6 parquets and maps a problem row -> (prompt messages, ground_truth tests, extra_info).

We evaluate with extra_info["split"]="test" so compute_score forces sparse_rewards
(reward 1.0 iff ALL tests pass) — the eval regime, matching P1a's AC/not-AC semantics.

Returns a normalized (verdict, reward, feedback) so the generation harness records the
same schema across both envs. Verdict vocabulary is normalized to the OJBench set where
possible: AC / WA / RE / TLE / NO_CODE.
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path("/home/sameersegal/Code/SparkyCoder")
SDPO_REPO = ROOT / "runs/replication-02/SDPO"
CODE_PY = SDPO_REPO / "verl/utils/reward_score/feedback/code.py"
DATA_DIR = ROOT / "runs/replication-02/data"

_code_mod = None
_rows_cache = {}


def _load_code_module():
    global _code_mod
    if _code_mod is not None:
        return _code_mod
    spec = importlib.util.spec_from_file_location("lcb_feedback_code", str(CODE_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _code_mod = mod
    return mod


def load_lcb_rows(which="test"):
    """Return list of dicts per LCB problem: {index, prompt(messages), ground_truth, extra_info}.

    which: "train" (public 50% tests) or "test" (full tests). We use "test" for judging.
    """
    if which in _rows_cache:
        return _rows_cache[which]
    import pyarrow.parquet as pq
    tbl = pq.read_table(str(DATA_DIR / f"{which}.parquet"))
    d = tbl.to_pylist()
    rows = []
    for r in d:
        # prompt is a list of {role, content}
        prompt = [{"role": m["role"], "content": m["content"]} for m in r["prompt"]]
        rm = r["reward_model"]
        ei = dict(r["extra_info"])
        rows.append({
            "index": ei.get("index"),
            "prompt": prompt,
            "ground_truth": rm["ground_truth"],
            "extra_info": ei,
        })
    _rows_cache[which] = rows
    return rows


def _normalize_verdict(result):
    """Map the LCB compute_score dict onto OJBench-style verdict labels."""
    if result["acc"] == 1.0:
        return "AC"
    if result.get("incorrect_format"):
        return "NO_CODE"
    if result.get("timed_out"):
        return "TLE"
    if result.get("error_in_test_cases"):
        return "RE"
    return "WA"


def judge_lcb(completion_text, ground_truth, extra_info, split="test"):
    """Judge one LCB completion. Returns (verdict, reward, feedback_text).

    split="test" -> compute_score forces sparse_rewards (all-pass=1.0). We use the
    same ground_truth tests as the test.parquet full set for an eval verdict.
    """
    mod = _load_code_module()
    ei = dict(extra_info)
    ei["split"] = split
    result = mod.compute_score(completion_text, ground_truth, extra_info=ei, sparse_rewards=(split == "test"))
    verdict = _normalize_verdict(result)
    return verdict, float(result["score"]), result.get("feedback", "") or ""


if __name__ == "__main__":
    rows = load_lcb_rows("test")
    print(f"loaded {len(rows)} LCB test rows")
    r = rows[0]
    print("index:", r["index"], "| n tests in gt:", len(json.loads(r["ground_truth"]).get("inputs", [])))
    # smoke: judge an obviously-wrong completion
    v, rew, fb = judge_lcb("```python\nprint('nope')\n```", r["ground_truth"], r["extra_info"])
    print("smoke verdict:", v, "reward:", rew, "| feedback[:120]:", repr(fb[:120]))
