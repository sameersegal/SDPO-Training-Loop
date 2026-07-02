"""Rollout-analysis toolkit for studying the SDPO brevity collapse.

Every SDPO training rollout (and eval sample) is a full model completion with the schema
    step, problem_id, language, sample_k, verdict, reward_fraction, reward_binary,
    success, teacher_eligible, n_tokens, n_chars, feedback, completion
Qwen3 completions are `<think> …long reasoning… </think> …prose… ```code``` …`, so the central
question — *what does the model drop as it collapses?* — is answered by splitting each completion into
its **thinking** vs **code** parts and tracking them across training steps.

These functions are the building blocks used by `notebooks/rollout_analysis.ipynb`. They take/return
pandas DataFrames and are side-effect-free except the `plot_*` helpers. Load with `load_rollouts()`.

    from rollout_analysis import load_rollouts, think_vs_code_trajectory, plot_think_vs_code
    df = load_rollouts("runs/iteration-10/evaldata/iter10_32k_rollouts.jsonl")
    plot_think_vs_code(think_vs_code_trajectory(df))
"""
from __future__ import annotations

import json
import re
from math import comb

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- parsing

_FENCE = re.compile(r"```(?:python|cpp|c\+\+|c|java)?\s*(.*?)```", re.S)


def split_think_code(completion: str) -> dict:
    """Decompose a completion into thinking / code / prose (post-think, non-code) char counts.

    thinking = everything up to the first `</think>` (the reasoning budget the model chose to spend);
    code = the LAST fenced block (the actual submission); prose = the rest after </think>.
    `think_unclosed` flags a completion that never closed <think> (often a truncated/degenerate one).
    """
    t = completion or ""
    if "</think>" in t:
        think, rest = t.split("</think>", 1)
        think = think.replace("<think>", "")
        unclosed = False
    else:
        think, rest, unclosed = t, "", ("<think>" in t)
    blocks = _FENCE.findall(t)
    code = blocks[-1] if blocks else ""
    prose = _FENCE.sub("", rest)
    return {
        "think_chars": len(think),
        "code_chars": len(code),
        "prose_chars": len(prose),
        "n_code_blocks": len(blocks),
        "think_frac": len(think) / max(1, len(t)),
        "think_unclosed": unclosed,
        "has_code": bool(blocks),
    }


# ---------------------------------------------------------------- loaders

def load_rollouts(path: str) -> pd.DataFrame:
    """Load a training-rollout JSONL into a DataFrame, enriched with think/code split columns."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    tc = df["completion"].apply(split_think_code).apply(pd.Series)
    return pd.concat([df.drop(columns=["completion"]), tc, df[["completion"]]], axis=1)


def load_eval_samples(path: str, tag: str | None = None) -> pd.DataFrame:
    """Load an eval-sample JSONL (id, verdict, n_chars, completion) with think/code split.

    NB: samples generated with `--no-judge` carry verdict='SKIP'; judge them first (judge_local.py)
    or pass the matching judged `sdpo_passk_<tag>.json` to `attach_verdicts()`.
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    tc = df["completion"].apply(split_think_code).apply(pd.Series)
    out = pd.concat([df.drop(columns=["completion"]), tc, df[["completion"]]], axis=1)
    if tag:
        out["tag"] = tag
    return out


def attach_verdicts(samples: pd.DataFrame, judged_json: str) -> pd.DataFrame:
    """Fill SKIP verdicts on --no-judge samples from a judged sdpo_passk_<tag>.json (per-problem verdict list)."""
    res = json.load(open(judged_json))["results"]
    vmap = {(r["id"], r["language"]): r["verdicts"] for r in res}
    out = samples.copy()
    out["verdict"] = [vmap.get((r.id, r.language), ["?"] * 99)[int(r.sample_k)]
                      if (r.id, r.language) in vmap else r.verdict
                      for r in out.itertuples()]
    return out


# ---------------------------------------------------------------- analyses (return tidy frames)

def length_trajectory(df: pd.DataFrame, col: str = "n_tokens") -> pd.DataFrame:
    """Per-step p10/median/p90 of completion length — the collapse curve."""
    g = df.groupby("step")[col]
    return pd.DataFrame({
        "p10": g.quantile(0.10), "median": g.median(), "p90": g.quantile(0.90),
        "mean": g.mean(), "n": g.size(),
    }).reset_index()


def think_vs_code_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """Per-step mean THINK chars vs CODE chars — does reasoning collapse while code holds?"""
    g = df.groupby("step")
    return pd.DataFrame({
        "think_chars": g["think_chars"].mean(),
        "code_chars": g["code_chars"].mean(),
        "think_frac": g["think_frac"].mean(),
        "unclosed_frac": g["think_unclosed"].mean(),
    }).reset_index()


def verdict_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Per-step fraction of each verdict (AC/WA/TLE/RE/NO_CODE/…) — the quality shift."""
    return (df.groupby("step")["verdict"].value_counts(normalize=True)
            .unstack(fill_value=0.0).reset_index())


def length_by_verdict(df: pd.DataFrame, col: str = "n_tokens") -> pd.DataFrame:
    """Length distribution grouped by verdict — do longer completions pass more often?"""
    return (df.groupby("verdict")[col]
            .agg(["count", "mean", "median",
                  lambda s: s.quantile(0.10), lambda s: s.quantile(0.90)])
            .rename(columns={"<lambda_0>": "p10", "<lambda_1>": "p90"}).reset_index())


def teacher_length_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Per-step mean length of teacher-eligible vs non-teacher rollouts.

    SDPO self-distills FROM the teacher-eligible (successful) rollouts. If teachers are systematically
    SHORTER, the model is being taught to be brief — the collapse mechanism, made measurable.
    """
    g = df.groupby(["step", "teacher_eligible"])["n_tokens"].mean().unstack()
    g.columns = [f"teacher={c}" for c in g.columns]
    return g.reset_index()


def truncation_analysis(df: pd.DataFrame, cap_tokens: int, near: float = 0.98) -> pd.DataFrame:
    """Per-step fraction of completions AT/near the token cap and their pass rate.

    Tests the cap-accelerant hypothesis: at a low cap, long reasoning gets truncated → fails → SDPO
    teaches brevity. `capped_frac` = fraction with n_tokens ≥ near*cap; `capped_ac_rate` = their AC rate.
    """
    d = df.copy()
    d["capped"] = d["n_tokens"] >= near * cap_tokens
    g = d.groupby("step")
    out = pd.DataFrame({
        "capped_frac": g["capped"].mean(),
        "capped_ac_rate": d[d.capped].groupby("step")["verdict"].apply(lambda s: (s == "AC").mean()),
        "uncapped_ac_rate": d[~d.capped].groupby("step")["verdict"].apply(lambda s: (s == "AC").mean()),
    }).reset_index()
    return out


def problem_trajectory(df: pd.DataFrame, pid: int) -> pd.DataFrame:
    """One problem across steps: mean length, AC rate, think/code split — how a single problem degrades."""
    d = df[df.problem_id == pid]
    g = d.groupby("step")
    return pd.DataFrame({
        "n_tokens": g["n_tokens"].mean(),
        "ac_rate": g["verdict"].apply(lambda s: (s == "AC").mean()),
        "think_chars": g["think_chars"].mean(),
        "code_chars": g["code_chars"].mean(),
        "n": g.size(),
    }).reset_index()


def _pass_at_k(n, c, k):
    k = min(k, n)
    return 1.0 if n - c < k else 1.0 - comb(n - c, k) / comb(n, k)


def pass_at_k(df: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    """pass@k per problem from an eval-sample frame (needs real verdicts, not SKIP)."""
    def agg(g):
        n = len(g); c = int((g.verdict == "AC").sum())
        return pd.Series({"n": n, "n_ac": c, f"pass@{k}": _pass_at_k(n, c, k)})
    return df.groupby(["id", "difficulty"] if "difficulty" in df else "id").apply(agg).reset_index()


def find_contrast_examples(base: pd.DataFrame, trained: pd.DataFrame, k: int = 8) -> pd.DataFrame:
    """Problems base solves but the trained model fails, ranked by AC drop — pick these to read the
    reasoning that got compressed away. Expects judged eval frames (real verdicts)."""
    # eval-sample frames carry n_chars (not n_tokens, which is training-rollout only)
    def acr(d):
        return d.groupby("id").apply(lambda g: pd.Series(
            {"ac": int((g.verdict == "AC").sum()), "n": len(g),
             "med_think": g["think_chars"].median(), "med_chars": g["n_chars"].median()}),
            include_groups=False)
    b, t = acr(base), acr(trained)
    m = b.join(t, lsuffix="_base", rsuffix="_trained")
    m["ac_drop"] = m["ac_base"] - m["ac_trained"]
    m["think_shrink"] = m["med_think_base"] / m["med_think_trained"].clip(lower=1)
    return m.sort_values("ac_drop", ascending=False).reset_index()


# ---------------------------------------------------------------- plot helpers (matplotlib)

def plot_length_trajectory(traj, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    ax.fill_between(traj.step, traj.p10, traj.p90, alpha=0.2, color="steelblue", label="p10–p90")
    ax.plot(traj.step, traj["median"], "o-", color="steelblue", label="median")
    ax.set_xlabel("training step"); ax.set_ylabel("completion length (tokens)")
    ax.set_title("Length collapse over training"); ax.legend(); ax.grid(alpha=0.3)
    return ax


def plot_think_vs_code(tvc, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    ax.plot(tvc.step, tvc.think_chars / 1000, "o-", color="crimson", lw=2, label="THINKING (k chars)")
    ax.plot(tvc.step, tvc.code_chars / 1000, "s-", color="seagreen", lw=2, label="CODE (k chars)")
    ax.set_xlabel("training step"); ax.set_ylabel("mean length (k chars)")
    ax.set_title("What collapses: reasoning vs code"); ax.legend(); ax.grid(alpha=0.3)
    return ax


def plot_verdict_mix(vm, ax=None):
    import matplotlib.pyplot as plt
    ax = ax or plt.gca()
    order = [c for c in ["AC", "WA", "TLE", "RE", "MLE", "NO_CODE", "ERR", "SKIP"] if c in vm]
    ax.stackplot(vm.step, *[vm[c] for c in order], labels=order, alpha=0.85)
    ax.set_xlabel("training step"); ax.set_ylabel("verdict fraction"); ax.set_ylim(0, 1)
    ax.set_title("Verdict mix over training"); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
    return ax
