"""LLM-as-critic environment feedback for SDPO (iteration 05).

The deterministic judge tells the SDPO self-teacher *that* a rollout failed and
hands it the failing I/O (`_format_feedback` in `sdpo_ojbench.py`). iteration-04's
per-token diagnostic showed that signal is RefSol-leaning: either diffuse/suppressive
or ~20x too weak and aimed at the prose, not the code (reports/iteration-04/INSIGHTS.md).

This module produces the StepAlignFB-style feedback the literature says actually teaches
(knowledge/summary_feedback_alignment_sd.md): a *trace-aligned critique* that anchors the
student's correct prefix and localizes the correction at the first wrong step, turning
self-distillation into a free process-reward signal. A frozen Claude model is the critic
(the "teacher-of-the-teacher"); Qwen3-8B itself remains the self-teacher conditioned on
`x + critique`.

THE TWO RULES (from the paper's induction-head analysis — get these wrong and the signal
diffuses or reinforces the bug):
  1. Anchor the student's CORRECT prefix; localize the correction at the first wrong step.
     Do NOT hand over a full alternative solution (that is RefSol / mode-b, net-harmful:
     a capable teacher fighting a correct rollout).
  2. DESCRIBE the buggy tail; never paste a verbatim "corrected line". A verbatim quote of
     the wrong region makes the model copy the bug.
For TLE (the dominant hard failure, no local error token) give APPROACH-level feedback
("your loop is O(n^2); n<=1e6 needs O(n log n)"), not a token-localized fix.

Usage in the live training path: `sdpo_feedback.make_feedback_reward_func` computes the
deterministic `(reward, verdict, fb)` per rollout; for failed rollouts, swap `fb` for
`critique(...)`. On ANY API error this returns the deterministic `fb` unchanged, so a
critic outage never stalls a training step.
"""
from __future__ import annotations

# Quality/cost balance for a per-rollout critic in a training loop (knowledge paper used a
# 32B critic; Sonnet 4.6 is the closest practical analog). Haiku is the cheap fallback.
DEFAULT_CRITIC_MODEL = "claude-sonnet-4-6"
CHEAP_CRITIC_MODEL = "claude-haiku-4-5"

_CRITIC_SYSTEM = """\
You are a competitive-programming tutor giving feedback on a student's FAILED attempt at a \
problem. You are given the problem, the student's submitted code, and the judge's verdict. You do \
NOT have the reference solution — diagnose the flaw from the attempt and the verdict alone.

The judge verdicts mean:
- AC  = Accepted (all tests pass)            - WA  = Wrong Answer (output mismatch)
- TLE = Time Limit Exceeded (too slow)       - RE  = Runtime Error (crash / non-zero exit)
- CE  = Compile Error (C++)                  - NO_CODE = no parseable code block

Your feedback becomes a teaching signal, so HOW you write it matters as much as what you say. \
Follow these rules exactly:

1. ANCHOR THE CORRECT PREFIX. Identify the part of the student's reasoning/code that is correct \
and say so explicitly, referring to their own code ("your reading of the input and your prefix-sum \
setup are correct"). This tells the learner which steps to keep.

2. LOCALIZE THE FIRST ERROR. Pinpoint the single place where the attempt first goes wrong — the \
specific line, condition, loop bound, or step — and explain WHY it is wrong, grounded in the \
failing test when one is given.

3. DESCRIBE THE FIX; DO NOT WRITE IT OUT. Explain in words what needs to change. Do NOT paste a \
corrected line of code, and do NOT provide a full or partial corrected solution. The learner must \
re-derive the fix themselves; handing over code teaches them to copy, not to reason.

4. MATCH THE FAILURE TYPE:
   - WA: point at the faulty logic that produces the wrong output on the shown input.
   - TLE: this is an APPROACH problem, not a single bad line. State the complexity of their \
approach and what asymptotic complexity the constraints require ("your solution is O(n^2); with \
n up to 1e6 you need about O(n log n)"), and name the technique class that gets there without \
writing the algorithm.
   - RE: identify the operation that faults (out-of-bounds index, null/empty access, overflow, \
division by zero) and the condition under which it triggers.
   - CE: name the compilation error and the language rule it violates.

5. OUTPUT ONLY THE FINAL FEEDBACK. Direct second-person ("your attempt"). No preamble, no restating \
the problem, no encouragement filler — and critically, do NOT narrate your reasoning or self-correct \
in the response (no "wait", "actually", "let me reconsider", no exploring then rejecting alternatives). \
State your conclusion directly and confidently."""

# Appended in terse mode — the A/B counterpart to the (verbose) default, to test in Phase 0
# whether a tighter signal teaches the student as well while saving tokens (reports/iteration-05).
_TERSE_DIRECTIVE = """

BREVITY MODE: Your entire response must be AT MOST 3 short sentences — one naming what is correct, \
one naming the specific error, one naming the fix direction. No code, no lists, nothing else."""

_USER_TEMPLATE = """\
PROBLEM ({lang}):
{problem}

STUDENT SUBMITTED CODE:
```{lang}
{student_code}
```

JUDGE RESULT:
{judge_feedback}

Write the trace-aligned feedback now, following the rules. Anchor what is correct, localize the \
first error, describe (do not write) the fix."""


def build_critic_messages(problem, student_code, verdict, judge_feedback, lang, style="verbose"):
    """Pure prompt assembly — returns (system, messages). Offline-testable (no API call).

    `judge_feedback` is the deterministic `_format_feedback` text (verdict + failing I/O),
    reused verbatim as the critic's raw material. `verdict` is passed through for callers
    that want to gate (e.g. skip the critic on NO_CODE/CE) but is not re-templated here.
    `style`: "verbose" (default) or "terse" (≤3 sentences) — the Phase-0 A/B knob.
    """
    system = _CRITIC_SYSTEM + (_TERSE_DIRECTIVE if style == "terse" else "")
    user = _USER_TEMPLATE.format(
        lang=lang,
        problem=problem,
        student_code=student_code or "(no parseable code block)",
        judge_feedback=judge_feedback,
    )
    return system, [{"role": "user", "content": user}]


def critique(
    problem,
    student_code,
    verdict,
    judge_feedback,
    lang,
    *,
    style="verbose",
    model=DEFAULT_CRITIC_MODEL,
    max_tokens=1024,
    thinking=False,
    timeout=30.0,
    max_retries=2,
    client=None,
):
    """Trace-aligned critique for one failed rollout.

    Returns the critique text on success, or `judge_feedback` (the deterministic fallback)
    on ANY error — so the training loop never stalls on a critic outage. AC/NO_CODE rollouts
    should not reach here; if they do, the deterministic text is returned unchanged.

    `client` may be injected (tests / shared client); otherwise a lazily-imported, env-keyed
    anthropic.Anthropic() is created. `thinking=True` uses adaptive thinking (higher quality,
    slower/costlier per rollout) — off by default to keep per-step latency bounded.
    """
    if verdict in ("AC", "NO_CODE"):
        return judge_feedback
    try:
        if client is None:
            import anthropic
            from _paths import load_env
            load_env()  # picks up ANTHROPIC_API_KEY from repo-root .env (no-op in Modal)
            client = anthropic.Anthropic(timeout=timeout, max_retries=max_retries)
        system, messages = build_critic_messages(
            problem, student_code, verdict, judge_feedback, lang, style=style
        )
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages)
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        resp = client.messages.create(**kwargs)
        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text").strip()
        return text or judge_feedback  # empty (e.g. refusal) -> deterministic fallback
    except Exception as e:  # noqa: BLE001 - any failure must fall back, never raise
        print(f"[critic] falling back to deterministic feedback: {type(e).__name__}: {e}",
              flush=True)
        return judge_feedback
