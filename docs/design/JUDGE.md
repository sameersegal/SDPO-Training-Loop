# Component design — the OJBench judge & SDPO feedback

> A **component design doc**, kept independent of any iteration (see `CLAUDE.md` → "Component
> design docs"). It captures *why* the judge and feedback are built this way, for the blog post
> and for future iterations. Code: `src/ojbench_eval.py`, `src/sdpo_ojbench.py`. Tested by
> `tests/test_judge_feedback.py`, `tests/test_hints.py`, `tests/test_splits.py`.

## 1. What the judge does
Given a model completion (a fenced code block) and a problem id, run the code against that
problem's test cases and return **`(reward, verdict, feedback_text)`**:
- **verdict** ∈ `AC` (all cases pass) · `WA` (wrong answer) · `RE` (runtime error / non-zero exit) ·
  `TLE` (timeout) · `CE` (C++ compile error) · `NO_CODE` (no parseable code block).
- **reward** — a scalar training signal (see §4).
- **feedback_text** — natural-language environment output for the SDPO teacher (see §5).

Two languages: **Python** = run `python3 sol.py`, normalized stdout diff (`judge_solution`).
**C++** = `g++ -O2 -std=c++17` then run the binary (`judge_cpp`). This is a *lightweight* judge,
**not** the official DMOJ sandbox (DMOJ needs root + pypy3, unavailable on our boxes).
**Judge fidelity = reward fidelity** — a false AC trains the wrong thing, so we only use
**diff-checkable** problems (no special-judge / interactive / float-tolerance).

## 2. Public / private split (anti-reward-hacking)
Each problem's test cases are split **deterministically 50/50** into **public** (the training
reward signal + the source of any leaked hint/feedback) and **private** (held back for eval). A
model that hardcodes public outputs fails private → caught at eval. The split is per-problem and
seeded (`public_private_cases`).

**Mirror-language invariant:** Python and C++ prompts are exact mirrors of the same problem (same
statement, same cases — only the code fence differs). So the **train/eval split is by problem id**,
never by `(problem, language)`; both languages of a problem live in the same split, or the model
would see a solution at train time. Enforced by `tests/test_splits.py`.

## 3. Case ordering — smallest input first
The judge sorts cases by **input size ascending** before running. Two wins:
- **Interpretable feedback:** a failure surfaces the *smallest* failing case (e.g. `5 8 7 3 4 →
  expected 7470, got 0`), which the model can actually reason about — not a 10 MB stress test
  whose expected output (`180`) is unlearnable.
- **Fail fast:** cheap cases run first; a buggy solution is rejected without paying for the big cases.

Ordering never changes the **AC** verdict (AC requires *all* cases) and is applied *after* the
public/private split, so it cannot move cases between splits.

## 4. Reward design — dense fraction vs binary
`reward_mode` (on `judge_completion` / `make_reward_func`):
- **`fraction`** (default): the **true** number of cases passed ÷ total (runs *all* cases,
  `count_all=True`). Distinguishes "passes 8/10" from "1/10".
- **`binary`**: `1.0` iff AC else `0.0`.
- AC is always `1.0`.

**Why dense is the default.** GRPO/SDPO advantages are computed *within a generation group* (the
G rollouts of one prompt) and are normalized by the group's reward std. With **binary** reward, a
group where no rollout fully solves the problem is **all-zeros → zero variance → zero advantage →
no policy-gradient signal** — exactly the all-fail medium/hard groups that stalled iteration 01.
A **dense** reward gives those groups *variance* (one rollout passes 6/10, another 2/10) → a
non-zero gradient that points toward more-correct trajectories, even before any rollout is AC.

**Costs / caveats.** `fraction` runs every case (no early-exit) → more compute, and pays the
timeout once per TLE case. It is a *partial-credit* signal, so monitor the **public-vs-private
pass gap** (hardcoding risk). Note dense reward feeds the *policy gradient*; SDPO's
`use_successful_as_teacher` still gates the *distillation teacher* on a full AC
(`success_reward_threshold`), so partial credit alone does not create a teacher — that's what live
judge feedback (§5) is for. **Eval is unaffected**: pass@k / pass@1 key on the *verdict* (`AC`),
not on this reward.

## 5. Feedback design — the SDPO teacher signal
`_format_feedback(verdict, detail)` turns a verdict into environment text the SDPO teacher can be
conditioned on (`docs/EXPERIMENT.md` §6, `src/sdpo_prompts.py`):
- **AC** → "All public tests passed."
- **WA** → `Verdict: WA. Failing test '<case>'. Input: … Expected output: … Your output: …`
  (the expected output is a *corrective leak* from a small public case).
- **RE** → verdict + failing input + stderr.
- **TLE** → verdict + "exceeded the time limit" (a complexity signal).
- **NO_CODE** → explains the required ```` ```python ```` fence.

This text is what distinguishes **copy-only** SDPO (teacher only shows a *successful* rollout) from
**live-feedback** SDPO (teacher is told *why the attempt failed*). Iteration 01 was copy-only
(`include_environment_feedback=False`) and mode-collapsed; iteration 02 wires this feedback in.
**Known limitation:** the `NO_CODE` message is hardcoded "Python" even for C++ (we prototype
feedback Python-first).

## 6. Worked-example hint (test-case extraction)
~17% of NOI problems have an `### Example` header with **no content**.
`augment_prompt_with_example` fills it with the **smallest public** (input, output) case — format
grounding, low risk (one small case ≠ a lookup table; suppressed if even the smallest case exceeds
`max_input_bytes`). What is *not* worth leaking: the giant stress-test inputs/outputs (unlearnable,
pure memorization). The detection regex must use a **lookahead** for the next header — an earlier
version consumed it and silently matched nothing (see `tests/test_hints.py`).

## 7. Liveness, not security
Model code runs **without a security sandbox** (ephemeral cloud containers — a bad generation just
kills a disposable container). But every execution runs as a **subprocess with a wall-clock timeout
+ memory rlimit** for *liveness*: one infinite-loop or fork-bomb generation must not stall the
training rollout (the cpp-judge hang that cost us time in iteration 01 was a liveness failure).
