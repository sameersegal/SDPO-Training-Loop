# Iteration 04 — Insights: when can SDPO actually work?

Distilled from the per-token advantage diagnostic ([REPORT.md](REPORT.md)) and the SD literature
([`knowledge/`](../../knowledge/)). The question this answers: *is it fair to say SDPO works if the
teacher (conditioned on privileged context `c`) can most likely generate the solution?*

**Short answer: that's a necessary condition, not a sufficient one.** Our iteration-04 data shows
SDPO failing in two distinct ways even when the teacher is perfectly capable of producing a solution.

---

## The precondition (what SDPO distills)

SDPO trains the student (no context) to behave as if it had seen the privileged context `c`. The
per-token signal is

> **A_t = log π(ŷ_t | x, c) − log π(ŷ_t | x)** — "how much does the hint change my mind about this token."

So the floor requirement for *any* learning signal is:

> **the teacher-with-`c` must be a better predictor of the correct continuation than the student-without-`c`.**

If conditioning on `c` makes the model reliably solve what it otherwise couldn't, there is a real
capability gap to transfer and the gradient has signal. The original SDPO paper's scaling curve warns
this gap **vanishes or reverses below ~8B**, which is the prime suspect for why Gemma-4-E2B (~2B) has
not worked for us — and is exactly what the accuracy probe below measures.

"Teacher can generate the solution" is the right instinct, but it splits into two abilities, and only
one of them makes SDPO work. The two insights below are those two failure modes.

---

## Insight 1 — The gap must come from `c` *resolving the student's specific gap*, not from `c` handing over the answer.

**Explanation.** There are two very different reasons a teacher-with-`c` can produce a correct
solution:
- **(a) `c` resolves the student's specific error** — a critique that says "your steps 1–3 are right,
  step 4 is where it breaks." The teacher now completes *the student's own trace* correctly, so the
  advantage is **negative exactly at the error and positive/zero on the correct prefix** — a free
  process-reward signal. This is what makes SDPO teach.
- **(b) `c` hands over an answer** — a full reference solution (or an AC group-mate's code). The
  teacher can trivially generate *a* correct solution, but it's *its own* solution in *its own*
  surface form, which diverges from the student's rollout **even where the student was right**. The
  advantage goes **diffusely negative across the whole rollout**, pushing against correct tokens. The
  signal conflates "you erred here" with "I'd have written it differently."

Capability to generate ≠ ability to correct *this* student. Mode (b) can be net-harmful.

**Our evidence (easy panel, loj-2314).** All 8 base rollouts AC'd, so our gating fed the teacher a
*different* AC group-mate's code (`solution` context — mode (b)). The teacher was maximally capable
(it had a correct solution in context), yet the advantage was **broadly negative across its own
correct code**: 45% of tokens negative, mean A_t = −0.38, spikes to −21.6. A capable teacher *fighting
a correct rollout* — the RefSol red flag from the feedback-alignment paper, observed directly in our
pipeline. This is also the token-level picture of iteration-01's collapse mechanism.

> Practical rule: condition on `c` that **anchors the student's correct prefix and localizes the
> correction**, never a free-standing alternative solution. For code: quote the student's own code up
> to the failing region, then *describe* (don't paste) the buggy tail.

---

## Insight 2 — The signal must be *strong and localized*.

**Explanation.** Even with the right *kind* of context, two things must hold for the gradient to teach:
- **Strong** — conditioning on `c` must actually shift the teacher's distribution. If `mean|A_t| ≈ 0`,
  the teacher-with-`c` is barely more confident about the right tokens than the student — there is
  nothing to distill, regardless of how correct `c` is. A near-zero advantage is the diagnostic
  reading of "the model can't *use* this context at this scale."
- **Localized** — the magnitude must concentrate at the actual error tokens, not smear across the
  rollout (Insight 1) and not sit in the reasoning prose while the buggy code goes untouched.

**Our evidence (medium loj-2086, hard loj-2083 — feedback context).** Both were all-fail groups, so
the teacher saw our judge feedback (`Verdict: TLE. Passed X/Y. Failing test… Input… Expected… Got…`).
The advantage was:

| arm | difficulty | mean \|A_t\| | mean A_t | reasoning \|A\| | code-region \|A\| | reading |
|---|---|---:|---:|---:|---:|---|
| solution (mode b) | easy | **0.449** | −0.381 | 0.559 | 0.178 | strong but **diffuse/suppressive** (Insight 1) |
| feedback | medium | **0.022** | −0.000 | 0.027 | 0.013 | **~20× too weak**, mean≈0, not at the code |
| feedback | hard | **0.019** | −0.001 | 0.022 | 0.006 | **~20× too weak**, mean≈0, not at the code |

The feedback arm is ~20× weaker than the solution arm *and* what little signal exists sits slightly
**more in the reasoning prose than in the code** (where the bug is). So the arm that fires on hard
problems — exactly where we need new capability — is a faint, mis-aimed nudge. That is the
token-level cause of why iteration-02 stopped the collapse but did not beat base.

A second-order point: the dominant base failure here is **TLE** (the algorithm is correct but too
slow), not WA. For TLE there is no single "error token" — the whole approach is wrong — so even
perfect trace alignment has nothing local to point at. Approach-level failures need approach-level
feedback ("your solution is O(n²); the n≤1e6 limit needs O(n log n)"), not a token-localized fix.

---

## Synthesis — the fair statement

SDPO can work when the teacher-with-`c`:
1. is **substantially more likely to produce the correct solution than the student** (capability gap —
   the scale-gated precondition), **and**
2. gets there by **correcting the student's own trace** so credit localizes (Insight 1), **and**
3. produces a signal **strong enough to shift the distribution** and aimed at the real error
   (Insight 2), on tokens the student can actually reach (learnability frontier).

Your phrasing nails (1). Iteration-04 shows (1) alone is not enough: we observed failure through
(2) (easy: capable teacher, diffuse/harmful signal) and through the strength half of (3) (medium/hard:
feedback too weak to lift the teacher).

---

## The accuracy probe (does the gap even exist at our scale?)

Condition (1) is directly measurable: for failed rollouts, compare **P(teacher solves | x, c)** against
**P(student solves | x)**. If the teacher-with-feedback's accuracy is meaningfully higher, the gap
exists and SDPO has fuel; if it is flat, SDPO has nothing to distill at this scale — itself a
publishable result (and the reason iteration-05 moves to Qwen3-8B). Script:
[`src/probe_teacher_accuracy.py`](../../src/probe_teacher_accuracy.py).

<!-- PROBE_RESULTS -->
**Result (Gemma-4-E2B, 1 problem/difficulty, 8 attempts each; full table:
[`data/teacher_accuracy_probe.md`](data/teacher_accuracy_probe.md)):**

| difficulty | context `c` | student AC | teacher-with-`c` AC | student mean-reward | teacher mean-reward | gap (reward) |
|---|---|---:|---:|---:|---:|---:|
| easy (loj-2314) | solution | 8/8 | — *(skipped)* | 1.00 | — | — |
| medium (loj-2086) | feedback | **0/8** | **0/8** | 0.521 | 0.438 | **−0.083** |
| hard (loj-2083) | feedback | **0/8** | **0/8** | 0.438 | 0.612 | +0.174 |

**Reading: there is no solve-rate gap.** Conditioning the teacher on our judge feedback let it solve
**0/8** on both medium and hard — *identical* to the student (0/8). Feedback does not turn a failure
into a solution at this scale. The partial-credit (mean-reward) signal is small and **sign-inconsistent
across the two problems** (medium −0.08, hard +0.17) — i.e. noise around zero, not a capability gain;
crucially, the hard bump converted **zero** rollouts to AC. The verdict distributions show only a
lateral shuffle, not progress: hard went student `TLE×5, WA×3` → teacher `TLE×7, WA×1` (a few WA→TLE,
still no AC); medium even regressed (`WA×2, TLE×6` → `TLE×5, WA×1, RE×2`).

This is the **independent, solve-rate confirmation** of the weak-signal finding (`mean|A_t| ≈ 0.02`):
**at Gemma-4-E2B (~2B) scale, the self-teacher conditioned on our feedback is not a better solver than
the student, so SDPO has essentially nothing to distill on the hard cases.** It matches the original
SDPO paper's scaling curve (the self-teacher edge vanishes below ~8B) and is the empirical basis for
iteration-05 moving to **Qwen3-8B** — and for pairing it with **trace-aligned feedback** (Insight 1)
so that, where a gap does exist, the signal localizes instead of diffusing.

**Caveats.** One problem per difficulty × 8 attempts is indicative, not definitive — the reward deltas
are within sampling noise and the decisive metric (AC) is flat zero. The solution arm (easy) is
skipped because the student already solves it (the context just hands over an answer). A stronger
version would sweep more problems and add the LLM-critic context as a third condition (the iteration-05
Phase-0 gate does exactly this). The AC-flat-at-zero result is robust to all of that.
