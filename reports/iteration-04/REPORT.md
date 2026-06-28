# Iteration 04 — Consolidated paper recommendations (rank-ordered for a <24h run)

**Status: PLANNED / planning doc.** This is not an experiment report yet — it consolidates the
**10 deep reads in `knowledge/`** into one rank-ordered action list and picks what to actually do in
the **<24-hour** window we have to show something substantial. It supersedes the scattered
"things to try" lists at the bottom of each `summary_*.md`.

Builds on [iteration 01](../iteration-01/REPORT.md) (real base pass@k frontier; 100-step easy-only
**collapse**), [iteration 02](../iteration-02/REPORT.md) (live feedback **stopped the collapse**, did
not beat base), and [iteration 03](../iteration-03/REPORT.md) (moving-frontier curriculum + dense
reward + 206-pool — **de-risked but not yet run**; open decisions still open).

---

## The strategic call (read this first)

With **<24h**, the deliverable is **one well-instrumented curriculum run that either beats base on
held-out pass@k or produces a clean, publishable null** — not a new trainer. So the prioritization is
ruthlessly **gain × feasibility-in-24h**, which pushes the big architectural papers (SD-Zero,
RLCSD, NCA) to iteration-05 and pulls *cheap single-knob changes + near-free instrumentation* to the
front. Crucially, **the free diagnostics turn even a null result into a finding** — that is the
insurance that makes the 24h non-wasted regardless of which way pass@k moves.

**The make-or-break gate, do it first (§T0-1):** the original SDPO paper's scaling curve says SDPO's
edge over GRPO *vanishes or reverses below ~8B* and Gemma-4-E2B is a ~2B-class model. If the
self-teacher is **not** more accurate than the student on our problems+feedback, SDPO has **nothing to
distill at our scale** — and *that* is itself the substantial result. Measure it before paying for a
long run.

---

## Cross-paper consensus (why these are high-conviction)

Ten papers, read independently, converge on the same handful of levers. The vote counts are the
signal — these are not one-paper hunches:

| Lever | Papers that vote for it | Our current state | Cost |
|---|---|---|---|
| **Teacher should be FIXED / periodically-refreshed, not continuous EMA** | selfdistill_degrades, SDPG, feedback_alignment, opd_length_inflation, rlcsd, SD-Zero (**6**); original SDPO dissents *only* for small-α EMA on code | `teacher_model_kind="ema"`, α unknown (`sdpo_train.py:133`) | 1 knob |
| **Distillation must be a small, scheduled, verifier-dominated nudge — not the objective** | SDPG ("single highest-leverage"), rlcsd (modulate, don't replace), opd_length_inflation, original SDPO (hybrid λ≈0.9) | `distillation_weight=1.0` "pure" (`sdpo_train.py:130`), unscheduled | 1 knob + schedule callback |
| **Add a frozen-base KL anchor** | SDPG, opd_length_inflation, when_context_returns (NCA variant); iter-02/03 plan already lists it | **not wired** (no `--kl` flag) | small |
| **Two-sided length/entropy/trunc/rep canary as a kill signal** | opd_length_inflation, SDPG, rlcsd, selfdistill_degrades, when_context_returns | length logged; `finish_reason` captured but not aggregated | ~free |
| **Feedback should be trace-aligned (anchor correct prefix, localize the fix), not a fresh solution** | feedback_alignment (headline), SD-Zero (verdict-keyed P_r), original SDPO (LeetCode-shaped) | outcome-aligned only (`_format_feedback`, `sdpo_ojbench.py:182`) | prompt logic |
| **Broaden coverage to the learnability frontier (mixed-variance groups)** | every paper; SDPG/feedback_alignment make it the filter | **implemented** (dense reward, solvability probe) | done |
| **Use the failed rollouts as a contrastive negative** | rlcsd (headline) | thrown away (positive-only teacher) | new trainer |
| **Self-teacher edge shrinks at small scale → SDPO+GRPO hybrid as insurance** | original SDPO | pure SDPO | depends on TRL |

---

## Rank-ordered recommendations

Ranked by **(expected gain on held-out pass@k or on the publishable story) × (feasibility in 24h)**.
T0 = do before launch (hours, near-free). T1 = cheap, do if time. T2 = the run itself. T3 = deferred
to iteration-05 (too big to build+smoke+Modal in 24h).

### T0 — Before launch: near-free, foundational, high-conviction

**T0-1. Self-teacher-vs-student accuracy probe — THE GATE.**
*Gain: decisive / Cost: ~1h, free.* Take a handful of failed OJBench rollouts, hand the model the
judge feedback in the reprompt template, and measure the **teacher's one-shot accuracy vs the
student's**. If the teacher is not meaningfully better, SDPO has no signal at Gemma-E2B scale →
**stop and report that** (a real result; predicted by the original SDPO scaling curve, Fig. 8).
Reuses the existing rollout + `judge_completion()` path. *(original SDPO)*

**T0-2. Teacher EMA → fixed (or pin α≈0.01 and verify).**
*Gain: high / Cost: 1 knob.* The **single most-voted change (6 papers)**. EMA *amplifies* the
collapse feedback loop (confident → teacher → more confident). Switch `teacher_model_kind` to
fixed-initial, or — if we keep EMA — first **confirm what α TRL actually uses** and pin it ≈0.01
(the original paper's value; under-regularized EMA is the dangerous case). `sdpo_train.py:133`.
*(selfdistill_degrades, SDPG, feedback_alignment, opd_length_inflation, rlcsd, SD-Zero)*

**T0-3. Demote distillation: weight 1.0 → ~1e-2, verifier-dominant, + warmup-decay schedule.**
*Gain: high / Cost: 1 knob + a `TrainerCallback`.* Called out as "probably the single highest-leverage
change." `distillation_weight=1.0` is ~100–1000× too strong relative to SDPG's β≈1e-3 with the
**outcome reward as the O(1) primary term**. Drop it 1–3 orders of magnitude and ramp 0→small→0
(`T_warm`≈first 10–15% of steps, decay to 0 near the end). `sdpo_train.py:130`.
*(SDPG, rlcsd, opd_length_inflation, original SDPO)*

**T0-4. Two-sided collapse canary (length + TruncRate + RepRate + entropy).**
*Gain: high on the *story* / Cost: ~free.* Iteration-01 collapsed *terse*; opd_length_inflation
collapses *verbose* — same failure class, opposite sign, and **the loss is blind to both**. Aggregate
`finish_reason=="length"` → TruncRate (already captured, `sdpo_eval_vllm.py:57`); zlib-compress the
last 10k chars → RepRate (~5 lines); log policy entropy per step. Treat a sharp move in **either**
length direction (or an entropy crash) as an early kill signal in the "watch the first few steps"
budget rule. **This is what makes a null run publishable.** *(opd_length_inflation, SDPG, rlcsd,
selfdistill_degrades)*

### T1 — Cheap, high-value, do if time before/with the launch

**T1-1. Measure context-induced harm on the existing iter-01/02 adapters — NO retraining.**
*Gain: high on the story / Cost: an eval variant, free.* Run held-out problems twice — as deployed
(no context) and with the solution/feedback re-fed (as the teacher saw it). Compute
**Harm = P(fail-with-context | pass-without-context)**, Acc_x vs Acc_x,c, Δlen. If iter-01 sits in
Regime A (high harm), that's independent confirmation our collapse is context-induced degradation —
a clean diagnostic finding with zero training cost. *(when_context_returns)*

**T1-2. Frozen-base KL anchor (β small).**
*Gain: medium-high / Cost: small.* Already in the iter-02/03 plan, **not yet wired**. Caps policy
drift, protects easy + GSM8K (iter-01's regression). KL alone is a "modest patch" (opd_length: 28.0→
29.7) — the heavy lifting is the mixture/curriculum we already have — but it's cheap regression
insurance. *(SDPG, opd_length_inflation)*

**T1-3. Trace-aligned, verdict-keyed feedback.**
*Gain: medium-high / Cost: prompt-assembly, unit-testable.* Upgrade `_format_feedback`
(`sdpo_ojbench.py:182`) toward StepAlignFB for code: include the **student's own code verbatim up to
the failing region**, then **describe (don't paste)** the buggy tail + the failing input/expected/got;
make the control phrase verdict-specific ("WA on test 3, fix it" / "TLE, make it faster"). The sharp
rule: **never quote the *erroneous* span verbatim** (induction-head copying reinforces the bug). Near-
miss mediums (passed 16/20) are the ideal case. Pure logic → unit test, no Modal. *(feedback_alignment,
SD-Zero, original SDPO)*

**T1-4. Audit the reprompt template for the known footgun.**
*Gain: prevents a silent regression / Cost: a code read.* Verify the teacher prompt does **not** paste
the student's raw attempt into the *user* turn (the original paper's clearest negative result: entropy
0.41→0.23, 44.5 vs 48.3). It should re-score the attempt as the assistant turn only. `sdpo_feedback.py`
`_LiveFeedbackBuilder`. *(original SDPO, feedback_alignment)*

### T2 — The run (the actual deliverable)

**T2-1. Launch the iteration-03 moving-frontier curriculum with T0/T1 fixes baked in.**
*This is what produces "something substantial."* Dense reward (✅ implemented), feedback-ON, variance>0
frontier band, **fixed teacher + low scheduled distillation + KL anchor + canaries** from T0. Resolve
iteration-03's open decisions for the 24h budget: **probe n=16**; **seed the band with the reachable
hards** from the pass@16/32 probe (don't wait for the frontier to migrate there); a **short schedule**
(one or two rounds); **early-stop on held-out pass@k**, not loss. Run the watchdog-protected Modal
pre-flight (`--max-steps 3 --save-steps 1`, ~15min/~$2) first, then the real run.

**T2-2. Eval base + adapter on the 53-problem held-out** (py & cpp × difficulty, pass@k n=16) + GSM8K,
on the **same** vLLM `--enable-lora` server. Report deltas, band trajectory, hard movement, and **all
the T0-4 canaries** so a null is still a finding.

### T3 — Deferred to iteration-05 (high gain, but cannot build+smoke+Modal in 24h)

- **RLCSD contrastive teacher** — use `G−` (the failed rollouts we already generate) as a negative
  reference to cancel style-drift; verifier-anchored modulation (judge sets direction, distillation
  only scales magnitude). Highest-gain *structural* change, but a new `SDPOTrainer` subclass + K
  negatives + Modal. *(rlcsd)*
- **SD-Zero reviser warm-start (two-phase)** — the principled fix for our **0-successful-rollout
  gotcha**: a reviser teacher conditions on the *wrong* attempt + its verdict, so it gives signal even
  when nothing passes. Two phases + new data path + SFT — too big for 24h. *(SD-Zero)*
- **No-Context Anchoring (NCA)** — `+β·KL(sg[q_x]‖q_c)`, one extra forward pass; bolt-on to
  `FeedbackSDPOTrainer`. Do **T1-1 (measure harm) first** — only worth it if harm is real. *(when_context_returns)*
- **Rock-token diagnostic + freeze** — ~1.5× Modal speedup by freezing gradients on inert structural
  tokens; but in *code* a "rock" (`{`, `;`) may be compilation-critical, so knockout-validate before
  freezing. Efficiency, not capability. *(opd_rock_tokens)*
- **SDPO+GRPO hybrid (λ≈0.9)** — the original paper's explicit recommendation for sub-8B models; the
  scale-insurance if T0-1 shows a weak self-teacher. Depends on TRL support. *(original SDPO)*
- **OPSDL distilled-spec teacher** — a "cleaner not smarter" teacher context (I/O spec + one worked
  example, story stripped); cheap prompt change, but secondary to the above. *(opsdl_long_context)*

---

## The structural risk that overhangs everything

The original SDPO paper's scaling result is the elephant: **the marginal improvement of SDPO over GRPO
is tightly coupled with base-model strength**, and at Qwen2.5-1.5B (closest data point to our
Gemma-4-E2B) **SDPO underperforms GRPO**. Everything above assumes the self-teacher has *something* to
distill. **T0-1 tests exactly that assumption** — which is why it is ranked first and gates the spend.
If the teacher edge is absent, the iteration-04 result is "SDPO doesn't help at this scale; here is the
measurement," plus the SDPO+GRPO hybrid as the iteration-05 path. That is a substantial, honest result
— not a failed run.

## One-line summary

In <24h: **probe the self-teacher edge (gate), flip EMA→fixed, cut distillation to a small scheduled
verifier-dominated nudge, add a KL anchor and a two-sided collapse canary, make feedback trace-aligned**
— then launch the already-de-risked moving-frontier curriculum and judge it on held-out pass@k, with
the canaries making even a null publishable. Contrastive/reviser/NCA trainers are iteration-05.
