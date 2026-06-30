# Retrospective — arriving at the iteration-07 insight faster and cheaper

> **Scope.** A cross-iteration post-mortem written after iteration-07. The question: *in
> hindsight, what could we have done differently to reach the same (or better) insight in fewer
> iterations and less cost?* Grounded in the committed per-iteration `REPORT.md`/`PROVENANCE.md`,
> `docs/FINDINGS.md`, and the `knowledge/` literature folder. Dollar figures are the measured
> Modal spend recorded in each iteration's provenance; the small/noisy eval probes carry the
> caveats noted in their own reports.

## The one-sentence version

Seven iterations and ~$130–180 of Modal spend converged on a single mechanism:

> **Flat reward groups kill the GRPO policy gradient, so self-distillation dominates and
> collapses output diversity — and training on the _frontier band_ (sometimes-solvable problems)
> restores within-group reward variance and reverses the collapse.**

That mechanism was **written down as the root cause at iteration-02** ("binary reward = zero
variance on all-pass groups = no policy gradient") and **designed as the fix at iteration-03**
(moving-frontier band + dense reward). The project then spent iterations 04→06 — and the large
majority of the money — rediscovering it at 8B scale before finally applying it at iteration-07.

**The honest headline: the answer was in our own notes by iteration-03; iterations 04–06 were
largely a detour around a fix we had already specified.**

## The arc, compressed

| Iter | One change | Outcome | Spend | Wasted |
|---|---|---|---|---|
| 01 | easy-only SDPO, Gemma-4-E2B | 20-step null; **100-step collapse** (pass@8 20→8, GSM8K −3.6pt; terse mode-collapse) | ~$15 | GB10 hang |
| 02 | live judge feedback | collapse **prevented**, no gain; **names the root cause** (flat binary groups → no policy gradient) | ~$10–20 | — |
| 03 | moving-frontier + dense reward | **designed & de-risked, never run** | $0 | (later ~$30 of integration false-starts) |
| 04 | per-token advantage diagnostic | judge feedback ~20× too weak & diffuse; no teacher>student gap at 2B | $0 (local) | — |
| 05 | Qwen3-8B + LLM critic | **held-out null HID a real train regression** (pass@8 0.83→0.50); mode-collapse at 8B | ~$45 | ~$22 |
| 06 | lower LR + cosine warmup | partial mitigation (Δ−0.33 → −0.167); `--beta` found **inert** in SDPOTrainer | ~$19 | ~$5.6 |
| 07 | **frontier band** (35 sometimes-solvable) | **first run to beat base** (pass@8 0.67→0.75, Δ+0.083); flat_group 0.75→0.44 | ~$15 | $0 |

Trend across the 8B runs: **−0.33 → −0.167 → +0.083**, monotonic with `flat_group_fraction`
falling 0.75→0.44. The lever was always the data (reward-group variance), not the model, the
critic, or the LR.

> Probe caveat (carried from the iter-05/06/07 reports): the train==eval probe is 12 problems,
> n=8; the *base* pass@8 swings ±0.15 across runs. Trust the within-run Δ and the monotonic ckpt
> trend, not cross-run base comparisons.

## What we would do differently — ranked by leverage

### 1. Trust the iter-02 root cause; run the iter-03 plan immediately
Iteration-02 already recorded that flat binary reward groups give zero advantage variance and no
policy gradient — *the* iter-07 finding. Iteration-03 then fully designed and GB10-de-risked the
fix (frontier band + dense reward) and shelved it. Running iter-03 as designed — cheaply, at 2B —
most plausibly collapses 04/05/06 into one step. **Estimated saving: ~3 iterations.**

### 2. Make `train==eval probe + pass@k + length/entropy` the day-one eval harness
The most expensive misdirection was iter-05's held-out result reading as a clean null
(pass@8 0.60→0.60) while the model had actually mode-collapsed (train-distribution 0.83→0.50).
That ambiguity is part of why iter-05 cost ~$45. The train==eval probe — the thing that revealed
*every* real result in 05/06/07 — is cheap and should have been standing from iter-01. Likewise
pass@k: we *knew* greedy was ±2/25 noise before iter-01 and still let it hide the 100-step
collapse. An entropy/length canary catches the brevity spiral by ~step 4.

### 3. Read `knowledge/` before iteration-01, not after iteration-02
The literature we collected already predicted what we hit empirically: SDPO is fragile below ~7B
(so 2B was always borderline); EMA teachers amplify the collapse feedback loop; the learnability
frontier is the correct training filter (Feedback-Alignment, SDPG); all-fail groups give no
signal (SD-Zero); pass@k/entropy are the collapse canaries (Length-Inflation, SDPG). iter-01's
collapse and the iter-07 fix were both foreseeable from a half-day read. The folder was assembled
*at iteration-03* — one collapse too late.

### 4. Front-load the free, local diagnostics before paying for training
The iter-04 per-token-advantage and teacher-vs-student probes cost $0 and ran locally — but only
*after* iters 01–03 had spent money, and the decisive train-distribution probe didn't appear
until iter-05. "The teacher must be measurably more accurate than the student, and its signal
must be localized" is a ~$0 precondition for SDPO having anything to distill at all.

### 5. Don't move two variables at once
Iteration-05 changed the model (2B→8B), added the LLM critic, *and* kept the flat-reward data in
one run. When it collapsed, the cause was ambiguous and needed a whole extra iteration (06) to
isolate. The data fix (frontier band) is independent of the model swap — testing it at 2B first
would have de-risked the expensive 8B run.

### 6. Check the trainer's actual capabilities first
Iteration-06 spent part of a run on a `--beta` KL anchor that is **inert in TRL's `SDPOTrainer`**
(no KL term, no ref model). Thirty minutes reading `sdpo_trainer.py` would have killed that
experiment before launch and redirected to the real lever (data selection). Several
literature-recommended mitigations (KL/reference anchoring) simply weren't available in our
trainer — worth knowing on day one.

## The cost story: ~a third was wasted, and it was ops, not science

`docs/FINDINGS.md` records that **~35% of all Modal spend was wasted compute**, almost entirely
pre-hardening *infrastructure* failures rather than failed experiments:

- **iter-05 eval ~$22** — two crashed eval runs (a 2h function-timeout on the 32k-thinking hard
  tail; a Modal auto-retry on a buggy loop) before the resilient per-problem try/except landed.
- **iter-03 ~$30** — six integration false-starts (judge hangs, missing volume data, CUDA-graph
  hang, watchdog logic) caught the expensive way.
- **iter-06 ~$5.6** — a watchdog false-kill (stdout buffering) and an eval crash from a `/` in a
  checkpoint tag.

The encouraging part: by iter-07 the waste was **$0** — the hardening (resilient eval, watchdog,
resume, `enforce_eager`, tag sanitization, decoupled launch) worked; it just arrived late. Same
for the "small watched 8-step run, kill early" discipline: iter-06/07 got the same signal for
~$10–15 that iter-05's 20-step run cost $45, because the collapse was visible by step 4. That
should have been the iter-01 default.

> Note: `docs/FINDINGS.md` points to `reports/comparison/SPEND.md` for the per-category
> segmentation; that file was never actually created. The numbers above are reconstructed from
> the per-iteration `PROVENANCE.md`/`REPORT.md`.

## In fairness — not everything was avoidable

- The **8B confirmation was reasonable**: the literature does say SDPO is fragile sub-7B, and
  reproducing the mechanism at the paper's own scale made the positive result credible.
- The **infra hardening had to be built sometime**; much of the "waste" was one-time learning
  cost that has now been paid and baked into the default launch path.

The point isn't carelessness — it's that the scientific conclusion and the cheap path to it were
already in our own notes by iteration-03.

## The compressed path (≈3 iterations, plausibly ~$60–70)

- **Iter A — $0, local:** read `knowledge/`; run the teacher-vs-student + per-token-advantage
  diagnostics; stand up the eval harness (pass@k + train==eval + length/entropy).
- **Iter B — cheap, 2B:** frontier-band + dense reward + fixed teacher, a small *watched* run
  with the full harness. Either reproduces the +Δ at 2B or cleanly establishes the 2B ceiling.
- **Iter C — 8B, the credible confirmation:** identical recipe at scale.

That skips the $45 iter-05 detour and most of the infra waste — roughly **3 iterations and ~$110
saved** for the same (arguably cleaner) conclusion.

## Durable process rules (candidates for `docs/FINDINGS.md` standing lessons)

1. **Front-load the literature read.** When the `knowledge/` folder predicts a failure mode,
   read it *before* the run that would hit it, not after.
2. **`train==eval probe + pass@k + length/entropy` is the day-one harness.** Held-out alone is
   dangerously insensitive; greedy pass@1 and the SDPO loss are blind to collapse.
3. **Verify the trainer's real capabilities before designing around them** (e.g. `--beta` is
   inert in TRL's `SDPOTrainer`).
4. **Change one variable per run**, and test the cheap variable (data/reward) at the cheap scale
   first.
5. **Small watched runs, kill early.** The signal that matters usually arrives in the first few
   steps; pay for length only once the mechanism is alive.

---

## Addendum — corrections after iteration-08

This retrospective was written after iter-07. iter-08 refines three of its claims:

1. **The fix is frontier band + BINARY reward, not "frontier band + dense reward."** iter-07/08 ran
   `--grpo-reward binary`. The lever is **data selection** (which problems split under the binary
   advantage). *Dense reward is a separate, gaming-prone alternative* (it harvests partial-solvers'
   variance) — it is **not** what reversed the collapse. Wherever this doc pairs "frontier band + dense
   reward," read "frontier band (+ binary reward)."
2. **"iter-03 would have nailed it in one step" overstates it — the band-selection *metric* was a real,
   later discovery.** iter-07's band (selected by the *dense* score from a coarse **n=4** probe) was
   **69% flat under the binary advantage** → `flat_group` only fell to **0.44** and pass@8 gained a
   noisy **+0.083**. iter-08 showed the band must be selected by **binary solve rate at the training
   temperature**: an **n=12, temp-1.0** re-probe of the 63 easy+med problems found only **10** true
   binary-frontier (`p∈[0.25,0.75]`) — training on them drove `flat_group` to **0.12**. So the iter-03
   plan as designed (dense reward + coarse frontier) would most plausibly have produced an *iter-07-like
   partial* result, not the clean fix. **Trend now: `flat_group` 0.75 → 0.44 → 0.12 across iter-06/07/08.**
3. **The +0.083 headline is RETRACTED — the definitive eval found NO effect.** The 30-problem, n=12
   matched eval (base + iter-05/06/07/08 ckpts, bootstrap 95% CI) came back: **base 0.73, iter-05 0.66,
   iter-06 0.75, iter-07 0.60, iter-08 0.71 — all CIs (~±0.15) overlap; no iteration is statistically
   distinguishable from base on pass@8.** The clean "−0.33 → −0.167 → +0.083" monotone in the arc table
   above (rows iter-05/06/07) was **12-probe noise and did not replicate** — iter-07 is actually the
   *lowest* point estimate. **What stands: the mechanism** (`flat_group` 0.75 → 0.44 → 0.12 across
   iter-06/07/08, a direct per-step measurement). **What dies: the capability claim** — reviving the
   policy gradient did not buy a measurable pass@8 gain at this scale/dose. *Treat every Δ in this
   document's tables as within-noise unless confirmed on a powered (≥30-problem, n≥12, bootstrap-CI)
   eval.* `reports/comparison/SPEND.md` **now exists** (the line calling it "never created" is stale).
   Cost total (~$130–180) predates iter-08 (+~$70: probe/train/definitive-eval).

4. **The biggest avoidable cost was believing small-probe trends.** Tying #2 of the main retrospective
   together with this null result: the entire "collapse → fix" arc that motivated iterations 06/07/08
   was visible *only* in a 12-problem probe whose ±0.15 noise could fabricate a ±0.25 swing. A
   30-problem bootstrap-CI eval — run *once, early* — would have shown the effect was unresolvable and
   redirected effort from mechanism-chasing to **powering the eval / increasing the dose**. New standing
   lesson: *never let a small-probe Δ drive an iteration; gate every capability claim on a CI that
   excludes zero.*

New standing lesson (now in `docs/FINDINGS.md`): *select the frontier band by binary solve rate at the
training temperature, not a dense/coarse probe — the right metric drives `flat_group`→0.*
