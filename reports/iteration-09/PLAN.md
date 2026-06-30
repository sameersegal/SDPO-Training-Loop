# Iteration-09 — apply a real dose: is iter-08's null under-training or no-effect?

> **Status: PLANNED, not started (2026-07-01).** iter-08 established the mechanism (`flat_group`
> 0.75→0.44→0.12) but the definitive eval found **no pass@8 effect** — and the within-run telemetry says
> why: the policy *barely moved* (peak LR 3e-5 decaying to 1.5e-6 over 8 steps, grad_norm pinned at
> ~0.02–0.03, effective batch 2 groups/step). iter-09 disambiguates the two live hypotheses:
> **(H1) under-training** — a healthy gradient was applied in homeopathic doses; vs **(H2) no real
> effect** — the frontier-band signal doesn't buy capability at this scale. We resolve it by applying an
> actual dose and measuring a **dose–response curve.**

## 1. The question (and why iter-08 can't answer it)
iter-08's anti-collapse recipe was *triple*-conservative: LR cut 3× from the 1e-4 default, **decayed to
~0 over just 8 steps**, on **2 prompt-groups/step**. The LR×steps integral is minuscule. But the frontier
band is *already* the collapse guard (that's the mechanism win) — so the extra LR throttle was redundant
and starved the update. **A policy that didn't move can't show a capability change; the null is
uninformative about whether SDPO works.** iter-09 removes the throttle and applies a real dose.

## 2. Changes vs iter-08 — one coherent "un-throttle the dose" bundle
These are correlated knobs serving a single goal (move the model), not independent variables — we change
the *dose*, hold the *mechanism* (frontier band + binary reward + critic) fixed.

| knob | iter-08 | **iter-09** | why |
|---|---|---|---|
| peak LR | 3e-5 | **1e-4** (default) | the band, not the LR, guards collapse now |
| schedule | linear → 0 | **constant_with_warmup** (flat after warmup) | don't decay to ~0 mid-learning |
| steps | 8 | **30** (watched, kill-early) | give the LR×steps integral room to move weights |
| `num_generations` (G) | 8 | **16** | cleaner group-relative advantage; more groups/step; H200 fits |
| eff. batch | 2 groups/step | grows with G | less gradient noise on a 10-problem band |
| band | `frontier_band_v2` (10) | **same** | hold the mechanism fixed |
| reward / critic / lang | binary / on / python | **same** | one thing at a time |

Held fixed deliberately: LoRA rank 32 / text-tower targets (capacity is a *separate* future lever — don't
move it with the dose), `frontier_band_v2`, binary reward, critic on, python-only.

## 3. The metric correction — measure pass@1, not pass@8 (pass@8 is saturated here)
**iter-05→08 measured the wrong quantity.** At n=12, pass@8 saturates: on the 63-problem probe pool it is
effectively **bimodal — 29 problems at exactly 0, 26 at exactly 1.0, only 4 in between** (because even
1/12 base successes ⇒ pass@8 = 0.67, and 2/12 ⇒ 0.91). **Every problem in the trained frontier band has
base pass@8 = 1.0** — so pass@8 *cannot* show improvement there by construction; that is the real reason
iter-05→08 saw nothing. **pass@1** has resolution 1/n, spreads the problems across 0.08–0.83, and is
*exactly what SDPO moves* — distilling pass@k success into the policy raises the floor (pass@1) toward the
ceiling (pass@k). (This does **not** contradict CLAUDE.md's "pass@k, not greedy pass@1": that warns against
*greedy single-shot* pass@1's ±2/25 batch noise. **Sampled-mean** pass@1 over n≥16 at temp>0 is a stable
expectation — the on-objective, resolvable metric.)

**Primary metric:** mean **sampled pass@1** (+ the pass@1/2/4/8 curve to show the gap closing), bootstrap
95% CI. **Dose–response artifact:** evaluate ckpt **0/10/20/30** and read the curve:
- **pass@1 rises with steps, CI separates from base on the trained band** → H1 confirmed (real effect).
- **pass@1 flat across 30 steps while the model demonstrably moved** (grad_norm/ΔW up, completions changed)
  → strong support for H2; pivot to *signal quality* (reward/teacher gap per iter-04's diffuse-advantage
  red flag, LoRA capacity, model scale), not more dose.
- **length/entropy collapse by ~step 6** → LR too hot; the canary catches it, back off to 5e-5.

## 4. The eval set — concrete, discriminative, train + held-out (`data/eval_iter09.json`)
The discriminative slice is **intrinsically small** (the learnability frontier is narrow): only **18** of
the 63 probe problems have base pass@1 ∈ (0, 0.85]. Built and committed as `data/eval_iter09.json`:
- **`train_eval` (10):** the trained frontier band — *did it learn the trained distribution?* (base pass@1
  0.25–0.75, all pass@8-saturated).
- **`heldout` (8):** non-trained, same easy+med pool (base pass@1 0.08–0.83) — *did it generalize?*
- **Power comes from pass@1 resolution + n + bootstrap + the dose–response, NOT problem count.** 18 is the
  honest ceiling on this pool; pushing to 30+ would need probing far more medium problems (mid-pass@1 ones
  are rare) — an optional pre-step, not a blocker. Run **n=24** for tight per-problem pass@1.
- **Bootstrap 95% CI**, matched base + each checkpoint (`eval_iterations` → `iters30_analysis.py`, retargeted
  to pass@1).
- **Lean + durable eval path (now baked in):** `--max-seqs 96`, `--max-tokens` 20–24k, **`--no-judge` on
  the H200 (generate only) + `judge_local.py` on the GB10**, and **every artifact committed to
  `sdpo-outputs:/evals/<run>/`** (results json + per-sample JSONL) — pull the whole folder; nothing strands
  in the container (the iter-08 lost-samples bug, now fixed).

## 5. Guardrails (apply known hazards by default)
- **Length/entropy canary ON** — flag/kill if `mean_length` or entropy drops sharply by ~step 6 (the
  brevity-spiral signature from iter-05). Cheap, already built.
- **Watch grad_norm in the first 3–5 steps** — it is the *leading indicator* the dose actually changed.
  If grad_norm is still ~0.02 at 1e-4, something else caps the update (investigate before paying for 30
  steps). If it's meaningfully larger, the dose is landing — let it run.
- Decoupled launch (`setsid nohup modal run --detach`), `RUNNING_APP_ID.txt`, `--save-steps 1`, `--resume`,
  watchdog (auto-kill silent hang), `enforce_eager`. Preserve every checkpoint to `sdpo-outputs:/iter09-*`.
- **Cost discipline:** ~12–18 min/step (G=16 is slower) × 30 ≈ 6–9 h ≈ real $. So *watch the first few
  steps* (grad_norm + canary + cadence) and **kill early** if grad_norm hasn't moved or it re-collapses —
  don't pay for 30 steps to confirm a dead first 5.

## 5b. Free pre-check (DONE 2026-07-01) — iter-08 barely moved the weights
We measured the actual functional weight perturbation `||ΔW||_F = ||(α/r)·B·A||_F` summed over all 252
LoRA layers, with **iter-05 (which moved the model enough to *collapse* it) as a built-in reference**:

| adapter | steps | `||ΔW||_F` | per-step | vs iter-05 |
|---|---|---|---|---|
| **iter-05** (collapse-grade move) | 20 | **2.62** | 0.131 | 1.00 |
| iter-07 (8-step frontier) | 8 | 0.49 | 0.061 | 0.19 |
| **iter-08** (8-step frontier v2) | 8 | **0.46** | 0.058 | **0.18** |

**iter-08 perturbed the weights ~5.7× less than collapse-grade iter-05 — and ~2.3× less *per step*** (the
3× LR cut + decay-to-zero tail). This is direct, serve-free confirmation of **H1 (under-training)**: the
model we evaluated had barely moved from base. The binding constraint is dose, and it is *recoverable* —
we know exactly what a behavior-changing ΔW looks like (~2.6).

## 5c. Use ΔW as a free leading indicator during the run
Compute `||ΔW||_F` at each saved checkpoint (free, local, no serve) as the live "is the dose landing"
gauge — better than grad_norm alone. **Target the model-moving regime (cumulative ΔW approaching iter-05's
~2.6) but let the length/entropy canary, not ΔW, be the *stop* — a big ΔW guarantees behavior change, not
improvement** (iter-05's 2.6 was *harmful*). iter-09's bundle (LR ×~3.3, steps ×~3.75, flat vs decaying
schedule) projects ΔW into the ~2.5–4+ range, i.e. at/above the threshold that demonstrably moves the model.

## 6. Honest risk — "will 30 steps make a difference?"
The pre-check sharpens the answer:
- **Steps alone, at iter-08's schedule: NO.** The LR decayed to ~0 by step 8 (per-step ΔW tailed off);
  steps 9–30 on that schedule would add almost nothing. *This is why iter-09 is a dose bundle, not just
  more steps.*
- **The iter-09 bundle (flat LR 1e-4 × 30 steps): very likely to MOVE the model** — the projected ΔW lands
  in or past the iter-05 regime we *know* changes behavior. **Necessary, not sufficient:** moving the model
  guarantees a *different* model, not a *better* one. iter-05 proves a large move can be *harmful*
  (collapse). The bet is that the frontier band makes the *direction* good this time.
- **Residual failure modes the dose can't fix:** (a) the gradient direction doesn't improve capability
  (entrenches a sideways move or re-collapses → caught by the canary + dose–response curve early), and
  (b) the distillation signal is too diffuse to teach (iter-04's red flag → would show as "model moved a
  lot, pass@8 still flat," pointing to reward/teacher-gap or capacity, not dose).

So: **confident the model will now move; genuinely uncertain whether the move helps.** That uncertainty is
the experiment — instrumented to resolve cheaply (grad_norm/ΔW early, canary by step 6, dose–response on
intermediate checkpoints) rather than after a full $6–9 run.

## 7. Provenance (recipe verified against `sdpo_train.py` + `modal_sdpo.main`, 2026-07-01)
All flags below exist and forward through `main()`. **Train (decoupled launch):**
```
setsid nohup .venv/bin/modal run --detach src/modal_sdpo.py::main \
  --frontier-band frontier_band_v2.json --grpo-reward binary --critic \
  --lr 1e-4 --lr-scheduler constant_with_warmup --warmup-ratio 0.1 \
  --num-generations 16 --max-steps 30 --save-steps 5 \
  --difficulties easy,medium --languages python \
  --output-dir sdpo_out/iter09-dose --max-completion-length 20480 \
  > runs/iteration-09/train.log 2>&1 < /dev/null &
```
(**`--output-dir sdpo_out/iter09-dose`** — the `sdpo_out/` prefix is MANDATORY: train chdir's to `/root/app`
and the volume mounts at `/root/app/sdpo_out`, so a bare `iter09-dose` writes OFF-volume and is lost. The
preflight caught exactly this. On the volume it appears as `/iter09-dose`; `eval_dose --output-dir
iter09-dose` reads it back. `--save-steps 5` ⇒ ckpts 5/10/…/30, ≤ interruption interval for `--resume`.
G=16 stays microbatch=1 so no OOM — adds grad-accum steps, not peak memory; ~2× slower/step than iter-08.)

**Eval (lean + durable, dose–response):** generate on H200, judge on GB10.
```
.venv/bin/modal run src/modal_sdpo.py::eval_iterations --judge False \
  --ids 3008,2590,2610,2498,2612,2499,2603,2132,2667,2415,4001,2608,2294,3565,2445,2129,2951,3896 \
  --checkpoints iter09-dose/checkpoint-10,iter09-dose/checkpoint-20,iter09-dose/checkpoint-30 \
  --n 24 --max-tokens 24576 --max-seqs 96 --tag-prefix iter09dose
.venv/bin/modal volume get sdpo-outputs /evals/iter09dose ./        # pull the whole folder
for f in iter09dose/sdpo_passk_*_samples.jsonl; do \
  python src/judge_local.py --samples "$f" \
    --tag "$(basename $f _samples.jsonl | sed s/sdpo_passk_//)"; done   # judge free on GB10
```
Then `iters30_analysis.py` **retargeted to pass@1** (primary) + the pass@1/2/4/8 curve, split
`train_eval` vs `heldout` (`data/eval_iter09.json`), with the per-checkpoint ΔW (`src/adapter_delta.py`)
overlaid as the dose axis. Live monitoring: ΔW + grad_norm in the first ~5 steps, length/entropy canary.

## 8. Execution & time budget (knobs FROZEN — this section is operational only)
Dominant cost is the ~10 h training run (G=16 ≈ 2× iter-08's ~12 min/step). Two time levers, neither
touches a frozen knob.

**De-risk ladder FIRST (conservative — do NOT skip; important changes = G 8→16, flat LR, 30 steps):**
1. `pytest tests/` — free, logic.
2. **Modal smoke** `modal run …::main --smoke` (~minutes, cents) — image + data + judge wiring. *(GB10
   smoke is deliberately skipped: the 8B thinking-ON backward OOM-cascades the GB10's 128 GB and can kill
   the box — `sdpo-gb10-8b-training-viable`; it would not represent the H200 run anyway.)*
3. **Representative preflight** — the key de-risk for G=16: **3 REAL steps at full
   `--max-completion-length 20480`, `--num-generations 16`, `--save-steps 1`** on the H200 (the `--smoke`
   512-cap would HIDE the real memory/timing surface). Validates: no OOM at the loss step with G=16, real
   per-step cadence, a checkpoint write **and a `--resume`**, and previews **Gate 0** (grad_norm at LR
   1e-4). ~15 min / ~$1 vs a multi-hour late failure. Only after this passes → the long run.

**#1 Pipeline-eval (overlap eval with training → ~0 added wall-clock):** as each checkpoint commits to the
volume, eval it *while training continues*. Driver = `modal_sdpo.py::eval_dose` (one-shot, **idempotent** —
skips base/ckpts already evaluated, skips ckpts not yet landed). Re-run on each save:
```
.venv/bin/modal run src/modal_sdpo.py::eval_dose --judge False \
  --output-dir iter09-dose --tag-prefix iter09dose --steps 10,20,30 --n 24 \
  --ids 3008,2590,2610,2498,2612,2499,2603,2132,2667,2415,4001,2608,2294,3565,2445,2129,2951,3896
```
Base evals immediately (no training dep); ckpt-10 evals the moment it lands while training runs to 20, etc.
All artifacts → `sdpo-outputs:/evals/iter09dose/`; pull + judge on GB10 (`judge_local.py`).

**#2 Early-kill gates (expected ~20–40% off; preserves the frozen recipe — just exit conditions):**
- **Gate 0 — dose landing? (~step 3–5, ~1 h in).** From W&B `grad_norm` (and/or pull `checkpoint-K` →
  `src/adapter_delta.py`): if grad_norm is still ~0.02 / ΔW per-step ≈ iter-08's 0.058 at LR 1e-4 →
  **kill**, the dose isn't landing (raise LR / check schedule) before burning ~9 h.
- **Gate 1 — answer in hand? (ckpt-10 then ckpt-20 dose–response).** If `train_eval` pass@1 is *already*
  cleanly separating from base (CI excludes 0) **or** clearly flat with the model demonstrably moved
  (ΔW up, completions changed) and the canary clean → **stop at 20, not 30**; the curve has answered H1 vs H2.
- **Canary (every step).** `mean_length`/entropy drops sharply by ~step 6 → LR too hot (re-collapse);
  kill, back off to 5e-5. (Watchdog + `--resume` already cap a silent hang at minutes, not hours.)
