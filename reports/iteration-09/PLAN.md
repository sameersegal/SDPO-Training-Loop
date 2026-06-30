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

## 3. The key new artifact — a dose–response curve
A single run yields the disambiguating evidence if we **eval intermediate checkpoints**, not just the
last one. Save every checkpoint (already mandatory); evaluate ckpt **0/10/20/30** on the powered set:
- **pass@8 rises monotonically with steps and the CI separates from base** → H1 confirmed, real effect,
  and we have its dose–response.
- **pass@8 stays flat across 30 steps on a powered eval, while the model demonstrably moved** (grad_norm
  up, completions changed) → H2 gains strong support; pivot to *signal quality* (reward/teacher gap per
  the iter-04 diffuse-advantage red flag, LoRA capacity, or model scale), not more dose.
- **length/entropy collapse by ~step 6** → LR too hot; the canary catches it, back off to 5e-5.

## 4. Powered eval (fix iter-08's second-order problem too)
- **Bigger, discriminative problem set:** 60–100 problems whose *base* pass@8 is mid-range (~0.3–0.8) —
  exclude the saturated (always-AC) and hopeless (always-fail) tails that dominated the 30-problem set and
  added noise without resolving power.
- **n≥12** (graded pass@8), **bootstrap 95% CI**, matched base + each evaluated checkpoint (`eval_iterations`).
- **Lean eval path (baked in this iteration):** `--max-seqs 96`, `--max-tokens` 20–24k, and **generate on
  the H200 (`--no-judge`) + judge locally on the GB10 (`judge_local.py`)** — keep the GPU generating, not
  idle-judging.

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

## 7. Provenance (to fill on run)
- Recipe: `modal_sdpo.py::main --frontier-band frontier_band_v2.json --lr 1e-4 --lr-scheduler
  constant_with_warmup --warmup-ratio 0.1 --num-generations 16 --max-steps 30 --save-steps 1
  --difficulties easy,medium --languages python` (confirm flag names against `sdpo_train.py`).
- Eval: `eval_iterations --ids <powered set> --checkpoints iter09-*/checkpoint-{10,20,30} --n 12
  --max-tokens 24576 --max-seqs 96` → `iters30_analysis.py` (dose–response variant).
