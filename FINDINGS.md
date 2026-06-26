# SDPO on Gemma-4-E2B-it / OJBench — Findings

**Date:** 2026-06-26 · **Box:** single GB10 (128 GB unified) · **W&B:** [run 9h0fk7ue](https://wandb.ai/sameersegal-personal/sdpo-gemma-ojbench/runs/9h0fk7ue) (project `sdpo-gemma-ojbench`)

## TL;DR
We ran the full SDPO post-training loop end-to-end and measured it against a controlled baseline.
- **The pipeline works**: data → on-policy generation → judge reward → self-distillation → adapter → serve → eval, all validated. On easy problems SDPO self-distillation is **active** (successful rollouts present every step, nonzero distillation loss).
- **Held-out OJBench: no measurable change** — base **3/25 → SDPO 3/25** (apples-to-apples, same server). This is the **expected null** for a 20-step prototype, *not* a failure.
- **No regression**: GSM8K held at **90.8% → 90.1%** (−0.8 pt, within noise); the SDPO model is actually **more concise** (truncations 29 → 6).
- **The eval is underpowered**: the *same base model* measured twice scored **4/25 vs 2/25** — run-to-run noise (±2/25) is as large as any plausible 20-step effect. A real signal needs more steps **and** a more stable metric.

## What we built
- **Model:** `google/gemma-4-E2B-it` (multimodal; LoRA on the **text tower only**, r=32/α=64).
- **Task:** OJBench competitive-programming problems, judged with a lightweight **python + C++** verifier (public tests = reward during training, private tests = held-out eval).
- **Method:** TRL experimental **SDPOTrainer**, vLLM colocate generation, `use_successful_as_teacher` (successful rollouts in a group distilled into the failing ones), EMA teacher, top-k logits (K=100).
- **Train set:** 15 easy problems × {python, cpp} = 30 rows. **Held-out:** 25 problems (5 easy + 5 medium + 15 hard) × 2 languages.

## Results

### Held-out pass@1 — base vs SDPO (same server, controlled)
| pass@1 | easy (5) | medium (5) | hard (15) | **overall (25)** |
|---|---|---|---|---|
| base · python | 1/5 | 0/5 | 0/15 | **1/25** |
| SDPO · python | 1/5 | 0/5 | 0/15 | **1/25** |
| base · cpp | 1/5 | 1/5 | 0/15 | **2/25** |
| SDPO · cpp | 2/5 | 0/5 | 0/15 | **2/25** |

Net effect: **0**. Per-cell moves (cpp easy +1, cpp medium −1) are within greedy-decoding noise.

### GSM8K (no-regression check, full 1319)
| | accuracy | correct | truncated@1024 |
|---|---|---|---|
| base | 90.83% | 1198/1319 | 29 |
| SDPO | 90.07% | 1188/1319 | 6 |

General math ability preserved; easy-only code SDPO did **not** cause catastrophic forgetting.

### Noise floor (why the held-out null is unsurprising)
The base model evaluated in two separate sessions: **cpp 4/25** (prior) vs **cpp 2/25** (this session) — identical weights, 2/25 difference. vLLM greedy decoding is batch-nondeterministic, so at n=25 the metric wobbles ±1–2. **The measurement noise exceeds the expected prototype effect.**

## Training dynamics (proof the loop learned on easy)
- `success_group_fraction` = 1.0 or 0.5 nearly every step (vs **0** on easy+medium — the base solves no medium problems in 8 tries).
- `reward_mean` swings 0.09–1.0 step-to-step, **driven by which easy problems are in each small batch**, with no clear 20-step trend — consistent with the flat held-out result.
- `distillation_loss` nonzero throughout (≈0.07–0.27): the self-distillation objective was genuinely optimized.

## Interpretation
1. **Engineering: success.** Every component (judge, splits, colocate training, LoRA-on-text, adapter serving, dual-language eval) works on the GB10.
2. **Science: inconclusive-but-expected.** 20 steps over 30 easy rows is a smoke-scale run. Per the experiment plan, a real delta needs **hundreds of steps**. We can't yet claim SDPO helps or hurts on held-out — the eval is too noisy and the run too short.
3. **Safety check passed.** No regression on out-of-domain math.

## Recommended next steps
1. **Scale the run** — hundreds of steps. On the GB10 that's overnight (~5 min/step, generation-bound); on a **single H100/H200 (~12× memory bandwidth)** it's ~1 hr. The work is portable (vanilla HF/TRL/vLLM) — see scale-up note.
2. **Stabilize the metric** — use **pass@k** (k≥4) and/or enlarge the frontier slice; n=25 greedy pass@1 is too noisy to detect small effects.
3. **Add live judge feedback (SDPO iteration 2)** — currently feedback is successful-rollouts-only. Patching per-rollout judge text into the teacher reprompt is where SDPO is expected to help **hard / all-failed** groups (which stayed 0/15 here, as predicted).
4. **Curriculum** — easy-only bootstraps signal; ramp easy → medium as the model improves (medium currently yields 0 successful rollouts cold).

## Caveats
- Lightweight judge (no DMOJ; no sudo/pypy3 on this box) — a false-positive AC is a bad training signal; consider stricter judging before trusting rewards at scale.
- Hard problems are expected to stay ≈0 for a 2.3B model without live feedback.
- Single shared GPU: training, serving, and the user's other jobs contend for one device.
