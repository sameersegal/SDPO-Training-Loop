# SDPO on Gemma-4-E2B-it / OJBench — Findings

**Date:** 2026-06-26 · **Compute:** prototype on GB10 (128 GB unified); scale + eval on Modal H100/H200 · **W&B:** project `sdpo-gemma-ojbench`

## TL;DR — a three-act story
1. **The opportunity is real.** Base `gemma-4-E2B-it` has a large **pass@k frontier** on held-out: Python pass@1 9.5% → **pass@8 20%** (easy 32.5→60, medium 15→40, hard 0). That pass@1→pass@8 gap is exactly what SDPO's successful-rollout distillation targets.
2. **20 steps = null.** Held-out greedy pass@1 base **3/25 → 3/25**; GSM8K **90.8% → 90.1%** (preserved). Within the ±2/25 noise floor — expected for a smoke-scale run. Pipeline fully validated.
3. **100 steps = regression (overfitting).** Scaling to 100 easy-only steps **hurt** the model: pass@k easy 60→40, **medium 40→0 (collapsed)**, overall pass@8 **20→8**, and **GSM8K fell too (~88% vs 90.8% — global, not coding-specific)**. ~50 epochs over 30 easy rows at LR 1e-4 → mode-collapse toward **terse** outputs (training completion length fell ~3–4×, ~3,500→~900 tokens). *More training on a too-narrow set is worse, not better.*

**Methodology wins:** **pass@k revealed both the opportunity and the regression** that greedy pass@1 (3/25 either way) hid — it's the metric to standardize on. The eval was underpowered: the *same base model* scored 4/25 vs 2/25 across sessions (±2/25 greedy noise).

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

### pass@k frontier — base model (Modal H100, n=8, temp 0.8, held-out)
The discriminative metric. Graphs: `slides/frontier_*.png`, `slides/opportunity_gap_*.png`.

| pass@k | easy | medium | hard | overall |
|---|---|---|---|---|
| python pass@1 | 32.5% | 15% | 0% | 9.5% |
| python pass@8 | **60%** | **40%** | 0% | **20%** |
| cpp pass@1 | 30% | 20% | 0% | 10% |
| cpp pass@8 | **60%** | **40%** | 0% | **20%** |

The pass@1→pass@8 gap is the SDPO opportunity. **Medium has a real frontier (15→40%)** even though greedy pass@1 showed it at 0/5 — the metric was hiding the signal.

### 100-step scaling: regression (overfitting) — base vs 100-step (Modal H100)
Graph: `slides/compare_python.png`.

| pass@k (held-out, python) | base | 100-step |
|---|---|---|
| easy pass@1 | 32.5% | 22.5% |
| easy pass@8 | 60% | 40% |
| medium pass@8 | 40% | **0%** |
| overall pass@8 | 20% | **8%** |

Regressed on every cell (n=8; medium = 0/40 attempts, not noise). **Cause: overfitting / mode-collapse toward terse outputs** — 100 steps over 30 easy rows (~50 epochs) at LR 1e-4; training completion length **fell ~3–4×** (mean ~3,500→~900 tokens), i.e. the model converged to short, narrow solutions, *not* verbose ones. The cpp-judge stall during eval was a separate issue (a pathological completion hangs the judge), not length. **The damage is global:** GSM8K also dropped (~88% vs base 90.8%). Training infra was healthy (45 s/step on H100, ~4× the GB10; `success_group_fraction` 0.5–1.0) — the *recipe*, not the pipeline, is the problem.

## Training dynamics (proof the loop learned on easy)
- `success_group_fraction` = 1.0 or 0.5 nearly every step (vs **0** on easy+medium — the base solves no medium problems in 8 tries).
- `reward_mean` swings 0.09–1.0 step-to-step, **driven by which easy problems are in each small batch**, with no clear 20-step trend — consistent with the flat held-out result.
- `distillation_loss` nonzero throughout (≈0.07–0.27): the self-distillation objective was genuinely optimized.

## Interpretation
1. **Engineering: success.** Every component (judge, splits, colocate training, LoRA-on-text, adapter serving, dual-language eval, Modal scale-up) works.
2. **Science: a clear lesson.** It's no longer "inconclusive" — 20 steps was null, **100 steps actively regressed**. The recipe (easy-only data + many epochs + LR 1e-4) overfits a 2.3B model. More compute ≠ better without fixing data breadth and regularization.
3. **The metric matters.** pass@k surfaced both the base frontier and the 100-step regression; greedy pass@1 was blind to both.

## Recommended next steps
Ordered by expected payoff. The regression reframes priority #1 from "scale" to "fix the recipe."

1. **Fix the recipe before scaling steps (highest priority).**
   - **Frontier-band data, not easy-only.** Select training problems by *measured solvability* (base pass@8: easy 60%, **medium 40%**, hard 0) — include the medium problems that sometimes pass. Drop always-fail (no teacher) and always-solved (nothing to learn).
   - **Regularize against collapse:** fewer epochs, lower LR (try 2e-5–5e-5), and/or a KL-to-base anchor to keep the policy near base and preserve output diversity (the model collapsed to terse, narrow solutions).
   - **Re-probe the frontier** every N steps (moving curriculum) and watch held-out pass@k as an early-stopping signal — stop before it regresses.
2. **Standardize on pass@k** (k≥4, n≥8) as the eval metric; retire greedy pass@1. Enlarge the held-out slice for tighter estimates.
3. **Live judge-text feedback (SDPO iteration 2).** Patch per-rollout judge text into the teacher reprompt — the path to non-zero on **hard / all-failed** groups (0 at every k today), and SDPO's main edge over GRPO.
4. **GRPO baseline on the frontier band** — show SDPO's advantage where GRPO's advantage collapses (all-same-reward groups), the paper's core claim.
5. **Reward fidelity / judge robustness** — stricter judging (toward DMOJ) and a hardened cpp judge (a verbose/looping completion stalled it); a false-positive AC trains the wrong thing.

## Caveats
- Lightweight judge (no DMOJ; no sudo/pypy3 locally) — false-positive AC = bad reward; also the cpp judge can stall on pathological completions (hardening needed).
- Hard problems stay ≈0 without live feedback — expected for a 2.3B model.
- GB10 is generation-bound and **hangs on high-concurrency multi-sample inference** → eval moved to Modal H100.
