# Iteration-10 — the goldilocks dose: LR 5e-5 collapses SLOWER, but still collapses

> **Status: DONE (2026-07-01). Result: the collapse is DOSE-DEPENDENT, not a threshold — and there is no
> goldilocks LR.** iter-08 (3e-5) under-dosed → null; iter-09 (1e-4) → full collapse. iter-10 tested the
> middle (**5e-5**) and got a **milder, slower collapse**: length 16k→~8k (vs iter-09's 16k→5.5k),
> ckpt-10 **pass@1 0.292 / pass@8 0.776** (vs iter-09 0.238/0.574, base 0.438/0.863) — **ΔW 1.34** (56% of
> iter-09's 2.40). pass@1 is no longer *significantly* below base (paired Δ −0.146 **[−0.30, +0.01]**), but
> it is **not above base either, and the length was still declining when we killed it at step 14.** Higher
> LR → more ΔW → more collapse → lower pass@1, **monotonically; no dose beats base.** The collapse is the
> SDPO self-distillation *direction*; LR only sets its *speed and floor*. **iter-11's lever must be a
> direction guard (entropy/length), not LR.**
>
> **NB on the cap:** true 32k (your ask) is **memory-infeasible at G=16 on one H200** — three OOMs proved it
> (a ~14 GiB peak roughly cap-independent 24–32k, wedged between loss-step OOM at high util and vLLM-init
> starvation at low util; G doesn't help). Ran **20k @ util 0.20** (iter-09's config), which makes this the
> *cleanest* goldilocks test anyway (same cap, only LR differs). See CLAUDE.md gotcha.

## 1. The three-dose picture
| run | LR | ΔW @ ckpt-10 | length trajectory | ckpt-10 pass@1 | pass@8 | verdict |
|---|---|---|---|---|---|---|
| iter-08 | 3e-5 | ~0.46 (8 steps) | **held ~16k** | (null vs base) | — | under-dosed |
| **iter-10** | **5e-5** | **1.34** | 16k → ~11k → **~8k** | **0.292** | **0.776** | **mild collapse** |
| iter-09 | 1e-4 | 2.40 | 16k → **5.5k** | 0.238 | 0.574 | full collapse |

The lever that moves everything is the **dose** (LR × steps → ΔW), and its effect is monotonic and one-signed:
a bigger dose collapses the model faster and further. The only dose that *doesn't* collapse (3e-5) is too
small to change anything (iter-08's null).

## 2. Length: a slower collapse with a higher floor
![length 5e-5 vs 1e-4](figures/iter10_length_vs_iter09.png)

Same 20k cap as iter-09 (both clip 25–31% early — the cap is *not* the variable). At 5e-5 the length
**plateaued ~11k for steps 6–10** — briefly looking like a stable, healthy-ish reduction — but the plateau
**broke at step 11** (7.8k → 6.3k → ~8k) and was trending down when we killed it at step 14. So 5e-5 buys a
slower descent to a higher floor (~8k vs 5.5k), not a stable operating point. Projected to step 30 (ΔW
would reach ~3.8, past iter-09's collapse-grade), it would most likely deepen toward iter-09 levels.

## 3. Outcome: a milder collapse, but still no gain
![dose-response](figures/iter10_dose_response.png)

Base vs ckpt-10 on the same 18-problem set (`data/eval_iter09.json`), n=24, paired bootstrap 95% CI. Base
is reused from iter-09 (identical model).

| comparison | paired Δ pass@1 (95% CI) | reading |
|---|---|---|
| iter-10 (5e-5) vs base | **−0.146 [−0.30, +0.01]** | not *significantly* below base (CI touches 0) |
| iter-10 (5e-5) vs iter-09 (1e-4) | **+0.053 [−0.05, +0.15]** | milder than 1e-4 (point est.), not significantly |

pass@8 tells the same story more cleanly: **0.776** (5e-5) sits between base 0.863 and iter-09 0.574 — a
partial collapse. **The honest summary: 5e-5 does less damage than 1e-4, but there is no evidence it helps,
and the trajectory says it was still degrading.**

## 4. What iter-08→10 together establish
- **The collapse is the SDPO self-distillation *direction* (toward brevity), and it is dose-dependent in
  *magnitude*.** LR/steps control how fast and how far you collapse; they do not turn collapse into a gain.
- **There is no goldilocks LR on this recipe.** The window is a false one: too little dose (3e-5) = null;
  enough dose to matter (5e-5, 1e-4) = collapse. Nothing is above base.
- **So the next lever is not LR — it's direction.** iter-11: add a **direction guard** — an entropy bonus
  or an explicit length/diversity regularizer (or a KL anchor, which needs code since `--beta` is inert) —
  and re-run the dose at 5e-5 with the guard, gated on the same sampled-pass@1 dose–response.

## 5. Process — the memory saga (and the gates that held)
- **32k infeasible (3 OOMs):** 32k/0.20 → loss-step OOM; 32k/0.15 → vLLM init starved; 24k/0.18 → OOM at
  step 2 (the ~14 GiB peak is roughly cap-independent). Reducing G doesn't help (per-microbatch=1). Fell
  back to the proven 20k/0.20. Now a CLAUDE.md gotcha — don't re-test >20k here.
- **Gate-0 (ΔW 0.77 @ ckpt-6)** confirmed the dose was landing (~half iter-09's rate). **Canary** flagged
  the length was declining (softer than iter-09). **Pipeline-eval** gave the pass@1 verdict; **early-kill**
  stopped at step 14 once the eval + the resuming length-decline showed a milder-but-real collapse (saved
  ~4 h / the remaining 16 steps).

## 6. Provenance
- Train: app `ap-BEfF2w6lynIcmI6sYdI0dc`, killed at step 14 (Gate-1). Recipe: iter-09 with **`--lr 5e-5`**
  and `--vllm-gpu-util 0.20` (20k cap; 32k infeasible). Prior OOM attempts: `ap-pUCKjjgssYwfJp850AoSh9`
  (32k/0.20), `ap-50tqf0gxW9GoGXZrPbUEPj` (32k/0.15), `ap-oGZwrFaZsynBSTVF13KNdX` (24k/0.18). Checkpoints:
  `sdpo-outputs:/iter10-dose/checkpoint-{2..14}`.
- Eval: `eval_dose --no-judge --no-base --steps 6,10 --n 24` (base reused from iter-09). Judged on GB10.
- Data/figures: `reports/iteration-10/data/` (eval jsons + `iter10_train_trace.json`), `figures/`.
  ΔW via `src/adapter_delta.py`.
