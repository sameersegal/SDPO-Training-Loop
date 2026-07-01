# Iteration-10 — the goldilocks dose + cap control (LR 5e-5, 32k cap)

> **Status: LAUNCHING (2026-07-01).** iter-08 (LR 3e-5) under-dosed → null; iter-09 (LR 1e-4) over-dosed →
> collapse (pass@1 −0.20, length 16k→3–8k). iter-10 tests the **untested middle** and the **cap confound**
> in one run: **LR 5e-5** and **max_completion_length 32768** (up from 20k). Two live outcomes:
> **(A)** it moves without collapsing → a working dose *and* it exonerates the 20k cap; **(B)** it still
> collapses at 32k → the collapse is the LR-driven self-distillation direction, not truncation → the next
> lever is a direction guard (entropy/length), not LR.

## 1. Why these two changes
- **LR 5e-5:** iter-09's clipped-ratio trace showed the collapse ran with the 20k cap *inactive* (0%
  clipped from step 6, outputs 3–8k ≪ 20k) — so it's the LR feeding SDPO's brevity spiral, not the cap.
  5e-5 is between under-dosing (3e-5, ΔW 0.46/8-steps) and collapse (1e-4, ΔW 2.40/10-steps): does a
  gentler dose move the model *without* triggering runaway brevity?
- **32k cap:** the cap control the analysis flagged. iter-09 shortened *below* 20k voluntarily, so the
  prediction is 32k still collapses — but running it removes the confound cleanly. Also gives thinking
  headroom for the problems that need >20k reasoning (iter-09 clipped 25–31% early).

## 2. Recipe (only 2 changes vs iter-09; mechanism held fixed)
| knob | iter-09 | **iter-10** |
|---|---|---|
| LR | 1e-4 | **5e-5** |
| max_completion_length | 20480 | **32768** |
| schedule / G / steps / band / reward / critic / lang | constant_with_warmup / 16 / 30 / v2 / binary / on / py | **same** |
| output_dir | sdpo_out/iter09-dose | **sdpo_out/iter10-dose** |
| watchdog / save-steps / resume | 4200 / 2 / on | **same** |

**New risk — 32k memory:** iter-08 note says "32k OOM'd the H200 at the loss step" at `vllm_gpu_util 0.30`;
0.20 (default) untested at 32k+G=16. **De-risk: watch step-1's loss step for OOM** (the 10GB+ logits
tensor at 32k is the failure point). Fallback: `--vllm-gpu-util 0.15`, then G=8 if still tight.

## 3. Execution (same gates as iter-09 — they all fired correctly)
- **De-risk:** `pytest` (free) → **watch step-1 for OOM** (the 32k memory test manifests at the loss step
  ~20 min in; kill+lower `--vllm-gpu-util` if it OOMs). iter-09 already smoked the near-identical wiring.
- **Gate-0 (ΔW):** at ckpt-2, is the dose landing? At 5e-5 expect ~0.5× iter-09's per-step ΔW.
- **Canary (every step):** `mean_length`/clipped_ratio — the whole point is to see if 5e-5 avoids the
  16k→3–8k drop. If length holds → success signal; if it collapses → outcome (B).
- **Pipeline-eval** (`eval_dose`, concurrent) at ckpt-10/20/30 + **Gate-1** on sampled-pass@1 dose–response
  (`data/eval_iter09.json` set, n=24, paired bootstrap vs base). Judge on GB10.
- **Decision:** length holds + pass@1 rises/holds → continue, real win. Collapse → kill, conclude (B).

## 4. Provenance (to fill on run)
- Train: `main --frontier-band frontier_band_v2.json --grpo-reward binary --critic --lr 5e-5
  --lr-scheduler constant_with_warmup --warmup-ratio 0.1 --num-generations 16 --max-steps 30
  --save-steps 2 --output-dir sdpo_out/iter10-dose --max-completion-length 32768
  --watchdog-stall-secs 4200 --resume`.
- Eval/analysis: `eval_dose ... --steps 10,20,30 --n 24` → `iter09_analysis.py` (reuse; retarget tags) +
  `adapter_delta.py` (ΔW dose axis).
