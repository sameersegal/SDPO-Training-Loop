# SDPO on Gemma-4-E2B-it / OJBench — Slide Data

Deck-ready numbers, graphs, and talking points. Graphs are in [`slides/`](./slides/).
Re-generate graphs with `python generate_slides.py` (auto-adds the 100-step comparison
once `sdpo_passk_sdpo_python.json` is present).

> Status: base pass@k + 20-step results are final. **100-step pass@k comparison is being
> finalized** (re-running cleanly after a cpp-judge stall) — the `compare_*.png` panels and
> the comparison rows below fill in automatically. Everything else is locked.

---

## Slide 1 — The opportunity (why SDPO)
**Graphs:** `slides/frontier_python.png`, `slides/opportunity_gap_python.png` (cpp variants too)

Base `gemma-4-E2B-it` pass@k on the held-out set (n=8 samples, temp 0.8, Modal H100):

| pass@k | easy | medium | hard | overall |
|---|---|---|---|---|
| **python** pass@1 | 32.5% | 15.0% | 0% | 9.5% |
| **python** pass@8 | **60.0%** | **40.0%** | 0% | **20.0%** |
| **cpp** pass@1 | 30.0% | 20.0% | 0% | 10.0% |
| **cpp** pass@8 | **60.0%** | **40.0%** | 0% | **20.0%** |

**Talking points:**
- The **pass@1 → pass@8 gap is large** (overall 9.5% → 20%; easy 32.5% → 60%). That gap = problems the model *can* solve with more tries.
- **This gap is precisely what SDPO targets**: `use_successful_as_teacher` distills the passing rollouts into the failing ones — so the frontier band is where it has signal.
- **Hard = 0 at every k** → hard needs *live judge feedback* (iteration 2), not just more sampling.

## Slide 2 — Setup
- **Model:** `google/gemma-4-E2B-it` (~2.3B, multimodal; LoRA r=32 on the **text tower only**).
- **Task:** OJBench competitive programming, **Python + C++**, lightweight judge (stdout diff / `g++ -O2`).
- **Method:** TRL experimental **SDPO** — successful rollouts teach failing ones (EMA teacher, top-100-logit KL).
- **Data:** train **easy-only** (easy+medium yields 0 successful rollouts cold → no signal); held-out = 15 hard + 5 easy + 5 medium × 2 langs.
- **Compute:** prototype on a local **GB10**, scale on **Modal H100/H200**.

## Slide 3 — 20-step prototype: works, no regression, expected null
| Metric | base | 20-step SDPO |
|---|---|---|
| Held-out pass@1 (32k, greedy) | 3/25 | 3/25 |
| GSM8K (full 1319) | **90.8%** | **90.1%** |

**Talking points:**
- Pipeline validated end-to-end; self-distillation **active** on easy (`success_group_fraction` 0.5–1.0, nonzero distill loss).
- **No held-out change** — but that's *within the noise floor*: the same base model scored 4/25 vs 2/25 across two greedy-pass@1 sessions (±2/25). Expected null for a 20-step run.
- **No regression** on out-of-domain math (GSM8K preserved; the SDPO model was actually *more concise* — truncations 29 → 6).

## Slide 4 — Scaling to 100 steps (Modal H100)
- **Completed**: 100 steps, ~**45 s/step** steady-state (76 min) — **~4× faster than the GB10** (~187 s/step). Cost ~$6.
- Training signal healthy throughout (`success_group_fraction` 0.5–1.0).
- **Base vs 100-step pass@k comparison:** `slides/compare_python.png` — *[filling in]*.
  - Headline question for the deck: *did 5× more training lift the frontier (pass@k) above the noise?*

## Slide 5 — Methodology insight (a strong "how we did it right" slide)
- **pass@k > greedy pass@1.** Greedy wobbles ±2/25 (batch nondeterminism); pass@k is clean & monotonic and *is* the metric SDPO exploits.
- **Select training data by *measured solvability*, not difficulty label.** Greedy showed medium 0/5, but pass@8 medium = **40%** — medium *is* trainable; it was the metric hiding the signal. The SDPO-coherent training set is the **frontier band** (problems solved sometimes-but-not-always), refreshed as the model improves (moving curriculum).

## Slide 6 — Infra / "why Modal" (honest engineering)
- GB10 (128 GB unified, ~270 GB/s) is **memory-bandwidth-bound** → generation-bound, ~187 s/step.
- Modal **H100** (HBM3) ~4× faster per step; **H200/multi-GPU did not reliably beat H100** at this dataset size (job is generation-bound; 4×H100 OOMs + starves 30 rows).
- The GB10 **hangs on high-concurrency multi-sample inference** (kernel-version issue) — we moved eval to Modal H100 for reliability.

## Slide 7 — Next steps
1. **Live judge-text feedback (iteration 2)** — the path to helping hard / all-failed groups (currently 0 at every k).
2. **Frontier-band curriculum** — train on measured-solvable problems, ramp difficulty as the model improves.
3. **Bigger / pass@k held-out** for a discriminative generalization metric.

---

### Source files
- pass@k base: `slides/passk_base.json` (graphs from it). 100-step: `slides/passk_sdpo.json` *(when ready)*.
- Held-out: `sdpo_eval_base_v2.json`, `sdpo_eval_sdpo.json`. GSM8K: `results_gsm8k_{base,sdpo}_full.json`.
- Full narrative: [`FINDINGS.md`](./FINDINGS.md); design: [`EXPERIMENT.md`](./EXPERIMENT.md).
