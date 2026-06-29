# Iteration 05 — Provenance

## Base-model opportunity (pass@k) — Qwen3-8B

- **What:** held-out pass@k (n=8, k=1,2,4,8) for `Qwen/Qwen3-8B` base, Python, private-test AC —
  the pass@1→pass@8 "opportunity graph" data.
- **Entry point:** `modal run src/modal_sdpo.py::passk_base` (Qwen3-8B, python, n=8, thinking-ON,
  32k cap, temp 0.8, system `cp_method`, split `ojb_splits.json` — 25 held-out: 5 easy/5 medium/15 hard).
- **Modal app:** `ap-HLQgYhaMXBRojgDzfgmhnR` (H200) — https://modal.com/apps/sameersegal/main/ap-HLQgYhaMXBRojgDzfgmhnR
- **Spend:** **$3.32** (H200 $2.83 · CPU $0.47 · Memory $0.02), via `src/modal_cost.py`.
- **W&B:** `passk-qwen3_base` (project `sdpo-gemma-ojbench`, job_type=eval).
- **Smoke (validation):** app `ap-CTe4pXSZv8Ki28czdEc5Mg` — `passk_base --smoke` (2 easy, n=2, 32k).
- **Artifacts (committed):**
  - `data/sdpo_passk_qwen3_base.json` — full per-problem verdicts + summary.
  - `figures/opportunity_gap_python.png`, `figures/frontier_python.png`, `figures/passk_opportunity.json`.
- **Reproduce figures:** `ITER=iteration-05 python src/plot_opportunity.py --passk reports/iteration-05/data/sdpo_passk_qwen3_base.json --label "Qwen3-8B (base)"`
- **Headline:** easy 0.675→0.80 (+12.5pp), medium 0.20→0.40 (×2), hard 0→0, overall 0.175→0.24.

### Code added/changed for this run
- `src/modal_sdpo.py` — `passk_model_remote` + `passk_base` (model-parameterized pass@k; serves any
  base model with `vllm serve --enforce-eager`, runs `sdpo_passk` at 16-way concurrency × n=8,
  writes durably to the `sdpo-outputs` volume).
- `src/sdpo_passk.py` — `--limit N` (easy-first) for cheap smokes.
- `src/plot_opportunity.py` — model-agnostic opportunity-gap + frontier plotter from any `sdpo_passk_*.json`.

## SDPO + critic training run (Qwen3-8B) and ckpt-20 eval  — 2026-06-29

- **What:** SDPO + LLM trace-aligned critic (`claude-sonnet-4-6`) on the OJBench easy+medium train
  band (63 problems → 126 py+cpp rows), binary-GRPO / fractional-SDPO reward split.
- **Config:** `Qwen/Qwen3-8B · H200 · easy,medium · python,cpp · G=8 · max_completion 20480 ·
  max_steps 20 · save_steps 5 · vllm_gpu_util 0.20 · lr 1e-4 · distillation_weight 0.1 ·
  teacher_kind base · --critic · --grpo-reward binary · --sdpo-threshold 0.5 · system cp_method`.
- **Modal app (train):** `ap-oeiw6nb406wU2KP1YlKYFM` (H200) — **$10.86** (H200 $9.26 · CPU $1.54).
- **W&B (train):** run `8281dbd7`, name `sdpo-qwen3-8b-critic-d0.1-grpobin-sdpo0.5-base-s20`,
  state **finished**, 20 steps, train_loss 0.0096, **epoch 0.3175** (~40/126 rows seen — the model
  did NOT see all train data in 20 steps).
- **Health (step-0 canaries, all green):** no OOM at 20k/0.20; `success_group_fraction` 0.5–1.0
  (binary-GRPO/fractional-SDPO split working — near-misses count); `reward_mean` (binary AC) noisy
  0.25→0; `reward_max` 1.
- **Checkpoints (volume `sdpo-outputs` root = the `sdpo_out` mount):** checkpoint-5/10/15/20, all
  fresh 2026-06-29 (08:02 / 08:22 / 08:49 / 09:19 IST) + top-level `adapter_model.safetensors` (09:19).
- **Adapter preserved:** copied checkpoint-20's `adapter_model.safetensors` + `adapter_config.json`
  to `sdpo-outputs:/iteration-05/checkpoint-20/` so the next run can't overwrite it. (V1 volume:
  no recursive cp; optimizer/scheduler states left in the root checkpoint-20, not preserved.)

### Eval: base vs checkpoint-20 — held-out pass@k (Python, n=8, temp 0.8, system cp_method)
- **Scope (decided):** **checkpoint-20 only** (no 15/10/5), and after a 2h-timeout crash on the full
  25, **easy+medium only** (`--limit 10`, easy-first ⇒ the 5 easy + 5 medium; the 15 hard are flat
  0/8 for both base + ckpt-20 and were the slow 32k-thinking tail).
- **Result (no measurable change — expected null for a 20-step / ~32%-of-data prototype):**

  | split | metric | base | ckpt-20 |
  |---|---|---|---|
  | overall | pass@1 / @2 / @4 / @8 | 0.425 / 0.518 / 0.586 / **0.600** | 0.388 / 0.514 / 0.590 / **0.600** |
  | easy | pass@1 / @8 | 0.650 / 0.800 | 0.575 / 0.800 |
  | medium | pass@1 / @8 | 0.200 / 0.400 | 0.200 / 0.400 |

  pass@8 identical at every level; pass@1 −0.037 overall is **within base's run-to-run noise** (base's
  earlier same-day run scored pass@1 0.388 vs 0.425 here — ±0.04 wobble on 5+5 problems at temp 0.8).
- **Modal apps (eval):**
  - `ap-3Z15ubX7ZNyXA01rphyFFX` — full-25 attempt, **$15.75**, **CRASHED** (sdpo container hit the 2h
    `passk_one` function timeout on the hard 32k tail; base side survived in W&B). Wasted.
  - `ap-cXQiWXakvlqzQbHcvfoI6I` — first easy+medium attempt, **$6.22**, crashed + Modal-retried on the
    same buggy `sdpo_passk` loop; killed. Wasted.
  - `ap-t0qNeih46t3H37Ju98unti` — easy+medium with the fixed loop, **$6.03**, **succeeded**.
- **W&B (eval):** `passk-base` (the easy+medium base, overall pass@1 0.425) + the paired ckpt-20 run.
- **Artifacts (committed):** `data/sdpo_passk_ckpt20eval_base_easymed.json`,
  `data/sdpo_passk_ckpt20_easymed.json` (per-problem verdicts + summary).
- **Total iteration spend this day:** **$39.74** (training $10.86 + eval $28.0; ~$22 of eval lost to
  the timeout crash + retry-loop before the fix).

### Code added/changed for this run
- `src/modal_sdpo.py` — `passk_one` / `eval_checkpoint` gained a `limit` passthrough (→ `sdpo_passk
  --limit N`), so eval can target the easy+medium band and skip the slow, zero-signal hard tail.
- `src/sdpo_passk.py` — **resilience fix:** the per-problem `one()` eval loop now wraps generation
  and judging in try/except; a failed request/judge degrades to an `ERR` (non-AC) verdict instead of
  raising and sinking the whole run (the buffer-all anti-pattern that caused `ap-cXQiWXak...` to crash
  after all generation and Modal to retry the same bug).

## train==eval 3-point generalization curve (the headline finding) — 2026-06-29

- **What:** does ckpt-20 differ from base on the problems it *optimized over*? Eval base vs ckpt-20 on
  **seen-train** + a difficulty-matched **unseen-train** sample, judged on **private** cases, paired
  with the held-out easy+medium → seen / unseen / held-out gradient.
- **Seen set (reconstructed exactly):** replayed the trainer's data order — `RepeatSampler(shuffle=True,
  seed=42)` over `build_dataset("train", easy+medium, py+cpp, cp_method).shuffle(seed=0)` — and took the
  first 40 consumed rows (= epoch 0.3175 × 126). Python subset = **19 seen** (5 easy + 14 medium).
  Mapping + the accelerated 6+6 subset saved in `data/train_eval_split.json` (reproducible).
- **Run (accelerated for a deadline):** `eval_checkpoint --ids <12: 6 seen + 6 unseen, 2 easy + 4
  medium each> --tag-suffix _train`, Python, n=8, **private** judge, **48-way concurrency** (raised from
  16 after KV-cache telemetry showed the H200 at 6–18% — idle). Modal app `ap-Wq3F8MHhELUW7xur89HIVF`,
  ~$5. (The full 19+19 run `ap-mxjKMxEW…` was killed for speed; 38 medium-heavy problems ≈ 2.5–3h.)
- **Result — held-out null HID a real regression on the trained distribution:**

  | bucket | n | base pass@1 / @8 | ckpt-20 pass@1 / @8 |
  |---|---|---|---|
  | seen-train | 6 | 0.58 / **0.83** | 0.40 / **0.50** |
  | unseen-train | 6 | 0.54 / **0.83** | 0.38 / **0.50** |
  | held-out | 10 | 0.42 / 0.60 | 0.39 / 0.60 |

  pass@8 **0.83 → 0.50 (Δ−0.33)** on BOTH seen and unseen train; **per-problem ckpt-20 ≤ base on 12/12**
  (never better) — marginal 2/8 problems collapse to 0/8, robust 8/8 erode to 6–7/8. Gap widens with k
  ⇒ **diversity loss / mode collapse** in 20 steps. Held-out missed it (base only 0.60 there). **Lesson:
  held-out eval alone is dangerously insensitive; the train==eval probe is the sensitive test.**
- **Artifacts (committed):** `data/sdpo_passk_traineval_{base,ckpt20}.json` (per-problem),
  `data/train_eval_split.json` (seen/unseen ids), figures
  `figures/train_eval_3point_curve.png` + `figures/iter05_story_heldout_vs_traineval.png`,
  `figures/heldout_passk_base_vs_ckpt20.png`.
- **Total iteration spend:** **~$45** (training $10.86 + held-out eval $28 + train==eval ~$5; ~$22 of
  the held-out eval was lost to the 2h-timeout crash + retry-loop before the resilience fix).

### Code added/changed (train==eval + acceleration)
- `src/sdpo_passk.py` — `--split` + `--ids` (eval an explicit problem-id list, e.g. the seen/unseen
  train subsets) on top of the resilience fix.
- `src/modal_sdpo.py` — `passk_one`/`eval_checkpoint` gained `ids` + `tag_suffix` passthrough; **fn
  timeout 2h → 5h** (thinking-ON mediums are slow); **vLLM `--max-num-seqs` + client `--concurrency`
  16 → 48** (KV-cache telemetry showed the H200 idling — see the CLAUDE.md utilization note).
- `src/plot_heldout_delta.py`, `src/plot_train_eval_curve.py`, `src/plot_iter05_story.py` — figures.
