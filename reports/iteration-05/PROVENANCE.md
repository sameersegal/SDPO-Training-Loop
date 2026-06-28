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
