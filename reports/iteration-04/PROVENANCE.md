# Iteration 04 — Provenance

**Type:** diagnostic (no training, no eval run). **Spend:** $0 (local GB10 only, ~35 min wall-clock).

## How it was produced
- Script: `src/plot_token_advantage.py` (added this iteration).
- Model: `google/gemma-4-E2B-it` (base), HF transformers, bf16, teacher-forced logprobs.
- Data universe: `data/ojb_splits_full.json` (iteration-03's 206-problem pool), via `OJB_SPLITS`.
- Problems (first of each difficulty, deterministic): easy=loj-2314, medium=loj-2086, hard=loj-2083 (python).
- Generation: G=8, temperature 1.0, top-p 0.95, max_new_tokens 8192, seed 0.
- Judge: `judge_completion(which="public", reward_mode="fraction")` with default reward case-caps
  (`SDPO_MAX_CASE_BYTES=1e6`, `SDPO_MAX_CASES=20`).
- Teacher context gating: `sdpo_prompts.decide_inputs` with the latest-run flags
  (`use_successful_as_teacher=True`, `success_reward_threshold=1.0`,
  `include_environment_feedback=True`, `environment_feedback_only_without_solution=True`).

## Command
```bash
cd runs/iteration-04
PYTHONPATH=../../src OJB_SPLITS=ojb_splits_full.json \
  ../../.venv/bin/python ../../src/plot_token_advantage.py \
  --difficulties easy,medium,hard --num-generations 8 --max-new-tokens 8192 \
  --out-prefix token_advantage
```

## Accuracy probe (follow-up, same setup)
- Script: `src/probe_teacher_accuracy.py` — reuses `data/token_advantage.json` (student verdicts +
  per-rollout feedback); generates teacher completions conditioned on that feedback and judges them.
- Command (decoupled, ~67 min on GB10):
  ```bash
  cd runs/iteration-04
  PYTHONPATH=../../src OJB_SPLITS=ojb_splits_full.json \
    ../../.venv/bin/python ../../src/probe_teacher_accuracy.py \
    --in-json ../../reports/iteration-04/data/token_advantage.json \
    --difficulties easy,medium,hard --max-new-tokens 8192
  ```
- Result: medium 0/8→0/8, hard 0/8→0/8 (no AC gap); easy skipped (solution-context). See
  `data/teacher_accuracy_probe.md` and INSIGHTS.md.

## Committed artifacts
- `figures/token_advantage.png` — main figure (8192-token budget, all 3 difficulties).
- `figures/token_advantage_2048.png` — earlier 2048-token pass (shows base NO_CODE on medium/hard).
- `data/token_advantage.json` — per-token A_t arrays + metadata + judge feedback for the group.
- `data/token_advantage_2048.json` — same for the 2048-token pass.
- `data/token_advantage_stats.csv` — summary stats per difficulty.
- `data/token_advantage_cases.md` — per-case student prompt + teacher prompt + completion.
- `data/teacher_accuracy_probe.md` / `.json` — accuracy probe (student vs teacher-with-feedback).
- `INSIGHTS.md` — when SDPO can/can't work; the two failure modes + probe synthesis.

## Raw (gitignored, under runs/iteration-04/)
- `tadv.log`, `probe.log`, `RUNNING_TADV.txt`, `RUNNING_PROBE.txt`, `token_advantage_2048.*`.
