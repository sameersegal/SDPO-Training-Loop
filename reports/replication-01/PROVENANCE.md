# Replication-01 provenance

Code at commit `2692fd4` (paper-faithful knobs) on top of `d593dc7` (teacher-context fix).

## Modal apps (all H200, workspace ac-W3DwsH8kQ2eULMoI1rp1zL)

| Stage | App | Cost |
|---|---|---|
| Preflight SDPO (2 real steps @16k) | ap-P7zVOJhicEpy2OSbWxQsCy | $1.83 |
| Preflight GRPO | ap-SYTaxBc5Nov2Cty9qTtoDE | $1.85 |
| Train SDPO seg-1 (steps 0→27, preempted) | ap-J7TQL25EatP6HgJhQVGrYS | $27.02 |
| Train GRPO seg-1 (steps 0→29, preempted) | ap-SYVApi54AJmg7FlsmIWaoJ | $27.02 |
| Concurrent evals (preempted; only base landed) | ap-FQYGSXyBkkHovgSaMITHKP, ap-l2epUMISBeaDNtIoRGHE08, ap-onQVVcxdfm9MDN02mYGEvk, ap-DobnvdJFxBpfTF86Pbc1eH | $37.06 (~$32 wasted) |
| Train SDPO resume (24→40) | ap-ul8UTGpjTihGdTxj65ttb8 | $17.26 |
| Train GRPO resume (28→40) | ap-K5ctxKRAOsWQskfVmes9rY | $11.95 |
| Evals SDPO arm (base+5 ckpts, sequenced) | ap-IRTToduKIsqQgjdUYWUn73 | $42.86 |
| Evals GRPO arm (5 ckpts) | ap-zvblCMAK17sdr9VxWIADuP | $46.18 |
| **Total** | | **≈ $213** |

Incidents: at ~5h the workspace **spend limit** stopped both training apps and the
**10-GPU concurrency cap** (2 train + 6 concurrent eval spawns) starved eval scheduling.
Recovery: budget raised by user; resumed from ckpt-24/28 (`--save-steps 4` + `--resume`);
evals re-run **sequenced after training**. Lesson: don't run concurrent `eval_dose` fleets
alongside training under a GPU-concurrency cap; sequence them or cap spawn width.

## Artifacts

- Adapters + checkpoints: volume `sdpo-outputs:/repl01-sdpo`, `/repl01-grpo` (10 ckpts each).
- Judged evals + summary: `data/` (12 json), raw sample JSONLs on the volume under
  `/evals/repl01sdpo`, `/evals/repl01grpo`; local copies in `runs/replication-01/evaldata/`.
- SDPO training rollouts: `runs/replication-01/evaldata/rollouts_sdpo.jsonl` (688 rows;
  GRPO arm has none — rollout capture rides the feedback bus, which GRPO doesn't use).
- Analysis: `src/repl01_analysis.py` (paired bootstrap B=10k, seed 0).
- Launch recipes + app IDs: `runs/replication-01/RUNNING_APP_ID.txt`, `EVAL_PLAN.txt`.
- W&B: project `sdpo-gemma-ojbench`, runs `sdpo-qwen3-8b-fb-d1.0-grpobin-sdpo1.0-ema-s40`
  (SDPO) and `...-verifier-d0.0-...-s40` (GRPO).
