# Iteration 01 — Provenance & artifact locations

Where every result came from, so the report is reproducible/auditable.

## Curated data (committed, in this folder)
- `data/train_metrics_100step.csv`, `data/train_metrics_20step.csv` — per-step loss, grad_norm,
  reward_mean, success_group_fraction, completion mean/max length (source for the loss & length graphs).
- `data/eval_summaries.json` — held-out pass@1, pass@k, and GSM8K summaries (base / 20-step / 100-step).
- `figures/passk_base.json`, `figures/passk_sdpo.json` — pass@k frontier numbers behind the graphs.

## Trained model (the actual output)
- **100-step adapter preserved at Modal volume `sdpo-outputs:/iteration-01/`** (adapter_config.json +
  adapter_model.safetensors, r=32). The volume *root* (`/sdpo_out`) is overwritten by the next training
  run — `/iteration-01/` is the safe copy.
  - Retrieve: `modal volume get sdpo-outputs /iteration-01/adapter_model.safetensors ./` (and the config).
  - Serve: `vllm serve google/gemma-4-E2B-it --enable-lora --lora-modules sdpo=<dir> --max-lora-rank 32`.

## W&B (project `sdpo-gemma-ojbench`, persists on cloud)
| Run ID | What |
|---|---|
| `9h0fk7ue` | 20-step easy-only training (GB10) |
| `dq2gjgqp` | 100-step easy-only training (Modal H100) |
| `dkdw57up`, `kp5e17cg` | base + 100-step eval (pass@k / held-out / GSM8K, Modal) |
| `7jevc32n` | 100-step pass@k re-run, python (Modal) |

## NOT preserved (regenerable / intentionally dropped)
- Raw vLLM/training stdout logs (`*.log`) — gitignored; curated metrics above capture the signal.
  Per-problem eval result arrays (~500 KB each) — only summaries kept.
- `hf-cache` and `ojbench-data` Modal volumes — re-fetchable (model weights; OJBench test cases).
- Intermediate/overwritten adapters (smoke + 2-step validations) — superseded by the 100-step adapter.

## Spend
Modal: ~$15 of the $30 cap (100-step train ~$6 + evals + smoke iterations).
