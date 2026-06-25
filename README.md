# SDPO Training Loop — Gemma-4-E2B-it on OJBench

Post-train **Gemma-4-E2B-it** with **SDPO (Self-Distillation Policy Optimization)** on
**OJBench** competitive-programming problems, with a hard-problem held-out generalization
set and a GSM8K regression probe.

**Read [`EXPERIMENT.md`](./EXPERIMENT.md) first** — it is the full, validate-before-running
spec (models, data splits, SDPO method, training/eval config, status, caveats).

## Layout
| File | Role |
|---|---|
| `EXPERIMENT.md` | experiment design (source of truth) |
| `ojb_splits.json` | train (easy+medium) / held-out (hard) split + py & cpp prompts |
| `sdpo_ojbench.py` | OJBench env adapter: prompts, public/private split, reward func, py+cpp judging |
| `ojbench_eval.py` | base judging primitives + the earlier Gemma-vs-VibeThinker OJBench comparison |
| `sdpo_train.py` | SDPO training (TRL `SDPOTrainer` + LoRA + vLLM colocate) |
| `sdpo_eval_vllm.py` | held-out pass@1 by language via a vLLM endpoint |
| `eval_runner.py` | GSM8K / math eval harness (regression probe) |
| `serve.sh`, `serve_vibe.sh` | vLLM serve helpers |

## Setup
```bash
uv venv .venv && source .venv/bin/activate
uv pip install vllm trl peft accelerate wandb datasets openai huggingface_hub
# put WANDB_API_KEY in .env (gitignored)
```

Data (OJBench test cases, ~2 GB) and model weights are downloaded on demand and are
gitignored. See the run order in `EXPERIMENT.md` §9.
