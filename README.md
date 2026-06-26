# SDPO Training Loop — Gemma-4-E2B-it on OJBench

Post-train **Gemma-4-E2B-it** with **SDPO (Self-Distillation Policy Optimization)** on
**OJBench** competitive-programming problems, with a hybrid held-out generalization set and a
GSM8K regression probe.

## Docs (read in this order)
| Doc | What it is |
|---|---|
| [`EXPERIMENT.md`](./EXPERIMENT.md) | **Design source of truth** — method, splits, configs, decisions, status |
| [`FINDINGS.md`](./FINDINGS.md) | **Results** — 20-step run, deltas, the eval noise floor, next steps |
| [`MODAL.md`](./MODAL.md) | **Cloud scale-up** — run training on Modal H100/H200 |
| [`HANDOFF.md`](./HANDOFF.md) | original next-steps handoff (executed — kept for history) |

## Code
| File | Role |
|---|---|
| `ojb_splits.json` | train / held-out split + py & cpp prompts |
| `sdpo_ojbench.py` | OJBench env adapter: prompts, public/private split, reward func, py+cpp judging |
| `ojbench_eval.py` | base judging primitives (extract code, run tests) |
| `sdpo_train.py` | SDPO training (TRL `SDPOTrainer` + LoRA + vLLM colocate) |
| `sdpo_eval_vllm.py` | held-out pass@1 by language via a vLLM endpoint (+W&B) |
| `eval_runner.py` | GSM8K / math eval harness (regression probe) |
| `modal_sdpo.py` | run training on Modal H100/H200 (see `MODAL.md`) |
| `serve.sh`, `serve_vibe.sh` | local vLLM serve helpers |

## TL;DR of the current approach
- **Train easy-only** (easy+medium yields 0 successful rollouts cold → no SDPO signal).
- **LoRA r=32** on the text tower; **8k** training completions, **32k** eval cap.
- **Serve the adapter via vLLM `--enable-lora`** (not a merged checkpoint — merge drops weights on this multimodal model).
- **Prototype on the GB10**, scale on a **single Modal H100** (the job is generation-bound; more GPUs don't reliably help at 30 rows).
- 20-step result: no held-out change (within the ±2/25 noise floor), no regression — see [`FINDINGS.md`](./FINDINGS.md).

## Setup
```bash
uv venv .venv && source .venv/bin/activate
uv pip install vllm trl peft accelerate wandb datasets openai huggingface_hub
# put WANDB_API_KEY in .env (gitignored)
```
Data (OJBench test cases, ~2.7 GB) and model weights download on demand and are gitignored.
Run order: see `EXPERIMENT.md` §9.
