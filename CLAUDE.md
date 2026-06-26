# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **research repo**, not a product: post-train **Gemma-4-E2B-it** with **SDPO**
(Self-Distillation Policy Optimization, TRL's experimental `SDPOTrainer`) on **OJBench**
competitive-programming problems, then measure generalization (held-out hard + easy/medium
thermometer) and capability regression (GSM8K). Most files in the root are **generated
experiment artifacts** (`.log`, `results_*.json`, `ojb_*.json*`, `sdpo_eval_*.json`,
`wandb/`) — all gitignored and regenerable via the scripts. The committed surface is ~8
Python files + the markdown docs + `ojb_splits.json`.

## Docs are the source of truth — read before changing anything

These are kept current and authoritative; defer to them over inference:

| Doc | Role |
|---|---|
| `EXPERIMENT.md` | **Design source of truth** — method, splits, configs, decisions, status (§10), caveats (§11) |
| `FINDINGS.md` | Results, deltas, the eval noise floor |
| `MODAL.md` | Cloud scale-up (Modal H100/H200) |
| `README.md` | Orientation / TL;DR |
| `HANDOFF.md` | Original handoff (executed, kept for history) |

When you change behavior, update the relevant doc in the same change — these files are the
contract for the next session.

## Code map

```
sdpo_ojbench.py     OJBench env adapter: load ojb_splits.json, public/private test split,
                    judge_completion() (py + cpp), build_dataset()/make_reward_func() for TRL
ojbench_eval.py     low-level judging primitives (extract_code, run tests, normalize)
sdpo_train.py       SDPO training entrypoint (SDPOTrainer + LoRA + vLLM colocate)
sdpo_eval_vllm.py   held-out pass@1 by language x difficulty via a vLLM OpenAI endpoint (+W&B)
sdpo_passk.py       held-out pass@k (k=1,2,4,8) — the discriminative, low-noise eval metric
eval_runner.py      GSM8K regression probe via a vLLM endpoint
modal_sdpo.py       reproduce env + run sdpo_train.py on Modal H100/H200 (code reused verbatim)
serve.sh            local vLLM serve helper (GB10)
ojb_splits.json     train/held-out split + per-id py & cpp prompts (the committed dataset)
```

Data flow: `ojb_splits.json` → `sdpo_ojbench.py` (prompts + judge) is imported by
`sdpo_train.py` (reward) **and** by every eval script (verdicts). Change the judge or the
split in `sdpo_ojbench.py` and both training reward and eval move together.

## Commands

Setup (uv-managed venv; `WANDB_API_KEY` goes in `.env`, gitignored):
```bash
uv venv .venv && source .venv/bin/activate
uv pip install vllm trl peft accelerate wandb datasets openai huggingface_hub
```

Local run order (GB10) — full sequence in `EXPERIMENT.md` §9:
```bash
# Always smoke-test integration first:
python sdpo_train.py --smoke

# 1. baseline (serve base via serve.sh on :8000 first):
python sdpo_eval_vllm.py --served-model google/gemma-4-E2B-it --tag base --max-tokens 32768 --wandb
python eval_runner.py --dataset gsm8k --sample-frac 1.0 --out results_gsm8k_base_full.json
# 2. train easy-only (free the GPU first — see "GB10 memory" below):
python sdpo_train.py --difficulties easy --languages python,cpp --max-steps 20 \
  --vllm-gpu-util 0.30 --output-dir sdpo_out
# 3. serve the ADAPTER (not a merge), then re-eval:
vllm serve google/gemma-4-E2B-it --enable-lora --lora-modules sdpo=sdpo_out \
  --max-lora-rank 32 --dtype bfloat16 --max-model-len 36864 --gpu-memory-utilization 0.85
python sdpo_eval_vllm.py --served-model sdpo --tag sdpo --max-tokens 32768 --wandb
python sdpo_passk.py --served-model sdpo --tag sdpo --wandb   # less noisy than pass@1
```

Cloud (faster, frees the local box) — full flow in `MODAL.md`:
```bash
.venv/bin/modal run modal_sdpo.py --smoke                        # validate image+data+judge
.venv/bin/modal run modal_sdpo.py --difficulties easy --max-steps 200
.venv/bin/modal volume get sdpo-outputs /sdpo_out ./sdpo_out_modal   # pull adapter back
```

## Critical gotchas (these have bitten us; respect them)

- **Train easy-only.** easy+medium yields **0 successful rollouts cold** →
  `success_group_fraction=0` → no distillation signal. Easy-only (15 problems × py+cpp = 30
  rows) is required to bootstrap. (`EXPERIMENT.md` §5)
- **GB10 silently OOM-kills** without `per_device_train_batch_size=1` + gradient
  checkpointing + `--vllm-gpu-util 0.30`. Microbatch = `num_generations` OOMs at step 0 on the
  LM-head logits tensor. **Free GPU memory before relaunching** (the GB10 has 128 GB *unified*
  memory shared with the system). On an 80 GB H100 use `0.25`; H200 (≥141 GB) can run
  `--no-grad-checkpointing`.
- **Serve the adapter via `vllm --enable-lora`, NOT a merged checkpoint.** `merge_and_unload`
  on this multimodal model silently drops upper-layer `k_norm` weights → vLLM "weights not
  initialized". Eval base and adapter on the *same* server for a clean delta.
- **LoRA targets the text tower only** — regex `.*language_model.*\.(q|k|v|o|gate|up|down)_proj$`.
  gemma4's vision/audio towers use `Gemma4ClippableLinear` which PEFT can't wrap.
- **Judge is lightweight, not the official DMOJ sandbox** (DMOJ needs root + pypy3). Python =
  stdout diff; C++ = `g++ -O2 -std=c++17`. TLE is a generous wall-clock, not DMOJ's limit.
  **Judge fidelity = reward fidelity** — a false AC trains the wrong thing.
- **Eval is noisy.** vLLM greedy pass@1 wobbles ±2/25 across sessions (batch nondeterminism);
  that noise floor ≳ a 20-step prototype's effect. Prefer `sdpo_passk.py` for a discriminative
  metric.
- **W&B signal to watch during training:** `self_distillation/success_group_fraction` (>0 ⇒
  learning), `reward_mean`, `distillation_loss`.

## Stopping GPU jobs

Kill training/serve jobs by **PID or tmux session**, never `pkill -f vllm` (collateral
damage). Long runs are launched in detached tmux.
