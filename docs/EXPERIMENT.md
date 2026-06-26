# SDPO post-training of Gemma-4-E2B-it on OJBench — Experiment Spec

> Status: **validated and executed.** The pipeline ran end-to-end (train → serve → eval).
> Headline results and the noise analysis live in **[`FINDINGS.md`](./FINDINGS.md)**; the
> cloud scale-up path is in **[`MODAL.md`](./MODAL.md)**. This file is the design source of
> truth — method, splits, configs, and the decisions behind them. §10 tracks what ran.

## 1. Goal

Post-train **Gemma-4-E2B-it** with **SDPO (Self-Distillation Policy Optimization)** so it
gets better at competitive-programming problems, using **OJBench** as the environment.
Measure generalization on a **held-out set** (hard + an easy/medium thermometer slice) and
check for capability regression on **GSM8K**.

## 2. Hardware / environment

- **Local dev:** 1× **NVIDIA GB10** (Grace-Blackwell, aarch64, 128 GB *unified* memory).
  Memory-bandwidth-bound (~270 GB/s) → generation-bound and slow (~5 min/step). Used for
  training the prototype and for all **inference/eval**.
- **Cloud scale-up:** Modal **H100 / H200** (see [`MODAL.md`](./MODAL.md)). Same code, reproduced
  env; ~2 min/step. The job is generation-bound, so H100↔H200 and multi-GPU differ less than
  run-to-run noise at this dataset size (30 rows) — single H100 is the working unit.
- Python venv at `.venv` (uv-managed). Key libs: **vLLM 0.23.0**, **transformers 5.12**,
  **trl 1.6.0** (`trl.experimental.sdpo.SDPOTrainer`), **peft 0.19**, **wandb**.
- W&B logging via `WANDB_API_KEY` in `.env` (project `sdpo-gemma-ojbench`).

## 3. Model under test

- **`google/gemma-4-E2B-it`** — dense ~2.3B effective params, multimodal arch
  (`Gemma4ForConditionalGeneration`); we train the **text tower only**.
- (We earlier benchmarked `WeiboAI/VibeThinker-3B` for reference; **not** part of SDPO.)

## 4. Benchmark: OJBench

- NOI/ICPC competition problems, difficulty **easy / medium / hard**, languages **Python & C++**.
  We use the **NOI** subset (clean `id → NOI/loj-<id>` mapping). Test cases ship as
  stdin→stdout pairs under `ojbench_data/NOI/loj-<id>/`.

### Judging (important caveat)
- **Lightweight judge** (`ojbench_eval.py` / `sdpo_ojbench.py`), NOT the official DMOJ sandbox
  (DMOJ needs root + pypy3, unavailable here).
  - Python: `python3 sol.py < input`, normalized stdout diff.
  - C++: `g++ -O2 -std=c++17` compile then run per test; verdicts AC/WA/RE/TLE/CE/NO_CODE.
  - Per-test wall-clock timeout + memory cap. **TLE ≈ generous wall-clock, not DMOJ's limits.**
- **Public/private split** (paper protocol): each problem's test cases split deterministically
  50/50 into **public** (training reward) and **private** (held-out eval), to reduce reward-hacking.

## 5. Data splits

`ojb_splits.json`:

| Split | Problems | Languages | Used for |
|---|---|---|---|
| **train** | NOI easy + medium (63: 15 easy + 48 medium) → 126 rows | python + cpp | SDPO training reward |
| **held-out** | 15 hard + 5 easy + 5 medium (25) → py+cpp | python + cpp | generalization + thermometer |

- **Held-out = hybrid:** hard is the headline generalization claim; the easy/medium slice is a
  sensitive gauge of whether the loop is learning at all (a 2.3B model is ~0 on hard). Disjoint
  from training (no leakage).
- **⚠️ Pivot — we train EASY-ONLY.** Cold, the base model solves **0** medium problems in G=8
  tries → `success_group_fraction=0` → no distillation signal ("SDPO inactive"). Easy-only
  (15 problems × py+cpp = 30 rows) is required to bootstrap any learning. easy+medium is a
  **future curriculum step** once the model is stronger (or once live judge feedback is wired —
  see §6). This is the single biggest change from the original design.
- **Frontier note:** no pre-computed pass@k map. SDPO's `use_successful_as_teacher` does implicit,
  online frontier selection — only groups with ≥1 success produce a gradient. An explicit pass@k
  map / larger held-out slice is the recommended fast-follow to make the eval discriminative.

## 6. SDPO method (what it actually is)

Source: *Reinforcement Learning via Self-Distillation* (ICML 2026); implemented by TRL's `SDPOTrainer`.

- **Setting:** RL with *rich feedback* — turns textual/implicit feedback into a dense, per-token signal.
- **Self-teacher:** the same policy conditioned on feedback `f`, `π(·|x, f, y_<t)`.
- **Loss:** `Σ_t KL( π(·|x, y_<t) ‖ stopgrad π(·|x, f, y_<t) )` over the model's own attempt.
- **Why fewer samples than GRPO:** when all rollouts in a group share a reward (e.g. all fail),
  GRPO's advantage → 0; SDPO still extracts a dense signal *if there is a teacher*.
- **Stability:** EMA teacher + top-K(=100) logit distillation.
- **Feedback mode we run (iteration 1):** **`use_successful_as_teacher=True`** — within each
  group, successful rollouts (public-tests AC) teach the failing ones. **The judge's textual
  feedback (verdict / failing case / expected-vs-got) is generated but NOT used**: TRL reads
  feedback only from a static `privileged_context` column (`sdpo_trainer.py:835`), so per-rollout
  judge text needs a trainer patch.
- **Consequence:** all-fail groups (medium/hard cold) get **no** teacher signal → easy-only is
  required, and hard stays ~0. Wiring live judge text (**iteration 2**) is what would let SDPO
  help all-fail / hard groups — the biggest fidelity gap to the paper.

## 7. Training configuration (`sdpo_train.py`)

- **LoRA** (**r=32, α=64**) on the **text model only**: regex
  `language_model.*(q|k|v|o|gate|up|down)_proj` (gemma4's vision/audio towers use
  `Gemma4ClippableLinear`, which PEFT can't wrap — skipped).
- `num_generations=8`, `distillation_weight=1.0`, `distillation_mode="topk_logits"`,
  `distillation_topk=100`, `teacher_model_kind="ema"`, temperature 1.0 / top_p 0.95, `lr=1e-4`, bf16.
- `max_completion_length=8192` (training). Completions average ~3.4k tokens.
- Generation via **vLLM colocate**.
- **Memory config (required — silent OOM otherwise):** `per_device_train_batch_size=1` +
  gradient accumulation + **gradient checkpointing**. Microbatch = `num_generations` OOMs at
  step 0 on the LM-head logits tensor. vLLM KV reservation: **GB10 `--vllm-gpu-util 0.30`**,
  **H100 `0.25`** (80 GB < 128 GB unified; 0.45 OOMs the H100). Tunable knobs:
  `--per-device-batch`, `--no-grad-checkpointing` (only fits on ≥141 GB H200).
- Logs to **W&B**; watch `self_distillation/success_group_fraction` (>0 ⇒ learning), `reward_mean`,
  `distillation_loss`.

## 8. Evaluation

**Token budgets:** Gemma supports 128K ctx. **Eval** uses a generous **32k** completion cap
(served at `max-model-len≈36k`) so long solutions don't truncate into false `NO_CODE`.
**Training** keeps completions at **8k** (32k × G=8 per-token KL is memory-prohibitive; easy
training solutions fit in 8k).

**Serving the adapter — via vLLM `--enable-lora` (NOT a merged checkpoint).** Merging the LoRA
into gemma-4 then serving the local dir is brittle: `merge_and_unload` via `AutoModelForCausalLM`
silently drops upper-layer `k_norm` weights (meta-device offload) → vLLM "weights not initialized".
Instead serve base + adapter directly:
```bash
vllm serve google/gemma-4-E2B-it --enable-lora --lora-modules sdpo=<adapter_dir> --max-lora-rank 32 \
  --dtype bfloat16 --max-model-len 36864 --gpu-memory-utilization 0.85
# then pass --served-model sdpo
```

Reported **separately for Python and C++**, pass@1 (greedy), on the **held-out** set, judged on
**private** test cases:
- `sdpo_eval_base*.json` — base Gemma (measure on the *same* server as the adapter for a clean delta)
- `sdpo_eval_sdpo.json` — after SDPO
- Delta = post − base, per language × difficulty.

**⚠️ Eval is noisy:** vLLM greedy is batch-nondeterministic; at n=5/cell (25 total) the same base
model wobbles ±2/25 across sessions. **Treat small cells as ±1–2; the noise floor ≈ any 20-step
effect.** Use pass@k / a larger held-out slice for a discriminative metric.

**Regression probe:** **full GSM8K** (1319) accuracy, base vs post, to catch capability degradation.

**W&B:** training streams to W&B; held-out metrics pushed via `sdpo_eval_vllm.py --wandb` as
`heldout/<tag>/<lang>/<diff>_pass@1`.

## 9. Pipeline / scripts (reproducible)

Code is in `src/`, committed inputs in `data/`, results in `reports/`. (Repo layout: `README.md`.)

| File | Role |
|---|---|
| `data/ojb_splits.json` | train/held-out split + py & cpp prompts |
| `src/sdpo_ojbench.py` | env adapter: prompts, public/private split, reward func, py+cpp judging |
| `src/ojbench_eval.py` | base judging primitives (extract code, run tests) |
| `src/sdpo_train.py` | SDPO training (TRL `SDPOTrainer` + LoRA + vLLM colocate) |
| `src/sdpo_eval_vllm.py` | held-out pass@1 by language via a vLLM endpoint (+W&B) |
| `src/sdpo_passk.py` | held-out pass@k (the discriminative metric) |
| `src/eval_runner.py` | GSM8K eval (regression probe) |
| `src/modal_sdpo.py` | run training + eval on Modal H100/H200 (see `MODAL.md`) |

**Run order (local GB10).** Run from `runs/iteration-NN/` so outputs stay per-iteration; scripts
are in `src/` (put it on the path). `S=../../src`:
```bash
# 1. baseline held-out + GSM8K (serve base):
PYTHONPATH=$S python $S/sdpo_eval_vllm.py --served-model google/gemma-4-E2B-it --tag base --max-tokens 32768 --wandb
PYTHONPATH=$S python $S/eval_runner.py --dataset gsm8k --sample-frac 1.0 --out results_gsm8k_base_full.json
# 2. train easy-only (free the GPU first):
PYTHONPATH=$S python $S/sdpo_train.py --difficulties easy --languages python,cpp --max-steps 20 \
  --vllm-gpu-util 0.30 --output-dir sdpo_out
# 3. serve adapter via --enable-lora, then re-eval + GSM8K (see §8):
PYTHONPATH=$S python $S/sdpo_passk.py --served-model sdpo --tag sdpo --wandb
PYTHONPATH=$S python $S/eval_runner.py --dataset gsm8k --sample-frac 1.0 --model sdpo --out results_gsm8k_sdpo_full.json
```
**Cloud (faster, from repo root):** `modal run src/modal_sdpo.py --difficulties easy --max-steps 100` — see `MODAL.md`.

## 10. Status (what actually ran)

- ✅ Judge + py/cpp + public/private split built and tested.
- ✅ Baseline measured (held-out + full GSM8K **90.83%**).
- ✅ **20-step easy-only run executed** (GB10). Self-distillation active (`success_group_fraction`
  mostly 1.0/0.5, nonzero distillation loss).
- ✅ **Post-eval (controlled):** held-out base 3/25 → SDPO 3/25 (within noise); GSM8K 90.8% → 90.1%
  (preserved). **No measurable held-out change — the expected null for a 20-step prototype.**
  Full results + tables: **[`FINDINGS.md`](./FINDINGS.md)**.
- ✅ **Modal scale-up validated** (H100/H200, correct learning signal). **100-step easy-only run
  on H100 in progress** at last update.
- ⏳ Pending: pass@k eval, live judge feedback (iteration 2), easy→medium curriculum.

## 11. Locked decisions & caveats

**Decisions:**
- Train **easy-only** (easy+medium → 0 successful rollouts cold). [§5]
- Held-out = **hybrid** (hard + easy/medium thermometer). [§5]
- Feedback = **successful-rollouts-only** (iteration 1); live judge text is iteration 2. [§6]
- Serve the adapter via **vLLM `--enable-lora`**, not a merged checkpoint. [§8]
- Compute: prototype on GB10; scale on a **single Modal H100** (job is generation-bound; more/bigger
  GPUs don't reliably help at this dataset size). [§2]

**Caveats:**
1. **Judge fidelity = reward fidelity.** A false-positive AC trains the wrong thing. DMOJ later.
2. **Eval noise ≥ prototype effect** at n=25 greedy pass@1 — fix with pass@k / larger slice. [§8]
3. **Compute reality:** a meaningful delta needs hundreds of steps, not the 20-step prototype.
4. **Hard stays ~0** without live judge feedback — expected for a 2.3B model. [§6]
