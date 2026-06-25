# SDPO post-training of Gemma-4-E2B-it on OJBench — Experiment Spec

> Status: **DRAFT for your validation.** This document is the single source of truth for
> what we are building. If anything below is wrong, that's the thing to fix before we burn
> more compute. Last updated mid-session; "Status" section at the end says what actually
> ran vs. what is pending.

## 1. Goal

Post-train **Gemma-4-E2B-it** with **SDPO (Self-Distillation Policy Optimization)** so it
gets better at competitive-programming problems, using **OJBench** as the environment.
Measure generalization on a **held-out set of HARD problems** and check for capability
regression on **GSM8K**.

## 2. Hardware / environment

- 1× **NVIDIA GB10** (Grace-Blackwell, aarch64, 128 GB unified memory), CUDA 13, driver 580.
- Python venv at `.venv` (uv-managed). Key libs: **vLLM 0.23.0**, **transformers** (gemma4-capable),
  **trl 1.6.0** (has `trl.experimental.sdpo.SDPOTrainer`), **peft 0.19**, **wandb**.
- W&B logging enabled via `WANDB_API_KEY` in `.env` (project `sdpo-gemma-ojbench`).

## 3. Model under test

- **`google/gemma-4-E2B-it`** — dense ~2.3B effective params (5.1B w/ embeddings), 128K ctx,
  multimodal arch (`Gemma4ForConditionalGeneration`); we train the **text tower only**.
- (For reference we earlier benchmarked `WeiboAI/VibeThinker-3B`; the user chose Gemma for
  speed/iteration. VibeThinker is **not** part of the SDPO experiment.)

## 4. Benchmark: OJBench

- 232 NOI/ICPC competition problems, difficulty **easy / medium / hard**, languages
  **Python & C++**. We use the **NOI** subset (clean `id → NOI/loj-<id>` mapping).
- Each problem ships `init.yml` + `tests.zip` (stdin→stdout test cases). All problems we use
  have **standard diff checkers** (no special judges).

### Judging (important caveat)
- We use a **lightweight judge** (`ojbench_eval.py` / `sdpo_ojbench.py`), NOT the official
  DMOJ sandbox (DMOJ needs root + pypy3, unavailable here).
  - Python: run `python3 sol.py < input`, normalized stdout diff.
  - C++: `g++ -O2 -std=c++17` compile, then run binary per test; verdicts AC/WA/RE/TLE/CE/NO_CODE.
  - Per-test wall-clock timeout + 2 GB memory cap. **TLE here ≈ generous CPython/native wall-clock,
    not DMOJ's exact per-problem limits.**
- **Public/private split** (paper protocol): for each problem we deterministically split its
  test cases 50/50 into **public** (used as the training reward signal) and **private**
  (held back for held-out evaluation), to reduce reward-hacking.

## 5. Data splits (locked: hybrid held-out)

| Split | Problems | Languages | Used for |
|---|---|---|---|
| **train** | NOI **easy + medium** minus the held-out slice (63 problems: 15 easy + 48 medium) | **python + cpp** → 126 rows | SDPO training reward |
| **held-out (hard)** | 15 NOI **hard** problems | **python + cpp** | generalization metric (expected ~0 early) |
| **held-out (frontier slice)** | 5 easy + 5 medium | **python + cpp** | **progress thermometer** (sensitive metric that moves) |

- **Decision:** hybrid held-out. Hard = the headline generalization claim; the easy/medium
  slice is a sensitive gauge so we can tell whether the loop is learning at all (a 2.3B model
  is expected near 0 on hard). Both slices are disjoint from training (no leakage).
- **Frontier note:** we do **not** pre-compute a `pass@k` frontier map. SDPO's
  `use_successful_as_teacher=True` does *implicit, online* frontier selection — each step it
  samples G rollouts and only groups with ≥1 success produce a learning signal; all-fail
  groups are skipped. Easy/medium training data is the coarse frontier. An explicit pass@k
  map is an optional efficiency fast-follow.

## 6. SDPO method (what it actually is)

Source: *Reinforcement Learning via Self-Distillation* (arXiv 2601.20802, ICML 2026);
implemented by TRL's `SDPOTrainer`.

- **Setting:** RL with *rich feedback*. Instead of GRPO's single scalar reward/attempt, SDPO
  turns textual/implicit feedback into a **dense, per-token** signal.
- **Self-teacher:** the *same* policy conditioned on feedback `f`, `π(·|x, f, y_<t)`.
  `f` = environment output (judge/runtime error) **and/or a correct rollout** from the group.
- **Loss:** `Σ_t KL( π(·|x, y_<t) ‖ stopgrad π(·|x, f, y_<t) )` over the model's own attempt `y`.
  Drop-in advantage swap vs GRPO; overhead is one extra (parallel) teacher log-prob pass.
- **Why fewer samples than GRPO:** when all rollouts in a group get the same reward (e.g. all
  fail a hard problem), GRPO's advantage collapses to 0 → no gradient; SDPO still extracts a
  dense signal. Paper: ~6× faster wall-clock to GRPO accuracy; 3× fewer attempts at test time.
- **Stability:** EMA teacher (α=0.01) + JSD/top-K(=100) logit distillation.
- **TRL limitation we accept:** TRL sources per-rollout feedback from the static
  `privileged_context` column, **not** dynamic per-attempt judge text. So our active signal is
  **`use_successful_as_teacher`** (successful rollouts teach the failing ones — the paper's
  "implicit feedback" mode, which already beats GRPO). Wiring our judge's stderr/failing-case
  text as live feedback is a **fast-follow** (needs a small trainer patch).

## 7. Training configuration (`sdpo_train.py`)

- **LoRA** (r=16, α=32) on the **text model only**: regex targets
  `language_model.*(q|k|v|o|gate|up|down)_proj` (gemma4's vision/audio towers use a custom
  `Gemma4ClippableLinear` that PEFT can't wrap — we deliberately skip them).
- `num_generations=8`, `distillation_weight=1.0` (pure SDPO), `distillation_mode="topk_logits"`,
  `distillation_topk=100`, `teacher_model_kind="ema"`, temperature 1.0 / top_p 0.95.
- Generation via **vLLM colocate** (`vllm_gpu_memory_utilization≈0.45`), bf16, `lr=1e-4`.
- `max_completion_length` ≈ 4096 (note: some C++ solutions exceed this → clipped → no reward).
- Logs to **W&B**; key metrics: `self_distillation/success_group_fraction`, `reward_mean`,
  `distillation_loss`.

## 8. Evaluation  ← please validate this

Reported **separately for Python and C++**, pass@1 (greedy), on the **15 hard held-out**
problems, judged on **private** test cases:

- `sdpo_eval_base.json` — base Gemma
- `sdpo_eval_sdpo.json` — after SDPO (merged LoRA)
- Delta = post − base, per language × difficulty.

**Regression probe:** GSM8K (2% sample) accuracy, base vs post, to catch capability
degradation from code-only training. Base GSM8K (measured earlier, greedy) = **84.6%**.

**W&B logging:** training streams to W&B (project `sdpo-gemma-ojbench`); held-out eval metrics
are also pushed (`sdpo_eval_vllm.py --wandb`) as `heldout/<tag>/<lang>/<diff>_pass@1`.

## 9. Pipeline / scripts (reproducible)

| File | Role |
|---|---|
| `ojb_splits.json` | the train/held-out split + prompts (py & cpp) |
| `sdpo_ojbench.py` | env adapter: prompts, public/private split, reward func, py+cpp judging |
| `ojbench_eval.py` | base judging primitives (extract code, run tests) |
| `sdpo_train.py` | SDPO training (TRL `SDPOTrainer` + LoRA + vLLM colocate) |
| `sdpo_eval_vllm.py` | held-out pass@1 by language via a vLLM endpoint |
| `eval_runner.py` | GSM8K eval (regression probe) |

**Run order:**
```bash
# 0. (done) build splits, download test data
# 1. baseline held-out (serve base, then):
python sdpo_eval_vllm.py --served-model google/gemma-4-E2B-it --tag base --max-tokens 8192
# 2. train (frees GPU first):
python sdpo_train.py --difficulties easy,medium --languages python,cpp --max-steps 40
# 3. merge + serve adapter, then re-eval + GSM8K:
python sdpo_eval_vllm.py --served-model <merged> --tag sdpo --max-tokens 8192
python eval_runner.py --dataset gsm8k --sample-frac 0.02   # vs base 84.6%
```

## 10. Status (what actually ran)

- ✅ Models downloaded; vLLM serving works for gemma4 on this box.
- ✅ Judge + py/cpp + public/private split built and tested.
- ✅ TRL SDPO smoke test ran end-to-end (vLLM-colocate + LoRA + gemma4) — pipeline validated.
- ✅ Splits rebuilt to **train=easy+medium, held-out=hard** (this change just landed).
- ⏳ Hard held-out **test data** downloading now.
- ⏳ **Baseline on the HARD held-out** not yet run (earlier 5/15 py, 6/15 cpp numbers were on an
  *easy+medium* held-out, now superseded).
- ⏳ Full SDPO training run + post-eval + GSM8K probe: **pending** (exceeds the 3-hour window;
  intended to run in background and be read off W&B).

## 11. Locked decisions & remaining caveats

**Decisions (this session):**
- Held-out = **hybrid** (hard + easy/medium thermometer slice). [§5]
- First-run feedback = **successful-rollouts-only** (TRL default); live judge-text feedback is
  iteration 2. [§6]
- First run = **quick prototype** (~10–20 steps) to validate the loop before a serious run.

**Remaining caveats:**
1. **Judge fidelity = reward fidelity.** Lightweight judge is fine for eval; for training it's
   the reward. A false-positive AC trains the wrong thing. Upgrade to real DMOJ later.
2. **C++ TLEs / Python TLEs** are partly language artifacts, not reasoning errors.
3. **Compute reality:** on-policy SDPO with G=8 and ~4k-token completions on one GPU is
   minutes per step; a *meaningful* delta needs hours, not the prototype's minutes.
4. **Hard likely stays ~0 early** — that's why the easy/medium thermometer exists.
