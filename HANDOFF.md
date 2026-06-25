# HANDOFF — SDPO post-training of Gemma-4-E2B-it on OJBench

Read **`EXPERIMENT.md`** first (full design + locked decisions). This file is the operational
state + exact next steps so another agent can continue without re-deriving anything.

---

## 0. One-paragraph status

The full SDPO loop is **built and validated end-to-end** (the TRL `SDPOTrainer` smoke test ran
on this box: vLLM-colocate generation + LoRA on gemma4 + SDPO loss + adapter saved). Data,
splits, judge (python+C++), and baseline are done. **The actual training run + post-eval +
full-GSM8K probe have NOT been run** — that's the next step. Baseline is logged to W&B.

## 1. What is DONE

- ✅ Models downloaded (`google/gemma-4-E2B-it`; also `WeiboAI/VibeThinker-3B` for an earlier
  comparison — not part of SDPO).
- ✅ Env: `.venv` with vLLM 0.23.0, trl 1.6.0 (`trl.experimental.sdpo`), peft 0.19, wandb, datasets.
- ✅ OJBench judge (lightweight): python (`python3 sol.py`) + C++ (`g++ -O2 -std=c++17`), normalized
  diff, public/private per-problem split. In `sdpo_ojbench.py` / `ojbench_eval.py`.
- ✅ Splits (`ojb_splits.json`): **train** = 63 easy+medium (15 easy, 48 medium); **held-out** =
  25 = 15 hard + 5 easy + 5 medium (frontier "thermometer" slice). All have py+cpp prompts + test data.
- ✅ SDPO trainer (`sdpo_train.py`): LoRA **r=32 / α=64** on text tower only, vLLM colocate,
  `use_successful_as_teacher`, EMA teacher, topk_logits K=100, num_generations=8, wandb.
- ✅ Held-out evaluator (`sdpo_eval_vllm.py`): pass@1 by language×difficulty, **32k** cap, wandb.
- ✅ **Baseline measured** (W&B run `eval-base`, project `sdpo-gemma-ojbench`):
  | pass@1 | easy(5) | medium(5) | hard(15) | overall(25) |
  |---|---|---|---|---|
  | python | 1/5 | 0/5 | 0/15 | 1/25 |
  | cpp | 3/5 | 1/5 | 0/15 | 4/25 |
- ✅ Smoke test of training pipeline passed (2 steps, tiny config).

## 2. What is NEXT (run order)

> The GPU is single; serving and colocate-training both need it — **only one at a time**.
> Stop any vLLM server before training (see §4 gotchas for the safe way to stop).

**A. Full-GSM8K BASELINE** (regression reference; do while base is served):
```bash
source .venv/bin/activate
# serve base if not already: python -m vllm.entrypoints.openai.api_server \
#   --model google/gemma-4-E2B-it --port 8000 --dtype bfloat16 --max-model-len 36864 --gpu-memory-utilization 0.85
python eval_runner.py --dataset gsm8k --sample-frac 1.0 --model google/gemma-4-E2B-it \
  --out results_gsm8k_base_full.json          # full 1319-problem GSM8K
```

**B. SDPO TRAINING** (stop the server first to free the GPU):
```bash
python sdpo_train.py --difficulties easy,medium --languages python,cpp \
  --num-generations 8 --max-completion-length 8192 --max-steps 20 --output-dir sdpo_out
# logs to W&B project sdpo-gemma-ojbench automatically (WANDB_API_KEY in .env)
```
Watch on W&B (CRITICAL — proves it's learning, not idling):
- `self_distillation/success_group_fraction` must be **> 0** (groups with ≥1 passing rollout).
- `self_distillation/reward_mean`, `self_distillation/distillation_loss` should be nonzero.
- If you see "SDPO self-distillation is inactive / no successful rollouts" repeatedly → the
  model isn't solving any training problem in 8 tries; bias the data more toward `easy`, raise
  `num_generations`, or raise `max_completion_length`.

**C. MERGE adapter + POST-EVAL** (gemma4 + LoRA may not serve directly in vLLM, so merge):
```bash
python - <<'PY'
import torch; from transformers import AutoModelForCausalLM, AutoTokenizer; from peft import PeftModel
b="google/gemma-4-E2B-it"
m=AutoModelForCausalLM.from_pretrained(b, dtype=torch.bfloat16)
m=PeftModel.from_pretrained(m,"sdpo_out").merge_and_unload()
m.save_pretrained("sdpo_merged"); AutoTokenizer.from_pretrained(b).save_pretrained("sdpo_merged")
PY
# serve sdpo_merged, then:
python sdpo_eval_vllm.py --served-model sdpo_merged --tag sdpo --max-tokens 32768 --wandb
python eval_runner.py --dataset gsm8k --sample-frac 1.0 --model sdpo_merged --out results_gsm8k_sdpo_full.json
```

**D. REPORT delta:** post (`sdpo_eval_sdpo.json`) − base (`sdpo_eval_base.json`) per language×diff;
GSM8K post vs base (no-regression check).

## 3. Decisions locked (don't relitigate without reason)
- Held-out = **hybrid** (hard + easy/med thermometer).
- First-run feedback = **successful-rollouts-only** (TRL default). **Iteration 2 = patch live
  judge-text feedback** into the teacher reprompt (TRL reads feedback only from a static
  `privileged_context` column at `sdpo_trainer.py:835`; per-rollout judge text needs a trainer
  patch — this is where SDPO's edge on all-failed/hard groups comes from).
- Eval cap **32k**, training completion **8k** (32k in training is compute/memory-prohibitive).
- LoRA **r=32/α=64**, text tower only.

## 4. Gotchas / environment specifics (will bite you)
- **Never** `pkill -f "vllm serve"` (or any pattern matching your own command line) — it kills the
  killer. Stop servers via the harness task API, or `pkill` by PID / a non-self-matching pattern.
- **gemma4 + LoRA:** target ONLY `language_model.*(q|k|v|o|gate|up|down)_proj` (regex in
  `sdpo_train.py`). The vision/audio towers use `Gemma4ClippableLinear`, which PEFT cannot wrap.
- **vLLM 0.23.0** prints a TRL "supported 0.12–0.19" warning — harmless, the loop runs.
- **No sudo / no pypy3** on this box → the official **DMOJ** judge is not usable; we use the
  lightweight judge. For training, judge fidelity = reward fidelity (false-positive AC = bad signal).
- **vLLM greedy is nondeterministic** across runs (batching) → small-n pass@1 wobbles ±1–2.
- Box = GB10 (aarch64, 128 GB unified). `nvidia-smi` reports memory as `[N/A]` here.
- `.env` holds `WANDB_API_KEY` (gitignored; scripts auto-load it). W&B project `sdpo-gemma-ojbench`.

## 5. File inventory (committed)
| File | Role |
|---|---|
| `EXPERIMENT.md` | design / source of truth |
| `HANDOFF.md` | this file |
| `ojb_splits.json` | train/held-out split + py & cpp prompts |
| `sdpo_ojbench.py` | env adapter: prompts, public/private split, reward func, py+cpp judging |
| `ojbench_eval.py` | judging primitives + earlier Gemma-vs-VibeThinker comparison harness |
| `sdpo_train.py` | SDPO training (TRL SDPOTrainer + LoRA + vLLM colocate) |
| `sdpo_eval_vllm.py` | held-out pass@1 by language×difficulty via vLLM (+wandb) |
| `sdpo_eval.py` | transformers-based held-out eval (fallback; slower) |
| `eval_runner.py` | GSM8K / math eval harness |
| `serve.sh`, `serve_vibe.sh` | vLLM serve helpers |

Gitignored (regenerate/re-download): `.venv/`, `ojbench_data/` (test cases, ~2 GB), `sdpo_out/`,
result `*.json/jsonl`, logs, `.env`, `ojbench_prompts/`, `OJBench_repo/`, `sdpo_paper/`.

## 6. Open risks / fast-follows
1. **Live judge feedback** (iteration 2) — the biggest faithfulness gap to the SDPO paper.
2. **Reward fidelity** — consider stricter judging / DMOJ before trusting training rewards.
3. **Thermometer noise** — bump frontier slice size or use pass@k for a stable signal.
4. **Hard likely stays ~0** for a 2.3B model in a short run; that's expected.
5. **Compute** — on-policy SDPO (G=8, 8k completions) is minutes/step on one GPU; a real delta
   needs hours/hundreds of steps, not the 20-step prototype.

## 7. Process state at handoff
- A base vLLM server (`--max-model-len 36864`) may still be running on :8000 from the baseline.
  Stop it before training. No training job is running. No uncommitted code (all pushed to
  `origin/main`).
