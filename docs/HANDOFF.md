# HANDOFF — iteration-05: monitor the running Qwen3-8B SDPO+critic run

> **Live operational handoff.** A training run is **RUNNING on Modal (cloud H200)** right now.
> This file is everything needed to (a) re-attach monitoring from any machine, (b) recover if it
> crashed, and (c) run the eval once it finishes. Design rationale lives in
> [`EXPERIMENT.md`](./EXPERIMENT.md) and [`../reports/iteration-05/REPORT.md`](../reports/iteration-05/REPORT.md);
> cloud mechanics in [`MODAL.md`](./MODAL.md). Older Gemma handoff is in git history.

---

## 0. One-paragraph status
Iteration-05 trains **Qwen/Qwen3-8B** with **SDPO + an LLM trace-aligned critic** (Claude
`claude-sonnet-4-6`) on the OJBench easy+medium train band. The run is **detached on Modal** and
survives any local/SSH disconnect (it runs server-side; the launcher is `setsid`-decoupled AND
`--detach`). It checkpoints to the `sdpo-outputs` volume every 5 steps, has a no-progress watchdog,
and **does not auto-relaunch** — per the standing instruction, a crash leaves it stopped for review.

## 1. The running job
- **Modal app id:** `ap-oeiw6nb406wU2KP1YlKYFM`  (also in `runs/iteration-05/RUNNING_APP_ID.txt`)
- **W&B:** project `sdpo-gemma-ojbench`, run name **`sdpo-qwen3-8b-critic-d0.1-grpobin-sdpo0.5-base-s20`**
- **Exact config (from the launch):**
  ```
  Qwen/Qwen3-8B · H200 · easy,medium · python,cpp · G=8 · max_completion 20480 · max_steps 20
  save_steps 5 · vllm_gpu_util 0.20 · lr 1e-4 · distillation_weight 0.1 · teacher_kind base
  --critic (sonnet) · --grpo-reward binary · --sdpo-threshold 0.5 · system cp_method
  ```
- **Launched with:** `setsid nohup .venv/bin/modal run --detach src/modal_sdpo.py::main --critic > runs/iteration-05/launch.log 2>&1 < /dev/null &`

### Re-attach monitoring (stateless — run from anywhere with the repo + `.venv` + Modal creds)
```bash
cd <repo>
.venv/bin/modal app list | grep oeiw6nb4                 # state: ephemeral=running, stopped=done/crashed
.venv/bin/modal app logs ap-oeiw6nb406wU2KP1YlKYFM       # live logs (Ctrl-C to detach; does NOT stop the run)
.venv/bin/modal volume ls sdpo-outputs/sdpo_out          # checkpoints appear at step 5/10/15/20
.venv/bin/python src/modal_cost.py --app ap-oeiw6nb406wU2KP1YlKYFM   # spend so far
```
Optional durable local mirror + filtered tail:
```bash
setsid nohup .venv/bin/modal app logs ap-oeiw6nb406wU2KP1YlKYFM > runs/iteration-05/train.log 2>&1 < /dev/null &
tail -f runs/iteration-05/train.log | grep -E "'loss':|success_group_fraction|saved adapter|OutOfMemory|Traceback|did not find"
```

### Step-0 health canaries (first per-step `{'loss': ...}` dict, ~5-10 min after launch)
- **No `OutOfMemory`** — 32k OOM'd; this run is 20k/0.20 to fit. If it OOMs again, see §2.
- `self_distillation/reward_mean` — now the **binary AC rate** on easy+medium (the GRPO signal).
- `self_distillation/success_group_fraction` — fraction of groups with a teacher at **fraction ≥ 0.5**
  (binary-GRPO/fractional-SDPO split; near-misses count, so this should beat AC-only).
- `completions/clipped_ratio` **< 1** — rollouts finishing under 20k (not all NO_CODE).
- `self_distillation/feedback_used_fraction` > 0 — the critic is conditioning all-fail/near-miss groups.

## 2. If it crashed / needs relaunch
1. Find the failure: `.venv/bin/modal app logs ap-oeiw6nb406wU2KP1YlKYFM | grep -iE "error|oom|traceback" | tail`.
2. **OOM** → lower further: `... main --critic --max-completion-length 16384 --vllm-gpu-util 0.18`
   (16k is the thinking-ON NO_CODE floor; easy problems — the signal source — still fit).
3. **Resume caveat (IMPORTANT):** the `sdpo_out` dir on the volume contains **stale checkpoints from a
   prior run** (`checkpoint-3,4,6,7`, possibly a different model). Do **NOT** pass `--resume` blindly —
   it picks the latest `checkpoint-*` and would resume Qwen3 from a stale/foreign checkpoint. Before any
   resume, verify the latest checkpoint is THIS run's (Qwen3, step ≥5) or wipe `sdpo_out` and start fresh.
4. Relaunch is the same decoupled command as §1 (omit `--resume`).

## 3. After training finishes → eval (pass@k only, per plan)
Eval **base vs each checkpoint** on the held-out split. `eval_checkpoint` serves Qwen3-8B at 40960
ctx / 32768 max-tokens on H200 (thinking-ON needs the room) and runs `sdpo_passk` (k=1,2,4,8):
```bash
# one checkpoint at a time (base re-served each call; fine, parallel base+sdpo containers):
.venv/bin/modal run src/modal_sdpo.py::eval_checkpoint --checkpoint checkpoint-20 --languages python
.venv/bin/modal run src/modal_sdpo.py::eval_checkpoint --checkpoint checkpoint-15 --languages python
.venv/bin/modal run src/modal_sdpo.py::eval_checkpoint --checkpoint checkpoint-10 --languages python
.venv/bin/modal run src/modal_sdpo.py::eval_checkpoint --checkpoint checkpoint-5  --languages python
```
Results land as `sdpo_passk_*.json` on the `sdpo-outputs` volume + W&B. The metric is **pass@k**, not
greedy pass@1 (greedy wobbles ±2/25; the SDPO loss is NOT a quality signal).

## 4. Preserve the adapter (so the next run can't overwrite it)
```bash
.venv/bin/modal volume get sdpo-outputs /sdpo_out/checkpoint-20 ./runs/iteration-05/
# and copy the chosen adapter into a per-iteration path on the volume, e.g. sdpo-outputs:/iteration-05/
```

## 5. Iteration-05 decisions locked (don't relitigate without reason)
- **Reward is split by consumer** (`src/sdpo_feedback.py`): GRPO policy **advantage = binary AC (1/0)**
  (`--grpo-reward binary`); SDPO **teacher gating = dense fraction** with `success_reward_threshold`
  (`--sdpo-threshold 0.5`) so near-miss rollouts (≥50% of public cases) can be teacher demonstrations.
  The reward func always judges dense (real fraction on the `FeedbackBus`) and derives binary from the
  verdict; `_LiveFeedbackBuilder` substitutes the fraction tensor into the gate. Teacher-demo selection
  is first-eligible in group order (a possible refinement: prefer the highest-fraction near-miss).
- **Hybrid loss** = `(1−0.1)·GRPO + 0.1·SDPO` via `distillation_weight` (verifier-dominant; no new code).
- **Teacher = base** (fixed/initial T0); **critic = sonnet, thinking per default**; **lr 1e-4**.
- **Memory fit:** 32k completion OOM'd the H200 at the loss step (logits `seq×vocab` + vLLM 0.30 >
  140 GiB). Defaults are now **20480 / vllm 0.20** (baked into `main()`); fragmentation flag
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` already set in the image.
- **GPU:** train AND eval run on **H200** (the `gpu="H100"` strings on `train`/`passk_one` are vestigial
  defaults always overridden via `.with_options(gpu="H200")`).
- **Eval scope:** pass@k only (no GSM8K this run). **Band:** entire train split (easy+medium) minus held-out.

## 6. Gotchas that have bitten us (apply by default)
- **Generate with vLLM, never transformers `model.generate()`** (~10 tok/s on the box; a thinking-ON
  rollout is ~14 min). Don't cap thinking-ON generation tight — Qwen3 NO_CODEs below ~16k.
- **Stream results to disk per item** (concurrent + per-completion write + `--resume`), never buffer-all.
- **Killing a vLLM parent orphans `EngineCore`** (pins ~`gpu_util`×VRAM) → next run OOMs at init. After
  any kill, `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` and `kill -9` leftovers.
- **`pkill -f <script>.py` can kill your own shell** — use a `[.]` bracket regex; never put the relaunch
  in the same line as the `pkill`.
- **Serve the adapter via `vllm --enable-lora`, not a merged checkpoint** (merge drops weights here).
- **Modal:** code+data ship to flat `/root/app`; in-process imports need `/root/app` on `sys.path`.

## 7. Process / repo state at handoff
- Training **running** on Modal (`ap-oeiw6nb406wU2KP1YlKYFM`); nothing local holds it.
- Code changes for the reward split + OOM fix + run name committed (`src/sdpo_feedback.py`,
  `src/sdpo_train.py`, `src/modal_sdpo.py`). No local GPU job on the box.
- `ANTHROPIC_API_KEY` is in repo-root `.env` (gitignored) and mirrored to the Modal secret `anthropic`
  (validated in-container via `modal run src/modal_sdpo.py::critic_check`).
