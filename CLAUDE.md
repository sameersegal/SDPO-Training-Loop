# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **research repo**, not a product: post-train **Gemma-4-E2B-it** with **SDPO**
(Self-Distillation Policy Optimization, TRL's experimental `SDPOTrainer`) on **OJBench**
competitive-programming problems, then measure generalization (held-out hard + easy/medium
thermometer) and capability regression (GSM8K). Work proceeds **iteration by iteration**.

## Repository layout
```
README.md  CLAUDE.md     # orientation + this guidance
docs/                    # design & process docs (source of truth)
src/                     # ALL Python (training, eval, judge, Modal, plotting)
scripts/                 # local vLLM serve helpers (serve.sh, serve_vibe.sh)
data/                    # committed inputs: ojb_splits.json, ojbench_selected.json
reports/iteration-NN/    # COMMITTED per-iteration results: REPORT.md, PROVENANCE.md, figures/, data/
reports/comparison/      # cross-iteration overlays (figures gitignored, regenerable)
knowledge/               # SDPO literature: citing_papers.csv (classified + abstracts) + summary_*.md deep reads
runs/iteration-NN/       # gitignored raw outputs (adapters, logs, raw eval JSONs)
ojbench_data/            # OJBench test cases (~2.7 GB, gitignored, re-fetchable)
```
Raw run artifacts (`*.log`, `results_*.json`, `sdpo_eval_*.json`, `sdpo_passk_*.json`, adapters,
`wandb/`) are gitignored. Curated, small, diffable results are committed under `reports/iteration-NN/`.

## Docs are the source of truth — read before changing anything
Kept current and authoritative; defer to them over inference. All in `docs/`:

| Doc | Role |
|---|---|
| `docs/EXPERIMENT.md` | **Design source of truth** — method, splits, configs, decisions, status, caveats |
| `docs/FINDINGS.md` | Results **index** → per-iteration reports + standing lessons |
| `docs/MODAL.md` | Cloud scale-up (Modal H100/H200) |
| `docs/HANDOFF.md` | Original handoff (executed, kept for history) |
| `reports/iteration-NN/REPORT.md` | Self-contained per-iteration report with embedded graphs |
| `knowledge/` | SDPO **literature**: `citing_papers.csv` (papers citing SDPO, classified by relevance + abstracts) and `summary_*.md` deep reads with a "how it applies here" section. Add a `summary_*.md` when you read a new paper. |

When you change behavior, update the relevant doc in the same change.

**Component design docs (`docs/design/`)** capture the design of *critical components* —
independent of any iteration — so they can seed a blog post and outlive a given run. Keep them
current when you change the component. Existing: `docs/design/JUDGE.md` (judge, reward modes,
public/private split, smallest-first ordering, feedback, worked-example hint); `docs/design/OBSERVABILITY.md`
(iter-06 rollout/logprob/metric capture so runs are reviewable; **implemented**: rollout JSONL + sampling
knobs + eval-completion text. Per-token logprobs are regenerated OFFLINE from checkpoints — so preserving
EVERY checkpoint is mandatory). When you build a new critical component (e.g. the live-feedback trainer subclass, the
dataset/filter pipeline), add a `docs/design/<COMPONENT>.md` rather than burying the rationale in an
iteration report.

## Code map (all in `src/`)
```
_paths.py           path resolver — finds data files + .env in BOTH the local src/ layout
                    and the flat /root/app Modal container. Use find_file()/ojbench_dir()/load_env().
sdpo_ojbench.py     OJBench env adapter: load ojb_splits.json, public/private test split,
                    judge_completion() (py + cpp), build_dataset()/make_reward_func() for TRL
ojbench_eval.py     low-level judging primitives (extract_code, run tests, normalize)
sdpo_train.py       SDPO training entrypoint (SDPOTrainer + LoRA + vLLM colocate)
sdpo_eval_vllm.py   held-out pass@1 by language x difficulty via a vLLM OpenAI endpoint (+W&B)
sdpo_passk.py       held-out pass@k (k=1,2,4,8) — the discriminative, low-noise eval metric
eval_runner.py      GSM8K regression probe via a vLLM endpoint
modal_sdpo.py       reproduce env on Modal + train()/evaluate()/passk_one() (code reused verbatim)
generate_slides.py  per-iteration figures from reports/<ITER>/data/*.csv + passk JSONs
compare_iterations.py  overlay loss/length/reward across iterations from committed CSVs
```
`data/ojb_splits.json` → `sdpo_ojbench.py` (prompts + judge) is imported by `sdpo_train.py`
(reward) **and** every eval script (verdicts). Change the judge/split and both move together.

**Path rule:** scripts read committed data via `_paths` (works locally + in the container) and
**write outputs to the CWD**. Run from `runs/iteration-NN/` so artifacts stay per-iteration.

## Commands
Setup (uv venv; `WANDB_API_KEY` in `.env` at repo root, gitignored):
```bash
uv venv .venv && source .venv/bin/activate
uv pip install vllm trl peft accelerate wandb datasets openai huggingface_hub matplotlib pyyaml
```

Local run order (GB10) — full sequence in `docs/EXPERIMENT.md` §9. Run from the iteration dir:
```bash
mkdir -p runs/iteration-NN && cd runs/iteration-NN
S=../../src                                   # scripts live in src/
PYTHONPATH=$S python $S/sdpo_train.py --smoke  # always smoke-test integration first
# 1. baseline (serve base on :8000 via ../../scripts/serve.sh):
PYTHONPATH=$S python $S/sdpo_eval_vllm.py --served-model google/gemma-4-E2B-it --tag base --max-tokens 32768 --wandb
PYTHONPATH=$S python $S/eval_runner.py --dataset gsm8k --sample-frac 1.0 --out results_gsm8k_base_full.json
# 2. train easy-only (free the GPU first — see gotchas):
PYTHONPATH=$S python $S/sdpo_train.py --difficulties easy --languages python,cpp --max-steps 20 \
  --vllm-gpu-util 0.30 --output-dir sdpo_out
# 3. serve the ADAPTER (not a merge), then re-eval:
vllm serve google/gemma-4-E2B-it --enable-lora --lora-modules sdpo=sdpo_out \
  --max-lora-rank 32 --dtype bfloat16 --max-model-len 36864 --gpu-memory-utilization 0.85
PYTHONPATH=$S python $S/sdpo_passk.py --served-model sdpo --tag sdpo --wandb   # pass@k, not greedy
```

Cloud (faster, frees the local box) — full flow in `docs/MODAL.md`, run from repo root:
```bash
.venv/bin/modal run src/modal_sdpo.py --smoke                       # validate image+data+judge
.venv/bin/modal run src/modal_sdpo.py --difficulties easy --max-steps 100
.venv/bin/modal run src/modal_sdpo.py::run_eval                     # base+adapter eval in parallel
.venv/bin/modal volume get sdpo-outputs /iteration-NN/adapter_model.safetensors ./   # pull adapter
```

Figures: `python src/generate_slides.py` (set `ITER=iteration-NN`); cross-iteration:
`python src/compare_iterations.py`.

## Critical gotchas (these have bitten us; respect them)
- **Train easy-only.** easy+medium yields **0 successful rollouts cold** → `success_group_fraction=0`
  → no distillation signal. But note iteration-01: **easy-only for 100 steps OVERFITS/regresses**
  (mode-collapse to terse outputs; held-out pass@k *and* GSM8K dropped). Iteration-02 plan: train the
  **learnability frontier** (easy + sometimes-solvable medium), lower LR, fewer epochs, KL anchor,
  early-stop on held-out pass@k. (`docs/EXPERIMENT.md` §5, `docs/FINDINGS.md`)
- **pass@k is the metric, not greedy pass@1.** Greedy wobbles ±2/25 (batch nondeterminism) and the
  SDPO **loss is not a quality signal** (it stayed flat while the model regressed). Use `sdpo_passk.py`.
- **GB10 silently OOM-kills** without `per_device_train_batch_size=1` + grad checkpointing +
  `--vllm-gpu-util 0.30`. Microbatch = `num_generations` OOMs at step 0 on the LM-head logits tensor.
  Free GPU memory before relaunching (128 GB *unified*). H100 (80 GB) uses `0.25`; H200 (≥141 GB) fits
  `--no-grad-checkpointing`. **The GB10 also hangs on high-concurrency multi-sample (n>1) inference** →
  run pass@k eval on Modal.
- **Generate with vLLM, NEVER transformers `model.generate()`.** Sequential HF decoding on the GB10 is
  ~10 tok/s — a single thinking-ON rollout (8k+ tokens) takes **~14 min**, so a 12-rollout probe set is
  *hours*, most of it wasted. vLLM (the same engine the serve/eval path already uses — `enforce_eager`
  + `n=1` to dodge the multi-sample hang above) batches and is ~10–30× faster. `gen_rollouts.py
  --engine vllm` is the fast path; `probe_teacher_accuracy.py`'s transformers `generate()` is the trap
  that cost us a wasted run. **Corollary: don't cap thinking-ON generation tight** — Qwen3-8B think-ON
  blew past an 8192 cap and returned **NO_CODE even on an *easy* problem** (the whole budget went to
  `<think>`); use ≥16k or uncapped, or the rollout is unusable.
- **Stream results to disk as each one is produced — NEVER buffer-all-then-write.** A single batched
  `llm.generate(all_prompts)` (or any "collect in a list, dump at the end") returns only after
  *everything* finishes: **zero live visibility** (you can't tell steady progress from a hang) and
  **zero resumability** (an interruption loses the whole batch). Generate→judge **per item in a loop,
  writing the JSON after each** (`gen_rollouts.py` `_persist()`), and support `--resume` to skip items
  already in the output file. **Don't conflate streaming-to-disk with *serial* generation.** The way to
  keep visibility/resume AND speed on a real GPU is **concurrent generation with per-completion writes**
  (`gen_rollouts.py --concurrent K`, vLLM v1 `AsyncLLM`): all requests run in vLLM's continuous batch,
  each writes the instant *it* finishes. Serial (`--concurrent 1`) is only the GB10 default because the
  GB10 hangs on high concurrency *and* batching buys it ~nothing there (aggregate ~14 tok/s ≈
  single-stream ~13 — a GB10 LPDDR-bandwidth artifact, **NOT a general truth**). On the H200 the KV
  cache showed **~45× concurrency at 16k**, so serial would waste most of the GPU — use `--concurrent`
  there (the Modal `gen` entrypoint defaults to 12). Same rule for eval sweeps.
- **Serve the adapter via `vllm --enable-lora`, NOT a merged checkpoint.** `merge_and_unload` on this
  multimodal model silently drops upper-layer `k_norm` weights → vLLM "weights not initialized". Eval
  base and adapter on the *same* server for a clean delta.
- **LoRA targets the text tower only** — regex `.*language_model.*\.(q|k|v|o|gate|up|down)_proj$`.
  gemma4's vision/audio towers use `Gemma4ClippableLinear` which PEFT can't wrap.
- **Judge is lightweight, not the official DMOJ sandbox** (DMOJ needs root + pypy3). Python = stdout
  diff; C++ = `g++ -O2 -std=c++17`. **Judge fidelity = reward fidelity** — a false AC trains the wrong
  thing. The cpp judge can stall on a pathological completion — harden before trusting at scale.
- **Modal:** code+data ship to a **flat `/root/app`** layout; OJBench test cases come from the
  `ojbench-data` Volume; adapters land in `sdpo-outputs` (preserve per iteration under `/iteration-NN/`).
- **`--output-dir` MUST be prefixed `sdpo_out/<name>` or the run's checkpoints are LOST.** `train()`
  chdir's to `/root/app` but the `sdpo-outputs` volume mounts at `/root/app/sdpo_out`. A bare
  `--output-dir iter09-dose` writes to `/root/app/iter09-dose` — *off* the mount — so the periodic
  commit captures nothing and a 10h run ends with zero checkpoints on the volume (and `eval_dose` finds
  nothing). Use `--output-dir sdpo_out/iter09-dose` → lands at `volume:/iter09-dose`. iter-09's preflight
  caught this; the bare-name form is also why loose `checkpoint-*` litter the volume root from old runs.
  Eval entrypoints take the BARE name (`eval_dose --output-dir iter09-dose`) — they prepend `sdpo_out/`.

## Preserving each iteration
Commit small/diffable artifacts to `reports/iteration-NN/`: `REPORT.md` (self-contained, embeds
figures), `PROVENANCE.md` (W&B run IDs, Modal adapter path, spend), `data/` (per-step metrics CSVs +
eval summaries), `figures/`. Copy the trained adapter to `sdpo-outputs:/iteration-NN/` so the next run
can't overwrite it. Raw logs/results stay in `runs/` (gitignored).

## Budget discipline (Modal costs real money)
**Use actuals, not napkin math.** `python src/modal_cost.py` reads `modal billing report` and prints
the real $ per app (GPU/CPU/Memory split): `--for today` / `--for "this month"` for a window,
`--this-run` for the app in `runs/iteration-NN/RUNNING_APP_ID.txt`, `--app <id|prefix>` for one run.
Check it after runs to ground spend decisions in measured cost (e.g. an H200 colocate train step ran
~$0.x/step; a single **hung** detached run cost **$3.16 for zero progress** — which is *why* the
de-risk ladder below matters). Billing lags the current partial interval, so re-run after a job ends
for the final number.

Cloud GPU time is billed — **de-risk new code before a long run, but don't over-test cheap changes.**
The escalation ladder, cheapest first: **unit tests** (`pytest tests/`, seconds, free) → **GB10 smoke**
(`sdpo_train.py --smoke`, local, free) → **Modal smoke** (`modal run …::main --smoke`, ~minutes, cents)
→ **long run** (hours, $). Climb only as far as the risk warrants:
- Pure-logic changes (judge, prompt assembly, splits, reward shaping) → unit tests are usually enough.
- Anything touching the train/eval *integration* (new flags, dataset wiring, trainer subclass, Modal
  image/volume) → at least a smoke on the tier you changed before paying for a long run. The
  iteration-03 false starts (serial judging stall, missing ICPC volume data, extraction race) were all
  caught by smokes / the preflight — each would have been an expensive multi-hour failure.
- **Make the smoke exercise the REAL failure surface, not just "does it start."** The `--smoke`
  config (4 shuffled problems, 512-token cap, 2 steps) wired things up but was too small/short to catch
  iteration-03's expensive failures: serial-judging slowness (the 512 cap hid it), a split problem with
  no volume data (4 problems missed it), the parallel-extraction race (needs the real concurrent path),
  the CUDA-graph hang (intermittent), and the per-step cost. Before a long run, do a **representative
  pre-flight**: a few REAL steps at full `--max-completion-length` with `--save-steps 1` (exercises real
  generation+judging timing, the parallel judge path, a checkpoint write, and resume) — ~15 min / ~$1
  vs the ~$18 of late-caught false starts.
- **Apply known hazards by default.** Bake mitigations we've already paid to learn into the default
  launch path (enforce_eager for the kernel hang, preflight-all-testdirs, group-kill judge timeouts,
  decoupled launch, `--resume`, and the **no-progress watchdog** that auto-kills a silent hang in
  minutes) so they aren't rediscovered at $3–10 each. **Time hurts more than cost** — a silent hang
  burns hours of wall-clock; the watchdog + resume turn that into a minutes-long auto-kill + restart.
- **Balance against speed:** the goal is correct results *quickly*. Don't gold-plate — skip a redundant
  Modal smoke when a GB10 smoke already exercised the same path, and prefer one well-instrumented long
  run (checkpoints, W&B, eval-ready) over several timid short ones. When unsure of per-step cost, launch
  the long run but **watch the first few steps** and kill early if the cadence/budget doesn't add up.
- **Watch machine utilization, not just progress — an idle GPU is wasted wall-clock.** When monitoring
  any GPU job, check the vLLM/engine telemetry (**GPU KV-cache usage %, Running vs Waiting request
  counts, tokens/s**) and `nvidia-smi`, and ask: *is the GPU actually saturated, or are we leaving
  performance on the table?* Low KV-cache % with a deep Waiting queue (e.g. eval ran 16 concurrent at
  6–18% KV on an H200 → ~3× headroom unused) means concurrency/batch is too low. **If a job is plainly
  under-utilizing the hardware, cancel it and relaunch with better params** (higher `--max-num-seqs` +
  client `--concurrency`, bigger batch, more GPUs/parallel shards) rather than letting it grind — the
  restart cost is minutes, the wasted idle time is hours. **Time is more valuable than money:** spending
  a little more on bigger/more concurrent hardware to finish sooner beats a cheap job that idles for
  hours. Pick the parameters that *saturate* the box (watch KV stays under the `gpu_memory_utilization`
  budget so you don't OOM), and re-check utilization in the first few steps after relaunch.
- **Eval defaults (baked in after iter-08): `--max-seqs 96` (not 48 — KV was still only ~25–46% on the
  H200), client `--concurrency` auto-matches it, and `--max-tokens ≥16k` (32k is safe but the 32k think
  tail dominates wall-clock; 20–24k is the speed/headroom trade).** And **judge OFF the GPU**: judging is
  pure CPU, so running it on the H200 idles a $/s card. Generate cloud-side with `sdpo_passk.py
  --no-judge` (streams completions to `sdpo_passk_<tag>_samples.jsonl`), pull that small JSONL, judge
  locally on the GB10 with `src/judge_local.py --tag <tag>` (same `sdpo_passk_<tag>.json` schema, free).
- **Never trust a <~25-problem pass@k probe for a capability claim.** A 12-problem × n=8 probe has ±0.15
  noise — enough to fabricate a "monotonic trend" from nothing (iter-06/07/08's "collapse→fix" arc was
  noise; it vanished on the 30-problem n=12 bootstrap-CI `eval_iterations`). Gate every capability claim
  on a CI that excludes zero; the per-step `flat_group` mechanism metric is trustworthy, the small-probe
  pass@k is not.

## Long-running runs MUST survive restarts & network drops
Training and eval runs take hours; they must outlive a dropped session, a network blip, or a
client restart, and be resumable. Non-negotiables:
- **Launch fully DECOUPLED, not just `--detach`.** `--detach` survives a clean client *disconnect*
  but NOT the client being **signal-killed mid-stream** — an attached `modal run` (or one held open in
  a session-bound shell/log stream) sends a *cancel* when killed. This cancelled a run at step 11
  ($4.86 lost). Launch so the client exits cleanly and nothing lingers to be killed:
  `setsid nohup .venv/bin/modal run --detach src/modal_sdpo.py::main ... > LOG 2>&1 < /dev/null &`.
  Local GB10: detached tmux. Confirm the client returned ("Done") and the app shows in `modal app list`.
- **Record the run id so it survives a restart.** Write the Modal **app id** (and the recipe) to
  `runs/iteration-NN/RUNNING_APP_ID.txt` right after launch. After any restart, reattach/monitor via
  that id + the `sdpo-outputs` Volume — do **not** rely on the local log (it freezes when the client
  dies; the remote keeps going).
- **Checkpoint cadence < interruption interval, and RESUME (don't restart).** `sdpo_train.py
  --save-steps N` (`save_strategy="steps"`, keep all) + the Modal `train()` commits the `sdpo-outputs`
  Volume periodically. **Relaunch with `--resume`** (Modal: `--resume`) — it picks up the latest
  `checkpoint-*` in the output dir, and is idempotent (fresh run with no checkpoint starts at step 0).
  Set `N` *smaller than how often runs actually die*: in iteration-03 every failure hit before the
  first checkpoint (step 20 ≈ 80 min) while interruptions came every ~30–45 min, so ~6 dead runs
  (~$18) each lost ALL progress. A checkpoint you can't reach (or resume from) saves nothing.
- **Monitor via durable, STATELESS queries — not a long-lived monitor.** Poll the Volume for
  `checkpoint-*` and `modal app list` for status with one-shot commands (`python src/modal_cost.py
  --this-run` for spend); a long-running monitor process dies with the session. The remote run keeps
  going regardless.

## Stopping GPU jobs
Kill training/serve jobs by **PID or tmux session**, never `pkill -f vllm` (collateral damage).
Long runs are launched in detached tmux (local) or `modal run --detach` (cloud).
- **Killing the Python parent ORPHANS the vLLM `EngineCore` subprocess, which keeps holding the GPU.**
  Killing `python …` (or `pkill -f gen_rollouts`) leaves a `VLLM::EngineCore` proc pinning ~`gpu_util`×VRAM
  (~101 GB at 0.85 on the GB10) → the next run **OOMs at engine init**. After any kill, verify the GPU is
  actually free with **`nvidia-smi --query-compute-apps=pid,used_memory --format=csv`** (this works on the
  GB10 even though the `memory.used` query returns N/A) and `kill -9` any leftover `EngineCore` before relaunching.
- **`pkill -f <script>.py` can kill your own shell** — the pattern matches the launching command's own
  cmdline. Use a regex that can't self-match (e.g. `pkill -f 'gen_rollouts[.]py'`) and never put the
  relaunch command in the same shell line as the `pkill`.
