# Running SDPO on Modal

Port of the SDPO training loop to a fast cloud GPU. The GB10 is memory-bandwidth-bound
(~270 GB/s); an H100/H200 (~3.3 TB/s, ~12×) runs this generation-bound loop several × faster
and frees the local box. `src/sdpo_train.py` / `src/sdpo_ojbench.py` are reused verbatim — `src/modal_sdpo.py`
only reproduces the environment.

## One-time setup

All commands use the venv's CLI: `.venv/bin/modal` (or activate the venv first).

```bash
# 1. Authenticate (opens a browser; needs a Modal account at modal.com)
.venv/bin/modal setup

# 2. Secrets — gated gemma-4 download + W&B logging
#    HF token must have accepted the gemma-4 license on huggingface.co.
.venv/bin/modal secret create huggingface HF_TOKEN=hf_xxxxxxxx
.venv/bin/modal secret create wandb WANDB_API_KEY=$(grep WANDB_API_KEY .env | cut -d= -f2)

# 3. Upload the 2.7 GB test cases into a Volume (once; ~a few min)
.venv/bin/modal volume create ojbench-data
.venv/bin/modal volume put ojbench-data ojbench_data /ojbench_data
```

## Run

```bash
# Smoke test first — 2 steps, validates image + data + judge end-to-end (~5-10 min,
# most of which is the one-time gemma-4 download into the hf-cache volume).
.venv/bin/modal run src/modal_sdpo.py --smoke

# Real run — the scaled experiment the GB10 can't do quickly
.venv/bin/modal run src/modal_sdpo.py --difficulties easy --max-steps 200
.venv/bin/modal run src/modal_sdpo.py --gpu H200 --difficulties easy --max-steps 500
```

Flags (see `main()` in `src/modal_sdpo.py`): `--gpu` (H100/H200/A100-80GB), `--difficulties`,
`--languages`, `--num-generations`, `--max-completion-length`, `--max-steps`, `--vllm-gpu-util`.

### Base-model pass@k / opportunity graph (any model)
`passk_base` serves an arbitrary **base** model with vLLM and runs `sdpo_passk` concurrently against
it (16-way × n=8, continuous-batched) — the pass@1→pass@8 "opportunity" data, model-parameterized
(`evaluate`/`passk_one` hardcode gemma + adapter semantics; this doesn't).
```bash
.venv/bin/modal run src/modal_sdpo.py::passk_base --smoke              # 2 easy, n=2 — validate plumbing
.venv/bin/modal run src/modal_sdpo.py::passk_base                      # Qwen3-8B python, 25 heldout, n=8, 32k
.venv/bin/modal run src/modal_sdpo.py::passk_base --model <id> --languages python,cpp --max-tokens 32768
```
Flags: `--model` (default `Qwen/Qwen3-8B`), `--languages`, `--n`, `--max-tokens`, `--temperature`,
`--system` (cp_method/expert/none), `--gpu`, `--ojb-splits`, `--out`. **Qwen3 think-ON needs ≥16k**
(8k → NO_CODE). Writes `sdpo_passk_<tag>.json` locally + to the `sdpo-outputs` volume. Plot with
`ITER=<iter> python src/plot_opportunity.py --passk <json> --label "<model>"`.

### Definitive cross-iteration eval (`eval_iterations`)
The matched, low-noise pass@k: base + a list of checkpoints on the **same** problem ids, all in
parallel, n>k so pass@k is *graded* (not binary-per-problem). This is the eval that settles a
small-probe trend — **don't trust a <~25-problem probe for any capability claim** (iter-06/07/08's
"collapse→fix" arc was 12-probe noise that vanished here; `reports/iteration-08/REPORT.md` §3.5).
```bash
.venv/bin/modal run src/modal_sdpo.py::eval_iterations \
  --ids "<comma pids>" --checkpoints "iteration-05/checkpoint-20,iter06-fast/checkpoint-8,..." \
  --n 12 --max-tokens 32768 --max-seqs 96
python src/iters30_analysis.py   # bootstrap 95% CI + figure
```
Flags: `--ids`, `--checkpoints` (comma volume paths), `--n`, `--temperature`, `--max-tokens`,
`--max-seqs`, `--gpu`. Each model spawns its own H200.

### Eval efficiency knobs (apply by default)
- **`--max-seqs` (default 96, was 48).** The iters30 eval ran KV cache only ~25–46% at 48 seqs on
  the H200 — under-utilized. 96 ~doubles in-flight requests + wall-clock throughput; the client
  `--concurrency` auto-matches `max_seqs` so the continuous batch stays full. **Watch KV-cache % stays
  under the 0.85 budget** (drop `max_seqs` if it pins/OOMs). Time > money: saturate the box.
- **`--max-tokens` (≥16k).** The 32k thinking tail dominates wall-clock (~9 min/seq). 32k is the safe
  default (Qwen3 think-ON NO_CODEs below ~16k); **20–24k trades a little headroom for a much shorter
  tail** when speed matters more than the last few hard problems.
- **Decouple generate (GPU) from judge (CPU).** Judging is pure CPU (stdout-diff / `g++`); running it
  on the H200 idles a $/s GPU. Generate on the cloud with **`sdpo_passk.py --no-judge`** (streams every
  completion to `sdpo_passk_<tag>_samples.jsonl`, skips judging + aggregation), pull that small JSONL,
  and judge **locally on the GB10 for free**:
  ```bash
  # cloud (GPU only generates):  sdpo_passk.py ... --no-judge --tag <tag>
  .venv/bin/modal volume get sdpo-outputs /<iter>/sdpo_passk_<tag>_samples.jsonl ./
  python src/judge_local.py --tag <tag> --workers 8   # writes the canonical sdpo_passk_<tag>.json
  ```
  `judge_local.py` produces the identical `sdpo_passk_<tag>.json` schema, so `iters30_analysis.py` /
  `generate_slides.py` consume it unchanged. Needs `ojbench_data/` present locally (re-fetchable).

## Get the trained adapter back

```bash
.venv/bin/modal volume get sdpo-outputs /sdpo_out ./sdpo_out_modal
```

Then evaluate locally exactly as before (serve via vLLM `--enable-lora --lora-modules sdpo=./sdpo_out_modal`).

## Notes / gotchas
- **First run is slow** — it downloads gemma-4 (~10 GB, gated) into the `hf-cache` volume; later runs reuse it.
- **Empty-data trap** — if you skip the `volume put`, every rollout scores 0 with no error. The
  function asserts the test-case dir is non-empty up front to catch this.
- **On a dedicated GPU the GB10 memory workarounds relax** — `--vllm-gpu-util` defaults to 0.45 here
  (vs 0.30 on the shared GB10). You can also raise the training microbatch in `sdpo_train.py`
  (currently 1 for the GB10) for more speed on an 80 GB card.
- **Multi-GPU (`--gpu H100:4`) is NOT wired yet** — it needs `accelerate launch`/`torchrun` instead of
  plain `python` for distributed training + vLLM colocate sharding. Single-GPU works today; multi-GPU
  is the next increment if one H100 isn't fast enough.
- **Cost** — H100 ≈ $4-8/hr on Modal; a 200-step run is well under an hour. Modal bills per-second while the function runs.
