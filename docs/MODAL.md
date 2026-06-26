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
