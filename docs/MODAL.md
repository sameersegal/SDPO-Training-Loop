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

# 3. Upload the 2.7 GB test cases into a Volume (once; ~a few min).
#    This uploads whatever is under local ojbench_data/ — BOTH parts must be there:
#      ojbench_data/NOI/loj-<id>/   (NOI-only split: ojb_splits.json)
#      ojbench_data/ICPC/<id>/      (full NOI+ICPC pool: ojb_splits_full.json, the it-03 default)
#    If you started NOI-only, the volume may be missing ICPC — see "Verify the volume" below.
.venv/bin/modal volume create ojbench-data
.venv/bin/modal volume put ojbench-data ojbench_data /ojbench_data
```

## Verify the volume has every part (NOI + ICPC) — before paying for a GPU

The full 206-pool split (`ojb_splits_full.json`, the iteration-03 default) needs **both**
NOI and ICPC test cases. A cheap CPU-only preflight reports exactly what's present, so a
missing-ICPC volume fails in seconds instead of after a billed H100 spins up:

```bash
.venv/bin/modal run src/modal_sdpo.py::check_data                              # full NOI+ICPC pool
.venv/bin/modal run src/modal_sdpo.py::check_data --ojb-splits ojb_splits.json # NOI-only
```

If it reports missing testdirs, sync the absent part: `modal volume put ojbench-data
ojbench_data/ICPC /ICPC` (the in-container `train()` preflight enforces this too, but on the GPU).

## Run

```bash
# Smoke test first — 2 steps, validates image + data + judge end-to-end. The gemma-4
# weights are staged into hf-cache at IMAGE-BUILD time (off-GPU), so this no longer
# pays a ~10 GB download on the billed GPU — the first *build* absorbs it instead.
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
- **Weights are prefetched at build, not on the GPU** — `image.run_function(_prefetch_weights, …)`
  pulls gemma-4 (~10 GB, gated) into the `hf-cache` volume during the image build (cheap CPU
  builder, `hf_transfer` parallel download). No GPU function ever burns billed H100/H200 time
  downloading weights; the first *build* absorbs it once and is a no-op (verify) on rebuilds.
  This needs the `huggingface` secret to exist before the first build.
- **Empty-data trap** — if you skip the `volume put`, every rollout scores 0 with no error. The
  function asserts the test-case dir is non-empty up front to catch this.
- **On a dedicated GPU the GB10 memory workarounds relax** — `--vllm-gpu-util` defaults to 0.45 here
  (vs 0.30 on the shared GB10). You can also raise the training microbatch in `sdpo_train.py`
  (currently 1 for the GB10) for more speed on an 80 GB card.
- **Multi-GPU (`--gpu H100:4`) is NOT wired yet** — it needs `accelerate launch`/`torchrun` instead of
  plain `python` for distributed training + vLLM colocate sharding. Single-GPU works today; multi-GPU
  is the next increment if one H100 isn't fast enough.
- **Cost** — H100 ≈ $4-8/hr on Modal; a 200-step run is well under an hour. Modal bills per-second while the function runs.
