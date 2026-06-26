"""Run SDPO training (sdpo_train.py) on a fast Modal GPU.

Why: the GB10 is memory-bandwidth-bound (~270 GB/s unified LPDDR); the SDPO loop
is generation-bound, so an H100/H200 (~3.3 TB/s HBM) runs it several x faster and
frees the local box. The training code (sdpo_train.py / sdpo_ojbench.py) is reused
verbatim; only the environment is reproduced here.

ONE-TIME SETUP (see MODAL.md for the walkthrough):
  modal setup                                              # browser auth
  modal secret create huggingface HF_TOKEN=hf_xxx          # gated gemma-4 access
  modal secret create wandb WANDB_API_KEY=xxx
  modal volume create ojbench-data
  modal volume put ojbench-data ojbench_data /ojbench_data # 2.7 GB test cases, once

RUN:
  modal run modal_sdpo.py --smoke                          # 2-step validation
  modal run modal_sdpo.py --difficulties easy --max-steps 200
  modal run modal_sdpo.py --gpu H200 --difficulties easy --max-steps 500

GET THE ADAPTER BACK:
  modal volume get sdpo-outputs /sdpo_out ./sdpo_out_modal
"""
import modal

APP_NAME = "sdpo-gemma-ojbench"
PYTHON_VERSION = "3.12"

# CUDA *devel* base: flashinfer JIT-compiles kernels at runtime and needs nvcc.
# Versions pinned to match the validated GB10 venv exactly.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python=PYTHON_VERSION
    )
    .apt_install("g++", "build-essential", "git")
    # Pin only the behavior-critical packages; let pip resolve transitive deps
    # (numpy/safetensors/accelerate/huggingface-hub are pulled by vLLM/transformers).
    # Over-pinning numpy==2.5.0 conflicts with numba (needs numpy<2.5) on x86.
    .pip_install(
        "torch==2.11.0",
        "vllm==0.23.0",
        "trl==1.6.0",
        "peft==0.19.1",
        "transformers==5.12.1",
        "datasets==5.0.0",
        "flashinfer-python==0.6.12",
        "wandb==0.28.0",
    )
    .env(
        {
            "TRL_EXPERIMENTAL_SILENCE": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        }
    )
    # Ship the training + judge code (data lives in a Volume, not the image).
    .add_local_file("ojbench_eval.py", "/root/app/ojbench_eval.py")
    .add_local_file("sdpo_ojbench.py", "/root/app/sdpo_ojbench.py")
    .add_local_file("sdpo_train.py", "/root/app/sdpo_train.py")
    .add_local_file("ojb_splits.json", "/root/app/ojb_splits.json")
    .add_local_file("ojbench_selected.json", "/root/app/ojbench_selected.json")
)

app = modal.App(APP_NAME)

# Persistent volumes: model weights cache, the 2.7 GB test cases, adapter outputs.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
ojbench_data = modal.Volume.from_name("ojbench-data", create_if_missing=True)
outputs = modal.Volume.from_name("sdpo-outputs", create_if_missing=True)

# ojbench_data volume holds ".../ojbench_data" contents at "/ojbench_data";
# the judge expects them under /root/app/ojbench_data (ROOT/ojbench_data/NOI/...).
VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/app/ojbench_data": ojbench_data,
    "/root/app/sdpo_out": outputs,
}


@app.function(
    image=image,
    gpu="H100",  # overridden per-run via .with_options(gpu=...)
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("huggingface"),  # -> HF_TOKEN (gated gemma-4)
        modal.Secret.from_name("wandb"),  # -> WANDB_API_KEY
    ],
    timeout=6 * 60 * 60,
)
def train(args: list[str]):
    import os
    import subprocess
    import sys

    os.chdir("/root/app")
    os.environ.setdefault("WANDB_PROJECT", APP_NAME)

    # Sanity: confirm the test data actually mounted (a silent-empty volume is
    # the #1 way this fails — every rollout would score 0 with no error).
    noi = "/root/app/ojbench_data/NOI"
    n = len(os.listdir(noi)) if os.path.isdir(noi) else 0
    print(f"[modal] ojbench test-case dirs visible: {n}", flush=True)
    assert n > 0, f"{noi} is empty — run `modal volume put ojbench-data ...` first"

    cmd = [sys.executable, "sdpo_train.py", *args]
    print("[modal] running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    outputs.commit()  # persist the saved adapter
    hf_cache.commit()  # persist downloaded weights for next run
    print("[modal] done — adapter saved to volume 'sdpo-outputs' (/sdpo_out)", flush=True)


@app.local_entrypoint()
def main(
    smoke: bool = False,
    gpu: str = "H100",  # e.g. H100, H200, A100-80GB
    difficulties: str = "easy",
    languages: str = "python,cpp",
    num_generations: int = 8,
    max_completion_length: int = 8192,
    max_steps: int = 20,
    vllm_gpu_util: float = 0.45,  # dedicated GPU -> more headroom than the GB10's 0.30
):
    if smoke:
        args = ["--smoke", "--difficulties", "easy", "--languages", languages]
    else:
        args = [
            "--difficulties", difficulties,
            "--languages", languages,
            "--num-generations", str(num_generations),
            "--max-completion-length", str(max_completion_length),
            "--max-steps", str(max_steps),
            "--vllm-gpu-util", str(vllm_gpu_util),
            "--output-dir", "sdpo_out",
        ]
    print(f"[modal] gpu={gpu}  args={args}")
    train.with_options(gpu=gpu).remote(args)
