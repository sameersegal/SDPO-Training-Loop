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
from pathlib import Path

import modal

APP_NAME = "sdpo-gemma-ojbench"
PYTHON_VERSION = "3.12"

SRC = Path(__file__).resolve().parent           # repo/src
REPO = SRC.parent                                # repo root
DATA = REPO / "data"
# Code + judge ship to a FLAT /root/app layout (data via Volume), so absolute
# local paths keep `modal run src/modal_sdpo.py` working from any CWD.
_CODE = ["_paths.py", "ojbench_eval.py", "sdpo_ojbench.py", "sdpo_train.py",
         "sdpo_feedback.py", "sdpo_passk.py", "sdpo_eval_vllm.py", "eval_runner.py"]
_DATA = ["ojb_splits.json", "ojb_splits_full.json", "ojbench_selected.json"]

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
)
# Ship the training + judge code + committed data into the flat /root/app layout
# (OJBench test cases come from a Volume, not the image).
for _f in _CODE:
    image = image.add_local_file(str(SRC / _f), f"/root/app/{_f}")
for _f in _DATA:
    image = image.add_local_file(str(DATA / _f), f"/root/app/{_f}")

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
    cpu=16.0,  # dense judging is subprocess-bound + thread-parallel — give it real cores
    volumes=VOLUMES,
    secrets=[
        modal.Secret.from_name("huggingface"),  # -> HF_TOKEN (gated gemma-4)
        modal.Secret.from_name("wandb"),  # -> WANDB_API_KEY
    ],
    timeout=10 * 60 * 60,  # dense reward + feedback on the full 206-pool is slower per step
)
def train(args: list[str], num_gpus: int = 1, ojb_splits: str = "ojb_splits_full.json"):
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time

    os.chdir("/root/app")
    os.environ.setdefault("WANDB_PROJECT", APP_NAME)
    # Iteration-03 trains on the full 206-pool. OJB_SPLITS is read at import time by
    # sdpo_ojbench, so set it in the env the subprocess inherits.
    os.environ["OJB_SPLITS"] = ojb_splits
    print(f"[modal] OJB_SPLITS={ojb_splits}", flush=True)
    # Pull the latest committed volume state so a --resume run sees checkpoints written
    # by a previous (dead) run.
    outputs.reload()

    # Log the GPU actually provisioned (Modal's UI shows the decorator's declared
    # gpu, not the per-run .with_options() override — so confirm from inside).
    try:
        import torch

        p = torch.cuda.get_device_properties(0)
        print(f"[modal] GPU: {p.name} x{torch.cuda.device_count()} "
              f"({p.total_memory / 1024**3:.0f} GiB each)", flush=True)
    except Exception as e:
        print(f"[modal] GPU probe failed: {e}", flush=True)

    # Sanity: confirm the test data actually mounted (a silent-empty volume is
    # the #1 way this fails — every rollout would score 0 with no error).
    noi = "/root/app/ojbench_data/NOI"
    n = len(os.listdir(noi)) if os.path.isdir(noi) else 0
    print(f"[modal] ojbench test-case dirs visible: {n}", flush=True)
    assert n > 0, f"{noi} is empty — run `modal volume put ojbench-data ...` first"

    # Preflight EVERY split problem's testdir BEFORE training — a missing dir otherwise
    # crashes mid-step-2 after minutes of generation (an ICPC dir missing did exactly
    # that). Fail fast with the full list instead.
    import json as _json
    _split = _json.load(open(f"/root/app/{ojb_splits}"))
    _missing = []
    for _pid, _td in _split.get("testdir_by_id", {}).items():
        if not os.path.exists(f"/root/app/ojbench_data/{_td}/init.yml"):
            _missing.append(_td)
    if _missing:
        raise RuntimeError(
            f"{len(_missing)} split testdirs missing from the volume "
            f"(e.g. {_missing[:5]}). Sync them: modal volume put ojbench-data <local> /<part>")
    print(f"[modal] preflight: all {len(_split.get('testdir_by_id', {}))} split testdirs present",
          flush=True)

    if num_gpus > 1:
        # Data-parallel: each process owns one GPU, runs its own colocate vLLM
        # and generates a shard of the rollouts. ~N x generation throughput.
        cmd = [
            "accelerate", "launch",
            "--num_processes", str(num_gpus),
            "--num_machines", "1",
            "--mixed_precision", "bf16",
            "sdpo_train.py", *args,
        ]
    else:
        cmd = [sys.executable, "sdpo_train.py", *args]
    print(f"[modal] num_gpus={num_gpus} running:", " ".join(cmd), flush=True)

    # Commit the outputs volume periodically so per-20-step checkpoints become
    # durable DURING the run (not just at the end) — a long run that dies mid-way
    # still leaves its latest checkpoint on the volume to resume/eval from.
    stop = threading.Event()

    def _committer():
        while not stop.wait(120):
            try:
                outputs.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[modal] periodic commit failed: {e}", flush=True)

    t = threading.Thread(target=_committer, daemon=True)
    t.start()

    # --- No-progress watchdog -------------------------------------------------
    # The expensive failure mode is a SILENT hang (a generation deadlock or a judge
    # that never returns) burning the GPU for hours with ZERO output (cost ~$10/2h once).
    # Heartbeat = ANY output line. A real hang produces no output at all, so "no line for
    # STALL seconds" is a clean, format-independent signal (a healthy step prints metrics +
    # tqdm well within STALL; model-load/first-gen also stream output). On stall, kill the
    # whole process group. (Earlier bug: matched "/100" but the bar is "/N" — never flipped,
    # so the startup grace expired mid-run and FALSE-fired on a healthy run.)
    import signal
    STALL = int(os.environ.get("WATCHDOG_STALL_SECS", "1200"))   # 20 min of TOTAL silence
    last = [time.time()]
    killed_by_watchdog = [False]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)

    def _reader():
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            last[0] = time.time()  # any output = alive

    def _watchdog():
        while proc.poll() is None:
            if stop.wait(30):
                return
            idle = time.time() - last[0]
            if idle > STALL:
                print(f"[modal] WATCHDOG: no output for {idle:.0f}s (>{STALL}s) "
                      f"— killing the hung run", flush=True)
                killed_by_watchdog[0] = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                return

    rt = threading.Thread(target=_reader, daemon=True)
    wt = threading.Thread(target=_watchdog, daemon=True)
    rt.start()
    wt.start()
    try:
        rc = proc.wait()
    finally:
        stop.set()
        outputs.commit()  # persist any checkpoint written before a hang/kill
        hf_cache.commit()
    if killed_by_watchdog[0]:
        raise RuntimeError("run killed by no-progress watchdog (hang) — latest checkpoint "
                           "committed; relaunch with --resume")
    if rc != 0:
        raise RuntimeError(f"sdpo_train.py exited {rc}")
    print("[modal] done — adapter + checkpoints saved to volume 'sdpo-outputs' (/sdpo_out)", flush=True)


@app.local_entrypoint()
def main(
    smoke: bool = False,
    # Single H100 is the right unit for this small (30-row) job: 4xH100 both
    # OOMs (each DDP rank loads a full policy+teacher+vLLM) AND starves the GPUs
    # (dataset split 4 ways). Speed comes from a faster single-GPU step, not more
    # GPUs. Multi-GPU (e.g. "H100:4") still works via accelerate if the dataset grows.
    gpu: str = "H100",
    difficulties: str = "easy",
    languages: str = "python,cpp",
    num_generations: int = 8,
    max_completion_length: int = 8192,
    max_steps: int = 20,
    vllm_gpu_util: float = 0.25,  # 0.45 OOM'd the 80 GB H100; 0.25 is validated
    per_device_batch: int = 1,
    grad_checkpointing: bool = True,  # off => faster backward on the roomy H100
    feedback: bool = False,           # iteration 02: live per-rollout judge feedback
    lr: float = 1e-4,                 # iteration 02 recipe: try 3e-5
    reward_mode: str = "fraction",    # "binary" = fast early-exit (medium's huge cases)
    system: str = "cp_method",        # iteration 03 default system prompt (train==eval)
    save_steps: int = 20,             # checkpoint cadence -> sdpo-outputs volume
    ojb_splits: str = "ojb_splits_full.json",  # iteration 03: the full 206-pool
    resume: bool = False,             # resume from latest checkpoint in sdpo_out if present
):
    num_gpus = int(gpu.split(":")[1]) if ":" in gpu else 1
    if smoke:
        # Smoke validates the REAL iteration-03 path (full split + dense + feedback +
        # cp_method + checkpointing), just tiny. --smoke shrinks data/steps internally;
        # --save-steps 1 forces a checkpoint write so the volume-commit path is exercised.
        args = ["--smoke", "--difficulties", difficulties, "--languages", languages,
                "--reward-mode", reward_mode, "--system", system, "--save-steps", "1"]
    else:
        args = [
            "--difficulties", difficulties,
            "--languages", languages,
            "--num-generations", str(num_generations),
            "--max-completion-length", str(max_completion_length),
            "--max-steps", str(max_steps),
            "--vllm-gpu-util", str(vllm_gpu_util),
            "--per-device-batch", str(per_device_batch),
            "--lr", str(lr),
            "--reward-mode", reward_mode,
            "--system", system,
            "--save-steps", str(save_steps),
            "--output-dir", "sdpo_out",
        ]
        if not grad_checkpointing:
            args.append("--no-grad-checkpointing")
        if resume:
            args.append("--resume")
    if feedback:
        args.append("--feedback")
    print(f"[modal] gpu={gpu} num_gpus={num_gpus} feedback={feedback} system={system} "
          f"reward={reward_mode} save_steps={save_steps} splits={ojb_splits}  args={args}")
    train.with_options(gpu=gpu).remote(args, num_gpus=num_gpus, ojb_splits=ojb_splits)


# ---------------------------------------------------------------------------
# Evaluation: serve vLLM in-container, run pass@k + held-out pass@1 + GSM8K.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=2 * 60 * 60,
)
def evaluate(which: str):
    """which in {"base","sdpo"}. Serves the model, runs the eval suite, returns
    the result JSONs as a dict."""
    import json
    import os
    import subprocess
    import time
    import urllib.request

    os.chdir("/root/app")
    os.environ.setdefault("WANDB_PROJECT", APP_NAME)
    base = "google/gemma-4-E2B-it"
    serve = ["vllm", "serve", base, "--port", "8000", "--dtype", "bfloat16",
             "--max-model-len", "36864", "--gpu-memory-utilization", "0.85",
             "--max-num-seqs", "32"]
    if which == "sdpo":
        serve += ["--enable-lora", "--lora-modules", "sdpo=/root/app/sdpo_out",
                  "--max-lora-rank", "32"]
        served, tag = "sdpo", "sdpo100"
    else:
        served, tag = base, "base_modal"

    print(f"[eval:{which}] starting server: {' '.join(serve)}", flush=True)
    srv = subprocess.Popen(serve)
    ready = False
    for _ in range(180):  # up to 15 min for startup
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
            ready = True
            break
        except Exception:
            if srv.poll() is not None:
                raise RuntimeError("vLLM server exited during startup")
            time.sleep(5)
    if not ready:
        raise RuntimeError("vLLM server did not become ready")
    print(f"[eval:{which}] server ready; running evals", flush=True)

    try:
        # pass@k (headline) first, then the slower 32k held-out, then GSM8K.
        subprocess.run(["python", "sdpo_passk.py", "--served-model", served,
                        "--tag", tag, "--n", "8", "--ks", "1,2,4,8",
                        "--max-tokens", "8192", "--temperature", "0.8",
                        "--concurrency", "16", "--wandb"], check=True)
        subprocess.run(["python", "sdpo_eval_vllm.py", "--served-model", served,
                        "--tag", tag, "--max-tokens", "32768", "--wandb"], check=True)
        subprocess.run(["python", "eval_runner.py", "--dataset", "gsm8k",
                        "--sample-frac", "1.0", "--model", served,
                        "--out", f"results_gsm8k_{tag}.json"], check=True)
    finally:
        srv.terminate()

    out = {}
    for f in [f"sdpo_passk_{tag}.json", f"sdpo_eval_{tag}.json",
              f"results_gsm8k_{tag}.json"]:
        if os.path.exists(f):
            out[f] = json.load(open(f))
    print(f"[eval:{which}] done — {list(out)}", flush=True)
    return out


@app.function(
    image=image,
    gpu="H100",
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=60 * 60,
)
def passk_one(which: str, languages: str = "python,cpp",
              adapter: str = "/root/app/sdpo_out", ojb_splits: str = "ojb_splits_full.json",
              tag: str = ""):
    """pass@k only (returns as soon as it finishes — no slow held-out/gsm8k tail).

    adapter: LoRA dir to serve when which=='sdpo' (e.g. .../sdpo_out/checkpoint-20).
    ojb_splits: which split's held-out to eval on (iteration-03 -> full 206-pool's 53).
    """
    import json
    import os
    import subprocess
    import time
    import urllib.request

    os.chdir("/root/app")
    os.environ.setdefault("WANDB_PROJECT", APP_NAME)
    os.environ["OJB_SPLITS"] = ojb_splits  # held-out comes from this split
    outputs.reload()  # see checkpoints committed by the (possibly stopped) train run
    base = "google/gemma-4-E2B-it"
    serve = ["vllm", "serve", base, "--port", "8000", "--dtype", "bfloat16",
             "--max-model-len", "16384", "--gpu-memory-utilization", "0.85",
             "--max-num-seqs", "32"]
    if which == "sdpo":
        serve += ["--enable-lora", "--lora-modules", f"sdpo={adapter}",
                  "--max-lora-rank", "32"]
        served, tag = "sdpo", (tag or "sdpo")
    else:
        served, tag = base, (tag or "base_modal")

    srv = subprocess.Popen(serve)
    for _ in range(180):
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
            break
        except Exception:
            if srv.poll() is not None:
                raise RuntimeError("vLLM exited during startup")
            time.sleep(5)
    try:
        subprocess.run(["python", "sdpo_passk.py", "--served-model", served,
                        "--tag", tag, "--n", "8", "--ks", "1,2,4,8",
                        "--max-tokens", "8192", "--temperature", "0.8",
                        "--languages", languages, "--concurrency", "16",
                        "--wandb"], check=True)
    finally:
        srv.terminate()
    return json.load(open(f"sdpo_passk_{tag}.json"))


@app.local_entrypoint()
def passk_run(which: str = "sdpo", languages: str = "python", out: str = ""):
    import json
    res = passk_one.remote(which, languages)
    fname = out or f"sdpo_passk_{which}_{languages.replace(',', '_')}.json"
    with open(fname, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[modal] wrote {fname}")
    print(json.dumps(res["summary"], indent=2))


@app.local_entrypoint()
def eval_checkpoint(checkpoint: str = "checkpoint-20", languages: str = "python",
                    ojb_splits: str = "ojb_splits_full.json"):
    """Held-out pass@k for base vs ONE checkpoint, in parallel — the cheap sanity check.
      modal run src/modal_sdpo.py::eval_checkpoint --checkpoint checkpoint-20 --languages python
    """
    import json
    adapter = f"/root/app/sdpo_out/{checkpoint}"
    print(f"[modal] eval base vs {checkpoint} ({adapter}) on {ojb_splits} held-out, langs={languages}")
    base_call = passk_one.spawn("base", languages, adapter, ojb_splits, "base")
    sdpo_call = passk_one.spawn("sdpo", languages, adapter, ojb_splits, checkpoint.replace("-", ""))
    base_res, sdpo_res = base_call.get(), sdpo_call.get()
    for name, res in [("base", base_res), (checkpoint, sdpo_res)]:
        with open(f"sdpo_passk_{name.replace('-', '')}.json", "w") as f:
            json.dump(res, f, indent=2)
    # compact comparison
    print(f"\n=== held-out pass@k: base vs {checkpoint} ({languages}) ===")
    for lang in languages.split(","):
        bo = base_res["by_language"].get(lang, {}).get("overall", {})
        so = sdpo_res["by_language"].get(lang, {}).get("overall", {})
        print(f"  [{lang}] " + "  ".join(
            f"pass@{k}: {bo.get(f'pass@{k}', 0):.3f}->{so.get(f'pass@{k}', 0):.3f}"
            for k in (1, 2, 4, 8)))


@app.local_entrypoint()
def run_eval():
    """Run base + 100-step adapter eval suites in parallel on two H100s."""
    import json

    print("[modal] spawning base + sdpo eval (parallel)...")
    base_call = evaluate.spawn("base")
    sdpo_call = evaluate.spawn("sdpo")
    results = {"base": base_call.get(), "sdpo": sdpo_call.get()}
    for _which, res in results.items():
        for fname, content in res.items():
            with open(fname, "w") as f:
                json.dump(content, f, indent=2)
            print(f"  wrote {fname}")
    print("[modal] eval complete — see sdpo_passk_*.json, sdpo_eval_*.json, results_gsm8k_*.json")
