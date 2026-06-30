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
         "sdpo_feedback.py", "sdpo_passk.py", "sdpo_eval_vllm.py", "eval_runner.py",
         "gen_rollouts.py", "sdpo_critic.py", "sdpo_prompts.py", "teacher_eval.py"]
_DATA = ["ojb_splits.json", "ojb_splits_full.json", "ojbench_selected.json",
         "frontier_band.json",      # iteration-07: 35 sometimes-solvable pids (n=4 probe)
         "frontier_band_v2.json"]   # iteration-08: 10 binary-frontier pids (n=12/temp-1.0 re-probe)

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
        "anthropic==0.112.0",  # iteration-05: the LLM critic (sdpo_critic) calls Claude per failed rollout
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
# Phase-0 probe inputs (committed under reports/) so teacher_eval reads them in-container.
for _f in ["qwen3_rollouts.json", "qwen3_critiques.json"]:
    _p = REPO / "reports" / "iteration-05" / "data" / _f
    if _p.exists():
        image = image.add_local_file(str(_p), f"/root/app/{_f}")

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
        modal.Secret.from_name("anthropic"),  # -> ANTHROPIC_API_KEY (the LLM critic)
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
        # -u = unbuffered stdout/stderr. Without it, sdpo_train's output is block-buffered
        # as a Modal subprocess, so a ~15-min step emits NO flushed line and the no-progress
        # watchdog FALSE-fires on a healthy run (this killed iter-06's first fast-run step 0).
        cmd = [sys.executable, "-u", "sdpo_train.py", *args]
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
    # 40 min of TOTAL silence. A real step is ~15 min (8B/20k on H200); with `-u` it streams,
    # but generation can be silent within a step, so the threshold must clear a slow step with
    # margin. 1200s (20 min) FALSE-fired on iter-06's first fast run (a ~15-min step at the 20k
    # cap crossed it). A genuine hang still trips 40 min.
    STALL = int(os.environ.get("WATCHDOG_STALL_SECS", "2400"))
    last = [time.time()]
    killed_by_watchdog = [False]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
    model: str = "Qwen/Qwen3-8B",     # iteration-05: in-regime ~8B model
    gpu: str = "H200",                # 8B + 32k thinking-ON + colocate vLLM is H200 territory
    difficulties: str = "easy,medium",  # train split is easy+medium (no hard); "" = all
    languages: str = "python,cpp",
    num_generations: int = 8,
    max_completion_length: int = 20480,  # 32k OOM'd the H200 at the loss step; 20k fits & is > ~16k NO_CODE floor
    max_steps: int = 20,
    vllm_gpu_util: float = 0.20,       # 0.30 + 32k logits overflowed 140GiB; 0.20 frees ~14GiB headroom
    per_device_batch: int = 1,
    grad_checkpointing: bool = True,
    feedback: bool = False,           # live per-rollout judge feedback into the teacher
    critic: bool = False,             # iteration-05: replace judge feedback w/ LLM critique (implies feedback)
    critic_model: str = "",           # "" -> sdpo_critic.DEFAULT_CRITIC_MODEL (sonnet)
    distillation_weight: float = 0.1,  # hybrid: loss=(1-w)*GRPO + w*SDPO (verifier-dominant)
    teacher_kind: str = "base",       # fixed/initial teacher (iteration-05 T0)
    lr: float = 1e-4,
    # iteration-06 anti-collapse levers (defaults = sdpo_train defaults = iter-05 behavior;
    # iter-06 invokes with --beta 0.04 --lr-scheduler cosine --warmup-ratio 0.1 --lr 3e-5).
    beta: float = 0.0,                # KL anchor to base (0 = off, as iter-05)
    lr_scheduler: str = "linear",     # iter-06: cosine
    warmup_ratio: float = 0.0,        # iter-06: 0.1
    temperature: float = 1.0,         # rollout sampling temperature
    top_p: float = 0.95,              # rollout nucleus top_p (held = iter-05)
    reward_mode: str = "fraction",
    grpo_reward: str = "binary",       # iteration-05: GRPO advantage = binary AC; SDPO gating = fraction
    sdpo_threshold: float = 0.5,       # near-miss rollouts (>=50% cases) can be SDPO teachers
    system: str = "cp_method",
    save_steps: int = 5,              # eval at 5/10/15/20
    output_dir: str = "sdpo_out",     # under the sdpo-outputs volume; per-iteration subdir
                                      # (e.g. sdpo_out/iter06-fast) isolates a run from leftovers
    frontier_band: str = "",          # iteration-07: e.g. "frontier_band.json" -> train ONLY on the
                                      # 35 sometimes-solvable pids (non-flat groups -> live policy grad)
    ojb_splits: str = "ojb_splits.json",  # iteration-05 88-pool (train=easy+medium, heldout has hard)
    resume: bool = False,
):
    num_gpus = int(gpu.split(":")[1]) if ":" in gpu else 1
    common = ["--model", model, "--system", system, "--reward-mode", reward_mode,
              "--grpo-reward", grpo_reward, "--sdpo-threshold", str(sdpo_threshold),
              "--distillation-weight", str(distillation_weight), "--teacher-kind", teacher_kind,
              # iter-06 anti-collapse + sampling knobs (apply to smoke + real)
              "--beta", str(beta), "--lr-scheduler", lr_scheduler,
              "--warmup-ratio", str(warmup_ratio),
              "--temperature", str(temperature), "--top-p", str(top_p)]
    if smoke:
        # Smoke validates the REAL path (Qwen3 LoRA attach + critic fires + dense reward +
        # checkpoint write), just tiny. --save-steps 1 exercises the volume-commit path.
        args = ["--smoke", "--difficulties", difficulties, "--languages", languages,
                "--save-steps", "1", *common]
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
            "--save-steps", str(save_steps),
            "--output-dir", output_dir,
            *common,
        ]
        if not grad_checkpointing:
            args.append("--no-grad-checkpointing")
        if frontier_band:
            args += ["--frontier-band", frontier_band]  # flat /root/app path, e.g. frontier_band.json
        if resume:
            args.append("--resume")
    if critic:
        args.append("--critic")
        if critic_model:
            args += ["--critic-model", critic_model]
    elif feedback:
        args.append("--feedback")
    print(f"[modal] gpu={gpu} model={model} critic={critic} distill_w={distillation_weight} "
          f"teacher={teacher_kind} system={system} save_steps={save_steps} splits={ojb_splits}\n"
          f"        anti-collapse: lr={lr} beta={beta} sched={lr_scheduler} warmup={warmup_ratio} "
          f"temp={temperature} top_p={top_p} langs={languages} frontier_band={frontier_band or 'no'}\n"
          f"        args={args}")
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
    timeout=5 * 60 * 60,  # headroom: thinking-ON 32k mediums are slow; the 2h default
)                         # crashed the full-25 held-out eval on the hard tail.
def passk_one(which: str, languages: str = "python",
              adapter: str = "/root/app/sdpo_out", ojb_splits: str = "ojb_splits.json",
              tag: str = "", model: str = "Qwen/Qwen3-8B",
              max_model_len: int = 40960, max_tokens: int = 32768, limit: int = 0,
              ids: str = "", n: int = 8, temperature: float = 0.8):
    """pass@k only (returns as soon as it finishes — no slow held-out/gsm8k tail).

    adapter: LoRA dir to serve when which=='sdpo' (e.g. .../sdpo_out/checkpoint-20).
    ojb_splits: which split's held-out to eval on (iteration-05 -> ojb_splits.json's 25).
    Qwen3-8B is thinking-ON: max_tokens MUST be ~32k or it NO_CODEs (CLAUDE.md gotcha).
    limit: cap to first N tasks (easy-first); with langs=python, --limit 10 = the 5 easy +
      5 medium held-out problems and SKIPS the 15 hard (flat 0/8, the slow 32k-thinking tail
      that overran the 2h fn timeout). 0 = full 25-problem split.
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
    base = model
    serve = ["vllm", "serve", base, "--port", "8000", "--dtype", "bfloat16",
             "--max-model-len", str(max_model_len), "--gpu-memory-utilization", "0.85",
             "--max-num-seqs", "48", "--enforce-eager"]  # eager: dodge the kernel-4.19 gen hang
    # 48-way: held-out logs showed KV cache only 6-18% at 16 seqs on H200 — badly
    # underutilized; 48 stays well under the 0.85 budget and ~3x's wall-clock throughput.
    if which == "sdpo":
        serve += ["--enable-lora", "--lora-modules", f"sdpo={adapter}",
                  "--max-lora-rank", "32"]
        served, tag = "sdpo", (tag or "sdpo")
    else:
        served, tag = base, (tag or "base")

    srv = subprocess.Popen(serve)
    for _ in range(240):  # 32k-ctx Qwen3 load + warmup takes longer than gemma
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
            break
        except Exception:
            if srv.poll() is not None:
                raise RuntimeError("vLLM exited during startup")
            time.sleep(5)
    try:
        _cmd = ["python", "sdpo_passk.py", "--served-model", served,
                "--tag", tag, "--n", str(n), "--ks", "1,2,4,8",
                "--max-tokens", str(max_tokens), "--temperature", str(temperature),
                "--languages", languages, "--concurrency", "48", "--wandb"]
        if limit:
            _cmd += ["--limit", str(limit)]
        if ids:
            _cmd += ["--ids", ids]
        subprocess.run(_cmd, check=True)
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
def probe_solve_rate(ids: str = "", languages: str = "python", n: int = 12,
                     temperature: float = 1.0, tag: str = "probe_v2", out: str = "",
                     gpu: str = "H200", model: str = "Qwen/Qwen3-8B"):
    """iteration-08: probe the BASE model's per-problem BINARY solve rate (n samples at the
    TRAINING temperature) over a given id list -> used to build a tightened frontier band.
    solve rate p = n_ac/n; the binary-frontier is p in ~[0.3,0.7] (minimizes flat groups).
    Probe easy+medium only (hard rollouts are 0 -> flat + slow 32k tail). Writes json +
    prints per-problem solve rates."""
    import json
    res = passk_one.with_options(gpu=gpu).remote(
        "base", languages, "/root/app/sdpo_out", "ojb_splits.json", tag,
        model=model, ids=ids, n=n, temperature=temperature)
    fname = out or f"sdpo_passk_{tag}.json"
    json.dump(res, open(fname, "w"), indent=2)
    rows = sorted(res["results"], key=lambda r: r["n_ac"] / max(1, r["n"]))
    print(f"[probe] {len(rows)} problems, n={n}, temp={temperature}")
    for r in rows:
        sr = r["n_ac"] / max(1, r["n"])
        print(f"  loj-{r['id']} {r['difficulty']:<6} {r['n_ac']:>2}/{r['n']} p={sr:.2f}")
    print(f"[probe] wrote {fname}")


@app.local_entrypoint()
def eval_iterations(ids: str = "", checkpoints: str = "", n: int = 12, temperature: float = 0.8,
                    languages: str = "python", tag_prefix: str = "iters30", gpu: str = "H200",
                    model: str = "Qwen/Qwen3-8B"):
    """iteration-08 DEFINITIVE eval: base + a list of checkpoints on the SAME ids (matched
    cross-iteration pass@k), all in PARALLEL. checkpoints = comma list of volume paths, e.g.
    'iteration-05/checkpoint-20,iter06-fast/checkpoint-8'. n>k => pass@k is graded (less noisy)."""
    import json
    p = passk_one.with_options(gpu=gpu)
    calls = [(f"{tag_prefix}_base",
              p.spawn("base", languages, "/root/app/sdpo_out", "ojb_splits.json",
                      f"{tag_prefix}_base", model=model, ids=ids, n=n, temperature=temperature))]
    for ck in [c.strip() for c in checkpoints.split(",") if c.strip()]:
        tag = f"{tag_prefix}_" + ck.replace("-", "").replace("/", "_")
        calls.append((tag, p.spawn("sdpo", languages, f"/root/app/sdpo_out/{ck}", "ojb_splits.json",
                                   tag, model=model, ids=ids, n=n, temperature=temperature)))
    for tag, c in calls:
        try:
            res = c.get()
            json.dump(res, open(f"sdpo_passk_{tag}.json", "w"), indent=2)
            ov = res["summary"]["overall"]
            print(f"[eval-iters] {tag}: pass@1={ov.get('pass@1')} pass@8={ov.get('pass@8')} "
                  f"n_problems={ov.get('n_problems')}")
        except Exception as e:  # noqa: BLE001
            print(f"[eval-iters] {tag} FAILED: {e}")


@app.local_entrypoint()
def eval_checkpoint(checkpoint: str = "checkpoint-20", languages: str = "python",
                    ojb_splits: str = "ojb_splits.json", model: str = "Qwen/Qwen3-8B",
                    gpu: str = "H200", limit: int = 0, ids: str = "", tag_suffix: str = ""):
    """Held-out pass@k for base vs ONE checkpoint, in parallel — the cheap sanity check.
      modal run src/modal_sdpo.py::eval_checkpoint --checkpoint checkpoint-10 --languages python
    limit: cap to first N tasks (easy-first). --limit 10 (python) = the 10 easy+medium held-out
      problems, skipping the 15 hard (0/8 + the slow 32k tail that overran the 2h timeout).
    ids: comma list of explicit problem ids (overrides the held-out split) — used for the
      train==eval generalization curve (seen / unseen-train subsets), judged on PRIVATE cases.
    tag_suffix: appended to the base/sdpo tags so train-eval outputs (e.g. base_train) don't
      clobber the held-out files/W&B runs.
    """
    import json
    adapter = f"/root/app/sdpo_out/{checkpoint}"
    # tag becomes a filename (sdpo_passk_<tag>.json) — strip path separators or the write
    # fails (a checkpoint like "iter06-fast/checkpoint-8" otherwise yields a slash in the tag).
    base_tag = "base" + tag_suffix
    sdpo_tag = checkpoint.replace("-", "").replace("/", "_") + tag_suffix
    print(f"[modal] eval base vs {checkpoint} ({adapter}) on {ojb_splits}, "
          f"langs={languages}, limit={limit or 'full'}, ids={'custom('+str(len(ids.split(',')))+')' if ids else 'split'}")
    p = passk_one.with_options(gpu=gpu)
    base_call = p.spawn("base", languages, adapter, ojb_splits, base_tag, model=model, limit=limit, ids=ids)
    sdpo_call = p.spawn("sdpo", languages, adapter, ojb_splits, sdpo_tag, model=model, limit=limit, ids=ids)
    base_res, sdpo_res = base_call.get(), sdpo_call.get()
    for name, res in [(base_tag, base_res), (sdpo_tag, sdpo_res)]:
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


# ---------------------------------------------------------------------------
# Model-parameterized pass@k — the "opportunity graph" data (pass@1..8 by
# difficulty) for ANY base model. evaluate()/passk_one() hardcode gemma +
# adapter semantics; this serves an arbitrary base model and runs sdpo_passk
# concurrently against it. vLLM continuous-batches the (concurrency x n)
# in-flight requests => concurrent generation on the served endpoint. Qwen3-8B
# think-ON needs a HIGH token cap (8192 -> NO_CODE), so max_tokens defaults to
# 32768 and max-model-len is sized to fit the prompt + full generation.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H200",  # overridden per-run via .with_options(gpu=...)
    cpu=16.0,    # private-test judging is subprocess- + thread-parallel
    volumes=VOLUMES,
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=3 * 60 * 60,
)
def passk_model_remote(model: str, tag: str, languages: str = "python",
                       n: int = 8, ks: str = "1,2,4,8", max_tokens: int = 32768,
                       temperature: float = 0.8, system: str = "cp_method",
                       limit: int = 0, ojb_splits: str = "ojb_splits.json",
                       wandb: bool = True):
    import json
    import os
    import subprocess
    import time
    import urllib.request

    os.chdir("/root/app")
    os.environ.setdefault("WANDB_PROJECT", APP_NAME)
    os.environ["OJB_SPLITS"] = ojb_splits   # held-out comes from this split
    outputs.reload()
    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        print(f"[passk] GPU: {p.name} ({p.total_memory / 1024**3:.0f} GiB)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[passk] GPU probe failed: {e}", flush=True)

    # max-model-len must fit longest prompt + full generation; mirror gen_rollouts
    # (max_new_tokens + 8192 headroom). enforce-eager dodges the kernel gen hang.
    max_model_len = max_tokens + 8192
    serve = ["vllm", "serve", model, "--port", "8000", "--dtype", "bfloat16",
             "--max-model-len", str(max_model_len), "--gpu-memory-utilization", "0.85",
             "--max-num-seqs", "32", "--enforce-eager"]
    print(f"[passk] serving: {' '.join(serve)}", flush=True)
    srv = subprocess.Popen(serve)
    ready = False
    for _ in range(240):  # up to 20 min (first-run weight download)
        try:
            urllib.request.urlopen("http://localhost:8000/v1/models", timeout=2)
            ready = True
            break
        except Exception:  # noqa: BLE001
            if srv.poll() is not None:
                raise RuntimeError("vLLM server exited during startup")
            time.sleep(5)
    if not ready:
        raise RuntimeError("vLLM server did not become ready")
    print("[passk] server ready; running sdpo_passk (concurrent)", flush=True)

    cmd = ["python", "sdpo_passk.py", "--served-model", model, "--tag", tag,
           "--n", str(n), "--ks", ks, "--max-tokens", str(max_tokens),
           "--temperature", str(temperature), "--languages", languages,
           "--system", system, "--concurrency", "16"]
    if limit:
        cmd += ["--limit", str(limit)]
    if wandb:
        cmd += ["--wandb"]
    try:
        subprocess.run(cmd, check=True)
    finally:
        srv.terminate()

    src = f"sdpo_passk_{tag}.json"
    res = json.load(open(src))
    # durable copy on the volume so a killed client doesn't lose the result
    json.dump(res, open(f"/root/app/sdpo_out/{src}", "w"), indent=2)
    outputs.commit()
    hf_cache.commit()  # persist newly-downloaded weights for next run
    print(f"[passk] wrote {src} (also volume sdpo-outputs:/{src})", flush=True)
    return res


@app.local_entrypoint()
def passk_base(model: str = "Qwen/Qwen3-8B", tag: str = "", smoke: bool = False,
               languages: str = "python", n: int = 8, max_tokens: int = 32768,
               temperature: float = 0.8, system: str = "cp_method", gpu: str = "H200",
               ojb_splits: str = "ojb_splits.json", out: str = ""):
    """Opportunity-graph data (pass@1..8 by difficulty) for a base model on Modal.
      modal run src/modal_sdpo.py::passk_base --smoke   # 2 easy problems, n=2 — validate plumbing
      modal run src/modal_sdpo.py::passk_base           # full: Qwen3-8B python, 25 heldout, n=8, 32k
    """
    import json
    tag = tag or ("qwen3_smoke" if smoke else "qwen3_base")
    limit, nn = (2, 2) if smoke else (0, n)
    print(f"[modal] passk_base gpu={gpu} model={model} tag={tag} smoke={smoke} "
          f"langs={languages} n={nn} max_tokens={max_tokens} system={system}", flush=True)
    res = passk_model_remote.with_options(gpu=gpu).remote(
        model, tag, languages, nn, "1,2,4,8", max_tokens, temperature, system,
        limit, ojb_splits, True)
    fname = out or f"sdpo_passk_{tag}.json"
    with open(fname, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[modal] wrote {fname}")
    print(json.dumps(res["summary"], indent=2))


# ---------------------------------------------------------------------------
# Generate + judge BASE rollouts (gen_rollouts.py) on a fast GPU — the Phase-0
# probe set. H200 batches thinking-ON generation far better than the GB10. Writes
# per-rollout to the sdpo-outputs volume (durable + resumable; committed every 60s).
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H200",  # overridden per-run via .with_options(gpu=...)
    cpu=16.0,    # dense judging is subprocess- + thread-parallel
    volumes=VOLUMES,
    timeout=4 * 60 * 60,  # base gen + judge; ungated model, no secrets needed
)
def gen_rollouts_remote(args: list[str], ojb_splits: str = "ojb_splits.json"):
    import json
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time

    os.chdir("/root/app")
    os.environ["OJB_SPLITS"] = ojb_splits   # read at import by sdpo_ojbench
    outputs.reload()                         # see any prior partial for --resume

    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        print(f"[modal] GPU: {p.name} ({p.total_memory / 1024**3:.0f} GiB)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[modal] GPU probe failed: {e}", flush=True)

    noi = "/root/app/ojbench_data/NOI"
    n = len(os.listdir(noi)) if os.path.isdir(noi) else 0
    print(f"[modal] ojbench test-case dirs visible: {n}", flush=True)
    assert n > 0, f"{noi} empty — populate the ojbench-data volume first"

    cmd = [sys.executable, "gen_rollouts.py", *args]
    print("[modal] running:", " ".join(cmd), flush=True)

    # Periodic commit so per-rollout writes to sdpo_out become durable DURING the run
    # (a dead run still leaves its rollouts on the volume to resume from). + no-progress
    # watchdog (a silent generation/judge hang otherwise burns the GPU for hours).
    stop = threading.Event()

    def _committer():
        while not stop.wait(60):
            try:
                outputs.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[modal] periodic commit failed: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    STALL = int(os.environ.get("WATCHDOG_STALL_SECS", "1800"))  # long thinking-ON rollouts
    last = [time.time()]
    killed = [False]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)

    def _reader():
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            last[0] = time.time()

    def _watchdog():
        while proc.poll() is None:
            if stop.wait(30):
                return
            if time.time() - last[0] > STALL:
                print(f"[modal] WATCHDOG: no output for >{STALL}s — killing hung gen", flush=True)
                killed[0] = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                return

    threading.Thread(target=_reader, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        rc = proc.wait()
    finally:
        stop.set()
        outputs.commit()   # persist rollouts written before any hang/kill
        hf_cache.commit()  # persist newly-downloaded weights for next run
    if killed[0]:
        raise RuntimeError("gen killed by no-progress watchdog — partial results committed; "
                           "rerun with --resume")
    if rc != 0:
        raise RuntimeError(f"gen_rollouts.py exited {rc}")
    out_path = args[args.index("--out") + 1] if "--out" in args else "rollouts.json"
    return json.load(open(out_path))


@app.local_entrypoint()
def gen(smoke: bool = False, gpu: str = "H200", model: str = "Qwen/Qwen3-8B",
        max_new_tokens: int = 32768, gpu_util: float = 0.85, resume: bool = True,
        concurrent: int = 12, ojb_splits: str = "ojb_splits.json", out: str = ""):
    """Generate + judge base Qwen3-8B rollouts on Modal (Phase-0 probe set).
      modal run src/modal_sdpo.py::gen --smoke   # 1 easy rollout — validate image+data+vllm+judge
      modal run src/modal_sdpo.py::gen           # full spread: easy/medium/hard, ≥3 each
    """
    import json
    from collections import Counter

    if smoke:
        specs = [["--spec", "easy:2314:1"]]
        max_new_tokens = 8192
        out = out or "sdpo_out/qwen3_smoke.json"
    else:
        specs = [["--spec", "easy:2314,2317,2420:1"],
                 ["--spec", "medium:2086,2129,2130:1"],
                 ["--spec", "hard:2083,2131,2133:2"]]
        out = out or "sdpo_out/qwen3_rollouts.json"
    args = ["--model", model, "--engine", "vllm", "--gpu-util", str(gpu_util),
            "--max-new-tokens", str(max_new_tokens), "--concurrent", str(concurrent), "--out", out]
    for s in specs:
        args += s
    if resume:
        args.append("--resume")
    print(f"[modal] gpu={gpu} smoke={smoke} args={args}", flush=True)
    res = gen_rollouts_remote.with_options(gpu=gpu).remote(args, ojb_splits=ojb_splits)
    r = res.get("results", [])
    print(f"[modal] {len(r)} rollouts · verdicts={dict(Counter(x['verdict'] for x in r))} "
          f"· failures={sum(1 for x in r if x['verdict'] != 'AC')}")
    local = out.split("/")[-1]
    with open(local, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[modal] saved -> {local} (also on volume sdpo-outputs:/{out.split('/',1)[-1]})")


# ---------------------------------------------------------------------------
# Phase-0 teacher eval (teacher_eval.py): solve-rate + per-token A_t for the teacher
# conditioned on the critic's feedback, on the committed probe set. H200; per-failure
# durable writes to the sdpo-outputs volume; committer + no-progress watchdog.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H200",
    cpu=16.0,
    volumes=VOLUMES,
    timeout=4 * 60 * 60,
)
def teacher_eval_remote(args: list[str], ojb_splits: str = "ojb_splits.json"):
    import json
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time

    os.chdir("/root/app")
    os.environ["OJB_SPLITS"] = ojb_splits
    outputs.reload()

    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        print(f"[modal] GPU: {p.name} ({p.total_memory / 1024**3:.0f} GiB)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[modal] GPU probe failed: {e}", flush=True)

    noi = "/root/app/ojbench_data/NOI"
    n = len(os.listdir(noi)) if os.path.isdir(noi) else 0
    print(f"[modal] ojbench test-case dirs visible: {n}", flush=True)
    assert n > 0, f"{noi} empty — populate the ojbench-data volume first"

    cmd = [sys.executable, "teacher_eval.py", *args]
    print("[modal] running:", " ".join(cmd), flush=True)

    stop = threading.Event()

    def _committer():
        while not stop.wait(60):
            try:
                outputs.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[modal] periodic commit failed: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    STALL = int(os.environ.get("WATCHDOG_STALL_SECS", "1800"))
    last = [time.time()]
    killed = [False]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)

    def _reader():
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            last[0] = time.time()

    def _watchdog():
        while proc.poll() is None:
            if stop.wait(30):
                return
            if time.time() - last[0] > STALL:
                print(f"[modal] WATCHDOG: no output for >{STALL}s — killing", flush=True)
                killed[0] = True
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                return

    threading.Thread(target=_reader, daemon=True).start()
    threading.Thread(target=_watchdog, daemon=True).start()
    try:
        rc = proc.wait()
    finally:
        stop.set()
        outputs.commit()
        hf_cache.commit()
    if killed[0]:
        raise RuntimeError("teacher_eval killed by watchdog — partial committed; rerun --resume")
    if rc != 0:
        raise RuntimeError(f"teacher_eval.py exited {rc}")
    out_path = args[args.index("--out") + 1] if "--out" in args else "teacher_eval.json"
    return json.load(open(out_path))


@app.local_entrypoint()
def teacher_eval(smoke: bool = False, gpu: str = "H200", model: str = "Qwen/Qwen3-8B",
                 critic_set: str = "sonnet_verbose", samples: int = 8,
                 max_new_tokens: int = 32768, gpu_util: float = 0.85, resume: bool = True,
                 out: str = ""):
    """Phase-0 teacher eval on Modal: solve-rate + A_t for teacher = x + critique.
      modal run src/modal_sdpo.py::teacher_eval --smoke   # 1 failure, K=2 — validate wiring
      modal run src/modal_sdpo.py::teacher_eval           # all 9 failures, K=8
    """
    import json

    out = out or ("sdpo_out/teacher_eval_smoke.json" if smoke else "sdpo_out/teacher_eval.json")
    args = ["--model", model, "--rollouts", "/root/app/qwen3_rollouts.json",
            "--critiques", "/root/app/qwen3_critiques.json", "--critic-set", critic_set,
            "--samples", str(2 if smoke else samples), "--max-new-tokens", str(max_new_tokens),
            "--gpu-util", str(gpu_util), "--out", out]
    if smoke:
        args += ["--limit", "1"]
    if resume:
        args.append("--resume")
    print(f"[modal] gpu={gpu} smoke={smoke} critic_set={critic_set} args={args}", flush=True)
    res = teacher_eval_remote.with_options(gpu=gpu).remote(args)
    r = res.get("results", [])
    print(f"\n[modal] {len(r)} failures evaluated (critic={critic_set}):")
    for x in r:
        a = x["advantage"]
        print(f"  {x['difficulty']} loj-{x['id']} s{x['sample']} [{x['base_verdict']}]: "
              f"solve-rate {x['solve_rate']} | A_t mean|.|={a['mean_abs']} "
              f"frac_neg={a['frac_neg']} code/reas={a['code_region_abs']}/{a['reasoning_abs']}")
    local = out.split("/")[-1]
    with open(local, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[modal] saved -> {local} (volume sdpo-outputs:/{out.split('/',1)[-1]})")


# ---------------------------------------------------------------------------
# Critic sanity check (CPU, no GPU): confirm the LLM critic actually calls Claude
# in-container — the silent failure mode is `import anthropic` / missing secret making
# critique() fall back to deterministic feedback WITHOUT crashing (critic silently off).
# ---------------------------------------------------------------------------
@app.function(image=image, secrets=[modal.Secret.from_name("anthropic")], timeout=5 * 60)
def critic_check_remote():
    import os
    import sys
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")  # in-process import (no subprocess) needs the flat dir on path
    from sdpo_critic import critique
    fb = "Verdict: WA.\nExpected output:\n7\nYour output:\n-1"
    out = critique("Read two integers a, b from stdin and print a+b.",
                   "a,b=map(int,input().split())\nprint(a-b)", "WA", fb, "python")
    fired = out != fb  # a real critique came back (not the deterministic fallback)
    print(f"[critic_check] ANTHROPIC_API_KEY set={bool(os.environ.get('ANTHROPIC_API_KEY'))} "
          f"critic_fired={fired} ({len(out)} chars)", flush=True)
    print("--- critique head ---\n" + out[:500], flush=True)
    return {"fired": fired, "chars": len(out)}


@app.local_entrypoint()
def critic_check():
    """modal run src/modal_sdpo.py::critic_check — confirms the critic calls Claude on Modal."""
    r = critic_check_remote.remote()
    print(f"[modal] critic_check: {r}  -> {'OK (critic live)' if r['fired'] else 'FALLBACK (critic OFF!)'}")


# ---------------------------------------------------------------------------
# Pre-warm model weights into the hf-cache volume on a CPU-only container.
# Downloading is GPU-free; doing it here means the first H200 train/eval run
# starts with Qwen3-8B already on disk instead of paying ~16 GB of download at
# H200 rates. The hf-cache volume persists, so this is a one-time cost per model.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},   # only the weights cache; no GPU
    secrets=[modal.Secret.from_name("huggingface")],  # harmless; Qwen3-8B is ungated
    timeout=60 * 60,
)
def prewarm_weights(model: str = "Qwen/Qwen3-8B"):
    """Download `model`'s weights into the hf-cache volume (no GPU). Idempotent:
    snapshot_download skips files already present, so re-running is cheap."""
    import os
    from huggingface_hub import snapshot_download

    hf_cache.reload()
    print(f"[prewarm] downloading {model} into hf-cache volume ...", flush=True)
    path = snapshot_download(
        repo_id=model,
        # bf16 safetensors weights + configs/tokenizer; skip duplicate .bin/.pth.
        ignore_patterns=["*.pt", "*.pth", "*.bin", "original/*", "*.gguf"],
    )
    hf_cache.commit()
    files = sorted(os.listdir(path))
    total = sum(os.path.getsize(os.path.join(path, f))
                for f in files if os.path.isfile(os.path.join(path, f)))
    print(f"[prewarm] done: {path}\n[prewarm] {len(files)} files, "
          f"{total / 1024**3:.2f} GiB resolved in snapshot", flush=True)
    print(f"[prewarm] files: {files}", flush=True)
    return {"model": model, "path": path, "num_files": len(files),
            "snapshot_gib": round(total / 1024**3, 2)}


@app.local_entrypoint()
def prewarm(model: str = "Qwen/Qwen3-8B"):
    """modal run src/modal_sdpo.py::prewarm --model Qwen/Qwen3-8B"""
    res = prewarm_weights.remote(model)
    print(f"[modal] prewarm complete: {res}")
