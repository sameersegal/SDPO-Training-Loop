"""Phase G0-build: Modal image + app skeleton for an EXACT replication of the SDPO
paper (arXiv 2601.20802) on their verl fork (https://github.com/lasgroup/SDPO).

This is a NEW file (does NOT reuse sdpo_train.py). It builds a Modal image that
git-clones the lasgroup/SDPO fork AT A PINNED COMMIT inside the image build and
`pip install --no-deps -e .`'s it on top of verl's OFFICIAL x86 base image.

WHY the verl-official base (not their Dockerfile):
  - Their CANONICAL env is GH200/aarch64 (Dockerfile.gh200 -> nvcr.io/nvidia/vllm
    25.12.post1 ARM64 + requirements-full.txt full of /opt/transfer/*aarch64.whl
    wheels). None of that runs on Modal's x86 H200/H100.
  - Their x86 path is README "Option 2 Local": torch 2.5.1/cu124 + requirements.txt
    + `pip install -e .[vllm]` + flash-attn compiled from source. Compiling flash-attn
    at image-build is slow/fragile.
  - Their own docker/README.md recommends verl's prebuilt images and `pip install
    --no-deps -e .` on top (which is EXACTLY what their Dockerfile.gh200 does with
    `--no-deps`). We follow that: a verlai/verl image ships torch+vllm+flash-attn+
    ray+transformers prebuilt; the fork's editable `verl/` package overrides the
    baked-in one so the SDPO `loss_mode: sdpo` code path is theirs.

PINS:
  fork commit : 7c457fc1b1f636ae794eb0362ba37d4743b06fbc  (HEAD of lasgroup/SDPO)
  base image  : verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2
                (vllm 0.10.0 is inside the fork's setup.py [vllm] range 0.8.5..0.12.0;
                 ships torch 2.7 + flash-attn + ray 2.x prebuilt).
  transformers: pin bumped to 4.57.1 to MATCH the fork's requirements.txt exactly
                (base ships 4.55.4).

SDPO ENTRY POINT (verified in the fork, NOT a standalone trainer class):
  SDPO is a POLICY-LOSS MODE inside verl's PPO trainer, selected by
  `actor_rollout_ref.actor.policy_loss.loss_mode=sdpo` (see verl/trainer/config/sdpo.yaml
  and verl/trainer/main_ppo.py:131). Training launches via `python -m verl.trainer.main_ppo`.

BUDGET: this phase is CPU-only cheap checks (~$2 cap). Do NOT launch GPU training here.

RUN (cheap CPU checks):
  modal run src/modal_verl_repl.py::smoke_imports     # imports verl + sdpo path, ray up/down
  modal run src/modal_verl_repl.py::shell_probe       # pip freeze of key packages
"""
import modal

APP_NAME = "sdpo-verl-repl"

# Pinned HEAD of lasgroup/SDPO (verl 0.6.x experimental fork).
FORK_REPO = "https://github.com/lasgroup/SDPO.git"
FORK_COMMIT = "7c457fc1b1f636ae794eb0362ba37d4743b06fbc"

# verl's official prebuilt x86 image: torch+vllm+flash-attn+ray+transformers baked in.
# vllm 0.10.0 sits inside the fork setup.py [vllm] range (0.8.5..0.12.0).
VERL_BASE = "verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2"

# Build the image: clone the fork at the pinned commit INSIDE the build (no local
# file shipping of their code), pin transformers to the fork's exact value, then
# editable-install the fork with --no-deps (deps already satisfied by the base image;
# this is what their own Dockerfile.gh200 does).
image = (
    modal.Image.from_registry(VERL_BASE)
    .apt_install("git")
    .run_commands(
        f"git clone {FORK_REPO} /opt/SDPO",
        f"cd /opt/SDPO && git checkout {FORK_COMMIT}",
        # match the fork's requirements.txt pins for verl-critical libs the base ships
        # too old: transformers (base 4.55.4), tensordict (base 0.6.2 — VIOLATES the
        # fork setup.py bound >=0.8,<=0.10,!=0.9), ray (base 2.47.1 vs their pin 2.53.0).
        "pip install --no-cache-dir 'transformers==4.57.1' 'tensordict==0.10.0' "
        "'ray[default]==2.53.0'",
        # runtime deps the fork's run scripts need that aren't guaranteed in the base:
        # hydra (config), codetiming/pylatexenc/word2number (verl reward utils), wandb,
        # sortedcontainers (in the code-reward sandbox's ALLOWLIST of modules a solution
        # may import in-process — reward_score/feedback/code.py:150; the judge is pure
        # Python multiprocessing + RLIMIT_AS, no sandbox binaries).
        "pip install --no-cache-dir hydra-core==1.3.2 codetiming==1.4.0 "
        "pylatexenc==2.10 word2number==1.1 'wandb==0.23.1' sortedcontainers "
        # fastapi/uvicorn at THEIR requirements.txt pins: the base ships fastapi 0.88
        # (pydantic-v1 era) but wandb pulls pydantic 2.x -> vllm's rollout import of
        # fastapi crashed: "cannot import name 'Undefined' from pydantic.fields"
        # (G1 attempt-3). fastapi 0.127.0 is pydantic-2 native + vllm-0.10 compatible.
        "'fastapi==0.127.0' 'uvicorn==0.40.0'",
        # their run_sdpo.sh setup_cmds pip line VERBATIM: the custom reward module
        # (reward_score/feedback/__init__.py) unconditionally imports math.py ->
        # `from math_verify import ...` (G1 attempt-1 crashed on this exact import).
        "pip install --no-cache-dir word2number latex2sympy2 'math-verify[antlr4_9_3]==0.8.0'",
        # editable install of the fork WITHOUT deps -> its verl/ overrides the baked one.
        "cd /opt/SDPO && pip install --no-deps -e .",
    )
    .env(
        {
            "VLLM_USE_V1": "1",                 # their verl_training.sh sets this
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
            "FORK_COMMIT": FORK_COMMIT,
        }
    )
)

app = modal.App(APP_NAME)

# Reuse our volume/secret patterns from src/modal_sdpo.py:
#   - hf-cache: model weights cache (shared with the existing app)
#   - verl-repl: NEW volume for this replication's checkpoints/data
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
verl_repl = modal.Volume.from_name("verl-repl", create_if_missing=True)

VOLUMES = {
    "/root/.cache/huggingface": hf_cache,
    "/root/verl-repl": verl_repl,
}
SECRETS = [
    modal.Secret.from_name("huggingface"),  # -> HF_TOKEN
    modal.Secret.from_name("wandb"),        # -> WANDB_API_KEY
]


@app.function(image=image, cpu=2.0, volumes=VOLUMES, secrets=SECRETS, timeout=15 * 60)
def smoke_imports():
    """CPU-only: import verl + the fork's SDPO code path, spin ray up/down, print versions.

    Verifies the image actually resolves the fork's verl (editable) and that the SDPO
    loss-mode plumbing imports. SDPO is NOT a separate trainer class — it's a policy-loss
    mode inside verl's PPO trainer (verl.trainer.main_ppo + verl.trainer.ppo.core_algos),
    selected via config sdpo.yaml (loss_mode: sdpo). So we import those modules + assert
    the fork's sdpo.yaml config is present.
    """
    import os

    print(f"[smoke] FORK_COMMIT (pinned) = {os.environ.get('FORK_COMMIT')}", flush=True)

    import torch
    print(f"[smoke] torch = {torch.__version__}", flush=True)

    import verl
    print(f"[smoke] verl module file = {verl.__file__}", flush=True)
    # the fork installs editable from /opt/SDPO -> its verl/ should be on the path
    from importlib.metadata import version as _v
    try:
        print(f"[smoke] verl (metadata) version = {_v('verl')}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] verl metadata lookup failed: {e}", flush=True)
    # read the fork's version file directly (the source of truth for the pin)
    try:
        with open("/opt/SDPO/verl/version/version") as f:
            print(f"[smoke] fork verl/version/version = {f.read().strip()}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] version file read failed: {e}", flush=True)

    # confirm verl resolves to the EDITABLE fork checkout, not the baked-in base copy
    assert "/opt/SDPO" in os.path.realpath(verl.__file__), (
        f"verl resolved to {verl.__file__}, NOT the /opt/SDPO fork checkout")
    print("[smoke] OK: verl resolves to the /opt/SDPO editable fork", flush=True)

    # SDPO code path: PPO trainer entrypoint + core algos (where loss modes live) + config
    import importlib
    for mod in ("verl.trainer.main_ppo", "verl.trainer.ppo.core_algos",
                "verl.trainer.ppo.ray_trainer"):
        importlib.import_module(mod)
        print(f"[smoke] imported {mod}", flush=True)
    cfg = "/opt/SDPO/verl/trainer/config/sdpo.yaml"
    assert os.path.exists(cfg), f"missing SDPO config {cfg}"
    print(f"[smoke] OK: SDPO config present at {cfg}", flush=True)

    # ray local init + shutdown (the training stack runs on ray)
    import ray
    print(f"[smoke] ray = {ray.__version__}", flush=True)
    ray.init(num_cpus=1, include_dashboard=False, ignore_reinit_error=True,
             logging_level="ERROR")
    print(f"[smoke] ray.init OK — nodes={len(ray.nodes())}", flush=True)
    ray.shutdown()
    print("[smoke] ray.shutdown OK", flush=True)

    import tensordict
    import transformers
    import vllm
    print(f"[smoke] transformers={transformers.__version__} vllm={vllm.__version__} "
          f"tensordict={tensordict.__version__}", flush=True)
    print("[smoke] ALL CHECKS PASSED", flush=True)
    return {
        "fork_commit": os.environ.get("FORK_COMMIT"),
        "torch": torch.__version__,
        "verl_file": verl.__file__,
        "ray": ray.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm.__version__,
    }


@app.function(image=image, cpu=1.0, timeout=10 * 60)
def shell_probe():
    """CPU-only: return `pip freeze` for the ~20 key packages so we can pin the env."""
    import subprocess

    freeze = subprocess.run(["pip", "freeze"], capture_output=True, text=True).stdout
    keys = ["torch", "torchvision", "torchaudio", "vllm", "flash", "flashinfer",
            "ray", "transformers", "tensordict", "peft", "accelerate", "datasets",
            "hydra", "numpy", "sglang", "xformers", "triton", "verl", "wandb",
            "deepspeed", "megatron", "transformer-engine", "codetiming"]
    lines = []
    for ln in freeze.splitlines():
        low = ln.lower()
        if any(k in low for k in keys):
            lines.append(ln)
    out = "\n".join(sorted(set(lines)))
    print(out, flush=True)
    return out


@app.local_entrypoint()
def probe():
    """Run shell_probe and save the pins to runs/replication-02/env_pins.txt."""
    import os

    out = shell_probe.remote()
    dest_dir = os.path.join(os.path.dirname(__file__), "..", "runs", "replication-02")
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.abspath(os.path.join(dest_dir, "env_pins.txt"))
    with open(dest, "w") as f:
        f.write(f"# SDPO verl-repl env pins — base {VERL_BASE}\n")
        f.write(f"# fork commit {FORK_COMMIT}\n\n")
        f.write(out + "\n")
    print(f"[probe] wrote {dest}")


# ---------------------------------------------------------------------------
# Training: their exact entry (`python -m verl.trainer.main_ppo --config-name sdpo`)
# from /opt/SDPO, with the CSCS-cluster red-flag overrides baked in:
#   - user.yaml hardcodes /users/${USER}/SDPO + /capstor/scratch/cscs/... paths and
#     reads ${oc.env:USER}/${oc.env:TASK}/${oc.env:EXPERIMENT} -> we export those env
#     vars AND override every leaf that points at the cluster (data files, ckpt dir,
#     custom_reward_function.path, wandb project/experiment names, critic model path).
#   - their verl_training.sh exports WANDB_ENTITY=sample-efficient-rlvr (their team)
#     -> we do NOT set it (defaults to our wandb account; project sdpo-verl-repl).
#   - VLLM_USE_V1=1 is set in the image env (their script sets it too).
#   - legacy FSDP worker: trainer.use_legacy_worker_impl stays "auto" (default);
#     SDPO only rejects "disable" (main_ppo.py:136) — do not touch it.
# `extra_overrides` lets G2/G3 reuse this function unchanged.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100",   # overridden per-run via .with_options(gpu=...)
    cpu=16.0,     # code-reward judging is multiprocessing CPU work
    volumes=VOLUMES,
    secrets=SECRETS,
    timeout=12 * 60 * 60,  # 15-step 8B run + serial vals overran the 6h preflight sizing (G3 died at step 12)
)
def train(run_name: str, model: str = "Qwen/Qwen3-0.6B", num_gpus: int = 1,
          task: str = "lcb_v6", extra_overrides: list[str] = None,
          watchdog_stall_secs: int = 2400):
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time

    os.chdir("/opt/SDPO")
    verl_repl.reload()  # see data/ckpts committed by other runs

    # env expected by their hydra config tree (user.yaml oc.env lookups) + wandb
    os.environ["USER"] = os.environ.get("USER") or "root"
    os.environ["TASK"] = task
    os.environ["EXPERIMENT"] = run_name
    os.environ.setdefault("WANDB_PROJECT", APP_NAME)
    os.environ.pop("WANDB_ENTITY", None)  # theirs is team-hardcoded in verl_training.sh
    os.environ["PYTHONPATH"] = "/opt/SDPO:" + os.environ.get("PYTHONPATH", "")

    try:
        import torch
        p = torch.cuda.get_device_properties(0)
        print(f"[train] GPU: {p.name} x{torch.cuda.device_count()} "
              f"({p.total_memory / 1024**3:.0f} GiB each)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[train] GPU probe failed: {e}", flush=True)

    data_dir = f"/root/verl-repl/data/{task}"
    for f in ("train.parquet", "test.parquet"):
        assert os.path.exists(f"{data_dir}/{f}"), f"missing {data_dir}/{f} — volume put it first"
    ckpt_dir = f"/root/verl-repl/ckpts/{run_name}"
    os.makedirs(ckpt_dir, exist_ok=True)

    overrides = [
        # data -> the verl-repl volume parquets (user.yaml points at /users/.../SDPO)
        f"data.train_files=['{data_dir}/train.parquet']",
        f"data.val_files=['{data_dir}/test.parquet']",
        # model (user.yaml hardcodes Qwen3-8B for actor AND critic; critic is disabled
        # by adv_estimator=grpo but keep its path consistent anyway)
        f"actor_rollout_ref.model.path={model}",
        f"critic.model.path={model}",
        # cluster-path + naming red flags
        f"trainer.default_local_dir={ckpt_dir}",
        f"trainer.project_name={APP_NAME}",
        f"trainer.group_name={run_name}",
        f"trainer.experiment_name={run_name}",
        "custom_reward_function.path=/opt/SDPO/verl/utils/reward_score/feedback/__init__.py",
        # topology
        f"trainer.n_gpus_per_node={num_gpus}",
        "trainer.nnodes=1",
        *(extra_overrides or []),
    ]
    cmd = [sys.executable, "-u", "-m", "verl.trainer.main_ppo",
           "--config-name", "sdpo", *overrides]
    print("[train] running:", " ".join(cmd), flush=True)

    # periodic volume commit (durable checkpoints DURING the run) + no-progress watchdog
    stop = threading.Event()

    def _committer():
        while not stop.wait(120):
            try:
                verl_repl.commit()
            except Exception as e:  # noqa: BLE001
                print(f"[train] periodic commit failed: {e}", flush=True)

    threading.Thread(target=_committer, daemon=True).start()

    STALL = int(os.environ.get("WATCHDOG_STALL_SECS", str(watchdog_stall_secs)))
    last = [time.time()]
    killed = [False]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
            idle = time.time() - last[0]
            if idle > STALL:
                print(f"[train] WATCHDOG: no output for {idle:.0f}s (>{STALL}s) — killing",
                      flush=True)
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
        verl_repl.commit()
        hf_cache.commit()
    if killed[0]:
        raise RuntimeError("run killed by no-progress watchdog (hang) — checkpoints committed")
    if rc != 0:
        raise RuntimeError(f"main_ppo exited {rc}")
    print(f"[train] done — checkpoints under verl-repl:/ckpts/{run_name}", flush=True)


@app.local_entrypoint()
def g1_smoke(run_name: str = "g1-smoke", gpu: str = "H100",
             model: str = "Qwen/Qwen3-0.6B"):
    """Phase G1: 0.6B / 1-GPU SDPO integration smoke (2 steps, tiny batch).

    SDPO knobs stay at their LCBv6 values (experiments/rich_feedback/run_sdpo.sh):
    alpha=1.0, topk=20, teacher ema 0.01, dont_reprompt_on_self_success=True,
    rollout_is=token, lr=1e-6, warmup 0. threshold 0.5 / remove_thinking /
    add_tail / max_reprompt_len 10240 right-trunc are already the actor.yaml
    defaults (verified) — not overridden. Smoke-sized: batch 4, n=4, mini=1,
    2 steps, resp len 2048, val n=2 at step 2 (test_freq=2; save_freq=2 for
    the checkpoint check).
    """
    overrides = [
        # 1-GPU topology: their rollout.yaml default is TP=2 (4xGH200 node) ->
        # "rollout world_size: 1 is not divisible by infer_world_size: 2" (attempt-2 crash)
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        # smoke sizing
        "data.train_batch_size=4",
        "data.max_response_length=2048",
        "actor_rollout_ref.rollout.n=4",
        "actor_rollout_ref.rollout.val_kwargs.n=2",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        "trainer.total_training_steps=2",
        "trainer.test_freq=2",
        "trainer.save_freq=2",
        # their LCBv6 SDPO hyperparameters (run_sdpo.sh)
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.self_distillation.distillation_topk=20",
        "actor_rollout_ref.actor.self_distillation.alpha=1.0",
        "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01",
        "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True",
        "algorithm.rollout_correction.rollout_is=token",
    ]
    print(f"[g1] launching {run_name} on {gpu} model={model}", flush=True)
    train.with_options(gpu=gpu).remote(run_name, model=model, num_gpus=1,
                                       extra_overrides=overrides)


@app.local_entrypoint()
def g2_preflight(run_name: str = "g2-preflight", gpu: str = "H200:4",
                 model: str = "Qwen/Qwen3-8B", total_steps: int = 3,
                 test_freq: int = 3, save_freq: int = 2,
                 watchdog_stall_secs: int = 7200):
    # watchdog default RAISED to 7200s after the first G2 run was FALSE-KILLED at 2401s:
    # the val loop's serial judging (524 samples, 40+ min) emits NO console output —
    # the iter-06 false-fire mode, now on the silent val phase.
    """Phase G2: Qwen3-8B preflight at the paper's REAL LCBv6 batch geometry on 4 GPUs.

    Their geometry VERBATIM (experiments/rich_feedback/run_sdpo.sh + sdpo.yaml/user.yaml
    defaults): batch 32, n=8, mini=1, max_prompt 2048, max_response 8192, TP=2 (their
    rollout.yaml default — NOT overridden here, unlike G1), lr 1e-6, warmup 0, topk 20,
    alpha 1.0, ema 0.01, dont_reprompt True, rollout_is token. Preflight-sized ONLY in
    run length: 3 steps, save at 2 (proves checkpoint), val ONCE at step 3 (test_freq=3,
    val n=4 = their FINAL sweep value) to measure 8B val wall-clock.

    RESUME CHECK (after this completes): relaunch with --total-steps 4 — trainer.resume_mode
    is 'auto' (ppo_trainer.yaml default): reads latest_checkpointed_iteration.txt in
    default_local_dir -> must load global_step_2 and run steps 3-4 only.
    """
    num_gpus = int(gpu.split(":")[1]) if ":" in gpu else 1
    overrides = [
        # their LCBv6 geometry (explicit even where it matches their defaults)
        "data.train_batch_size=32",
        "data.max_prompt_length=2048",
        "data.max_response_length=8192",
        "actor_rollout_ref.rollout.n=8",
        "actor_rollout_ref.rollout.val_kwargs.n=4",
        "actor_rollout_ref.actor.ppo_mini_batch_size=1",
        # preflight run length + cadence
        f"trainer.total_training_steps={total_steps}",
        f"trainer.test_freq={test_freq}",
        f"trainer.save_freq={save_freq}",
        # their LCBv6 SDPO hyperparameters (run_sdpo.sh)
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.self_distillation.distillation_topk=20",
        "actor_rollout_ref.actor.self_distillation.alpha=1.0",
        "actor_rollout_ref.actor.self_distillation.teacher_update_rate=0.01",
        "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True",
        "algorithm.rollout_correction.rollout_is=token",
    ]
    print(f"[g2] launching {run_name} on {gpu} model={model} steps={total_steps}", flush=True)
    train.with_options(gpu=gpu, cpu=32.0).remote(run_name, model=model, num_gpus=num_gpus,
                                                 extra_overrides=overrides,
                                                 watchdog_stall_secs=watchdog_stall_secs)
