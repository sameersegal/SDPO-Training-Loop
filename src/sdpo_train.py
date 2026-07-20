#!/usr/bin/env python3
"""SDPO training for Qwen3-8B on the OJBench frontier.

(Iterations 01-04 trained google/gemma-4-E2B-it; we moved to the paper's in-regime
~8B scale at iteration-05 and removed the gemma-specific code paths — see git history.)

Uses TRL's experimental SDPOTrainer with our OJBench judge as the verifier
reward. Core SDPO signal = successful rollouts in each group distilled into the
failing ones (use_successful_as_teacher), the paper's implicit-feedback mode.

Run a tiny smoke test first:
  python sdpo_train.py --smoke
Full run:
  python sdpo_train.py --max-steps 60
"""
import argparse
import os

from _paths import load_env

load_env()  # WANDB_API_KEY etc. from repo-root .env (no-op in Modal; uses Secrets)
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from peft import LoraConfig  # noqa: E402
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer  # noqa: E402

from sdpo_ojbench import build_dataset, make_reward_func  # noqa: E402
from sdpo_feedback import FeedbackSDPOTrainer, FeedbackBus, make_feedback_reward_func  # noqa: E402
from sdpo_reprompt_guard import install_reprompt_guard  # noqa: E402


def build_parser():
    """Construct the CLI argument parser (extracted so tests can assert defaults)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--smoke", action="store_true", help="tiny config to debug integration")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    # --- anti-collapse regularization (iteration-06) ---
    ap.add_argument("--beta", type=float, default=0.0,
                    help="KL anchor to the frozen base/ref model (GRPO/SDPO beta). "
                         "0.0=off (iteration-05); >0 (e.g. 0.04) penalizes drift from base "
                         "to counter diversity loss / mode collapse.")
    ap.add_argument("--lr-scheduler", default="linear",
                    help="HF lr_scheduler_type: linear|cosine|constant|"
                         "constant_with_warmup|cosine_with_restarts")
    ap.add_argument("--warmup-ratio", type=float, default=0.0,
                    help="fraction of total steps spent warming the LR up from 0 "
                         "(e.g. 0.1). Avoids slamming a hot LR cold at step 0.")
    # --- sampling knobs (exposed for transparency; iteration-05 effective values
    #     were temperature=1.0, top_p=0.95 hardcoded). top_p<1 truncates the tail
    #     (diversity lever); temperature is the primary exploration knob. ---
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="rollout sampling temperature (exploration; higher = more diverse)")
    ap.add_argument("--top-p", type=float, default=0.95,
                    help="rollout nucleus sampling top_p (1.0 = no truncation)")
    # --- observability: persist every rollout to JSONL for offline review (P0-1,
    #     docs/design/OBSERVABILITY.md). On by default; the iter-05 gap was saving none. ---
    ap.add_argument("--no-rollout-log", dest="rollout_log", action="store_false",
                    help="disable per-rollout JSONL capture (default: ON -> "
                         "<output-dir>/rollouts.jsonl)")
    ap.set_defaults(rollout_log=True)
    ap.add_argument("--distillation-weight", type=float, default=1.0,
                    help="hybrid blend: loss=(1-w)*GRPO + w*SDPO. 1.0=pure SDPO, 0.0=pure GRPO")
    ap.add_argument("--teacher-kind", default="ema", choices=["ema", "base", "live"],
                    help="SDPO teacher: base=fixed/initial (iteration-05), ema=weight-EMA, live=current")
    ap.add_argument("--max-completion-length", type=int, default=8192)
    ap.add_argument("--max-prompt-length", type=int, default=3072)
    ap.add_argument("--output-dir", default="sdpo_out")
    ap.add_argument("--difficulties", default="easy,medium", help="comma list, e.g. 'easy' or 'easy,medium'")
    ap.add_argument("--languages", default="python,cpp", help="comma list: python,cpp")
    ap.add_argument("--system", default="cp_method", choices=["cp_method", "expert", "none"],
                    help="system prompt prepended to every train prompt (keeps train==eval prompt; "
                         "iteration-03 default: cp_method)")
    ap.add_argument("--frontier-band", default=None,
                    help="path to a frontier_band.json; train on its pids instead of the whole split")
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45)
    ap.add_argument("--reward-mode", default="fraction", choices=["fraction", "binary"],
                    help="dense passed/total (default) or strict AC=1/else 0")
    ap.add_argument("--grpo-reward", default="fraction", choices=["fraction", "binary"],
                    help="reward feeding the GRPO policy ADVANTAGE (feedback path only). "
                         "binary=AC 1/0 (iteration-05: clean policy signal); the SDPO teacher "
                         "gating always uses the dense fraction (see --sdpo-threshold).")
    ap.add_argument("--sdpo-threshold", type=float, default=1.0,
                    help="success_reward_threshold for SDPO teacher selection, applied to the "
                         "dense FRACTION. 1.0=AC-only; <1.0 (e.g. 0.5) lets near-miss rollouts teach.")
    ap.add_argument("--feedback", action="store_true",
                    help="live per-rollout judge feedback into the SDPO teacher (iteration 02)")
    # --- teacher-context integrity (iteration-11; the iter-01..10 silent bug) ---
    ap.add_argument("--keep-demo-thinking", action="store_true",
                    help="keep the sibling demo's <think> block in the teacher prompt "
                         "(pre-iter-11 behavior). A full-completion demo is 10-17k tokens, "
                         "overflows max_reprompt_len=8192, and LEFT-truncation then cuts "
                         "BOS+system+PROBLEM from the teacher context — the malformed-teacher "
                         "bug behind the iter-01..10 brevity collapse. Default: strip to code.")
    ap.add_argument("--feedback-only-without-solution", action="store_true",
                    help="use judge feedback only when the group has NO successful demo "
                         "(pre-iter-11 behavior). Default combines solution+feedback, the "
                         "SDPO paper's best config (48.3%% vs either alone).")
    ap.add_argument("--critic", action="store_true",
                    help="replace deterministic feedback with an LLM trace-aligned critique "
                         "(iteration 06; implies --feedback; needs ANTHROPIC_API_KEY)")
    ap.add_argument("--critic-model", default=None,
                    help="critic model id (default: sdpo_critic.DEFAULT_CRITIC_MODEL)")
    ap.add_argument("--critic-thinking", action="store_true",
                    help="use adaptive thinking for the critic (higher quality, slower/costlier)")
    # --- paper-faithful SDPO knobs (replication audit, 2026-07) ---------------
    # An audit of TRL's experimental SDPOConfig against the SDPO paper's ACTUAL
    # LCBv6 run config (experiments/rich_feedback/run_sdpo.sh in lasgroup/SDPO —
    # which overrides the method-section prose) found defaults that diverge. We
    # adopt the run-script values as OUR defaults (except --distillation-is-clip,
    # kept at TRL's 2.0 to preserve prior behavior) and expose each so a
    # paper-faithful run is one flag away.
    ap.add_argument("--teacher-update-rate", type=float, default=0.01,
                    help="EMA teacher decay (teacher_update_rate). Paper best=0.01 (OUR default); "
                         "TRL default is 0.05.")
    ap.add_argument("--distillation-alpha", type=float, default=1.0,
                    help="divergence interpolation (distillation_alpha): 1.0=reverse KL (paper's "
                         "LCBv6 run script AND TRL default — OUR default), 0.5=symmetric JSD "
                         "(paper's method-section prose), 0.0=forward KL.")
    ap.add_argument("--distillation-topk", type=int, default=20,
                    help="top-K logits for the distillation support (distillation_topk). Paper's "
                         "LCBv6 run: K=20 (+ tail bucket); TRL default None; our pre-replication "
                         "runs hardcoded 100.")
    ap.add_argument("--distillation-tail", dest="distillation_tail",
                    action="store_true", default=True,
                    help="add a tail bucket for non-top-k probability mass (distillation_add_tail). "
                         "Paper keeps the tail (OUR default: ON); TRL default is OFF.")
    ap.add_argument("--no-distillation-tail", dest="distillation_tail", action="store_false",
                    help="drop the tail bucket (TRL default behavior).")
    ap.add_argument("--distillation-is-clip", default="2.0",
                    help="importance-sampling clip coefficient (distillation_is_clip). Default 2.0 "
                         "preserves current behavior; the paper's plain per-token divergence has NO "
                         "IS correction — pass 'none' (or 0) to disable clipping for a paper-faithful run.")
    # Memory/speed knobs. Defaults are GB10-safe (microbatch=1 + grad ckpt).
    # On a roomier GPU (80 GB H100), --per-device-batch 2+ and/or
    # --no-grad-checkpointing trade memory for a faster step.
    ap.add_argument("--per-device-batch", type=int, default=1,
                    help="completions processed per device per micro-step")
    ap.add_argument("--grad-checkpointing", dest="grad_checkpointing",
                    action="store_true", default=True)
    ap.add_argument("--no-grad-checkpointing", dest="grad_checkpointing",
                    action="store_false",
                    help="disable activation checkpointing (faster backward, more memory)")
    ap.add_argument("--save-steps", type=int, default=0,
                    help="checkpoint every N steps (0 = save only the final adapter)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint-* in --output-dir if one exists "
                         "(safe on a fresh run: starts from step 0 when there's no checkpoint)")
    ap.add_argument("--enforce-eager", dest="enforce_eager", action="store_true", default=True,
                    help="disable vLLM CUDA graphs in the colocate engine (default on). The Modal "
                         "host kernel (4.19) intermittently HANGS the first colocate generation with "
                         "CUDA graphs on — eager makes it deterministic (~10-20%% slower generation).")
    ap.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    ap.add_argument("--no-wandb", action="store_true")
    return ap


def main():
    ap = build_parser()
    args = ap.parse_args()
    # --distillation-is-clip: accept the literal "none" (or 0) -> None (IS off, paper-faithful).
    _is_clip = str(args.distillation_is_clip).strip().lower()
    args.distillation_is_clip = None if _is_clip in ("none", "0", "0.0") else float(args.distillation_is_clip)
    if args.critic:
        args.feedback = True  # the critic rewrites the live feedback signal; it needs the bus

    # Teacher prompts must never be silently truncated again (iter-01..10 bug):
    # measure + log + warn on any overflow of max_reprompt_len.
    install_reprompt_guard()

    # Inject enforce_eager into TRL's colocate vLLM (it doesn't expose the flag in
    # colocate mode — only the server path does). Patch the LLM symbol TRL calls.
    if args.enforce_eager:
        try:
            import trl.generation.vllm_generation as _vg
            _orig_LLM = _vg.LLM

            def _eager_LLM(*a, **kw):
                kw.setdefault("enforce_eager", True)
                return _orig_LLM(*a, **kw)

            _vg.LLM = _eager_LLM
            print("[sdpo] vLLM colocate: enforce_eager=True (CUDA graphs off — kernel-hang guard)")
        except Exception as e:  # noqa: BLE001
            print(f"[sdpo] could not patch enforce_eager: {e}")

    difficulties = args.difficulties.split(",") if args.difficulties else None
    languages = tuple(args.languages.split(","))
    import sdpo_ojbench as S
    system = S.SYSTEM_PROMPTS[args.system]
    band_ids = None
    if args.frontier_band:
        import json
        band_ids = [int(p) for p in json.load(open(args.frontier_band))["frontier_band"]]
    ds = build_dataset("train", difficulties=difficulties, languages=languages,
                       system=system, ids=band_ids).shuffle(seed=0)
    print(f"[sdpo] dataset: {len(ds)} (problem,language) rows "
          f"diff={difficulties} langs={languages} system={args.system} "
          f"frontier_band={'yes('+str(len(band_ids))+')' if band_ids else 'no'}")
    if args.smoke:
        ds = ds.select(range(min(4, len(ds))))
        args.max_steps = 2
        args.num_generations = 4
        args.max_completion_length = 512

    # Per-rollout JSONL capture (P0-1): stream every rollout's text+verdict+reward+length
    # to <output-dir>/rollouts.jsonl so a collapse is reviewable after the fact.
    rollout_path = os.path.join(args.output_dir, "rollouts.jsonl") if args.rollout_log else None

    bus = None
    if args.feedback:
        bus = FeedbackBus()
        reward = make_feedback_reward_func(bus, which="public", timeout=6.0, reward_mode=args.reward_mode,
                                           critic=args.critic, critic_model=args.critic_model,
                                           critic_thinking=args.critic_thinking,
                                           grpo_reward=args.grpo_reward,
                                           rollout_log=rollout_path,
                                           sdpo_threshold=args.sdpo_threshold)
    else:
        reward = make_reward_func(which="public", timeout=6.0, reward_mode=args.reward_mode)

    # LoRA targets: Qwen3 is a plain text decoder — target the proj suffixes across
    # ALL layers. (The gemma-era text-tower regex that scoped these under
    # `language_model.*` was removed with the gemma path; see git history.)
    _proj = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=_proj,
    )

    report_to = "none" if (args.no_wandb or not os.environ.get("WANDB_API_KEY")) else "wandb"
    if report_to == "wandb":
        # Historical project name (gemma era) — kept so all iterations' runs stay
        # in ONE W&B project; renaming would orphan the iter-01..10 history.
        os.environ.setdefault("WANDB_PROJECT", "sdpo-gemma-ojbench")

    # Informative run name: model · signal source · distill weight · reward mode · teacher · steps
    # e.g. sdpo-qwen3-8b-critic-d0.1-binary-base-s20 — so runs are self-describing in the W&B list.
    _model_short = args.model.split("/")[-1].lower().replace(".", "")
    _signal = "critic" if args.critic else ("fb" if args.feedback else "verifier")
    # e.g. sdpo-qwen3-8b-critic-d0.1-grpobin-sdpo0.5-base-s20
    run_name = (f"sdpo-{_model_short}-{_signal}-d{args.distillation_weight}"
                f"-grpo{'bin' if args.grpo_reward == 'binary' else 'frac'}"
                f"-sdpo{args.sdpo_threshold}-{args.teacher_kind}-s{args.max_steps}"
                + ("-smoke" if args.smoke else ""))

    cfg = SDPOConfig(
        output_dir=args.output_dir,
        # --- SDPO core ---
        distillation_weight=args.distillation_weight,   # hybrid: (1-w)*GRPO + w*SDPO
        distillation_mode="topk_logits",
        teacher_model_kind=args.teacher_kind,           # "base"=fixed/initial (iteration-05 T0)
        # --- paper-faithful SDPO knobs (replication audit, 2026-07) ---
        # Defaults follow the paper's LCBv6 RUN SCRIPT (lasgroup/SDPO
        # experiments/rich_feedback/run_sdpo.sh), not the method-section prose.
        teacher_update_rate=args.teacher_update_rate,   # paper best=0.01 (TRL default 0.05)
        distillation_alpha=args.distillation_alpha,     # 1.0=reverse KL (LCBv6 run script; JSD=0.5 is prose-only)
        distillation_topk=args.distillation_topk,       # LCBv6 run: K=20 (pre-replication runs hardcoded 100)
        distillation_add_tail=args.distillation_tail,   # paper keeps tail bucket (TRL default False)
        distillation_is_clip=args.distillation_is_clip, # None=paper's plain per-token divergence (no IS clip)
        use_successful_as_teacher=True,
        success_reward_threshold=args.sdpo_threshold,  # applied to the dense FRACTION (feedback path)
        # --- teacher-context integrity (iteration-11) ---
        # The demo is a sibling's ENTIRE completion; un-stripped it is 10-17k tokens and
        # overflows max_reprompt_len, whose LEFT truncation (ids[-N:]) then cuts BOS+system+
        # PROBLEM from the teacher context — the iter-01..10 malformed-teacher bug. Stripping
        # <think> keeps the demo to code (~0.5-1k tok): problem + demo + feedback ≈ 2-3k tok,
        # never truncated (judge feedback fields are _clip'd to 600 chars each upstream).
        remove_thinking_from_demonstration=not args.keep_demo_thinking,
        # Judge feedback into the teacher. Default combines solution+feedback (the paper's
        # best config); --feedback-only-without-solution restores the pre-iter-11 gating
        # that dropped feedback whenever a demo existed. Either way feedback still covers
        # the ALL-FAIL groups (iteration-01's gap: no successful rollout -> no teacher).
        include_environment_feedback=args.feedback,
        environment_feedback_only_without_solution=args.feedback_only_without_solution,
        max_reprompt_len=8192,  # backstop only — the guard logs/warns if it ever binds
        # --- generation / RL ---
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=args.temperature,
        top_p=args.top_p,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
        # --- optim ---
        learning_rate=args.lr,
        # KL anchor to the frozen base + warmup-decay schedule: the iteration-06
        # anti-collapse levers (iteration-05 ran beta=0.0, linear-from-step-0).
        beta=args.beta,
        lr_scheduler_type=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        # Microbatch keeps the LM-head logits tensor [bs*seq*vocab] small enough
        # to fit alongside colocate vLLM (bs=num_generations OOMs the GB10 at
        # step 0). Effective batch is held at 2*num_generations via accumulation.
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=max(
            1, (2 * args.num_generations) // args.per_device_batch
        ),
        gradient_checkpointing=args.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy=("steps" if args.save_steps > 0 else "no"),
        save_steps=(args.save_steps if args.save_steps > 0 else 500),
        save_total_limit=None,  # keep every checkpoint (we want the per-20-step history)
        bf16=True,
        report_to=report_to,
        run_name=run_name,
    )

    if args.feedback:
        trainer = FeedbackSDPOTrainer(
            model=args.model, reward_funcs=reward, args=cfg,
            train_dataset=ds, peft_config=peft_cfg, feedback_bus=bus,
        )
    else:
        trainer = SDPOTrainer(
            model=args.model, reward_funcs=reward, args=cfg,
            train_dataset=ds, peft_config=peft_cfg,
        )
    # Resume from the latest checkpoint if asked AND one exists. Guarding on existence
    # keeps a single command idempotent: first launch starts fresh, a relaunch after a
    # death picks up the last checkpoint instead of re-burning GPU from step 0.
    resume = False
    if args.resume:
        import glob
        cks = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        resume = len(cks) > 0
        print(f"[sdpo] resume requested: {len(cks)} checkpoint(s) in {args.output_dir} -> "
              f"{'RESUMING from latest' if resume else 'none found, starting fresh'}")

    print(f"[sdpo] training: {len(ds)} problems, max_steps={args.max_steps}, "
          f"G={args.num_generations}, feedback={args.feedback}, critic={args.critic}, "
          f"resume={resume}, report_to={report_to}")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(args.output_dir)
    print(f"[sdpo] saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
