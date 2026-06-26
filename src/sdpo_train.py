#!/usr/bin/env python3
"""SDPO training for Gemma-4-E2B-it on the OJBench frontier.

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--smoke", action="store_true", help="tiny config to debug integration")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
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
    ap.add_argument("--feedback", action="store_true",
                    help="live per-rollout judge feedback into the SDPO teacher (iteration 02)")
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
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

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

    bus = None
    if args.feedback:
        bus = FeedbackBus()
        reward = make_feedback_reward_func(bus, which="public", timeout=6.0, reward_mode=args.reward_mode)
    else:
        reward = make_reward_func(which="public", timeout=6.0, reward_mode=args.reward_mode)

    # Restrict LoRA to the TEXT model (language_model). gemma4's vision/audio
    # towers wrap projections in Gemma4ClippableLinear which PEFT can't target;
    # the text projections are plain nn.Linear. Regex => re.fullmatch in PEFT.
    peft_cfg = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )

    report_to = "none" if (args.no_wandb or not os.environ.get("WANDB_API_KEY")) else "wandb"
    if report_to == "wandb":
        os.environ.setdefault("WANDB_PROJECT", "sdpo-gemma-ojbench")

    cfg = SDPOConfig(
        output_dir=args.output_dir,
        # --- SDPO core ---
        distillation_weight=1.0,            # pure self-distillation
        distillation_mode="topk_logits",
        distillation_topk=100,
        teacher_model_kind="ema",
        use_successful_as_teacher=True,
        success_reward_threshold=1.0,
        # Live judge feedback into the teacher. feedback-only-without-solution targets
        # the ALL-FAIL groups (iteration-01's gap: no successful rollout -> no teacher);
        # success groups still use the solution. This also bounds teacher-prompt length
        # (prompt+solution OR prompt+feedback, never both) so it fits the 80 GB H100.
        include_environment_feedback=args.feedback,
        environment_feedback_only_without_solution=args.feedback,
        max_reprompt_len=8192,
        # --- generation / RL ---
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        temperature=1.0,
        top_p=0.95,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_util,
        # --- optim ---
        learning_rate=args.lr,
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
        run_name="sdpo-gemma-ojbench" + ("-smoke" if args.smoke else ""),
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
    print(f"[sdpo] training: {len(ds)} problems, max_steps={args.max_steps}, "
          f"G={args.num_generations}, feedback={args.feedback}, report_to={report_to}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[sdpo] saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
