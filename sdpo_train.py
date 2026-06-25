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

# load WANDB_API_KEY (and anything else) from .env
from pathlib import Path

ROOT = Path(__file__).parent
envf = ROOT / ".env"
if envf.exists():
    for line in envf.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from peft import LoraConfig  # noqa: E402
from trl.experimental.sdpo import SDPOConfig, SDPOTrainer  # noqa: E402

from sdpo_ojbench import build_dataset, make_reward_func  # noqa: E402


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
    ap.add_argument("--vllm-gpu-util", type=float, default=0.45)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    difficulties = args.difficulties.split(",") if args.difficulties else None
    languages = tuple(args.languages.split(","))
    ds = build_dataset("train", difficulties=difficulties, languages=languages).shuffle(seed=0)
    print(f"[sdpo] dataset: {len(ds)} (problem,language) rows "
          f"diff={difficulties} langs={languages}")
    if args.smoke:
        ds = ds.select(range(min(4, len(ds))))
        args.max_steps = 2
        args.num_generations = 4
        args.max_completion_length = 512

    reward = make_reward_func(which="public", timeout=6.0)

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
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=2,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        report_to=report_to,
        run_name="sdpo-gemma-ojbench" + ("-smoke" if args.smoke else ""),
    )

    trainer = SDPOTrainer(
        model=args.model,
        reward_funcs=reward,
        args=cfg,
        train_dataset=ds,
        peft_config=peft_cfg,
    )
    print(f"[sdpo] training: {len(ds)} problems, max_steps={args.max_steps}, "
          f"G={args.num_generations}, report_to={report_to}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"[sdpo] saved adapter to {args.output_dir}")


if __name__ == "__main__":
    main()
