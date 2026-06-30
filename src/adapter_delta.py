#!/usr/bin/env python3
"""Measure how far a LoRA adapter actually moved the weights — serve-free, no base model needed.

For each LoRA layer the functional perturbation is ΔW = (alpha/r)·B·A; ||ΔW||_F summed over layers is a
single scalar for "how much did training change the model." Because B initializes to 0, this is a pure
measure of movement. Compare across checkpoints/iterations (same LoRA config => apples-to-apples) to tell
under-training (tiny ΔW) from a real move.

Reference scale (Qwen3-8B, r=32, alpha=64): iter-05 ckpt-20 ΔW≈2.62 moved the model enough to COLLAPSE it;
iter-08 ckpt-8 ΔW≈0.46 barely moved it. Use as iter-09's live "is the dose landing?" gauge.

  python src/adapter_delta.py path/to/adapter_dir [more_dirs...]
  python src/adapter_delta.py runs/iteration-09/sdpo_out/checkpoint-*   # dose curve across a run
"""
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open


def delta_norm(adapter_dir: Path):
    cfg = json.load(open(adapter_dir / "adapter_config.json"))
    scaling = cfg["lora_alpha"] / cfg["r"]
    f = safe_open(adapter_dir / "adapter_model.safetensors", "pt")
    keys = list(f.keys())
    tot2 = 0.0
    nlayer = 0
    for ak in [k for k in keys if "lora_A" in k]:
        bk = ak.replace("lora_A", "lora_B")
        if bk not in keys:
            continue
        A = f.get_tensor(ak).float()        # [r, in]
        B = f.get_tensor(bk).float()        # [out, r]
        tot2 += torch.linalg.norm(scaling * (B @ A)).item() ** 2
        nlayer += 1
    return dict(scaling=scaling, r=cfg["r"], nlayer=nlayer, dW=tot2 ** 0.5)


def main():
    dirs = [Path(p) for p in sys.argv[1:]] or [Path(".")]
    print(f"{'adapter':<48} {'r':>3} {'scale':>5} {'layers':>6} {'||ΔW||_F':>10}")
    for d in dirs:
        if not (d / "adapter_model.safetensors").exists():
            print(f"{str(d):<48} (no adapter_model.safetensors)")
            continue
        r = delta_norm(d)
        print(f"{str(d):<48} {r['r']:>3} {r['scaling']:>5.1f} {r['nlayer']:>6} {r['dW']:>10.3f}")


if __name__ == "__main__":
    main()
