#!/usr/bin/env python3
"""Held-out pass@1 evaluator for SDPO (base model or LoRA adapter).

Generates one greedy solution per held-out problem via transformers and judges
on the PRIVATE test cases (held back from training). Reports pass@1 by difficulty.

  python sdpo_eval.py --tag base
  python sdpo_eval.py --adapter sdpo_out --tag sdpo
"""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdpo_ojbench import (SPLITS, PROMPT_BY_ID, CPP_PROMPT_BY_ID, DIFF_BY_ID,
                          judge_completion)

ROOT = Path(__file__).parent


def generate(model, tok, prompt, max_new_tokens):
    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to("cuda")
    plen = enc["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = tok.decode(out[0][plen:], skip_special_tokens=True)
    return gen, round(time.perf_counter() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--languages", default="python,cpp")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    languages = args.languages.split(",")
    ids = SPLITS["heldout"]
    results = []
    for lang in languages:
        pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
        for pid in ids:
            if pid not in pmap:
                continue
            gen, dt = generate(model, tok, pmap[pid], args.max_new_tokens)
            reward, verdict, _ = judge_completion(gen, int(pid), which="private", language=lang)
            results.append({"id": pid, "language": lang, "difficulty": DIFF_BY_ID[pid],
                            "verdict": verdict, "is_passed": verdict == "AC",
                            "private_reward": reward, "gen_s": dt})
            print(f"  [{lang:<6}|{DIFF_BY_ID[pid]:<6}] loj-{pid}: {verdict} "
                  f"(reward={reward:.2f})", flush=True)

    def rate(lang, d=None):
        sub = [r for r in results if r["language"] == lang and (d is None or r["difficulty"] == d)]
        return sum(r["is_passed"] for r in sub), len(sub)

    summary = {"tag": args.tag, "adapter": args.adapter, "by_language": {}}
    for lang in languages:
        e, m, o = rate(lang, "easy"), rate(lang, "medium"), rate(lang)
        summary["by_language"][lang] = {
            "easy_pass": f"{e[0]}/{e[1]}", "medium_pass": f"{m[0]}/{m[1]}",
            "overall_pass": f"{o[0]}/{o[1]}"}
    out = ROOT / f"sdpo_eval_{args.tag}.json"
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print("\n=== HELD-OUT SUMMARY (by language) ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
