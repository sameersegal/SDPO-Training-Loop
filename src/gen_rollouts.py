#!/usr/bin/env python3
"""Generate + judge BASE (student, no-context) rollouts for a model — the Phase-0 probe set.

We had no Qwen3-8B generations (iteration-04's saved rollouts are Gemma-4-E2B). This script
samples completions across easy/medium/hard, judges each, and saves id/difficulty/language/
verdict/reward/completion/feedback so we can (a) eyeball the LLM critic on real Qwen failures
across verdict types and (b) seed the Phase-0 gate (teacher-with-critic solve-rate / A_t).

Reuses the same prompts + judge as training/eval (sdpo_ojbench), so verdicts match. Writes
incrementally (resilient to a crash) and prints the verdict distribution.

  S=../../src
  PYTHONPATH=$S python $S/gen_rollouts.py --model Qwen/Qwen3-8B \
    --spec easy:2314,2317,2420:1 --spec medium:2086,2129,2130:2 --spec hard:2083,2131,2133:2 \
    --max-new-tokens 8192 --out qwen3_rollouts.json
"""
# HF download path: rely on huggingface_hub>=1.x defaults, which auto-use the Xet protocol
# when `hf_xet` is installed (the modern fast path, ~16-way concurrency). DO NOT enable the
# aggressive fast modes here — BOTH HF_HUB_ENABLE_HF_TRANSFER=1 and HF_XET_HIGH_PERFORMANCE=1
# STALL in this environment: each opens ~150-200 parallel sockets that never complete (0 B/s),
# evidently hitting a connection-count limit. Plain hf_xet defaults download reliably.
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sdpo_ojbench import (PROMPT_BY_ID, CPP_PROMPT_BY_ID, SYSTEM_PROMPTS,
                          judge_completion, reward_case_caps)

ROOT = Path.cwd()


def gen_one(model, tok, messages, args, seed, thinking):
    kw = {}
    if thinking is not None:
        kw["enable_thinking"] = thinking  # Qwen3 chat-template switch
    try:
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, **kw).to("cuda")
    except TypeError:  # template doesn't accept enable_thinking
        enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
    plen = enc["input_ids"].shape[1]
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=True,
                             temperature=args.temperature, top_p=args.top_p,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new = out[0][plen:]
    dt = time.perf_counter() - t0
    n_new = int(new.shape[0])
    hit_cap = n_new >= args.max_new_tokens
    return tok.decode(new, skip_special_tokens=True), n_new, hit_cap, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--system", default="cp_method", choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--spec", action="append", required=True,
                    help="difficulty:id1,id2,...:samples  (repeatable)")
    ap.add_argument("--language", default="python", choices=["python", "cpp"])
    ap.add_argument("--engine", default="vllm", choices=["vllm", "transformers"],
                    help="vllm = fast batched (GB10-proven via eval); transformers = slow sequential")
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=32768,
                    help="generation cap; 32768 + 8192 prompt = Qwen3-8B's 40960 native context "
                         "(don't cap thinking-ON tight — it NO_CODEs, see CLAUDE.md)")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thinking", dest="thinking", action="store_true", default=True)
    ap.add_argument("--no-thinking", dest="thinking", action="store_false")
    ap.add_argument("--out", default="rollouts.json")
    ap.add_argument("--resume", action="store_true",
                    help="skip (id,sample,language) rollouts already in --out (restart-safe)")
    ap.add_argument("--concurrent", type=int, default=1,
                    help="vLLM only: max in-flight requests. 1 = serial per-request (safe on GB10, "
                         "which hangs on high concurrency). >1 = async concurrent with per-completion "
                         "writes — keeps streaming/resume AND uses real-GPU batching (use on Modal/H200).")
    args = ap.parse_args()

    jobs = []  # (difficulty, id, samples)
    for spec in args.spec:
        diff, ids, n = spec.split(":")
        for pid in ids.split(","):
            jobs.append((diff, int(pid), int(n)))
    total = sum(n for _, _, n in jobs)
    sys_text = SYSTEM_PROMPTS[args.system]
    pmap = CPP_PROMPT_BY_ID if args.language == "cpp" else PROMPT_BY_ID
    mb, mc = reward_case_caps()

    out = ROOT / args.out
    meta = {"model": args.model, "system": args.system, "language": args.language,
            "engine": args.engine, "thinking": args.thinking,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature}

    # Flatten to per-sample tasks; drop problems with no prompt in this language.
    tasks = []  # (diff, pid, sample, messages)
    for diff, pid, n in jobs:
        prompt = pmap.get(pid)
        if prompt is None:
            print(f"  skip loj-{pid}: no {args.language} prompt", flush=True)
            continue
        messages = ([{"role": "system", "content": sys_text}] if sys_text else []) + \
                   [{"role": "user", "content": prompt}]
        for s in range(n):
            tasks.append((diff, pid, s, messages))

    # Resume-safe: keep prior results and skip tasks already done (restart after an interruption).
    results = []
    if args.resume and out.exists():
        results = json.load(open(out)).get("results", [])
        done = {(x["id"], x["sample"], x["language"]) for x in results}
        before = len(tasks)
        tasks = [t for t in tasks if (t[1], t[2], args.language) not in done]
        print(f"resume: {len(results)} existing rollouts; {before - len(tasks)} skipped, "
              f"{len(tasks)} to generate", flush=True)
    if not tasks:
        print("nothing to do (all rollouts already present)", flush=True)
        return

    print(f"loading {args.model} via {args.engine} (thinking={args.thinking})…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)

    def chat_to_text(messages):
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=args.thinking)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _judge(comp, pid):
        return judge_completion(comp, pid, which="public", language=args.language,
                                reward_mode="fraction", max_case_bytes=mb, max_cases=mc)

    def _persist(diff, pid, s, comp, n_new, hit_cap, dt, r, v, fb):
        results.append({"difficulty": diff, "id": pid, "language": args.language,
                        "sample": s, "verdict": v, "reward": round(r, 3),
                        "n_new_tokens": n_new, "hit_token_cap": hit_cap,
                        "gen_seconds": round(dt, 1), "completion": comp, "feedback": fb})
        json.dump({**meta, "results": results}, open(out, "w"), indent=2)  # incremental write
        print(f"  [{diff}] loj-{pid} s{s}: {v} reward={r:.2f} "
              f"({n_new} tok{', CAP' if hit_cap else ''}, {dt:.0f}s)", flush=True)

    def record(diff, pid, s, comp, n_new, hit_cap, dt):  # serial path
        r, v, fb = _judge(comp, pid)
        _persist(diff, pid, s, comp, n_new, hit_cap, dt, r, v, fb)

    t_start = time.perf_counter()
    if args.engine == "vllm" and args.concurrent > 1:
        # Concurrent generation with PER-COMPLETION writes: keeps the streaming/resume
        # guarantee AND uses real-GPU continuous batching (the GB10's "batching buys
        # nothing" was a GB10 memory-bandwidth artifact — see CLAUDE.md). vLLM schedules
        # the in-flight requests together; each coroutine writes the moment ITS request
        # finishes. enforce_eager stays (Modal host-kernel-4.19 CUDA-graph hang).
        import asyncio
        from vllm import AsyncEngineArgs, SamplingParams
        from vllm.v1.engine.async_llm import AsyncLLM

        async def _run_concurrent():
            engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
                model=args.model, dtype="bfloat16", enforce_eager=True,
                gpu_memory_utilization=args.gpu_util,
                max_model_len=args.max_new_tokens + 8192))
            sem = asyncio.Semaphore(args.concurrent)
            lock = asyncio.Lock()
            done = [0]
            print(f"generating {len(tasks)} rollouts (vLLM async, concurrent={args.concurrent}, "
                  f"per-completion writes, max_new={args.max_new_tokens})…", flush=True)

            async def _heartbeat():
                while done[0] < len(tasks):
                    await asyncio.sleep(120)
                    print(f"  [gen] {done[0]}/{len(tasks)} done, still generating…", flush=True)

            async def _one(diff, pid, s, messages):
                sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                    max_tokens=args.max_new_tokens, seed=args.seed + 1009 * pid + s)
                t0 = time.perf_counter()
                final = None
                async with sem:
                    async for out in engine.generate(chat_to_text(messages), sp,
                                                     request_id=f"{diff}-{pid}-{s}"):
                        final = out
                g = final.outputs[0]
                r, v, fb = await asyncio.to_thread(_judge, g.text, pid)  # judge is blocking subprocess
                async with lock:
                    _persist(diff, pid, s, g.text, len(g.token_ids),
                             g.finish_reason == "length", time.perf_counter() - t0, r, v, fb)
                    done[0] += 1

            hb = asyncio.create_task(_heartbeat())
            try:
                await asyncio.gather(*[_one(*t) for t in tasks])
            finally:
                hb.cancel()

        asyncio.run(_run_concurrent())
    elif args.engine == "vllm":
        from vllm import LLM, SamplingParams
        # enforce_eager + n=1 per request avoids the GB10 CUDA-graph / multi-sample hang.
        llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
                  gpu_memory_utilization=args.gpu_util,
                  max_model_len=args.max_new_tokens + 8192)
        # Per-request (NOT one batched generate): write+judge each rollout the moment it finishes,
        # so we have live visibility (hang vs progress) and a resumable run. On GB10 batching gives
        # ~no throughput gain (see CLAUDE.md), so this costs nothing.
        print(f"generating {len(tasks)} rollouts (vLLM, per-request writes, "
              f"max_new={args.max_new_tokens})…", flush=True)
        for diff, pid, s, messages in tasks:
            sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                                max_tokens=args.max_new_tokens, seed=args.seed + 1009 * pid + s)
            t0 = time.perf_counter()
            o = llm.generate([chat_to_text(messages)], sp, use_tqdm=False)[0]
            g = o.outputs[0]
            record(diff, pid, s, g.text, len(g.token_ids), g.finish_reason == "length",
                   time.perf_counter() - t0)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     device_map="cuda")
        model.eval()
        print(f"generating {len(tasks)} rollouts sequentially (transformers)…", flush=True)
        for diff, pid, s, messages in tasks:
            comp, n_new, hit_cap, dt = gen_one(model, tok, messages, args,
                                               args.seed + 1009 * pid + s, args.thinking)
            record(diff, pid, s, comp, n_new, hit_cap, dt)

    dist = Counter(x["verdict"] for x in results)
    fails = sum(1 for x in results if x["verdict"] != "AC")
    print(f"\ndone in {time.perf_counter()-t_start:.0f}s · {len(results)} rollouts · "
          f"verdicts={dict(dist)} · failures={fails}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
