#!/usr/bin/env python3
"""Phase-0 teacher evaluation: does conditioning Qwen3-8B on the critic's feedback (a) make
it more likely to SOLVE (solve-rate) and (b) produce a localized per-token ADVANTAGE on the
student's own failed attempt (A_t — the actual SDPO training signal)?

For each FAILED rollout in the probe set, with one critique (default sonnet_verbose) and the
SDPO-faithful teacher prompt `x + critique` (NO student attempt in the prompt — that's the
original-SDPO entropy-collapse footgun):

  solve-rate : generate K samples from (x + critique), judge -> fraction AC.
  A_t        : score the student's attempt y under student (x) and teacher (x + critique):
                 A_t = log q(y_t | x, critique, y_<t) - log pi(y_t | x, y_<t)
               via vLLM prompt_logprobs (teacher-forced, same y tokens under both prompts).

Records completions + logprobs (per-token student_lp/teacher_lp/A_t) for offline analysis.
vLLM only; generation is concurrent (n=K batched per failure); writes incrementally per
failure (visibility + --resume). Run on Modal H200 (modal_sdpo.py::teacher_eval).
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from transformers import AutoTokenizer

from ojbench_eval import extract_code  # noqa: F401  (kept for parity / future code-region use)
from sdpo_ojbench import PROMPT_BY_ID, CPP_PROMPT_BY_ID, SYSTEM_PROMPTS, judge_completion, reward_case_caps
import sdpo_prompts as SP

ROOT = Path.cwd()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", default="reports/iteration-05/data/qwen3_rollouts.json")
    ap.add_argument("--critiques", default="reports/iteration-05/data/qwen3_critiques.json")
    ap.add_argument("--critic-set", default="sonnet_verbose")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--system", default="cp_method", choices=list(SYSTEM_PROMPTS))
    ap.add_argument("--samples", type=int, default=8, help="K generations for solve-rate")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="teacher_eval.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N failures (smoke)")
    args = ap.parse_args()

    out = ROOT / args.out
    sys_text = SYSTEM_PROMPTS[args.system]
    mb, mc = reward_case_caps()

    roll = json.load(open(args.rollouts))
    crit = json.load(open(args.critiques))
    crit_by = {(r["id"], r["sample"]): r["critiques"] for r in crit["rows"]}
    fails = [x for x in roll["results"] if x["verdict"] not in ("AC", "NO_CODE")]
    fails = [x for x in fails if (x["id"], x["sample"]) in crit_by]

    meta = {"model": args.model, "critic_set": args.critic_set, "system": args.system,
            "samples": args.samples, "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature, "teacher_prompt": "x + critique (no attempt)"}

    results = []
    if args.resume and out.exists():
        results = json.load(open(out)).get("results", [])
        done = {(r["id"], r["sample"]) for r in results}
        fails = [x for x in fails if (x["id"], x["sample"]) not in done]
        print(f"resume: {len(results)} done, {len(fails)} to do", flush=True)
    if args.limit:
        fails = fails[:args.limit]
    if not fails:
        print("nothing to do", flush=True)
        return

    from vllm import LLM, SamplingParams
    try:
        from vllm import TokensPrompt
    except ImportError:
        from vllm.inputs import TokensPrompt

    print(f"loading {args.model} (vLLM)…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", enforce_eager=True,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_new_tokens + 8192)

    def chat_text(msgs):
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def chat_ids(msgs):
        # Render to string then encode. Verified faithful for Qwen3 (Qwen2Tokenizer): decode
        # round-trips exactly and special tokens map to single ids (encode('<|im_start|>')==[151644]),
        # so this equals the canonical apply_chat_template(tokenize=True)['input_ids']. (That call
        # returns a BatchEncoding dict, not a bare list — the earlier "2 ids" was len()-on-dict.)
        return tok.encode(chat_text(msgs), add_special_tokens=False)

    def attempt_token_logprobs(prompt_msgs, y_ids):
        """Teacher-forced per-token logprob of y_ids appended to prompt_msgs (vLLM prompt_logprobs)."""
        plen = len(chat_ids(prompt_msgs))
        full = chat_ids(prompt_msgs) + y_ids
        o = llm.generate(TokensPrompt(prompt_token_ids=full),
                         SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1),
                         use_tqdm=False)[0]
        pls = o.prompt_logprobs  # list aligned to `full`; [0] is None
        lp = []
        for i in range(plen, len(full)):
            d = pls[i] or {}
            tid = full[i]
            lp.append(d[tid].logprob if tid in d else float("nan"))
        return lp

    def code_fence_tok_index(text, y_ids):
        pos = text.find("```")
        if pos < 0:
            return len(y_ids)
        return len(tok.encode(text[:pos], add_special_tokens=False))

    def summarize(a_t, fence):
        import math
        n = len(a_t)
        absa = [abs(v) for v in a_t if not math.isnan(v)]
        neg = sum(1 for v in a_t if not math.isnan(v) and v < 0)
        code = [abs(v) for v in a_t[fence:] if not math.isnan(v)]
        reas = [abs(v) for v in a_t[:fence] if not math.isnan(v)]
        mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else 0.0
        return {"n_tokens": n, "mean_abs": mean(absa), "frac_neg": round(neg / max(1, n), 3),
                "code_region_abs": mean(code), "reasoning_abs": mean(reas), "code_fence_tok": fence}

    t0 = time.perf_counter()
    for x in fails:
        pid, s, lang = x["id"], x["sample"], x["language"]
        problem = (CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID)[pid]
        critique = crit_by[(pid, s)][args.critic_set]
        attempt = x["completion"]
        base = [{"role": "system", "content": sys_text}] if sys_text else []
        student_msgs = base + [{"role": "user", "content": problem}]
        teacher_msgs = SP.build_teacher_messages(student_msgs, feedback_raw=critique)

        # --- solve-rate: K concurrent samples from (x + critique) ---
        sp = SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_new_tokens, seed=args.seed + pid, logprobs=0)
        go = llm.generate(chat_text(teacher_msgs), sp, use_tqdm=False)[0]
        samples = []
        with ThreadPoolExecutor(max_workers=min(16, args.samples)) as ex:
            judged = list(ex.map(
                lambda o: judge_completion(o.text, pid, which="public", language=lang,
                                           reward_mode="fraction", max_case_bytes=mb, max_cases=mc),
                go.outputs))
        for o, (r, v, _fb) in zip(go.outputs, judged):
            nt = len(o.token_ids)
            samples.append({"verdict": v, "reward": round(float(r), 3), "n_tokens": nt,
                            "finish": o.finish_reason,
                            "mean_logprob": round(o.cumulative_logprob / max(1, nt), 4),
                            "completion": o.text})
        solve_rate = round(sum(sm["verdict"] == "AC" for sm in samples) / len(samples), 3)

        # --- A_t: score the student's attempt under student (x) vs teacher (x + critique) ---
        y_ids = tok.encode(attempt, add_special_tokens=False)
        student_lp = attempt_token_logprobs(student_msgs, y_ids)
        teacher_lp = attempt_token_logprobs(teacher_msgs, y_ids)
        a_t = [t - sl for t, sl in zip(teacher_lp, student_lp)]
        fence = code_fence_tok_index(attempt, y_ids)

        results.append({
            "id": pid, "sample": s, "difficulty": x["difficulty"], "language": lang,
            "base_verdict": x["verdict"], "base_reward": x["reward"],
            "critique": critique,
            "solve_rate": solve_rate, "samples": samples,
            "attempt_n_tokens": len(y_ids),
            "advantage": {**summarize(a_t, fence),
                          "student_lp": [round(v, 4) for v in student_lp],
                          "teacher_lp": [round(v, 4) for v in teacher_lp],
                          "A_t": [round(v, 4) for v in a_t]},
        })
        json.dump({**meta, "results": results}, open(out, "w"), indent=2)  # incremental
        adv = results[-1]["advantage"]
        print(f"  {x['difficulty']} loj-{pid} s{s} [{x['verdict']}]: solve-rate {solve_rate} "
              f"(K={args.samples}) | A_t mean|.|={adv['mean_abs']} frac_neg={adv['frac_neg']} "
              f"code/reas={adv['code_region_abs']}/{adv['reasoning_abs']}", flush=True)

    print(f"\ndone in {time.perf_counter()-t0:.0f}s · {len(results)} failures · wrote {out}", flush=True)


if __name__ == "__main__":
    main()
