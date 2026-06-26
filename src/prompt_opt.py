"""Prompt optimization on base gemma: try prompt variants on sample OJBench problems,
generate + judge, and compare solve rates. Serve the base model on :8000 first.

  PYTHONPATH=src python src/prompt_opt.py --n 4 --temperature 0.6
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

import sdpo_ojbench as S

EXPERT_SYS = ("You are an expert competitive programmer. First reason briefly about the "
              "algorithm and its time complexity given the input limits, then write a single "
              "correct, efficient solution that reads from stdin and writes to stdout in the "
              "exact required format.")


def variants(prompt, pid):
    """name -> (system_or_None, user_prompt)."""
    hinted = S.augment_prompt_with_example(prompt, pid)  # no-op unless empty Example + small case
    return {
        "base": (None, prompt),
        "expert_sys": (EXPERT_SYS, prompt),
        "hint": (None, hinted),
        "hint+expert": (EXPERT_SYS, hinted),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--which", default="private", choices=["public", "private"])
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--out", default="prompt_opt_results.json")
    args = ap.parse_args()

    # sample: a spread of easy + medium train problems, prioritising ones where the
    # worked-example hint actually fires (empty Example + small case) so we can see its effect.
    def hint_fires(pid):
        if not S.example_section_is_empty(S.PROMPT_BY_ID.get(pid, "")):
            return False
        pub, _ = S.public_private_cases(pid)
        return min(c[0].stat().st_size for c in pub) <= 400

    easy = [p for p in S.SPLITS["train"] if S.DIFF_BY_ID[p] == "easy"]
    med = [p for p in S.SPLITS["train"] if S.DIFF_BY_ID[p] == "medium"]
    pids = sorted(easy, key=lambda p: not hint_fires(p))[:4] + sorted(med, key=lambda p: not hint_fires(p))[:3]
    print(f"problems: {pids}  hint_fires: {[p for p in pids if hint_fires(p)]}")

    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    def gen_and_judge(pid, vname, system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        resp = client.chat.completions.create(model=args.model, messages=msgs,
                                              temperature=args.temperature, max_tokens=args.max_tokens, n=args.n)
        ac = 0
        verdicts = []
        for ch in resp.choices:
            _, verdict, _ = S.judge_completion(ch.message.content or "", pid, which=args.which, language="python")
            verdicts.append(verdict)
            ac += int(verdict == "AC")
        return pid, vname, ac, len(resp.choices), verdicts

    jobs = []
    for pid in pids:
        for vname, (system, user) in variants(S.PROMPT_BY_ID[pid], pid).items():
            jobs.append((pid, vname, system, user))

    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(gen_and_judge, *j) for j in jobs]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(f"  loj-{r[0]} {r[1]:<12} AC {r[2]}/{r[3]}  {r[4]}", flush=True)

    # aggregate per variant: pass@1 (mean AC over all samples) and #problems solved (>=1 AC)
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, 0])  # ac, total, problems_with_any_ac
    for pid, vname, ac, n, _ in results:
        agg[vname][0] += ac
        agg[vname][1] += n
        agg[vname][2] += int(ac > 0)
    print("\n=== SUMMARY (variant: pass@1, problems-solved/total) ===")
    summary = {}
    nprob = len(pids)
    for v in ["base", "expert_sys", "hint", "hint+expert"]:
        ac, tot, solved = agg[v]
        p1 = ac / tot if tot else 0
        summary[v] = {"pass@1": round(p1, 3), "problems_solved": solved, "n_problems": nprob}
        print(f"  {v:<12} pass@1={p1:.3f}  solved={solved}/{nprob}")
    json.dump({"summary": summary, "results": results, "pids": pids}, open(args.out, "w"), indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
