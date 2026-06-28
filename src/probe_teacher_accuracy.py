#!/usr/bin/env python3
"""Accuracy probe — does the SDPO capability gap exist at our scale?

SDPO can only teach if the teacher (conditioned on privileged context `c`) is a BETTER
predictor of the correct solution than the student (no context). This probe measures it
directly: for the all-fail rollout groups from the per-token diagnostic, condition the
teacher on our judge feedback and have it GENERATE a fresh solution, then compare its
AC rate to the student's.

  P(student solves | x)        — base rollouts, no context (from the diagnostic group)
  P(teacher solves | x, c)     — same model, conditioned on the per-rollout feedback `c`

If teacher >> student, the gap exists and SDPO has fuel. If flat, there is nothing to
distill at this scale — itself a result (cf. the original SDPO scaling curve; iteration-05
moves to Qwen3-8B). See reports/iteration-04/INSIGHTS.md.

Reuses the committed diagnostic JSON (student verdicts + per-rollout feedback), so no
student regeneration — only teacher generations. For groups that HAD an AC, the gating
context is the AC group-mate's code ("answer handed over"), which is not an informative
probe (the student already solves), so those are skipped.

  S=../../src
  PYTHONPATH=$S OJB_SPLITS=ojb_splits_full.json python $S/probe_teacher_accuracy.py \
    --in-json ../../reports/iteration-04/data/token_advantage.json
"""
import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sdpo_prompts as SP
from sdpo_ojbench import (PROMPT_BY_ID, CPP_PROMPT_BY_ID, SYSTEM_PROMPTS,
                          judge_completion, reward_case_caps)

ROOT = Path.cwd()


def build_messages(prompt_text, system_text):
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": prompt_text})
    return msgs


def generate_one(model, tok, messages, max_new_tokens, temperature, top_p, seed):
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    plen = enc["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p, num_return_sequences=1,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0][plen:], skip_special_tokens=True)


def probe_one(model, tok, res, meta, args, mb, mc):
    diff, pid = res["difficulty"], int(res["id"])
    lang = res.get("language", meta["language"])
    pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
    messages = build_messages(pmap[pid], SYSTEM_PROMPTS[meta["system"]])

    # student side — straight from the diagnostic group (no regeneration)
    group = res["group"]
    s_verdicts = Counter(g["verdict"] for g in group)
    s_rewards = [g["reward"] for g in group]
    student = {"n": len(group), "ac": sum(g["verdict"] == "AC" for g in group),
               "mean_reward": round(sum(s_rewards) / len(s_rewards), 3),
               "verdicts": dict(s_verdicts)}

    if res["group_has_success"]:
        teacher = {"mode": "solution(answer handed over)", "skipped": True,
                   "note": "student already solves (group has an AC); solution-context is not "
                           "an informative gap probe"}
        print(f"[{diff}] loj-{pid}: student AC {student['ac']}/{student['n']} — "
              f"teacher SKIPPED (solution-context / answer handed over)", flush=True)
        return {"difficulty": diff, "id": pid, "context": "solution",
                "student": student, "teacher": teacher}

    # teacher side — condition on each rollout's own feedback, generate, judge
    feedbacks = res["feedback_all"]
    attempts = []
    print(f"[{diff}] loj-{pid}: student AC {student['ac']}/{student['n']} "
          f"(mean_reward {student['mean_reward']}); probing teacher on {len(feedbacks)} "
          f"feedback(s) x {args.samples_per_feedback}…", flush=True)
    for i, fb in enumerate(feedbacks):
        tmsgs = SP.build_teacher_messages(messages, feedback_raw=fb)
        for s in range(args.samples_per_feedback):
            txt = generate_one(model, tok, tmsgs, args.max_new_tokens,
                               args.temperature, args.top_p, args.seed + 1009 * i + s)
            r, v, _ = judge_completion(txt, pid, which="public", language=lang,
                                       reward_mode="fraction", max_case_bytes=mb, max_cases=mc)
            attempts.append({"fb_idx": i, "sample": s, "verdict": v, "reward": round(r, 3)})
            print(f"    teacher[fb={i},s={s}]: {v} reward={r:.2f}", flush=True)
    t_rewards = [a["reward"] for a in attempts]
    teacher = {"mode": "feedback", "n": len(attempts),
               "ac": sum(a["verdict"] == "AC" for a in attempts),
               "mean_reward": round(sum(t_rewards) / len(t_rewards), 3) if t_rewards else 0.0,
               "verdicts": dict(Counter(a["verdict"] for a in attempts)),
               "attempts": attempts}
    return {"difficulty": diff, "id": pid, "context": "feedback",
            "student": student, "teacher": teacher}


def write_md(meta, rows, out_md):
    L = ["# Accuracy probe — does the SDPO capability gap exist?", "",
         f"Model `{meta['model']}` · system `{meta['system']}` · language `{meta['language']}`. "
         "Student = base rollouts (no context). Teacher = same model conditioned on the per-rollout "
         "judge feedback `c`, generating a fresh solution. AC = all public tests pass; mean reward = "
         "fraction of public tests passed (partial-credit signal).", "",
         "| difficulty | problem | context | student AC | student mean-reward | teacher AC | teacher mean-reward | gap (AC) | gap (reward) |",
         "|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        s, t = r["student"], r["teacher"]
        if t.get("skipped"):
            L.append(f"| {r['difficulty']} | loj-{r['id']} | {r['context']} | "
                     f"{s['ac']}/{s['n']} | {s['mean_reward']} | — (skipped) | — | — | — |")
        else:
            gap_ac = f"{t['ac']}/{t['n']} − {s['ac']}/{s['n']}"
            gap_rw = round(t["mean_reward"] - s["mean_reward"], 3)
            L.append(f"| {r['difficulty']} | loj-{r['id']} | {r['context']} | "
                     f"{s['ac']}/{s['n']} | {s['mean_reward']} | {t['ac']}/{t['n']} | "
                     f"{t['mean_reward']} | {gap_ac} | {gap_rw:+} |")
    L += ["", "**Verdict distributions** (student → teacher):", ""]
    for r in rows:
        s, t = r["student"], r["teacher"]
        tline = "skipped (solution-context)" if t.get("skipped") else f"{t['verdicts']}"
        L.append(f"- **{r['difficulty']} loj-{r['id']}** — student {s['verdicts']} → teacher {tline}")
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="default: model from --in-json meta")
    ap.add_argument("--in-json", default="../../reports/iteration-04/data/token_advantage.json")
    ap.add_argument("--difficulties", default="easy,medium,hard")
    ap.add_argument("--samples-per-feedback", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="teacher_accuracy_probe")
    args = ap.parse_args()

    data = json.load(open(args.in_json))
    meta = {"model": args.model or data["model"], "system": data["system"],
            "language": data["language"]}
    want = set(args.difficulties.split(","))
    results = [r for r in data["results"] if r["difficulty"] in want]

    print(f"loading {meta['model']}…", flush=True)
    tok = AutoTokenizer.from_pretrained(meta["model"])
    model = AutoModelForCausalLM.from_pretrained(meta["model"], dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    mb, mc = reward_case_caps()

    t0 = time.perf_counter()
    rows = [probe_one(model, tok, r, meta, args, mb, mc) for r in results]

    out_json = ROOT / f"{args.out_prefix}.json"
    json.dump({**meta, "rows": rows}, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}", flush=True)
    write_md(meta, rows, ROOT / f"{args.out_prefix}.md")
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
