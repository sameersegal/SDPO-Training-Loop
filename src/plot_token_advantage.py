#!/usr/bin/env python3
"""Per-token SDPO advantage diagnostic — recreate Fig. 4 of "The Role of Feedback
Alignment in Self-Distillation" (arXiv 2606.11173) for OUR base model.

The paper plots, along a single rollout, the per-token advantage

    A_t = log pi(y_t | x, c, y_<t)  -  log pi(y_t | x, y_<t)
          \\___ teacher: sees context c ___/    \\__ student: question only __/

i.e. how much a privileged context c shifts the SAME model's next-token logprob on
the SAME rollout tokens. Red (A_t>0) = "keep this token"; blue (A_t<0) = "change it".
Localized blue at the error  => StepAlignFB-like (good); diffuse blue everywhere =>
RefSol-like (the signal fights correct code too). See
knowledge/summary_feedback_alignment_sd.md (diagnostic #3).

This recreates it with EXACTLY the context our latest SDPO run feeds the teacher
(sdpo_train.py: use_successful_as_teacher + success_reward_threshold=1.0,
include_environment_feedback, environment_feedback_only_without_solution). Per
rollout group:
  - group has an AC rollout  -> c = that AC rollout's full code   (solution slot)
  - all-fail group           -> c = our judge feedback (_format_feedback)
The student's own completion tokens are always what gets scored (TRL appends them
to the teacher prompt; sdpo_trainer.py:296). A partial-pass (e.g. 16/20) attempt is
NEVER used as the solution context — only a full AC is.

Difficulty sweeps the two arms naturally: easy tends to land in the AC-solution
regime (~ paper panel ii / RefSol), hard tends to land in the feedback regime
(~ panel i / StepAlignFB).

Usage (from runs/iteration-NN/ so figures land per-iteration):
  S=../../src
  PYTHONPATH=$S python $S/plot_token_advantage.py --difficulties easy,medium,hard
  PYTHONPATH=$S python $S/plot_token_advantage.py --smoke      # 1 diff, G=2, short
"""
import argparse
import json
import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sdpo_prompts as SP
from sdpo_ojbench import (PROMPT_BY_ID, CPP_PROMPT_BY_ID, DIFF_BY_ID, SPLITS,
                          SYSTEM_PROMPTS, judge_completion, reward_case_caps)

ROOT = Path.cwd()


def pick_ids(difficulties, language, explicit_ids):
    """One problem id per requested difficulty (deterministic: first available).

    Drawn from train ∪ heldout pools so it works regardless of which split a
    difficulty lives in (hard only exists in heldout_hard)."""
    pmap = CPP_PROMPT_BY_ID if language == "cpp" else PROMPT_BY_ID
    if explicit_ids:
        return {DIFF_BY_ID[i]: i for i in explicit_ids}
    pool = []
    for sp in ("train", "heldout", "heldout_hard", "heldout_frontier"):
        pool += SPLITS.get(sp, [])
    seen, chosen = set(), {}
    for pid in pool:
        if pid in seen:
            continue
        seen.add(pid)
        d = DIFF_BY_ID.get(pid)
        if d in difficulties and d not in chosen and pid in pmap:
            chosen[d] = pid
    return chosen


def build_messages(prompt_text, system_text):
    msgs = []
    if system_text:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": prompt_text})
    return msgs


def generate_group(model, tok, messages, n, max_new_tokens, temperature, top_p, seed):
    """Sample n rollouts for one prompt. Returns (list[completion_ids_tensor], list[text])."""
    enc = tok.apply_chat_template(messages, add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to("cuda")
    plen = enc["input_ids"].shape[1]
    torch.manual_seed(seed)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True,
                             temperature=temperature, top_p=top_p,
                             num_return_sequences=n,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    comp_ids, texts = [], []
    pad = tok.pad_token_id or tok.eos_token_id
    for g in range(out.shape[0]):
        ids = out[g][plen:]
        # strip trailing pad (right-padding from shorter siblings in the batch)
        keep = (ids != pad).nonzero()
        last = int(keep[-1]) + 1 if len(keep) else 0
        ids = ids[:last]
        comp_ids.append(ids.unsqueeze(0))                       # [1, C]
        texts.append(tok.decode(ids, skip_special_tokens=True))
    return comp_ids, texts


def completion_token_logprobs(model, prompt_ids, completion_ids):
    """Teacher-forced logprob of each completion token given prompt_ids ++ completion_ids."""
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)   # [1, P+C]
    with torch.no_grad():
        logits = model(input_ids).logits[:, :-1, :]             # predict token t+1
    targets = input_ids[:, 1:]
    C = completion_ids.shape[1]
    logits_c = logits[:, -C:, :].float()                        # only completion positions
    targets_c = targets[:, -C:]
    lp = torch.log_softmax(logits_c, dim=-1)
    return lp.gather(-1, targets_c.unsqueeze(-1)).squeeze(-1)[0]  # [C]


def prompt_ids_for(tok, messages):
    return tok.apply_chat_template(messages, add_generation_prompt=True,
                                   return_tensors="pt", return_dict=True)["input_ids"].to("cuda")


def render_prompt_text(tok, messages):
    """The exact string the model is fed (chat template applied, with the generation
    prompt appended) — so the dumped prompt corroborates the scored token positions."""
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def code_block_token_index(tok, completion_ids):
    """Token index where the ```python (or ```) code fence starts, for a marker line."""
    text = tok.decode(completion_ids[0], skip_special_tokens=True)
    m = re.search(r"```", text)
    if not m:
        return None
    # map char offset -> token index by re-encoding the prefix
    prefix = text[: m.start()]
    pref_ids = tok(prefix, add_special_tokens=False)["input_ids"]
    return min(len(pref_ids), completion_ids.shape[1] - 1)


def analyze_difficulty(model, tok, diff, pid, language, system_text, args):
    pmap = CPP_PROMPT_BY_ID if language == "cpp" else PROMPT_BY_ID
    messages = build_messages(pmap[pid], system_text)
    print(f"[{diff}] loj-{pid} ({language}) — sampling {args.num_generations} rollouts…",
          flush=True)
    comp_ids, texts = generate_group(model, tok, messages, args.num_generations,
                                     args.max_new_tokens, args.temperature, args.top_p,
                                     args.seed)

    mb, mc = reward_case_caps()
    judged = []
    for k, text in enumerate(texts):
        r, v, fb = judge_completion(text, int(pid), which="public",
                                    language=language, reward_mode="fraction",
                                    max_case_bytes=mb, max_cases=mc)
        judged.append({"reward": r, "verdict": v, "feedback": fb})
        print(f"    rollout {k}: {v} reward={r:.2f} ({len(comp_ids[k][0])} tok)", flush=True)

    # --- replicate our gating to choose the teacher context c -------------------
    group_has_success = any(j["reward"] >= 1.0 for j in judged)
    ac_idx = next((k for k, j in enumerate(judged) if j["reward"] >= 1.0), None)
    # visualize a NON-AC rollout (the AC one gets no reprompt -> A_t≈0); pick the
    # highest partial reward (the most informative near-miss). Fall back to rollout 0.
    cand = [k for k in range(len(judged)) if k != ac_idx]
    target = max(cand, key=lambda k: judged[k]["reward"]) if cand else 0

    has_solution, use_feedback = SP.decide_inputs(
        group_has_success=group_has_success,
        is_self_demo=(target == ac_idx),
        feedback_available=bool(judged[target]["feedback"]),
        use_successful_as_teacher=True,
        dont_reprompt_on_self_success=True,
        include_environment_feedback=True,
        environment_feedback_only_without_solution=True,
    )
    demo = texts[ac_idx] if (has_solution and ac_idx is not None) else None
    fb_raw = judged[target]["feedback"] if use_feedback else None
    ctx_kind = ("solution(AC group-mate)" if has_solution
                else "feedback(judge)" if use_feedback else "none(no reprompt)")
    print(f"    -> group_has_success={group_has_success}; visualizing rollout "
          f"{target} ({judged[target]['verdict']}, reward={judged[target]['reward']:.2f}); "
          f"context c = {ctx_kind}", flush=True)

    teacher_msgs = SP.build_teacher_messages(messages, demo_text=demo, feedback_raw=fb_raw)
    student_ids = prompt_ids_for(tok, messages)
    teacher_ids = prompt_ids_for(tok, teacher_msgs)
    ct = comp_ids[target]

    student_lp = completion_token_logprobs(model, student_ids, ct)
    teacher_lp = completion_token_logprobs(model, teacher_ids, ct)
    adv = (teacher_lp - student_lp).cpu().tolist()

    # per-rollout group summary (lets the reader see why this context was chosen)
    group = [{"idx": k, "verdict": judged[k]["verdict"], "reward": judged[k]["reward"],
              "n_tok": int(comp_ids[k].shape[1]),
              "is_target": k == target, "is_ac_demo": k == ac_idx and has_solution}
             for k in range(len(judged))]

    return {
        "difficulty": diff, "id": pid, "language": language,
        "verdict": judged[target]["verdict"], "reward": judged[target]["reward"],
        "target_idx": target, "ac_demo_idx": (ac_idx if has_solution else None),
        "group_has_success": group_has_success, "context_kind": ctx_kind,
        "context_text": (demo if demo else fb_raw) or "",
        "n_tokens": len(adv), "advantages": adv,
        "code_fence_idx": code_block_token_index(tok, ct),
        "feedback_all": [j["feedback"] for j in judged],
        # full prompt + completion for corroboration with the graph (chat-template
        # applied = exactly what the model is fed). The graph's token positions index
        # into `completion`; teacher and student score that SAME completion.
        "student_prompt": render_prompt_text(tok, messages),
        "teacher_prompt": render_prompt_text(tok, teacher_msgs),
        "completion": texts[target],
        "group": group,
    }


def plot(results, out_png):
    n = len(results)
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n + 0.6), squeeze=False)
    for ax, r in zip(axes[:, 0], results):
        adv = r["advantages"]
        x = list(range(len(adv)))
        pos = [a if a > 0 else 0 for a in adv]
        neg = [a if a < 0 else 0 for a in adv]
        ax.fill_between(x, pos, color="#d62728", linewidth=0)   # red = reinforce
        ax.fill_between(x, neg, color="#1f77b4", linewidth=0)   # blue = suppress
        ax.axhline(0, color="black", linewidth=0.6)
        if r.get("code_fence_idx") is not None:
            ax.axvline(r["code_fence_idx"], color="black", linestyle="--", linewidth=0.9)
            ax.text(r["code_fence_idx"], ax.get_ylim()[1], " code starts",
                    va="top", ha="left", fontsize=8)
        ax.set_ylabel(f"{r['difficulty']}\nA_t (nats)")
        ax.set_title(
            f"loj-{r['id']} [{r['language']}] — {r['verdict']} "
            f"(reward={r['reward']:.2f}) — context c = {r['context_kind']}",
            fontsize=10)
        ax.margins(x=0.005)
    axes[-1, 0].set_xlabel("token position in rollout")
    fig.suptitle("Per-token SDPO advantage  A_t = log π(y|x,c) − log π(y|x)   "
                 "(red: keep · blue: change)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_png, dpi=130)
    print(f"wrote {out_png}", flush=True)


def write_cases_md(meta, results, out_md):
    """Human-readable per-case dump: group summary + the exact STUDENT prompt,
    TEACHER prompt, and the scored COMPLETION, so the graph can be corroborated."""
    L = [f"# Per-token advantage — prompts & completions",
         "",
         f"Model `{meta['model']}` · system `{meta['system']}` · language `{meta['language']}`.",
         "",
         "For each difficulty: the **student prompt** (question only), the **teacher prompt** "
         "(question + privileged context `c`, assembled by our SDPO gating), and the **completion** "
         "`ŷ` that both score. The figure's per-token advantage "
         "`A_t = log π(ŷ_t | teacher_prompt) − log π(ŷ_t | student_prompt)` indexes into that "
         "completion (token 0 = first token after the prompt). The teacher and student prompts "
         "differ **only** by the inserted context `c`; the completion is identical.",
         ""]
    for r in results:
        L += [f"---", "",
              f"## {r['difficulty'].upper()} — loj-{r['id']} [{r['language']}]", "",
              f"- **Visualized rollout:** #{r['target_idx']} — **{r['verdict']}** "
              f"(reward {r['reward']:.2f}), {r['n_tokens']} tokens",
              f"- **Context `c`:** {r['context_kind']}"
              + (f" (AC demo = rollout #{r['ac_demo_idx']})" if r['ac_demo_idx'] is not None else ""),
              f"- **Code fence starts at token:** {r['code_fence_idx']}",
              "",
              "**Group rollouts (why this context):**", "",
              "| # | verdict | reward | tokens | role |",
              "|---|---|---|---|---|"]
        for g in r["group"]:
            role = "← visualized" if g["is_target"] else ("AC demo (context)" if g["is_ac_demo"] else "")
            L.append(f"| {g['idx']} | {g['verdict']} | {g['reward']:.2f} | {g['n_tok']} | {role} |")
        L += ["",
              "### Student prompt (question only)", "", "````text", r["student_prompt"], "````", "",
              "### Teacher prompt (question + context `c`)", "", "````text", r["teacher_prompt"], "````", "",
              "### Completion ŷ (scored by both; graph x-axis indexes these tokens)", "",
              "````text", r["completion"], "````", ""]
    out_md.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out_md}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--difficulties", default="easy,medium,hard")
    ap.add_argument("--language", default="python", choices=["python", "cpp"])
    ap.add_argument("--system", default="cp_method", choices=["cp_method", "expert", "none"])
    ap.add_argument("--ids", default=None, help="comma list of explicit pids (overrides pick)")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-prefix", default="token_advantage")
    ap.add_argument("--smoke", action="store_true",
                    help="1 difficulty, G=2, 256 tokens — wire test")
    args = ap.parse_args()

    if args.smoke:
        args.difficulties = args.difficulties.split(",")[0]
        args.num_generations, args.max_new_tokens = 2, 256

    difficulties = args.difficulties.split(",")
    explicit = [int(x) for x in args.ids.split(",")] if args.ids else None
    chosen = pick_ids(difficulties, args.language, explicit)
    missing = [d for d in difficulties if d not in chosen]
    if missing:
        print(f"WARN: no {args.language} problem found for difficulties {missing}")

    system_text = SYSTEM_PROMPTS[args.system]
    print(f"loading {args.model}…", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()

    t0 = time.perf_counter()
    results = []
    for diff in difficulties:
        if diff not in chosen:
            continue
        results.append(analyze_difficulty(model, tok, diff, chosen[diff],
                                           args.language, system_text, args))

    meta = {"model": args.model, "system": args.system, "language": args.language}
    out_json = ROOT / f"{args.out_prefix}.json"
    json.dump({**meta, "results": results}, open(out_json, "w"), indent=2)
    print(f"wrote {out_json}", flush=True)
    write_cases_md(meta, results, ROOT / f"{args.out_prefix}_cases.md")
    plot(results, ROOT / f"{args.out_prefix}.png")
    print(f"done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
