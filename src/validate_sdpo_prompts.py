"""Validate that src/sdpo_prompts.py matches TRL's real SDPO prompt construction.

Checks, against the installed `trl.experimental.sdpo` (not a copy):
  1. our DEFAULT_* templates == SDPOConfig() defaults            (catches template drift)
  2. our build_reprompt()      == builder._build_reprompt_text() (the library's own method)
  3. our format_solution/feedback == args.<tpl>.format(...)      (the library's formatting)
  4. custom templates are honoured identically by both
Then prints the rendered teacher prompts for the four gating cases (transparency).

  PYTHONPATH=src python src/validate_sdpo_prompts.py
"""
from trl.experimental.sdpo import SDPOConfig
from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder

import sdpo_prompts as P


class _StubTrainer:
    """Minimal trainer with just `.args` — enough to call _build_reprompt_text."""
    def __init__(self, args):
        self.args = args


def _trl_builder(cfg):
    return SuccessfulRolloutTeacherContextBuilder(_StubTrainer(cfg))


def main():
    cfg = SDPOConfig(output_dir="/tmp/_sdpo_validate")
    fails = []

    # 1. our copied defaults == the library's defaults
    for ours, theirs, name in [
        (P.DEFAULT_REPROMPT_TEMPLATE, cfg.reprompt_template, "reprompt_template"),
        (P.DEFAULT_SOLUTION_TEMPLATE, cfg.solution_template, "solution_template"),
        (P.DEFAULT_FEEDBACK_TEMPLATE, cfg.feedback_template, "feedback_template"),
    ]:
        ok = ours == theirs
        print(f"[{'ok' if ok else 'FAIL'}] default {name} matches library")
        if not ok:
            fails.append(f"{name}: ours={ours!r} != trl={theirs!r}")

    # 2 & 3. our functions reproduce the library's method/format output, default + custom templates
    samples = [
        ("Sum two ints from stdin.", "```python\nprint(sum(map(int,input().split())))\n```",
         "Verdict: WA.\nExpected:\n7\nGot:\n12"),
        ("Edge case {braces} and %s in text", "", ""),  # ensure no accidental .format on content
    ]
    configs = {
        "default": cfg,
        "custom": SDPOConfig(
            output_dir="/tmp/_sdpo_validate2",
            reprompt_template="### Q: {prompt}{solution}{feedback}\n>>> Solve it.\n",
            solution_template="\n[demo]\n{successful_previous_attempt}\n",
            feedback_template="\n[judge] {feedback_raw}\n",
        ),
    }
    for cname, c in configs.items():
        builder = _trl_builder(c)
        for prompt_text, demo, fb in samples:
            sol = P.format_solution(demo, solution_template=c.solution_template) if demo else ""
            fbk = P.format_feedback(fb, feedback_template=c.feedback_template) if fb else ""
            # library formatting of the slots
            trl_sol = c.solution_template.format(successful_previous_attempt=demo) if demo else ""
            trl_fbk = c.feedback_template.format(feedback_raw=fb) if fb else ""
            # library's reprompt method vs ours
            trl_out = builder._build_reprompt_text(prompt_text, trl_sol, trl_fbk)
            our_out = P.build_reprompt(prompt_text, sol, fbk, reprompt_template=c.reprompt_template)
            ok = (sol == trl_sol) and (fbk == trl_fbk) and (our_out == trl_out)
            print(f"[{'ok' if ok else 'FAIL'}] {cname} reprompt matches library  (prompt={prompt_text[:24]!r})")
            if not ok:
                fails.append(f"{cname}/{prompt_text[:20]}: our={our_out!r} != trl={trl_out!r}")

    print("\n" + "=" * 70)
    if fails:
        print("VALIDATION FAILED:")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print("VALIDATION PASSED — src/sdpo_prompts.py matches trl.experimental.sdpo exactly.")

    # Transparency: the four gating cases, rendered with the LIBRARY's default templates.
    print("\n--- rendered teacher prompts (default templates) ---")
    q = "Print the sum of two integers read from stdin."
    demo = "```python\na,b=map(int,input().split());print(a+b)\n```"
    fb = "Verdict: WA.\nExpected output:\n7\nYour output:\n12"
    for name, kw in {
        "copy-only (iter-01)": dict(demo_text=demo, feedback_raw=None),
        "feedback-only (all-fail group)": dict(demo_text=None, feedback_raw=fb),
        "solution + feedback": dict(demo_text=demo, feedback_raw=fb),
    }.items():
        msgs = P.build_teacher_messages([{"role": "user", "content": q}], **kw)
        print(f"\n[{name}]\n{msgs[-1]['content'] if isinstance(msgs, list) else msgs}")


if __name__ == "__main__":
    main()
