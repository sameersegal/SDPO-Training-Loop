"""De-abstracted TRL SDPO teacher-prompt construction.

TRL's `SDPOTrainer` builds the *teacher* prompt (the feedback-conditioned reprompt that
the EMA teacher sees, against which the student is distilled) deep inside
`SuccessfulRolloutTeacherContextBuilder.build()`. This module makes that construction
EXPLICIT and controllable, while using the *same* templates and assembly as TRL — so we
build on the library, not around it. `validate_sdpo_prompts.py` asserts byte-identity.

Reference (trl 1.6.0, .venv/.../trl/experimental/sdpo/):
  sdpo_trainer.py:835   feedbacks = [ex.get("privileged_context") for ex in inputs]   # the feedback source
  sdpo_trainer.py:281   solution_text  = args.solution_template.format(successful_previous_attempt=demo)
  sdpo_trainer.py:285   feedback_text  = args.feedback_template.format(feedback_raw=raw_feedback)
  sdpo_trainer.py:149   reprompt_text  = args.reprompt_template.format(prompt=, solution=, feedback=)
  sdpo_trainer.py:291   teacher_msgs   = system_messages + [{"role":"user","content": reprompt_text}]
  sdpo_trainer.py:296   teacher_input  = teacher_prompt_ids + the student's OWN completion ids

Assembly (per rollout):
  reprompt = reprompt_template.format(
      prompt   = <last user turn text>,
      solution = solution_template.format(successful_previous_attempt=<demo>)  if a group-mate AC'd else "",
      feedback = feedback_template.format(feedback_raw=<judge text>)           if feedback used      else "",
  )

Gating (sdpo_trainer.py:220-265), which decides solution/feedback per rollout:
  has_solution = use_successful_as_teacher and group has an AC rollout
                 (and not (dont_reprompt_on_self_success and this rollout is itself the AC demo))
  use_feedback = include_environment_feedback and feedback_text != ""
                 and (not environment_feedback_only_without_solution or not has_solution)
  if not has_solution and not use_feedback: teacher prompt == original prompt (no reprompt)

KEY for iteration 02 (live per-rollout feedback): `feedbacks` is already per-ROLLOUT length
(inputs are pre-expanded to prompt x num_generations), TRL just fills it with the *static*
`privileged_context` repeated across the group. To inject live feedback we pass a per-rollout
list (each completion's own judge verdict) and set include_environment_feedback=True — no
template/loss changes needed.
"""
from typing import Any

# Copied verbatim from trl.experimental.sdpo.SDPOConfig defaults (sdpo_config.py:554-564),
# so this module is inspectable without importing TRL. validate_sdpo_prompts.py checks they match.
DEFAULT_REPROMPT_TEMPLATE = "{prompt}{solution}{feedback}\n\nCorrectly solve the original question.\n"
DEFAULT_SOLUTION_TEMPLATE = "\nCorrect solution:\n\n{successful_previous_attempt}\n\n"
DEFAULT_FEEDBACK_TEMPLATE = "\nThe following is feedback from your unsuccessful earlier attempt:\n\n{feedback_raw}\n\n"


def format_solution(demo_text: str, *, solution_template: str = DEFAULT_SOLUTION_TEMPLATE) -> str:
    """The `solution` slot — a successful group-mate rollout (sdpo_trainer.py:281)."""
    return solution_template.format(successful_previous_attempt=demo_text)


def format_feedback(feedback_raw: str, *, feedback_template: str = DEFAULT_FEEDBACK_TEMPLATE) -> str:
    """The `feedback` slot — environment/judge text (sdpo_trainer.py:285)."""
    return feedback_template.format(feedback_raw=feedback_raw)


def build_reprompt(
    prompt_text: str,
    solution_text: str = "",
    feedback_text: str = "",
    *,
    reprompt_template: str = DEFAULT_REPROMPT_TEMPLATE,
) -> str:
    """Mirror of SuccessfulRolloutTeacherContextBuilder._build_reprompt_text (sdpo_trainer.py:149)."""
    return reprompt_template.format(prompt=prompt_text, solution=solution_text, feedback=feedback_text)


def decide_inputs(
    *,
    group_has_success: bool,
    is_self_demo: bool,
    feedback_available: bool,
    use_successful_as_teacher: bool = True,
    dont_reprompt_on_self_success: bool = True,
    include_environment_feedback: bool = False,
    environment_feedback_only_without_solution: bool = False,
) -> tuple[bool, bool]:
    """Replicates TRL's per-rollout gating (sdpo_trainer.py:220-265) -> (has_solution, use_feedback)."""
    has_solution = (
        use_successful_as_teacher
        and group_has_success
        and not (dont_reprompt_on_self_success and is_self_demo)
    )
    use_feedback = (
        include_environment_feedback
        and feedback_available
        and (not environment_feedback_only_without_solution or not has_solution)
    )
    return has_solution, use_feedback


def build_teacher_messages(
    original_prompt: Any,
    *,
    demo_text: str | None = None,
    feedback_raw: str | None = None,
    reprompt_template: str = DEFAULT_REPROMPT_TEMPLATE,
    solution_template: str = DEFAULT_SOLUTION_TEMPLATE,
    feedback_template: str = DEFAULT_FEEDBACK_TEMPLATE,
):
    """Full teacher message(s) for one rollout (sdpo_trainer.py:287-293).

    `original_prompt` is a conversational message list (gemma chat) or a plain string.
    Returns the same structure TRL appends to `local_teacher_messages`.
    """
    solution_text = format_solution(demo_text, solution_template=solution_template) if demo_text else ""
    feedback_text = format_feedback(feedback_raw, feedback_template=feedback_template) if feedback_raw else ""
    if not solution_text and not feedback_text:
        return original_prompt  # no reprompt: teacher sees the original prompt unchanged
    if isinstance(original_prompt, list):
        system_messages = original_prompt[:-1]
        prompt_text = _last_user_text(original_prompt)
        reprompt = build_reprompt(prompt_text, solution_text, feedback_text, reprompt_template=reprompt_template)
        return system_messages + [{"role": "user", "content": reprompt}]
    reprompt = build_reprompt(original_prompt, solution_text, feedback_text, reprompt_template=reprompt_template)
    return reprompt


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """Mirror of _extract_last_user_text (sdpo_trainer.py:127)."""
    last = messages[-1]
    if last.get("role") != "user":
        raise ValueError(f"expected conversation to end with a user turn, got role '{last.get('role')}'")
    content = last.get("content", "")
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return content


if __name__ == "__main__":
    # Transparency: show the actual teacher prompts for the four gating cases.
    q = "Print the sum of two integers read from stdin."
    demo = "```python\na,b=map(int,input().split());print(a+b)\n```"
    fb = "Verdict: WA.\nFailing test 'loj-1/2.in'.\nExpected output:\n7\nYour output:\n12"
    cases = {
        "copy-only (iter-01)": dict(demo_text=demo, feedback_raw=None),
        "feedback-only (all-fail group)": dict(demo_text=None, feedback_raw=fb),
        "solution + feedback": dict(demo_text=demo, feedback_raw=fb),
        "neither (no reprompt)": dict(demo_text=None, feedback_raw=None),
    }
    for name, kw in cases.items():
        print("=" * 70, f"\n[{name}]")
        msgs = build_teacher_messages([{"role": "user", "content": q}], **kw)
        print(msgs if isinstance(msgs, str) else msgs[-1]["content"])
