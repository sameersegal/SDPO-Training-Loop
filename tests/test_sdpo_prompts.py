"""Unit tests for the de-abstracted SDPO teacher-prompt construction (src/sdpo_prompts.py)."""
import pytest

import sdpo_prompts as P

Q = "Print the sum of two integers read from stdin."
DEMO = "```python\na,b=map(int,input().split());print(a+b)\n```"
FB = "Verdict: WA.\nExpected output:\n7\nYour output:\n12"


# --- slot formatting ---------------------------------------------------------
def test_format_solution_uses_template():
    assert P.format_solution("X") == "\nCorrect solution:\n\nX\n\n"


def test_format_feedback_uses_template():
    assert P.format_feedback("BAD") == "\nThe following is feedback from your unsuccessful earlier attempt:\n\nBAD\n\n"


def test_build_reprompt_default_assembly():
    out = P.build_reprompt("P", "S", "F")
    assert out == "PSF\n\nCorrectly solve the original question.\n"


def test_build_reprompt_custom_template():
    out = P.build_reprompt("P", "S", "F", reprompt_template="<{prompt}|{solution}|{feedback}>")
    assert out == "<P|S|F>"


# --- braces/% in content must not be re-interpreted by .format ---------------
def test_content_with_braces_is_literal():
    weird = "use {n} and {x} and 100% here"
    out = P.build_reprompt(weird, P.format_solution(weird), P.format_feedback(weird))
    assert weird in out  # appears (3x), never raises KeyError/IndexError


# --- full teacher messages, the four gating cases ----------------------------
def test_copy_only_has_solution_no_feedback():
    msgs = P.build_teacher_messages([{"role": "user", "content": Q}], demo_text=DEMO, feedback_raw=None)
    content = msgs[-1]["content"]
    assert "Correct solution:" in content and DEMO in content
    assert "feedback from your unsuccessful" not in content
    assert content.endswith("Correctly solve the original question.\n")


def test_feedback_only_no_solution():
    msgs = P.build_teacher_messages([{"role": "user", "content": Q}], demo_text=None, feedback_raw=FB)
    content = msgs[-1]["content"]
    assert "feedback from your unsuccessful" in content and FB in content
    assert "Correct solution:" not in content


def test_solution_and_feedback_both():
    msgs = P.build_teacher_messages([{"role": "user", "content": Q}], demo_text=DEMO, feedback_raw=FB)
    content = msgs[-1]["content"]
    assert "Correct solution:" in content and "feedback from your unsuccessful" in content


def test_neither_returns_original_prompt_unchanged():
    original = [{"role": "user", "content": Q}]
    out = P.build_teacher_messages(original, demo_text=None, feedback_raw=None)
    assert out is original  # no reprompt constructed


def test_preserves_system_messages():
    convo = [{"role": "system", "content": "sys"}, {"role": "user", "content": Q}]
    msgs = P.build_teacher_messages(convo, demo_text=DEMO)
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[-1]["role"] == "user"


def test_string_prompt_returns_string():
    out = P.build_teacher_messages(Q, demo_text=DEMO)
    assert isinstance(out, str) and Q in out


def test_last_message_must_be_user():
    with pytest.raises(ValueError):
        P.build_teacher_messages([{"role": "assistant", "content": "x"}], demo_text=DEMO)


# --- gating logic (decide_inputs) -------------------------------------------
def test_gating_copy_only_default():
    # iteration-01 default: include_environment_feedback=False -> feedback never used
    has_sol, use_fb = P.decide_inputs(group_has_success=True, is_self_demo=False, feedback_available=True)
    assert has_sol and not use_fb


def test_gating_dont_reprompt_on_self_success():
    has_sol, _ = P.decide_inputs(group_has_success=True, is_self_demo=True, feedback_available=False)
    assert not has_sol  # the AC rollout itself is not reprompted with its own solution


def test_gating_live_feedback_enabled():
    has_sol, use_fb = P.decide_inputs(
        group_has_success=False, is_self_demo=False, feedback_available=True,
        include_environment_feedback=True)
    assert not has_sol and use_fb  # all-fail group still gets a teacher via feedback


def test_gating_feedback_only_without_solution():
    # when a solution exists AND environment_feedback_only_without_solution=True, feedback is suppressed
    has_sol, use_fb = P.decide_inputs(
        group_has_success=True, is_self_demo=False, feedback_available=True,
        include_environment_feedback=True, environment_feedback_only_without_solution=True)
    assert has_sol and not use_fb


# --- parity with the actual TRL library (the source of truth) ----------------
def test_matches_trl_library():
    trl = pytest.importorskip("trl.experimental.sdpo")
    from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder

    cfg = trl.SDPOConfig(output_dir="/tmp/_sdpo_test")
    # our copied defaults must equal the library's
    assert P.DEFAULT_REPROMPT_TEMPLATE == cfg.reprompt_template
    assert P.DEFAULT_SOLUTION_TEMPLATE == cfg.solution_template
    assert P.DEFAULT_FEEDBACK_TEMPLATE == cfg.feedback_template

    class _Stub:
        args = cfg
    builder = SuccessfulRolloutTeacherContextBuilder(_Stub())
    sol = cfg.solution_template.format(successful_previous_attempt=DEMO)
    fbk = cfg.feedback_template.format(feedback_raw=FB)
    assert P.build_reprompt(Q, sol, fbk) == builder._build_reprompt_text(Q, sol, fbk)
