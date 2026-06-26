"""Tests for the empty-Example worked-example hint (src/sdpo_ojbench.py)."""
import sdpo_ojbench as S

EMPTY = "### Problem\nDo X.\n\n### Example\n\n### Constraints\nN<=10\n"
FILLED = "### Problem\nDo X.\n\n### Example\n\nInput: 1\nOutput: 1\n\n### Constraints\nN<=10\n"
NO_HDR = "### Problem\nDo X.\n\n### Constraints\nN<=10\n"


# --- detection ---------------------------------------------------------------
def test_empty_example_detected():
    assert S.example_section_is_empty(EMPTY)


def test_filled_example_not_empty():
    assert not S.example_section_is_empty(FILLED)


def test_no_example_header_not_empty():
    assert not S.example_section_is_empty(NO_HDR)


def test_example_at_end_of_prompt():
    assert S.example_section_is_empty("### Problem\nDo X.\n\n### Example\n")


# --- injection (pure) --------------------------------------------------------
def test_inject_fills_empty_section():
    out = S.inject_example(EMPTY, "5 8 7 3 4", "7470")
    assert "Input:" in out and "5 8 7 3 4" in out and "7470" in out
    assert "### Constraints" in out  # next header preserved, not clobbered


def test_inject_noop_when_not_empty():
    assert S.inject_example(FILLED, "x", "y") == FILLED
    assert S.inject_example(NO_HDR, "x", "y") == NO_HDR


def test_inject_handles_braces_and_percent():
    out = S.inject_example(EMPTY, "{n} 100%", "result {x}")
    assert "{n} 100%" in out and "result {x}" in out  # never str.format'd


# --- integration on a real problem (loj-2599, known empty Example) ------------
def test_augment_real_problem_injects_smallest_case():
    pid = 2599
    base = S.PROMPT_BY_ID[pid]
    assert S.example_section_is_empty(base)  # precondition
    aug = S.augment_prompt_with_example(base, pid, which="public")
    assert aug != base
    assert "5 8 7 3 4" in aug and "7470" in aug  # the smallest public case


def test_augment_noop_when_input_too_large():
    pid = 2599
    base = S.PROMPT_BY_ID[pid]
    # tiny cap -> even the smallest case is "too large" -> no injection
    assert S.augment_prompt_with_example(base, pid, max_input_bytes=1) == base
