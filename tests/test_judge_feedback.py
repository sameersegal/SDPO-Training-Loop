"""Judge + feedback edge cases: AC / WA / RE / TLE / CE / NO_CODE.

Uses synthetic (input, expected-output) cases written to temp files, so tests are fast,
deterministic, and don't need real OJBench data. Exercises both the verdict logic and the
feedback text that becomes the SDPO teacher signal.
"""
import pytest

from ojbench_eval import judge_solution
from sdpo_ojbench import judge_cpp, _format_feedback


def cases(tmp_path, pairs):
    """pairs = [(stdin, expected_stdout), ...] -> [(in_path, out_path), ...]."""
    out = []
    for i, (inp, exp) in enumerate(pairs):
        ip, op = tmp_path / f"{i}.in", tmp_path / f"{i}.out"
        ip.write_text(inp)
        op.write_text(exp)
        out.append((ip, op))
    return out


# --- _format_feedback: the text the teacher sees, per verdict ----------------
def test_feedback_ac():
    assert _format_feedback("AC", {}) == "All public tests passed."


def test_feedback_no_code_explains_format():
    txt = _format_feedback("NO_CODE", {})
    assert "did not contain a Python code block" in txt


def test_feedback_wa_leaks_expected_and_got():
    txt = _format_feedback("WA", {"failing_case": "2.in", "expected": "7", "got": "12"})
    assert "Verdict: WA." in txt
    assert "Failing test '2.in'." in txt
    assert "Expected output:\n7" in txt and "Your output:\n12" in txt


def test_feedback_re_includes_stderr():
    txt = _format_feedback("RE", {"failing_case": "1.in", "stderr": "ZeroDivisionError"})
    assert "Runtime error:\nZeroDivisionError" in txt


def test_feedback_tle_message():
    txt = _format_feedback("TLE", {"failing_case": "3.in"})
    assert "exceeded the time limit" in txt


def test_feedback_pass_rate_line_in_fraction_mode():
    # dense mode passes passed/total -> teacher sees how close the attempt was
    txt = _format_feedback("WA", {"failing_case": "2.in", "expected": "7", "got": "12"},
                           passed=16, total=20)
    assert "Passed 16/20 tests (80%)." in txt
    # verdict line first, pass-rate right after it
    assert txt.index("Verdict: WA.") < txt.index("Passed 16/20")


def test_feedback_no_pass_rate_when_omitted():
    # binary mode omits passed/total -> no (misleading) percentage
    txt = _format_feedback("WA", {"failing_case": "2.in", "expected": "7", "got": "12"})
    assert "Passed" not in txt and "%)" not in txt


# --- judge_solution (Python) end-to-end --------------------------------------
SUM = "a,b=map(int,input().split())\nprint(a+b)\n"


def test_python_ac(tmp_path):
    verdict, passed, total, _ = judge_solution(SUM, cases(tmp_path, [("3 4", "7"), ("10 20", "30")]), timeout=5)
    assert verdict == "AC" and passed == total == 2


def test_python_wa_reports_expected_and_got(tmp_path):
    wrong = "a,b=map(int,input().split())\nprint(a*b)\n"
    verdict, _, _, detail = judge_solution(wrong, cases(tmp_path, [("3 4", "7")]), timeout=5)
    assert verdict == "WA"
    assert detail["expected"] == "7" and detail["got"] == "12"
    # the feedback the teacher would receive
    fb = _format_feedback(verdict, detail)
    assert "Expected output:\n7" in fb and "Your output:\n12" in fb


def test_python_runtime_error(tmp_path):
    boom = "raise ValueError('boom')\n"
    verdict, _, _, detail = judge_solution(boom, cases(tmp_path, [("1", "1")]), timeout=5)
    assert verdict == "RE" and "boom" in detail["stderr"]


def test_python_tle(tmp_path):
    loop = "while True:\n    pass\n"
    verdict, _, _, detail = judge_solution(loop, cases(tmp_path, [("1", "1")]), timeout=1)
    assert verdict == "TLE" and "failing_case" in detail


def test_python_no_code():
    verdict, _, _, detail = judge_solution(None, [], timeout=5)
    assert verdict == "NO_CODE"


# --- judge_cpp: AC + compile error -------------------------------------------
CPP_SUM = "#include <iostream>\nint main(){long a,b;std::cin>>a>>b;std::cout<<a+b;}\n"


def test_cpp_ac(tmp_path):
    verdict, passed, total, _ = judge_cpp(CPP_SUM, cases(tmp_path, [("3 4", "7")]), timeout=5)
    assert verdict == "AC" and passed == total == 1


def test_cpp_compile_error(tmp_path):
    bad = "int main(){ this is not c++ }\n"
    verdict, _, _, detail = judge_cpp(bad, cases(tmp_path, [("1", "1")]), timeout=5)
    assert verdict == "CE" and "stderr" in detail


# --- B: judge smallest-input-first -> interpretable failure feedback ----------
def test_smallest_failing_case_is_surfaced(tmp_path):
    # huge failing case listed FIRST, tiny failing case listed second; both WA
    (tmp_path / "small.in").write_text("1 1")
    (tmp_path / "small.out").write_text("999")
    (tmp_path / "big.in").write_text("1 1\n" + "0\n" * 20000)  # ~40 KB
    (tmp_path / "big.out").write_text("999")
    ordered = [(tmp_path / "big.in", tmp_path / "big.out"),       # huge listed first
               (tmp_path / "small.in", tmp_path / "small.out")]
    verdict, _, _, detail = judge_solution(SUM, ordered, timeout=5)  # SUM prints 2 -> both WA
    assert verdict == "WA"
    assert detail["failing_case"] == "small.in"   # smallest-first, not "big.in"
    assert detail["input"] == "1 1"               # small, interpretable feedback


# --- reward modes: count_all -> true passed/total dense signal ----------------
def test_count_all_counts_true_passes(tmp_path):
    # smallest case FAILS, two larger cases PASS
    cs = cases(tmp_path, [("1 1", "999"), ("10 20", "30"), ("100 200", "300")])
    v, p, t, d = judge_solution(SUM, cs, timeout=5, count_all=True)
    assert v == "WA" and p == 2 and t == 3 and d["failing_case"].endswith(".in")  # true 2/3
    # default early-exit fails on the smallest first -> prefix passed = 0
    v2, p2, _, _ = judge_solution(SUM, cs, timeout=5)
    assert v2 == "WA" and p2 == 0
