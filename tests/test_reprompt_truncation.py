"""Tests for the teacher-context truncation bug (iter-01..10) and its fix.

The bug: TRL's SuccessfulRolloutTeacherContextBuilder left-truncates every teacher
prompt to the LAST max_reprompt_len tokens (ids[-N:], after the chat template). A
demonstration carrying a sibling's full <think> block overflows the cap, so the cut
removes the HEAD of the context — BOS, system prompt, and the PROBLEM STATEMENT —
and the teacher scores the student's rollout against a context with no problem in it.

The fix (iter-11): remove_thinking_from_demonstration=True (demo -> code only, fits
intact) + sdpo_reprompt_guard (truncation measured, logged, and warned about).
"""
import json
import re
from pathlib import Path

import pytest

import sdpo_prompts as P

REPO = Path(__file__).resolve().parent.parent
CAP = 8192  # max_reprompt_len in sdpo_train.py

SYSTEM = ("You are an expert competitive programmer. Follow this method, then output "
          "only the final code.")
PROBLEM_SENTINEL = "Remove all the stones from the piles"
PROBLEM = (f"### Problem Description\n\n{PROBLEM_SENTINEL}. There are n piles; "
           "operation 1 removes >=2 stones from one pile...\n### Format: read stdin, "
           "write stdout.\n```python\ndef main():\n<Your code is here>\n```")
CODE = "```python\ndef main():\n    n = int(input())\n    print(n)\n\nmain()\n```"


def _long_demo(n_words: int = 30_000) -> str:
    """A realistic overflow demo: a huge <think> block followed by code (iter-10
    demos ran 40-60k chars of thinking; 30k words tokenizes well past 8192)."""
    think = "I need to consider the parity of the stone piles carefully. " * (n_words // 10)
    return f"<think>\n{think}\n</think>\n\n{CODE}"


# --- strip_thinking mirrors TRL exactly --------------------------------------
def test_strip_thinking_removes_think_block():
    assert P.strip_thinking(_long_demo()) == CODE


def test_strip_thinking_matches_trl_regex():
    # TRL applies: re.sub(r"<think>.*?</think>", "", demo, flags=re.DOTALL).strip()
    # (sdpo_trainer.py, remove_thinking_from_demonstration). Assert byte-identity on
    # tricky inputs, and that the expression is still present in the installed TRL.
    cases = [
        _long_demo(100),
        "<think>a</think>x<think>b</think>y",   # multiple blocks
        "no think block at all",
        "<think>unclosed...",                   # unclosed -> untouched (non-greedy needs the closer)
    ]
    for text in cases:
        assert P.strip_thinking(text) == re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    import trl.experimental.sdpo.sdpo_trainer as T
    src = Path(T.__file__).read_text()
    assert r"<think>.*?</think>" in src, "TRL's demo-stripping regex changed — update strip_thinking"


# --- the bug and the fix, on the real tokenizer + chat template --------------
@pytest.fixture(scope="module")
def tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    except Exception as e:  # noqa: BLE001 — offline box / no HF cache
        pytest.skip(f"gemma tokenizer unavailable: {e}")


def _apply_template(tok, msgs):
    """Tokenize exactly as TRL's _tokenize_prompts_untruncated does (sdpo_trainer.py:969):
    batch of conversations, return_dict=True, take input_ids per conversation."""
    out = tok.apply_chat_template(
        conversation=[msgs], add_generation_prompt=True, tokenize=True, return_dict=True,
    )
    return list(out["input_ids"][0])


def _teacher_ids(tok, demo_text):
    """Replicate the trainer's path: build_teacher_messages -> chat template -> ids[-CAP:]."""
    msgs = P.build_teacher_messages(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": PROBLEM}],
        demo_text=demo_text,
    )
    ids = _apply_template(tok, msgs)
    return ids, ids[-CAP:]


def test_bug_full_demo_overflows_and_loses_problem(tokenizer):
    """Reproduces the iter-01..10 bug: an un-stripped demo overflows the cap and the
    surviving context has NO problem statement, NO system prompt, and NO BOS."""
    full_ids, kept_ids = _teacher_ids(tokenizer, _long_demo())
    assert len(full_ids) > CAP, "demo not long enough to reproduce the overflow"
    kept_text = tokenizer.decode(kept_ids, skip_special_tokens=False)
    assert PROBLEM_SENTINEL not in kept_text          # problem statement gone
    assert SYSTEM[:40] not in kept_text               # system prompt gone
    assert kept_ids[0] != full_ids[0]                 # BOS/head gone -> starts mid-text
    # what remains is the tail of the sibling's reasoning + its code + the trailer
    assert "Correctly solve the original question." in kept_text


def test_fix_code_only_demo_fits_intact(tokenizer):
    """With remove_thinking_from_demonstration semantics, the same context fits whole."""
    full_ids, kept_ids = _teacher_ids(tokenizer, P.strip_thinking(_long_demo()))
    assert len(full_ids) <= CAP                       # never truncated
    assert kept_ids == full_ids                        # ids[-CAP:] is a no-op
    kept_text = tokenizer.decode(kept_ids, skip_special_tokens=False)
    assert PROBLEM_SENTINEL in kept_text
    assert SYSTEM[:40] in kept_text
    assert CODE.strip("`\n")[:30] in kept_text


def test_fix_worst_case_real_prompt_budget(tokenizer):
    """Regression budget: longest committed OJBench prompt + a generous code demo +
    worst-case judge feedback (fields _clip'd to 600 chars each) stays well under cap."""
    splits = json.load(open(REPO / "data" / "ojb_splits.json"))
    longest = max(splits["py_prompt_by_id"].values(), key=len)
    demo = "```python\n" + ("x = solve_case(read_input())\n" * 60) + "```"
    feedback = ("Verdict: WA. Passed 3/20 tests (15%). Failing test 'x.in'.\n"
                + "Input:\n" + "9" * 600 + "\nExpected output:\n" + "8" * 600
                + "\nYour output:\n" + "7" * 600)
    msgs = P.build_teacher_messages(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": longest}],
        demo_text=demo, feedback_raw=feedback,
    )
    ids = _apply_template(tokenizer, msgs)
    # measured worst case: ~4.4k tokens (1.9k problem + 0.7k demo + 1.8k feedback —
    # digit-heavy feedback tokenizes at ~1 tok/char). Keep >=2k headroom under the cap.
    assert len(ids) <= CAP - 2048, f"teacher prompt budget blown: {len(ids)} tokens"


# --- the guard: metrics + slicing behavior ------------------------------------
class _StubTokenizer:
    pad_token_id = 0


class _StubArgs:
    max_reprompt_len = 10


class _StubAccel:
    device = "cpu"


class _StubTrainer:
    args = _StubArgs()
    _tokenizer = _StubTokenizer()
    accelerator = _StubAccel()

    def _tokenize_prompts_untruncated(self, msgs):
        return [[7] * 15, [7] * 5]  # one over the cap of 10, one under


def test_guard_measures_truncation_and_slices_like_trl():
    from sdpo_reprompt_guard import install_reprompt_guard
    from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder

    install_reprompt_guard()
    install_reprompt_guard()  # idempotent
    b = SuccessfulRolloutTeacherContextBuilder(_StubTrainer())
    out = b._tokenize_teacher_messages(["a", "b"])

    stats = b._reprompt_guard_stats
    assert stats["self_distillation/reprompt_truncated_frac"] == pytest.approx(0.5)
    assert stats["self_distillation/reprompt_len_max"] == 15.0
    assert stats["self_distillation/reprompt_overflow_max"] == 5.0
    # original slicing behavior is preserved: both padded to the cap
    assert tuple(out["prompt_ids"].shape) == (2, 10)
    assert tuple(out["prompt_mask"].shape) == (2, 10)
    # untruncated healthy batch -> frac 0
    class _Healthy(_StubTrainer):
        def _tokenize_prompts_untruncated(self, msgs):
            return [[7] * 4, [7] * 6]
    b2 = SuccessfulRolloutTeacherContextBuilder(_Healthy())
    b2._tokenize_teacher_messages(["a", "b"])
    assert b2._reprompt_guard_stats["self_distillation/reprompt_truncated_frac"] == 0.0


# --- the student-prompt guard (replication audit, 2026-07) --------------------
from collections import defaultdict  # noqa: E402


class _StubModel:
    training = True


class _StubTrainerForPrompts:
    """Minimal SDPOTrainer stand-in exercising _tokenize_prompts's truncation path."""
    max_prompt_length = 10
    model = _StubModel()

    def __init__(self, ids_list):
        self._ids_list = ids_list
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

    def _tokenize_prompts_untruncated(self, prompts):
        return self._ids_list


def test_student_prompt_guard_measures_and_slices():
    from sdpo_reprompt_guard import install_reprompt_guard
    from trl.experimental.sdpo.sdpo_trainer import SDPOTrainer

    install_reprompt_guard()
    install_reprompt_guard()  # idempotent

    # over-length prompt present: one of two prompts exceeds cap=10
    t = _StubTrainerForPrompts([[7] * 15, [7] * 5])
    out = SDPOTrainer._tokenize_prompts(t, ["a", "b"])
    m = t._metrics["train"]
    assert m["prompts/student_prompt_truncated_frac"] == [pytest.approx(0.5)]
    assert m["prompts/student_prompt_len_max"] == [15.0]
    # original slicing preserved: ids[-cap:] applied
    assert out == [[7] * 10, [7] * 5]

    # under-length batch: no truncation -> no metrics appended, untouched ids
    t2 = _StubTrainerForPrompts([[7] * 4, [7] * 6])
    out2 = SDPOTrainer._tokenize_prompts(t2, ["a", "b"])
    assert "prompts/student_prompt_truncated_frac" not in t2._metrics["train"]
    assert out2 == [[7] * 4, [7] * 6]


# --- sdpo_train argparse defaults match paper values (replication audit) ------
def test_sdpo_train_paper_faithful_defaults():
    import sdpo_train
    ap = sdpo_train.build_parser()
    defaults = {a.dest: a.default for a in ap._actions}
    assert defaults["teacher_update_rate"] == 0.01     # paper best (TRL default 0.05)
    assert defaults["distillation_alpha"] == 1.0       # reverse KL (paper's LCBv6 RUN SCRIPT + TRL
                                                       # default; the method-section JSD=0.5 is prose-only)
    assert defaults["distillation_topk"] == 20         # LCBv6 run K=20 (pre-replication hardcoded 100)
    assert defaults["distillation_tail"] is True       # tail bucket on (TRL default False)
    assert defaults["distillation_is_clip"] == "2.0"   # preserve current behavior; "none"->None at parse


def test_sdpo_train_is_clip_none_parsing():
    import sdpo_train
    ap = sdpo_train.build_parser()
    for raw in ("none", "None", "0", "0.0"):
        args = ap.parse_args(["--distillation-is-clip", raw])
        clip = str(args.distillation_is_clip).strip().lower()
        assert (None if clip in ("none", "0", "0.0") else float(args.distillation_is_clip)) is None
    args = ap.parse_args(["--distillation-is-clip", "2.0"])
    assert float(args.distillation_is_clip) == 2.0


def test_sdpoconfig_accepts_paper_fields():
    try:
        from trl.experimental.sdpo import SDPOConfig
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"SDPOConfig import requires optional deps: {e}")
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        # distillation_mode="topk_logits" as in our real run (sdpo_train.py). alpha=0.5 here is
        # DELIBERATE (not the 1.0 default): it keeps the JSD path covered — the config's
        # __post_init__ rejects distillation_alpha!=1.0 ONLY in the "sampled_token" mode.
        cfg = SDPOConfig(
            output_dir=d,
            distillation_mode="topk_logits",
            distillation_topk=100,
            teacher_update_rate=0.01,
            distillation_alpha=0.5,
            distillation_add_tail=True,
            distillation_is_clip=None,
        )
    assert cfg.teacher_update_rate == 0.01
    assert cfg.distillation_alpha == 0.5
    assert cfg.distillation_add_tail is True
    assert cfg.distillation_is_clip is None
