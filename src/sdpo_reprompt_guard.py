"""Reprompt-truncation guard for TRL's SDPO teacher-context builder.

THE BUG THIS GUARDS AGAINST (iter-01..10): `SuccessfulRolloutTeacherContextBuilder.
_tokenize_teacher_messages` silently LEFT-truncates every teacher prompt to the last
`max_reprompt_len` tokens (`ids[-N:]`, sdpo_trainer.py:162) — applied AFTER the chat
template. When the prompt overflows (a demonstration carrying a sibling's full <think>
block is 10-17k tokens on OJBench), what gets cut is the HEAD: BOS, chat-turn markers,
the system prompt, and the PROBLEM STATEMENT. The teacher then re-scores the student's
rollout against a malformed context that never states the problem — in iter-10, ~100%
of early-step solution-teacher contexts were malformed this way, and NOTHING logged it.

This module monkeypatches the builder (base class — `_LiveFeedbackBuilder` inherits) to:
  1. measure pre-truncation teacher-prompt lengths every step,
  2. merge them into `builder.last_metrics` (the trainer already forwards those to
     W&B / logs at sdpo_trainer.py:881), and
  3. print a LOUD warning the moment any teacher prompt is truncated.

Metrics added (per step, under the trainer's usual prefixes):
  self_distillation/reprompt_truncated_frac   fraction of teacher prompts over the cap
  self_distillation/reprompt_len_max          longest teacher prompt (tokens, pre-cut)
  self_distillation/reprompt_overflow_max     worst overflow past max_reprompt_len

SECOND, SAME-CLASS HAZARD (replication audit, 2026-07): the trainer-level
`SDPOTrainer._tokenize_prompts` (sdpo_trainer.py:981-985) left-truncates the STUDENT
prompt ids to `max_prompt_length` (`ids[-N:]`) — the identical head-cutting bug class.
When a student prompt (system + PROBLEM STATEMENT) overflows max_prompt_length the cut
removes BOS/system/problem the same way. We wrap it with the same measure-before-truncate
pattern, logging into the trainer's own per-step `_metrics[mode]`:
  prompts/student_prompt_truncated_frac       fraction of student prompts over max_prompt_length
  prompts/student_prompt_len_max              longest student prompt (tokens, pre-cut)

Usage (sdpo_train.py does this at startup):
    from sdpo_reprompt_guard import install_reprompt_guard
    install_reprompt_guard()

The pre-flight expectation for any healthy run: reprompt_truncated_frac == 0.
"""

from trl.experimental.sdpo.sdpo_trainer import (
    SuccessfulRolloutTeacherContextBuilder as _Builder,
    SDPOTrainer as _Trainer,
)

_orig_tokenize = _Builder._tokenize_teacher_messages
_orig_build = _Builder.build
_orig_tokenize_prompts = _Trainer._tokenize_prompts


def install_reprompt_guard():
    """Idempotently instrument the teacher-context builder with truncation metrics."""
    if getattr(_Builder, "_reprompt_guard_installed", False):
        return

    def _tokenize_teacher_messages(self, teacher_messages_list):
        # Measure BEFORE the original applies ids[-max_reprompt_len:]. The extra
        # tokenization pass is a few ms per step — nothing next to generation.
        ids_list = self.trainer._tokenize_prompts_untruncated(teacher_messages_list)
        cap = self.trainer.args.max_reprompt_len
        lens = [len(ids) for ids in ids_list]
        overflow = [max(0, n - cap) for n in lens]
        n_trunc = sum(1 for o in overflow if o > 0)
        self._reprompt_guard_stats = {
            "self_distillation/reprompt_truncated_frac": n_trunc / max(1, len(lens)),
            "self_distillation/reprompt_len_max": float(max(lens, default=0)),
            "self_distillation/reprompt_overflow_max": float(max(overflow, default=0)),
        }
        if n_trunc:
            print(
                f"[reprompt-guard] WARNING: {n_trunc}/{len(lens)} teacher prompts exceed "
                f"max_reprompt_len={cap} (longest {max(lens)} tok, worst overflow "
                f"{max(overflow)} tok). Left-truncation is cutting the HEAD — BOS/system/"
                f"PROBLEM STATEMENT — so the teacher context is malformed. Shrink the "
                f"demo (remove_thinking_from_demonstration) or feedback.",
                flush=True,
            )
        return _orig_tokenize(self, teacher_messages_list)

    def build(self, *args, **kwargs):
        out = _orig_build(self, *args, **kwargs)
        # _orig_build overwrites last_metrics at its end; merge our stats after.
        stats = getattr(self, "_reprompt_guard_stats", None)
        if stats:
            self.last_metrics.update(stats)
        return out

    def _tokenize_prompts(self, prompts):
        # Measure BEFORE the original applies ids[-max_prompt_length:] to STUDENT prompts.
        # Same head-cutting hazard as the teacher-reprompt bug: an over-length student
        # prompt loses BOS/system/PROBLEM. Log into the trainer's own per-step _metrics.
        ids_list = self._tokenize_prompts_untruncated(prompts)
        cap = self.max_prompt_length
        if cap is not None:
            lens = [len(ids) for ids in ids_list]
            n_trunc = sum(1 for n in lens if n > cap)
            if n_trunc:
                mode = "train" if self.model.training else "eval"
                self._metrics[mode]["prompts/student_prompt_truncated_frac"].append(
                    n_trunc / max(1, len(lens))
                )
                self._metrics[mode]["prompts/student_prompt_len_max"].append(
                    float(max(lens, default=0))
                )
                print(
                    f"[reprompt-guard] WARNING: {n_trunc}/{len(lens)} STUDENT prompts exceed "
                    f"max_prompt_length={cap} (longest {max(lens)} tok). Left-truncation "
                    f"(ids[-N:]) is cutting the HEAD — BOS/system/PROBLEM STATEMENT — so the "
                    f"student sees a malformed prompt. Raise max_prompt_length or shrink the "
                    f"prompt.",
                    flush=True,
                )
        return _orig_tokenize_prompts(self, prompts)

    _Builder._tokenize_teacher_messages = _tokenize_teacher_messages
    _Builder.build = build
    _Trainer._tokenize_prompts = _tokenize_prompts
    _Builder._reprompt_guard_installed = True
    _Trainer._reprompt_guard_installed = True
    print("[reprompt-guard] installed: teacher-prompt AND student-prompt truncation are now "
          "measured (self_distillation/reprompt_truncated_frac + "
          "prompts/student_prompt_truncated_frac — expect 0)")
