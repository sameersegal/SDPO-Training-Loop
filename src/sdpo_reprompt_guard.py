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

Usage (sdpo_train.py does this at startup):
    from sdpo_reprompt_guard import install_reprompt_guard
    install_reprompt_guard()

The pre-flight expectation for any healthy run: reprompt_truncated_frac == 0.
"""

from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder as _Builder

_orig_tokenize = _Builder._tokenize_teacher_messages
_orig_build = _Builder.build


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

    _Builder._tokenize_teacher_messages = _tokenize_teacher_messages
    _Builder.build = build
    _Builder._reprompt_guard_installed = True
    print("[reprompt-guard] installed: teacher-prompt truncation is now measured "
          "(self_distillation/reprompt_truncated_frac — expect 0)")
