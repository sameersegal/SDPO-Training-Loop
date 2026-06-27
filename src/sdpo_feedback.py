"""Live judge feedback for SDPO (iteration 02).

TRL's SDPOTrainer feeds the teacher reprompt from a STATIC `privileged_context`
dataset column (`sdpo_trainer.py:835`, `feedbacks=privileged_contexts`). We want
the teacher to see **per-rollout judge text** ("Verdict: WA, expected … got …") so
it learns to *correct failures*, not just copy successes (iteration 01 was copy-only
and collapsed). See `src/sdpo_prompts.py` for the validated prompt assembly and
`reports/iteration-02/REPORT.md` for the rationale.

Mechanism (no framework internals copied — single-process / single-GPU):
  - the reward function already calls judge_completion -> (reward, verdict, feedback);
    we capture the per-rollout feedback into a shared FeedbackBus.
  - a thin teacher-context-builder subclass substitutes bus.feedbacks for the static
    feedbacks argument in build(). The reward function runs (line 844) before build()
    (line 874) in the same _prepare_training_batch call, so the bus is fresh.
  - set include_environment_feedback=True in SDPOConfig so build() actually uses it.
"""
from trl.experimental.sdpo import SDPOTrainer
from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder

from sdpo_ojbench import judge_completion


class FeedbackBus:
    """Carries per-rollout feedback from the reward function to the teacher builder."""
    def __init__(self):
        self.feedbacks = []


def make_feedback_reward_func(bus, which="public", timeout=6.0, reward_mode="fraction"):
    """Like make_reward_func, but also stashes per-rollout judge feedback on the bus.

    Dense (fraction) judging runs EVERY public case with no early-exit, so judging is
    the per-step bottleneck on hard problems (a TLE rollout pays timeout x cases).
    judge_completion is subprocess-based (releases the GIL), so we judge the group's
    completions CONCURRENTLY in a thread pool. ex.map preserves order, so rewards[k] /
    feedbacks[k] still line up with completion_ids[k] for the teacher builder.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from sdpo_ojbench import reward_case_caps
    workers = int(os.environ.get("SDPO_JUDGE_WORKERS", "16"))
    mb, mc = reward_case_caps()  # cap in-loop cases (huge test sets stall/hang a step)

    def reward_func(completions, id=None, language=None, **kwargs):
        def judge_k(k):
            comp = completions[k]
            lang = language[k] if language is not None else "python"
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            r, _, fb = judge_completion(text, int(id[k]), which=which, timeout=timeout,
                                        language=lang, reward_mode=reward_mode,
                                        max_case_bytes=mb, max_cases=mc)
            return float(r), fb

        n = len(completions)
        with ThreadPoolExecutor(max_workers=min(workers, max(1, n))) as ex:
            results = list(ex.map(judge_k, range(n)))  # ex.map preserves input order
        bus.feedbacks = [fb for _, fb in results]
        return [r for r, _ in results]
    reward_func.__name__ = f"ojbench_{which}_reward_fb"
    return reward_func


class _LiveFeedbackBuilder(SuccessfulRolloutTeacherContextBuilder):
    def __init__(self, trainer, bus):
        super().__init__(trainer)
        self.bus = bus

    def build(self, output, prompts, rewards, feedbacks=None):
        fb = self.bus.feedbacks
        n = output["completion_ids"].shape[0]
        # local feedbacks must align 1:1 with this process's completions; else fall back
        use = fb if len(fb) == n else feedbacks
        return super().build(output, prompts, rewards, feedbacks=use)


class FeedbackSDPOTrainer(SDPOTrainer):
    """SDPOTrainer that conditions the teacher on live per-rollout judge feedback.
    Pass the SAME FeedbackBus used to build the reward function."""
    def __init__(self, *args, feedback_bus, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_context_builder = _LiveFeedbackBuilder(self, feedback_bus)
