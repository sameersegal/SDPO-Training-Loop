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
import json
import os

from trl.experimental.sdpo import SDPOTrainer
from trl.experimental.sdpo.sdpo_trainer import SuccessfulRolloutTeacherContextBuilder

from sdpo_ojbench import judge_completion


class _RolloutLogger:
    """Append-only JSONL sink for per-rollout records (P0-1, docs/design/OBSERVABILITY.md).

    Streams one line per rollout (open+append+close each write) so a crash/OOM keeps
    everything up to the last completed write, and a --resume run just keeps appending.
    Best-effort: a logging failure must NEVER sink a training step.
    """
    def __init__(self, path):
        self.path = path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        except Exception as e:  # noqa: BLE001
            print(f"[rollout-log] mkdir failed ({e}); capture disabled")
            self.path = None

    def write_many(self, records):
        if not self.path:
            return
        try:
            with open(self.path, "a") as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"[rollout-log] write failed ({e}); skipping this step's records")


def _log_rollouts(logger, bus, completions, id, language, results, sdpo_threshold,
                  completion_ids):
    """Build per-rollout JSONL records for one generation group and append them.

    One record per rollout: text + verdict + dense/binary reward + success flag +
    teacher-eligibility (fraction >= the SDPO gate) + token/char length + step. This
    is the artifact that lets us answer "what did the completions look like as they
    shortened?" offline (length-split, diversity) without touching the loss path.
    Wrapped end-to-end so a logging failure never sinks the training step.
    """
    try:
        step = -1
        tr = getattr(bus, "trainer", None)
        if tr is not None and getattr(tr, "state", None) is not None:
            step = int(getattr(tr.state, "global_step", -1))
        tok = getattr(bus, "tokenizer", None)
        records = []
        for k, (frac, binary, fb, verdict) in enumerate(results):
            comp = completions[k]
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            pid = int(id[k]) if id is not None else -1
            lang = language[k] if language is not None else "python"
            n_tok = None
            try:
                if completion_ids is not None and k < len(completion_ids):
                    n_tok = len(completion_ids[k])
                elif tok is not None:
                    n_tok = len(tok(text, add_special_tokens=False).input_ids)
            except Exception:  # noqa: BLE001
                n_tok = None
            records.append({
                "step": step, "problem_id": pid, "language": lang, "sample_k": k,
                "verdict": verdict, "reward_fraction": float(frac),
                "reward_binary": float(binary), "success": binary == 1.0,
                "teacher_eligible": float(frac) >= sdpo_threshold,
                "n_tokens": n_tok, "n_chars": len(text),
                "feedback": fb, "completion": text,
            })
        logger.write_many(records)
    except Exception as e:  # noqa: BLE001
        print(f"[rollout-log] record build failed ({e}); skipping step")


class FeedbackBus:
    """Carries per-rollout data from the reward function to the teacher builder.

    feedbacks: per-rollout judge text / critique (teacher conditioning).
    fractions: per-rollout dense passed/total. The SDPO teacher gating reads THIS
      (near-miss rollouts can teach), while the GRPO advantage uses the binary AC
      reward the func returns — see make_feedback_reward_func / _LiveFeedbackBuilder.
    tokenizer / trainer: set by FeedbackSDPOTrainer after construction so the reward
      func can record exact completion token counts and the current global_step.
    """
    def __init__(self):
        self.feedbacks = []
        self.fractions = []
        self.tokenizer = None
        self.trainer = None


def make_feedback_reward_func(bus, which="public", timeout=6.0, reward_mode="fraction",
                              critic=False, critic_model=None, critic_thinking=False,
                              grpo_reward="fraction", rollout_log=None, sdpo_threshold=1.0):
    """Like make_reward_func, but also stashes per-rollout judge feedback on the bus.

    Dense (fraction) judging runs EVERY public case with no early-exit, so judging is
    the per-step bottleneck on hard problems (a TLE rollout pays timeout x cases).
    judge_completion is subprocess-based (releases the GIL), so we judge the group's
    completions CONCURRENTLY in a thread pool. ex.map preserves order, so rewards[k] /
    feedbacks[k] still line up with completion_ids[k] for the teacher builder.

    iteration-05 splits the reward by consumer (`grpo_reward`):
      - the GRPO policy advantage gets a BINARY reward (AC=1.0 else 0.0) when
        grpo_reward="binary" — a clean "did it actually solve it" signal, no partial
        credit polluting the policy gradient;
      - the SDPO teacher gating always gets the dense FRACTION (stashed on the bus),
        so near-miss rollouts (fraction >= success_reward_threshold, e.g. 0.5) can
        serve as teacher demonstrations even when not AC.
    We ALWAYS judge dense here (so the fraction is real) and derive binary from the
    verdict; `grpo_reward` only selects what the func RETURNS for the advantage.

    With `critic=True` (iteration 05), each FAILED rollout's deterministic feedback is
    replaced by an LLM trace-aligned critique (sdpo_critic) before it goes on the bus,
    converting the RefSol-leaning signal into a StepAlignFB one (reports/iteration-05).
    The Claude client is created once and shared across the thread pool (thread-safe);
    any critic error falls back to the deterministic feedback, so a step never stalls.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor
    from sdpo_ojbench import reward_case_caps, PROMPT_BY_ID, CPP_PROMPT_BY_ID, extract_code_cpp
    from ojbench_eval import extract_code
    workers = int(os.environ.get("SDPO_JUDGE_WORKERS", "16"))
    mb, mc = reward_case_caps()  # cap in-loop cases (huge test sets stall/hang a step)

    critic_client = None
    if critic:
        import anthropic
        from _paths import load_env
        from sdpo_critic import DEFAULT_CRITIC_MODEL
        load_env()
        critic_client = anthropic.Anthropic(timeout=30.0, max_retries=2)
        critic_model = critic_model or DEFAULT_CRITIC_MODEL

    logger = _RolloutLogger(rollout_log) if rollout_log else None

    def reward_func(completions, id=None, language=None, **kwargs):
        def judge_k(k):
            comp = completions[k]
            lang = language[k] if language is not None else "python"
            text = comp[-1]["content"] if isinstance(comp, list) else comp
            pid = int(id[k])
            # Always judge DENSE so `frac` is the true passed/total (binary mode would
            # early-exit on first failure and lose it); derive the binary AC reward from
            # the verdict. The bus carries `frac` for SDPO gating; the func returns the
            # GRPO reward (binary or fraction) per `grpo_reward`.
            frac, v, fb = judge_completion(text, pid, which=which, timeout=timeout,
                                           language=lang, reward_mode="fraction",
                                           max_case_bytes=mb, max_cases=mc)
            binary = 1.0 if v == "AC" else 0.0
            if critic and v not in ("AC", "NO_CODE"):
                from sdpo_critic import critique  # local import: optional dependency
                pmap = CPP_PROMPT_BY_ID if lang == "cpp" else PROMPT_BY_ID
                code = extract_code_cpp(text) if lang == "cpp" else extract_code(text)
                fb = critique(pmap.get(pid, ""), code, v, fb, lang,
                              model=critic_model, thinking=critic_thinking,
                              client=critic_client)
            return float(frac), binary, fb, v

        n = len(completions)
        with ThreadPoolExecutor(max_workers=min(workers, max(1, n))) as ex:
            results = list(ex.map(judge_k, range(n)))  # ex.map preserves input order
        bus.fractions = [frac for frac, _, _, _ in results]   # SDPO teacher gating
        bus.feedbacks = [fb for _, _, fb, _ in results]       # teacher conditioning

        # P0-1: stream this group's rollouts to JSONL (best-effort; never sinks the step).
        if logger is not None:
            _log_rollouts(logger, bus, completions, id, language, results,
                          sdpo_threshold, kwargs.get("completion_ids"))

        # GRPO advantage reward: binary AC by default for iteration-05, fraction otherwise.
        return [(b if grpo_reward == "binary" else f) for f, b, _, _ in results]
    reward_func.__name__ = (f"ojbench_{which}_reward_fb"
                            + ("_critic" if critic else "")
                            + ("_bin" if grpo_reward == "binary" else ""))
    return reward_func


class _LiveFeedbackBuilder(SuccessfulRolloutTeacherContextBuilder):
    def __init__(self, trainer, bus):
        super().__init__(trainer)
        self.bus = bus

    def build(self, output, prompts, rewards, feedbacks=None):
        import torch
        fb = self.bus.feedbacks
        fr = self.bus.fractions
        n = output["completion_ids"].shape[0]
        # local feedbacks/fractions must align 1:1 with this process's completions; else fall back
        use = fb if len(fb) == n else feedbacks
        # SDPO teacher gating (success_reward_threshold) runs on the dense FRACTION, so
        # near-miss rollouts can be teachers. The GRPO advantage already consumed the binary
        # reward upstream (rewards arg); we substitute fractions here purely for selection.
        gate = rewards
        if len(fr) == n:
            gate = torch.tensor(fr, dtype=rewards.dtype, device=rewards.device)
        return super().build(output, prompts, gate, feedbacks=use)


class FeedbackSDPOTrainer(SDPOTrainer):
    """SDPOTrainer that conditions the teacher on live per-rollout judge feedback.
    Pass the SAME FeedbackBus used to build the reward function."""
    def __init__(self, *args, feedback_bus, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_context_builder = _LiveFeedbackBuilder(self, feedback_bus)
        # Wire the bus so the reward func can stamp each rollout record (P0-1) with the
        # exact completion token count (tokenizer) and the current global_step (trainer).
        feedback_bus.trainer = self
        feedback_bus.tokenizer = getattr(self, "processing_class", None)
