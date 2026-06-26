"""Live-feedback plumbing: the reward function must populate the bus with per-rollout
judge feedback aligned 1:1 with completions (what the teacher builder substitutes in)."""
import pytest


def test_feedback_reward_populates_bus_aligned():
    pytest.importorskip("trl.experimental.sdpo")
    import sdpo_ojbench as S
    from sdpo_feedback import FeedbackBus, make_feedback_reward_func

    bus = FeedbackBus()
    rf = make_feedback_reward_func(bus, which="public", reward_mode="fraction")
    pids = S.SPLITS["train"][:3]
    comps = ["```python\nprint(0)\n```"] * len(pids)          # wrong stub -> not AC
    rewards = rf(comps, id=pids, language=["python"] * len(pids))

    assert len(rewards) == len(pids)
    assert len(bus.feedbacks) == len(pids)                    # aligned 1:1 with completions
    assert all(isinstance(f, str) and f.strip() for f in bus.feedbacks)
    # a non-AC rollout's feedback carries a verdict (the corrective signal the teacher sees)
    assert any("Verdict" in f for f in bus.feedbacks)
