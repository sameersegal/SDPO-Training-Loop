"""Rollout capture (iteration 06 P0-1, docs/design/OBSERVABILITY.md): the per-rollout
JSONL records and the streaming logger. Pure-logic, offline, free — no trainer/model."""
import json

from sdpo_feedback import _RolloutLogger, _log_rollouts, FeedbackBus


class _FakeState:
    def __init__(self, step):
        self.global_step = step


class _FakeTrainer:
    def __init__(self, step):
        self.state = _FakeState(step)


def _read(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def test_logger_streams_one_line_per_record(tmp_path):
    p = tmp_path / "sub" / "rollouts.jsonl"   # nested dir must be created
    lg = _RolloutLogger(str(p))
    lg.write_many([{"a": 1}, {"a": 2}])
    lg.write_many([{"a": 3}])                 # append, not overwrite
    rows = _read(p)
    assert [r["a"] for r in rows] == [1, 2, 3]


def test_log_rollouts_record_shape_and_values(tmp_path):
    p = tmp_path / "rollouts.jsonl"
    lg = _RolloutLogger(str(p))
    bus = FeedbackBus()
    bus.trainer = _FakeTrainer(step=7)
    bus.tokenizer = None
    # chat-style completions (list of messages) + plain-string completion both supported
    completions = [[{"role": "assistant", "content": "AC code here"}], "fail code"]
    ids = [101, 202]
    langs = ["python", "cpp"]
    # results: (fraction, binary, feedback, verdict)
    results = [(1.0, 1.0, "", "AC"), (0.4, 0.0, "Verdict: WA", "WA")]
    _log_rollouts(lg, bus, completions, ids, langs, results,
                  sdpo_threshold=0.5, completion_ids=None)
    rows = _read(p)
    assert len(rows) == 2
    a, b = rows
    assert a["step"] == 7 and a["problem_id"] == 101 and a["language"] == "python"
    assert a["verdict"] == "AC" and a["success"] is True
    assert a["teacher_eligible"] is True            # frac 1.0 >= 0.5
    assert a["n_chars"] == len("AC code here") and a["completion"] == "AC code here"
    assert b["verdict"] == "WA" and b["success"] is False
    assert b["teacher_eligible"] is False           # frac 0.4 < 0.5
    assert b["reward_fraction"] == 0.4 and b["feedback"] == "Verdict: WA"


def test_log_rollouts_uses_completion_ids_for_token_count(tmp_path):
    p = tmp_path / "r.jsonl"
    lg = _RolloutLogger(str(p))
    bus = FeedbackBus()                              # no trainer -> step defaults to -1
    completions = ["x"]
    results = [(1.0, 1.0, "", "AC")]
    _log_rollouts(lg, bus, completions, [5], ["python"], results,
                  sdpo_threshold=1.0, completion_ids=[[1, 2, 3, 4]])
    row = _read(p)[0]
    assert row["n_tokens"] == 4 and row["step"] == -1


def test_log_rollouts_never_raises_on_bad_input(tmp_path):
    # a malformed group must be swallowed (never sink a training step)
    lg = _RolloutLogger(str(tmp_path / "r.jsonl"))
    _log_rollouts(lg, FeedbackBus(), None, None, None, [("bad",)],
                  sdpo_threshold=1.0, completion_ids=None)  # must not raise


# ---- P1-6 per-token logprob capture (pure tensor logic; CPU, no trainer/GPU) ----

def test_entropy_chunked_matches_unchunked():
    import torch
    from sdpo_logprobs import _entropy_chunked
    torch.manual_seed(0)
    logits = torch.randn(2, 1000, 50)               # [B, T, vocab]
    lp = torch.log_softmax(logits, dim=-1)
    ref = -(lp.exp() * lp).sum(-1)                   # full-tensor reference
    got = _entropy_chunked(logits, chunk=128)        # chunked over time
    assert torch.allclose(got, ref, atol=1e-4)
    # uniform logits -> entropy = log(vocab); peaked -> ~0
    uni = _entropy_chunked(torch.zeros(1, 4, 50), chunk=2)
    assert torch.allclose(uni, torch.full((1, 4), float(torch.log(torch.tensor(50.0)))), atol=1e-4)


def test_at_advantage_teacher_minus_base():
    import numpy as np
    from sdpo_logprobs import at_advantage
    rec = {"teacher_logp": np.array([-1.0, -2.0, -0.5], dtype=np.float16),
           "base_logp": np.array([-1.5, -1.0, -0.5], dtype=np.float16)}
    at = at_advantage(rec)
    assert np.allclose(at, [0.5, -1.0, 0.0], atol=1e-3)
    assert at_advantage({"base_logp": np.zeros(3)}) is None   # no teacher -> None
