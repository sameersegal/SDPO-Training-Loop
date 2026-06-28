"""LLM-critic feedback (iteration 05): prompt assembly, the StepAlignFB rules baked into
the system prompt, and the non-negotiable fallback-on-error / skip-on-AC behavior. No
network: the anthropic client is injected as a fake, so these run offline and free."""
import pytest

import sdpo_critic as C


def test_build_messages_embeds_code_and_judge_feedback():
    system, messages = C.build_critic_messages(
        problem="Print a+b.",
        student_code="a,b=map(int,input().split())\nprint(a-b)",
        verdict="WA",
        judge_feedback="Verdict: WA.\nInput:\n3 4\nExpected output:\n7\nYour output:\n-1",
        lang="python",
    )
    assert len(messages) == 1 and messages[0]["role"] == "user"
    user = messages[0]["content"]
    assert "print(a-b)" in user                      # the student's own code is anchored
    assert "Expected output:\n7" in user             # the judge result is the raw material
    assert "```python" in user                       # fenced in the rollout's language
    # the StepAlignFB rules must be in the system prompt, or the signal diffuses
    assert "do not write" in user.lower() or "describe" in user.lower()
    low = system.lower()
    assert "anchor the correct prefix" in low        # rule 1
    assert "do not paste a corrected line" in low or "do not write it out" in low  # rule 3
    assert "tle" in low and "o(n^2)" in low.replace(" ", "")  # approach-level TLE guidance


class _FakeBlock:
    def __init__(self, text):
        self.type, self.text = "text", text


class _FakeResp:
    def __init__(self, text):
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, outer):
        self._outer = outer

    def create(self, **kwargs):
        self._outer.calls.append(kwargs)
        if self._outer.boom:
            raise RuntimeError("api down")
        return _FakeResp(self._outer.reply)


class _FakeClient:
    def __init__(self, reply="anchored critique: your prefix is right; the comparison is wrong.",
                 boom=False):
        self.reply, self.boom, self.calls = reply, boom, []
        self.messages = _FakeMessages(self)


def test_returns_critique_on_success():
    client = _FakeClient()
    out = C.critique("Print a+b.", "print(a-b)", "WA",
                     "Verdict: WA.\nExpected output:\n7\nYour output:\n-1", "python",
                     client=client)
    assert out == client.reply
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == C.DEFAULT_CRITIC_MODEL


def test_falls_back_to_judge_feedback_on_api_error():
    fb = "Verdict: WA.\nExpected output:\n7\nYour output:\n-1"
    out = C.critique("Print a+b.", "print(a-b)", "WA", fb, "python",
                     client=_FakeClient(boom=True))
    assert out == fb                                  # outage must never stall a step


def test_empty_reply_falls_back():
    fb = "Verdict: RE.\nRuntime error:\nIndexError"
    out = C.critique("x", "code", "RE", fb, "python", client=_FakeClient(reply="   "))
    assert out == fb


def test_skips_critic_for_ac_and_no_code():
    client = _FakeClient()
    assert C.critique("x", "c", "AC", "All public tests passed.", "python", client=client) \
        == "All public tests passed."
    assert C.critique("x", "", "NO_CODE", "no code block", "python", client=client) \
        == "no code block"
    assert client.calls == []                         # no API call for AC / NO_CODE


def test_reward_func_routes_failed_rollouts_through_critic(monkeypatch):
    """Phase 2a wiring: a failed rollout's bus feedback is the critique, not the raw verdict;
    the critic sees a non-AC verdict and the student's extracted code. API fully stubbed."""
    pytest.importorskip("trl.experimental.sdpo")
    import anthropic
    import sdpo_critic
    import sdpo_ojbench as S
    from sdpo_feedback import FeedbackBus, make_feedback_reward_func

    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: object())  # no real client

    seen = []

    def fake_critique(problem, code, verdict, judge_feedback, lang, **kw):
        seen.append((verdict, code))
        return f"CRITIQUE[{verdict}]"

    monkeypatch.setattr(sdpo_critic, "critique", fake_critique)

    bus = FeedbackBus()
    rf = make_feedback_reward_func(bus, which="public", reward_mode="fraction", critic=True)
    pids = S.SPLITS["train"][:2]
    comps = ["```python\nprint(0)\n```"] * len(pids)   # wrong stub -> WA/RE, not AC
    rf(comps, id=pids, language=["python"] * len(pids))

    assert all(f.startswith("CRITIQUE[") for f in bus.feedbacks)  # critic replaced the signal
    assert seen and all(v not in ("AC", "NO_CODE") for v, _ in seen)
    assert all("print(0)" in code for _, code in seen)            # extracted student code, fenced upstream
