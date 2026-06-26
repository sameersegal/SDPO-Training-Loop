"""Guard against py/cpp data leakage across train/eval.

py and cpp prompts are EXACT mirrors of the same problem (same statement, same test
cases — only the code-fence language differs). So a problem-id must live entirely in
ONE split; if loj-X's python went to train and its cpp to eval, the model would have
seen the solution at train time. The split is therefore by PROBLEM ID, never by
(problem, language). These tests enforce that invariant.
"""
import sdpo_ojbench as S


def test_train_heldout_disjoint():
    assert set(S.SPLITS["train"]).isdisjoint(S.SPLITS["heldout"])


def test_every_pid_in_exactly_one_split():
    tr, ho = S.SPLITS["train"], S.SPLITS["heldout"]
    assert len(tr) == len(set(tr)) and len(ho) == len(set(ho))  # no dup within a split


def test_py_and_cpp_share_the_same_split():
    # both languages are keyed by the same pid, so for every dataset row the pid is in
    # exactly one split -> py and cpp of a problem can never be on opposite sides.
    train_pids = set(r["id"] for r in S.build_dataset("train", languages=("python", "cpp")))
    held_pids = set(r["id"] for r in S.build_dataset("heldout", languages=("python", "cpp")))
    assert train_pids.isdisjoint(held_pids)


def test_full_splits_invariants_if_present():
    """If the combined NOI+ICPC splits exist, enforce the same invariants there."""
    import json
    from pathlib import Path
    p = Path(__file__).resolve().parent.parent / "data" / "ojb_splits_full.json"
    if not p.exists():
        import pytest
        pytest.skip("ojb_splits_full.json not built")
    full = json.load(open(p))
    tr, ho = set(full["train"]), set(full["heldout"])
    assert tr.isdisjoint(ho)                                  # no leak
    parts = full["part_by_id"]
    assert {parts[str(i)] for i in ho} == {"NOI", "ICPC"}     # held-out spans both parts
    for i in tr | ho:                                         # py & cpp present for every pid
        assert str(i) in full["py_prompt_by_id"] and str(i) in full["cpp_prompt_by_id"]
        assert str(i) in full["testdir_by_id"]


def test_dataset_has_both_languages_per_pid_when_available():
    rows = S.build_dataset("train", languages=("python", "cpp"))
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r["id"], set()).add(r["language"])
    # a pid present for python should also be present for cpp (mirrors) and vice versa
    for pid, langs in by_pid.items():
        assert langs == {"python", "cpp"}, f"loj-{pid} missing a language: {langs}"
