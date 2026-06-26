"""Build the FULL OJBench split (NOI + ICPC) from prompts + downloaded test data.

Writes data/ojb_splits_full.json (reviewable; does NOT overwrite the live NOI-only
ojb_splits.json). Guards:
  - diff-checkable only: drop problems whose init.yml has a 'checker'/'interactive'/'spj'
    key (special judges would false-negative correct solutions and corrupt the reward).
  - must have local test data.
  - pid-level split (py & cpp of a problem share a split -> no cross-language leak).
  - held-out spans every (part x difficulty) cell.

ICPC ids are strings (nwerc2022_A); we assign synthetic INT ids (>= 900000) so the rest
of the int-keyed pipeline is unchanged, and record testdir_by_id / part_by_id so
extract_tests can resolve either part. Run: PYTHONPATH=src python src/build_splits_full.py
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import yaml

from _paths import ojbench_dir, find_file

PROMPTS = find_file("full.jsonl") if (find_file("full.jsonl")).exists() else Path("ojbench_prompts/prompts/full.jsonl")
DATA = ojbench_dir()
SPECIAL = {"checker", "interactive", "spj", "grader", "custom_judge"}


def load_prompts():
    """{(part, id): {language: prompt, difficulty}}"""
    rec = defaultdict(dict)
    for line in open("ojbench_prompts/prompts/full.jsonl"):
        d = json.loads(line)
        part = "NOI" if d["dataset"] == "NOI" else "ICPC"
        key = (part, str(d["id"]))
        rec[key][d["language"]] = d["prompt"]
        rec[key]["difficulty"] = d["difficulty"]
    return rec


def testdir(part, raw_id):
    return DATA / part / (f"loj-{raw_id}" if part == "NOI" else raw_id)


def diff_checkable(part, raw_id):
    d = testdir(part, raw_id)
    if not (d / "init.yml").exists():
        return False, "no test data"
    y = yaml.safe_load(open(d / "init.yml"))
    bad = set(y.keys()) & SPECIAL
    return (not bad), (f"special: {bad}" if bad else "ok")


def main():
    rec = load_prompts()
    kept, dropped = [], defaultdict(int)
    next_icpc_id = 900000
    by_id, py, cpp, testdir_by_id, part_by_id, diff_by_id = {}, {}, {}, {}, {}, {}
    for (part, raw_id), info in sorted(rec.items()):
        ok, reason = diff_checkable(part, raw_id)
        if not ok:
            dropped[reason.split(":")[0]] += 1
            continue
        if "python" not in info or "cpp" not in info:
            dropped["missing language"] += 1
            continue
        iid = int(raw_id) if part == "NOI" else next_icpc_id
        if part == "ICPC":
            next_icpc_id += 1
        by_id[iid] = info["difficulty"]
        py[iid] = info["python"]
        cpp[iid] = info["cpp"]
        diff_by_id[iid] = info["difficulty"]
        part_by_id[iid] = part
        testdir_by_id[iid] = f"{part}/{'loj-'+raw_id if part=='NOI' else raw_id}"
        kept.append((iid, part, info["difficulty"]))

    # pid-level held-out spanning every (part x difficulty) cell (~25%, >=3 each)
    rng = random.Random(0)
    cells = defaultdict(list)
    for iid, part, d in kept:
        cells[(part, d)].append(iid)
    train, heldout = [], []
    for cell, ids in sorted(cells.items()):
        rng.shuffle(ids)
        k = max(3, round(0.25 * len(ids)))
        heldout += ids[:k]
        train += ids[k:]

    splits = {
        "by_id": {str(k): v for k, v in by_id.items()},
        "py_prompt_by_id": {str(k): v for k, v in py.items()},
        "cpp_prompt_by_id": {str(k): v for k, v in cpp.items()},
        "testdir_by_id": {str(k): v for k, v in testdir_by_id.items()},
        "part_by_id": {str(k): v for k, v in part_by_id.items()},
        "train": sorted(train),
        "heldout": sorted(heldout),
    }
    out = Path("data/ojb_splits_full.json")
    json.dump(splits, open(out, "w"))
    # summary
    print(f"kept {len(kept)} problems (py+cpp, diff-checkable). dropped: {dict(dropped)}")
    bycell = defaultdict(lambda: [0, 0])
    tr, ho = set(train), set(heldout)
    for iid, part, d in kept:
        bycell[(part, d)][0 if iid in tr else 1] += 1
    print(f"{'part':5} {'diff':7} {'train':6} {'held':5}")
    for cell in sorted(bycell):
        t, h = bycell[cell]
        print(f"{cell[0]:5} {cell[1]:7} {t:<6} {h}")
    print(f"TOTAL train={len(train)} heldout={len(heldout)}  (split-disjoint: {tr.isdisjoint(ho)})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
