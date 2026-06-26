"""Review an OJBench problem end-to-end: task prompt, test cases, judge feedback,
and the SDPO teacher reprompt that would be generated.

Helps review what the model sees and what feedback the teacher gets (incl. the
smallest-first failing case). Optionally judge a candidate solution.

  PYTHONPATH=src python src/inspect_problem.py --pid 2599
  PYTHONPATH=src python src/inspect_problem.py --pid 2599 --solution mysol.py
"""
import argparse

import sdpo_ojbench as S
import sdpo_prompts as P


def clip(s, n=400):
    return s if len(s) <= n else s[:n] + f"\n...<+{len(s)-n} chars>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--language", default="python", choices=["python", "cpp"])
    ap.add_argument("--which", default="public", choices=["public", "private"])
    ap.add_argument("--solution", default=None, help="path to a candidate solution (else a wrong stub)")
    args = ap.parse_args()
    pid = args.pid

    pmap = S.CPP_PROMPT_BY_ID if args.language == "cpp" else S.PROMPT_BY_ID
    prompt = pmap[pid]
    pub, prv = S.public_private_cases(pid)
    cases = pub if args.which == "public" else prv
    cases_sorted = sorted(cases, key=lambda c: c[0].stat().st_size)

    print("=" * 78)
    print(f"loj-{pid}  difficulty={S.DIFF_BY_ID[pid]}  language={args.language}  "
          f"public/private = {len(pub)}/{len(prv)} cases")

    print("\n----- TASK PROMPT (what the model sees) -----")
    print(clip(prompt, 1200))

    print(f"\n----- TEST CASES ({args.which}, smallest-first) -----")
    for inp, out in cases_sorted[:3]:
        print(f"  [{inp.name}] input {inp.stat().st_size}B:  {clip(inp.read_text(errors='replace'), 120)!r}")
        print(f"            expected:    {clip(out.read_text(errors='replace'), 120)!r}")
    smallest = cases_sorted[0]
    print(f"  (smallest public case = {smallest[0].name}, {smallest[0].stat().st_size}B — this is what failure feedback uses)")

    # Judge a candidate solution (or a deliberately-wrong stub to show the feedback path)
    code = open(args.solution).read() if args.solution else "print(0)\n"
    reward, verdict, feedback = S.judge_completion(
        f"```{'cpp' if args.language=='cpp' else 'python'}\n{code}\n```",
        pid, which=args.which, language=args.language)
    print(f"\n----- JUDGE (solution={'file' if args.solution else 'wrong stub print(0)'}) -----")
    print(f"  verdict={verdict}  reward={reward:.3f}")
    print(f"  feedback text (the SDPO environment signal):\n    " + feedback.replace("\n", "\n    "))

    print("\n----- TEACHER REPROMPT that would be generated (feedback path) -----")
    msgs = P.build_teacher_messages([{"role": "user", "content": prompt}], demo_text=None, feedback_raw=feedback)
    print(clip(msgs[-1]["content"], 1600))


if __name__ == "__main__":
    main()
