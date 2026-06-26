# FINDINGS — Results index

Results are tracked **iteration by iteration** under [`reports/`](../reports/). Each iteration
has a self-contained report with embedded graphs.

| Iteration | Summary | Report |
|---|---|---|
| **01 — easy-only SDPO** | Base has a real pass@k frontier (py 9.5→20%); 20 steps = null; **100 steps = global regression** (overfit/collapse to terse outputs, GSM8K 90.8→87.3%). pass@k caught what greedy pass@1 + the loss hid. | [reports/iteration-01/REPORT.md](../reports/iteration-01/REPORT.md) |
| **02 — live judge feedback** | **run 1 done.** Live per-rollout judge text → SDPO teacher, validated (`feedback_used` fires on all-fail groups; **0** in iter-01). Feedback-ON (easy+medium, 25 steps, LR 3e-5) **stopped iter-01's collapse**: medium pass@8 **held 40→40** (iter-01 →0), **GSM8K held 90.5%** (iter-01 87.3%). But **did not beat base** (easy 60→40); 7/25 steps degenerate under binary reward → next: **dense reward** + more steps. | [reports/iteration-02/REPORT.md](../reports/iteration-02/REPORT.md) |
| **03 — moving-frontier curriculum** | **PLANNED.** Downloaded 59 missing NOI-hard test sets → usable pool **147→206** (the other 26 to 232 are special-judge, deferred). Fixed **53-problem held-out** (2.1× iter-01/02), train pool 153 with **80 hard** (was 0). Plan: hard-reachability probe (pass@16/32 fractional) + **moving-frontier curriculum** + **dense reward** + feedback-ON + KL anchor. Goal: lift easy/medium/**hard**. | [reports/iteration-03/REPORT.md](../reports/iteration-03/REPORT.md) |

**Standing lessons (carry forward):**
- **pass@k (k≥4, n≥8) is the metric** — greedy pass@1 (±2/25 noise) and the SDPO loss are both blind to regression.
- **Select training data by measured solvability** (learnability frontier), not difficulty label.
- SDPO on the frontier = **capability elicitation/sharpening** (pass@k→pass@1), bounded by base pass@k; new capability (hard) needs **live judge feedback** (iteration 2).
- Design: [`EXPERIMENT.md`](./EXPERIMENT.md) · Cloud scale-up: [`MODAL.md`](./MODAL.md).
