# FINDINGS — Results index

Results are tracked **iteration by iteration** under [`reports/`](../reports/). Each iteration
has a self-contained report with embedded graphs.

| Iteration | Summary | Report |
|---|---|---|
| **01 — easy-only SDPO** | Base has a real pass@k frontier (py 9.5→20%); 20 steps = null; **100 steps = global regression** (overfit/collapse to terse outputs, GSM8K 90.8→87.3%). pass@k caught what greedy pass@1 + the loss hid. | [reports/iteration-01/REPORT.md](../reports/iteration-01/REPORT.md) |
| 02 — frontier-band + regularized | *(planned)* train the learnability frontier (easy + sometimes-solvable medium), lower LR / fewer epochs / KL anchor, early-stop on held-out pass@k. | — |

**Standing lessons (carry forward):**
- **pass@k (k≥4, n≥8) is the metric** — greedy pass@1 (±2/25 noise) and the SDPO loss are both blind to regression.
- **Select training data by measured solvability** (learnability frontier), not difficulty label.
- SDPO on the frontier = **capability elicitation/sharpening** (pass@k→pass@1), bounded by base pass@k; new capability (hard) needs **live judge feedback** (iteration 2).
- Design: [`EXPERIMENT.md`](./EXPERIMENT.md) · Cloud scale-up: [`MODAL.md`](./MODAL.md).
