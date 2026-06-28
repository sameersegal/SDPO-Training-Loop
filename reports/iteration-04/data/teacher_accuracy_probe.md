# Accuracy probe — does the SDPO capability gap exist?

Model `google/gemma-4-E2B-it` · system `cp_method` · language `python`. Student = base rollouts (no context). Teacher = same model conditioned on the per-rollout judge feedback `c`, generating a fresh solution. AC = all public tests pass; mean reward = fraction of public tests passed (partial-credit signal).

| difficulty | problem | context | student AC | student mean-reward | teacher AC | teacher mean-reward | gap (AC) | gap (reward) |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| easy | loj-2314 | solution | 8/8 | 1.0 | — (skipped) | — | — | — |
| medium | loj-2086 | feedback | 0/8 | 0.521 | 0/8 | 0.438 | 0/8 − 0/8 | -0.083 |
| hard | loj-2083 | feedback | 0/8 | 0.438 | 0/8 | 0.612 | 0/8 − 0/8 | +0.174 |

**Verdict distributions** (student → teacher):

- **easy loj-2314** — student {'AC': 8} → teacher skipped (solution-context)
- **medium loj-2086** — student {'WA': 2, 'TLE': 6} → teacher {'TLE': 5, 'WA': 1, 'RE': 2}
- **hard loj-2083** — student {'TLE': 5, 'WA': 3} → teacher {'TLE': 7, 'WA': 1}
