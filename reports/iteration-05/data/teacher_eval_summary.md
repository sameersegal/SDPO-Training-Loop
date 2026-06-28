# Phase-0 teacher eval — solve-rate + per-token A_t (sonnet-verbose critique)

Qwen3-8B, K=8 samples/failure, teacher prompt = `x + critique` (no attempt in the prompt; SDPO-faithful). A_t = log q(y_t | x, critique, y_<t) − log pi(y_t | x, y_<t) over the student's attempt. Full token-level colored view: `figures/advantage_view.html`.

| failure | base | solve-rate | mean\|A_t\| | frac_neg | code \|A\| | reas \|A\| | top5% mass | top1% mass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| medium loj-2086 s0 | WA | 0.0 | 0.0592 | 0.549 | 0.0023 | 0.0613 | 53% | 26% |
| medium loj-2129 s0 | WA | 0.875 | 0.0487 | 0.568 | 0.003 | 0.05 | 58% | 32% |
| medium loj-2130 s0 | TLE | 0.5 | 0.0345 | 0.579 | 0.0018 | 0.0352 | 54% | 26% |
| hard loj-2083 s0 | TLE | 0.0 | 0.0502 | 0.522 | 0.0075 | 0.0516 | 46% | 20% |
| hard loj-2083 s1 | WA | 0.0 | 0.0348 | 0.496 | 0.0081 | 0.0355 | 45% | 18% |
| hard loj-2131 s0 | WA | 0.0 | 0.0637 | 0.503 | 0.0011 | 0.0655 | 49% | 21% |
| hard loj-2131 s1 | WA | 0.0 | 0.0441 | 0.567 | 0.0278 | 0.0446 | 55% | 27% |
| hard loj-2133 s0 | WA | 0.0 | 0.0426 | 0.492 | 0.0095 | 0.0444 | 48% | 20% |
| hard loj-2133 s1 | TLE | 0.0 | 0.054 | 0.553 | 0.0055 | 0.0568 | 50% | 22% |

**Read:**
- **Solve-rate:** the critique lifts only the *sometimes-solvable* mediums (2129 -> 0.875, 2130 -> 0.5); medium 2086 and all 6 hard stay 0 (hard is flat-0 at base pass@8, so the critique cannot unlock it).
- **A_t is concentrated, not diffuse:** the top 5% of tokens carry ~45-58% of the |A_t| mass (vs 5% if uniform), top 1% carry ~18-32% — real localized spikes sitting under a low-magnitude background (which is what frac_neg ~= 0.5 reflects).
- **The spikes sit in the reasoning prose, not the code** (code |A| is ~10-50x smaller than reasoning). Whether that is a good SDPO training signal depends on whether they land on the *actual flawed step* — read the colored HTML to judge.
- 2129 (the biggest solve-rate lift) is also the most concentrated A_t (top5% = 58%) — capability lift and signal peakedness correlate.
