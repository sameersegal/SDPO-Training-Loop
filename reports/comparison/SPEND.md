# Cross-iteration spend — methodology & ledger

Tracks **what every Modal dollar was spent on**, segmented into productive vs wasted compute, so
spend can be reported later and the waste tied back to specific, fixable failure modes. This is a
**cross-iteration** artifact (lives in `reports/comparison/`, not any single iteration). Update it
when a new iteration's runs land.

Last reconciled: **2026-06-30** (covers iterations 01–06). Total Modal spend to date: **~$154**.

## TL;DR

| Category | Spend | Share |
|---|--:|--:|
| **Not productive** (crashed / hung / cancelled / killed) | **~$53** | **35%** |
| Eval (held-out + train==eval pass@k, GSM8K, base-opportunity, probes) | ~$51 | 33% |
| Train (SDPO runs that produced an adapter/checkpoints) | ~$39 | 26% |
| Infra (smokes, prewarm, image builds, CPU-only ops) | ~$10 | 6% |
| **Total** | **~$154** | |

**Headline:** the single largest bucket is **wasted compute (~35%)**, not training. Almost all of it
is two iterations — **iter-03 integration false-starts (~$30)** and **iter-05 eval crash + retry-loop
(~$23)** — and both are exactly the failure modes the CLAUDE.md hazard defaults were later written to
prevent (no-progress watchdog, checkpoint-cadence < interruption-interval + `--resume`, decoupled
`setsid+nohup --detach` launch, the 2h→5h function timeout). The waste is the ROI case for that
hardening.

## Why this can't come from billing alone

`modal billing report` (wrapped by `src/modal_cost.py`) gives the **authoritative dollar total per
app** and the GPU/CPU/Memory split — but it **cannot classify** a run. Every app shares one
description (`sdpo-gemma-ojbench`); there is no per-function label and `modal app list` drops old
stopped apps, so neither the purpose (train vs eval vs smoke) nor the outcome (succeeded vs crashed)
is recoverable from billing metadata.

**Classification therefore comes from the committed reports, not billing:**
`reports/iteration-NN/PROVENANCE.md` (itemized run ledgers, where they exist), `REPORT.md` (run
narrative, crash post-mortems), and `CLAUDE.md` / `memory/` (the documented waste events with their
dollar figures). Billing supplies the *magnitudes*; the docs supply *what each run was and whether it
worked*. An earlier pass that classified by cost-band heuristics on billing alone **undercounted
waste** — it scored iter-03 mostly as "Train" when the reports describe that day as almost entirely
integration debugging.

## Category definitions

- **Train** — an SDPO/GRPO training run that produced a usable adapter or checkpoints.
- **Eval** — measurement: held-out pass@k, the train==eval probe, GSM8K regression, the base-model
  opportunity/frontier graph, and capability probes (teacher-accuracy, rollout probe sets). Productive
  even when the result is a null — a measured null is signal.
- **Infra** — setup/validation that isn't itself train or eval: `--smoke` runs, `prewarm_weights`,
  image builds, CPU-only ops.
- **Not productive** — compute that yielded no usable output: crashed runs, hung runs, runs cancelled
  or killed mid-stream before a checkpoint, and retry-loops on a since-fixed bug. A run deliberately
  killed **and** superseded by a better relaunch counts here (the spend bought nothing that survived).

## Confidence tiers

Read the per-iteration numbers through these tiers (column `confidence` in
[`data/spend_by_iteration.csv`](./data/spend_by_iteration.csv)):

- **pinned** — exact dollar from a PROVENANCE ledger or a CLAUDE.md post-mortem. ~$70 of the total is
  pinned: iter-05 train $10.86 · iter-05 waste $23.34 · iter-05 base-opportunity $3.32 · iter-06 train
  $9.40 + eval $8.17 · iter-04 $0 · iter-01 ~$15.
- **documented-magnitude** — the bucket total is stated in a report, the per-app split is inferred:
  iter-03 "~$30 across SIX integration failures".
- **estimate** — no PROVENANCE exists (iter-02, 03, 06 have none); the Train/Eval/Infra split inside
  the iteration is reconstructed from the REPORT narrative. Could shift ±$5; the four **bucket totals
  are robust** to it.

## Per-iteration ledger

Data: [`data/spend_by_iteration.csv`](./data/spend_by_iteration.csv). Waste line-items:
[`data/spend_not_productive_detail.csv`](./data/spend_not_productive_detail.csv).

| Iteration | Date | Total | Train | Eval | Infra | Not productive | Notes |
|---|---|--:|--:|--:|--:|--:|---|
| iter-01 | 06-26 | ~$15 | $6 | $7 | $2 | $0 | 100-step train ~$6 + evals + smokes (PROVENANCE) |
| iter-02 | 06-26 | ~$16 | ~$8 | ~$6 | ~$2 | $0 | RUN 1 COMPLETE — one feedback-ON run + eval; no crash |
| iter-03 | 06-27 | ~$40 | ~$5 | $0 | ~$5 | **~$30** | six integration failures; status still PLANNED |
| iter-04 | 06-28 | $0 | — | — | — | — | local GB10 only (PROVENANCE: $0) |
| iter-05 ph-0 | 06-28 | ~$15 | $0 | ~$14 | ~$1 | $0 | Qwen3-8B base opportunity $3.32 + teacher/rollout probes |
| iter-05 main | 06-29 | ~$50 | $10.86 | ~$16 | $0 | **$23.34** | train $10.86; crash $15.75 + retry $6.22 + killed $1.37 |
| iter-06 | 06-30 | ~$18 | $9.40 | $8.17 | $0 | $0 | fast train + 12-probe eval (partial billing interval) |
| **Total** | | **~$154** | **~$39** | **~$51** | **~$10** | **~$53** | |

Note: iter-04's PROVENANCE records $0 (diagnostic, local). The Modal apps billed on 06-28 are
iter-05 **phase-0** prep (base opportunity + probes), not iter-04 — do not attribute 06-28 spend to
iter-04.

## How to update / reconcile

1. **Pull the authoritative totals** (not napkin math):
   `python src/modal_cost.py --for "this month"` (or `--this-run` for the current iteration's app).
   Re-run **after** a job fully stops — billing reports full intervals only, so an in-flight run
   undercounts (this is why iter-06's ~$18 may tick up).
2. **Classify from the iteration's own docs**, not billing: read that iteration's `PROVENANCE.md` /
   `REPORT.md` for which app was train vs eval and whether it crashed. Record the new row in
   [`data/spend_by_iteration.csv`](./data/spend_by_iteration.csv) with a `confidence` tag, and add any
   waste line-items to [`data/spend_not_productive_detail.csv`](./data/spend_not_productive_detail.csv).
3. **Always write a `PROVENANCE.md` with a spend ledger for new iterations** — iter-02/03/06 lack one,
   which is the only reason their splits are estimates. A per-run `app_id → $ → purpose → outcome`
   table at iteration time makes this artifact exact instead of reconstructed.
4. Refresh the TL;DR totals and the donut (4 slices: Not productive / Eval / Train / Infra).

## Caveats

- **iter-01 / iter-02 share 06-26** ($31.38 day total) and can't be split from billing alone — only
  from iter-01's PROVENANCE "~$15", which leaves ~$16 for iter-02.
- **iter-06 is a partial billing interval** (launched 06-30) — re-reconcile once the run settles.
- The Train↔Eval boundary inside the no-PROVENANCE iterations (02, 03, 06) is the softest number here;
  the **Not-productive total is the firmest** because every dollar in it is tied to a documented crash.
