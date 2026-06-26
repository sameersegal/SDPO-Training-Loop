# Iteration 03 — Moving-frontier curriculum on the full OJBench pool (206)

**Status: PLANNED.** Finalize before any run. Builds on
[iteration 01](../iteration-01/REPORT.md) (real base pass@k frontier; 100-step easy-only
**collapse**) and [iteration 02](../iteration-02/REPORT.md) (live judge feedback **stopped the
collapse** — medium pass@8 held 40→40, GSM8K held — but **did not beat base**; root cause:
binary reward gave all-fail medium groups zero advantage variance for 7/25 steps).

**Compute:** Modal H100/H200 · prototype on local GB10 · **W&B:** `sdpo-gemma-ojbench`

---

## Goal
**Improve held-out pass@1 across easy, medium, *and* hard** — not just avoid regression. Move the
model up the learnability ladder by (a) training on a curriculum that follows the model's own
frontier as it shifts, (b) giving every group a live gradient via **dense reward**, and (c) keeping
the iteration-02 protections (live feedback, KL anchor, length early-stop) so easy/GSM8K don't
regress.

## What changed since iteration 02 — the data pool tripled the hard set
We downloaded the **59 missing NOI-hard test sets** from `He-Ren/OJBench_testdata` (all
diff-checkable — none are checker/interactive) and verified the judge extracts and runs them.
The usable, diff-checkable, test-backed pool went **147 → 206**.

### The three nested sets (answering "what is the shape of train vs eval?")
| Set | Count | What it is |
|---|---|---|
| **OJBench total** | **232** | every problem (py+cpp mirror each) |
| **− special-judge** | −26 | 20 `checker` + 6 `interactive` ICPC problems — our stdout-diff judge would false-AC them and **corrupt the reward**. Reaching 232 needs a custom checker harness (deferred — judge-fidelity risk). |
| **= usable pool** | **206** | diff-checkable, test-backed, py+cpp present. **This is the iteration-3 universe.** |
| **→ held-out (fixed)** | **53** | the eval set — **2.1× iteration-01/02's 25** → the ±2/25 greedy noise shrinks |
| **→ train pool** | **153** | everything else; the moving frontier is selected **from here each round** |

### Pool shape by difficulty (`data/ojb_splits_full.json`)
| difficulty | train | held-out | total |
|---|---|---|---|
| easy | 23 | 8 | 31 |
| medium | 50 | 17 | 67 |
| **hard** | **80** | **28** | **108** |
| **TOTAL** | **153** | **53** | **206** |

Held-out spans every **(part × difficulty)** cell (≥3 each), and each held-out problem is evaluated
in **both py & cpp** → **106 eval instances**, far more discriminative than iter-01/02. Split is
**pid-level** (py & cpp of a problem share a side → no cross-language leak; verified disjoint).

**Key shift vs iter-01/02:** those trained on the 88-problem NOI-only split with **zero hard**
problems in train. Iteration-3's train pool has **80 hard** — the raw material for any hard-problem
improvement.

## Pre-flight diagnosis — where base actually fails on hard (and what conditioning buys)
Before committing compute we generated base rollouts on small-case train-hard problems
([`data/hard_rollouts.md`](./data/hard_rollouts.md), `src/diagnose_hard.py`), judged with the
**dense** reward so partial credit + the exact failing case are visible. Three failure archetypes,
with very different prognoses:

| archetype | example | signature | prompt-fixable? | curriculum value |
|---|---|---|---|---|
| **right idea, undelivered** | loj-2442 (0.70 WA) | passes most cases; **over-theorizes** (hunts a closed-form) instead of simulating the *stated* rule + tiering by the Data Range table | **partly** | **high** — best targets |
| **complexity wall** | loj-2131 (0.20 TLE) | exponential brute force where a poly DP is needed | no | low (needs capability) |
| **wrong algorithm** | loj-2356/3537 (0.00) | fundamentally off | no | low |

**Prompt-conditioning A/B** ([`data/prompt_condition.md`](./data/prompt_condition.md),
`src/prompt_condition.py`; n=6, dense reward) on the two reachable problems:

| problem | base | expert_sys | **cp_method** (restate+implement stated rule, tier by input size, output code) |
|---|---|---|---|
| loj-2442 | 0.70 | 0.70 | **0.80** best |
| loj-900011 | 0.50 | 0.48 | **0.62** best |

**Read:** `cp_method` **lifts the ceiling** (+0.10–0.12 best fraction) by getting the model to
*simulate* rather than theorize — but **reaches no AC in 6 samples**. The residual gap on 2442 is
(a) a **subtle simulation bug** (now fails at n=62 345 with a wrong count, not a timeout) and (b) the
**unimplemented n≤10¹⁸ tier** (matrix exponentiation / period detection). Both residuals are
**judge-visible** ("expected 94 220, got 577 877") and **partial-credit-bearing** — i.e. exactly the
signal **dense reward + live judge feedback** consume. So conditioning and the SDPO feedback loop are
**complementary, not substitutes**: bake `cp_method` into the prompt for a free ceiling lift, then let
feedback-driven training close the residual bug-level gap.

**Design consequences (folded in below):** (1) the hard-reachability probe ranks by **partial-credit
fraction**, not just pass@k>0 — a 0.70-WA hard is one bug-fix away; a 0.05-TLE hard needs a capability.
(2) Use the **`cp_method` system prompt** as the iteration-3 default (applied to base + treatment).

## Design

### 1. Moving-frontier curriculum (the centerpiece)
Solvability is measured, not assumed, and **re-measured as the model improves**:
1. **Probe** base/current policy on the **train pool** (n samples, fractional reward) → classify each
   problem **saturated** (pass≈1), **hopeless** (pass≈0), **frontier** (0 < pass < 1).
2. **Train** the frontier band for N steps (feedback-ON, dense reward, KL anchor).
3. **Re-probe** → the band **shifts** (newly-solvable problems saturate and drop out; previously
   hopeless ones become frontier and enter). Repeat.

Over rounds the band migrates **easy→medium→hard**, so the curriculum tracks capability instead of a
fixed difficulty label. Probe always on the **train pool only** (held-out stays clean).

### 2. Dense reward (fixes iteration-02's dead gradient)
`reward_mode="fraction"` (passed/total) — already implemented. An all-fail group now has **partial-
credit variance** → a live GRPO advantage, where iteration-02's binary reward gave it none (the 7/25
degenerate steps). Run all cases for the fraction; cap completion length / use H200 for the bigger
hard-problem judging cost.

### 3. Hard-reachability probe (`sampling @16/32`, your request)
Before trusting the curriculum to reach hard, **measure which hard problems are *ever* sometimes-
solvable** by base: pass@16 and pass@32 with fractional reward over the **80 train-pool hard
problems**. Output: the list of "reachable" hards (pass@32 > 0) = the **first hard curriculum
targets**. Hards with pass@32 = 0 are out of reach for elicitation (SDPO sharpens base ability; it
can't synthesize a capability that never appears in 32 samples) — flagged, not trained, this round.

### 4. Keep the iteration-02 protections
- **Live feedback ON** (`include_environment_feedback`, feedback-only-without-solution) — the
  no-collapse mechanism.
- **KL-to-base anchor** to protect easy + GSM8K (iteration-01's regression insurance).
- **Length early-stop** — kill the run if `completions/mean_length` starts collapsing.
- **Lower LR** (3e-5), **early-stop on held-out pass@k** (not loss — the loss is blind to regression).

## Success criteria
- **Primary:** held-out **pass@1 ↑ vs base on medium**, **easy held ≥ base**, and **measurable hard
  movement** (≥1 reachable hard goes pass@1: 0→>0, or hard pass@8 ↑).
- **No regression:** GSM8K within ~1 pt; no length collapse; pass@8 not below base.
- **Mechanism:** frontier band **shifts** across rounds (evidence the curriculum is moving), and
  dense reward removes the degenerate zero-variance steps (track `success_group_fraction` /
  advantage variance per step).
- **Informative negative:** if even with hard in the pool + dense reward + moving frontier there's no
  hard movement, the bottleneck is base capability/LoRA capacity, not data/curriculum — a real result.

## Risks & mitigations
| Risk | Likelihood | Mitigation |
|---|---|---|
| Hard problems all pass@32 = 0 (nothing reachable) | Med | The reachability probe **measures this first** — don't train unreachable hards; report the ceiling honestly |
| Dense-reward judging on hard is slow/expensive (big test sets) | Med | H200, cap completion length, smallest-first case ordering, public-subset for the in-loop reward |
| Moving frontier never reaches hard within budget | Med | Seed the band with reachable hards from the probe; longer schedule; report band trajectory regardless |
| Larger held-out (53) raises eval cost | Low | pass@k on Modal in parallel; it's still cheap vs training |
| 26 special-judge problems excluded → not "full 232" | Accepted | Documented; building a checker judge is a separate, fidelity-risky effort, deferred |
| Budget | — | confirm before launch (probe + reachability + curriculum rounds + eval) |

## Plan of record (run order, once finalized)
1. **Promote the full split** to the live config (point training/eval at `ojb_splits_full.json` / 206).
2. **Reachability probe** — base pass@16/32 fractional over 80 train-pool hard → `data/hard_reachable.json`.
3. **Frontier probe** round 0 over the 153 train pool → `data/frontier_band_full.json`.
4. **Curriculum run** (feedback-ON, dense, KL, length+pass@k early-stop), re-probe between rounds.
5. **Eval** base + adapter on the **53-problem held-out**, py & cpp × difficulty, pass@k n=16 + GSM8K.
6. **Report** deltas, band trajectory, hard movement; update [`docs/FINDINGS.md`](../../docs/FINDINGS.md).

## Provenance (to fill on run)
- Pool: `data/ojb_splits_full.json` (206; rebuilt after downloading 59 NOI-hard test sets).
- Probes: `data/hard_reachable.json`, `data/frontier_band_full.json` (train-pool only).
- Adapter: Modal `sdpo-outputs:/iteration-03/`. Design source of truth: [`docs/EXPERIMENT.md`](../../docs/EXPERIMENT.md).

---
**Open decisions to finalize before launch:** (a) curriculum rounds × steps-per-round and total
budget; (b) probe n for the frontier band (cost vs classification noise); (c) whether to start the
curriculum easy→hard or seed directly with reachable hards; (d) KL coefficient.
