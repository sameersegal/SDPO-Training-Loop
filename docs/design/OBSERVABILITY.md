# Component design — training/eval observability (rollout + logprob capture)

> A **component design doc**, kept independent of any iteration (see `CLAUDE.md` → "Component
> design docs"). Status: **P0 core IMPLEMENTED + verified** (rollout JSONL, sampling knobs, eval
> completion text) — unit-tested (`tests/test_observability.py`, 61 suite green) and confirmed on a
> GB10 `--feedback` smoke. **P1-6 LIVE capture DROPPED** (cost 2–3 extra forwards/step — unaffordable);
> per-token logprobs/entropy/A_t are instead **regenerated OFFLINE from the saved `checkpoint-*`**
> (deterministic given checkpoint + tokens). **This makes P0-4 — preserve EVERY checkpoint — load-
> bearing, not optional.** Pure math helpers kept in `src/sdpo_logprobs.py` (`realized_token_logp`,
> `per_token_entropy`/`_entropy_chunked`, `at_advantage`) for that offline pass. **Deferred:** P0-4
> wiring (per-iteration checkpoint copy), the offline regen script, P1-7 (`review_rollouts.py`). Code:
> `src/sdpo_feedback.py` (`_RolloutLogger`, `_log_rollouts`, bus wiring), `src/sdpo_train.py`
> (`--temperature`/`--top-p`/`--no-rollout-log`), `src/sdpo_passk.py` (`--no-save-completions`).

> **trl-1.6.0 note:** `log_completions`/`num_completions_to_print` (P0-5) do **not** exist on
> `SDPOConfig` — superseded by P0-1's durable on-disk JSONL (better than a sampled W&B table anyway).
> With **`beta>0`, TRL logs `kl` natively** (iter-05 at beta=0 had no kl column), and length-split /
> diversity are computed **offline from the rollout JSONL** — so P0-2 needs no loss-path override.

## 0. Why this exists — the gap that motivated it

iteration-05 mode-collapsed (length 17.3k → 4.4k tokens; pass@k diversity lost) and when we went to
**review the actual completions to see whether the shorter responses were still good**, we found:

- training **completions were never persisted** (no `log_completions`, rollouts ephemeral on Modal);
- the eval/pass@k scripts **judge each completion then discard the text** — only verdicts survive;
- only **one checkpoint** (ckpt-20) was retained — no mid-trajectory to inspect;
- the mechanistic signals that explain the collapse — **per-token entropy and the A_t advantage**
  (iter-04's diagnostic, `A_t = log π(ŷ|x,c) − log π(ŷ|x)`) — were never logged, only aggregate scalars.

Root cause of the collapse (see memory `sdpo-iter05-collapse-rootcause`): a **self-distillation
brevity spiral** — flat reward groups zeroed the policy gradient (`policy_loss≈0` on 18/20 steps), so
training became ~100% self-distillation toward the model's own shortening rollouts. **The signals that
would have shown this live are exactly the ones below.** Never run blind again.

## 1. Principles

1. **Stream to disk, never buffer-all-then-write** (`CLAUDE.md` rule). One append per rollout, so a
   crash/OOM keeps everything up to the last completed write. Mirror to the Modal volume each step.
2. **Right format for the data.** Human-reviewable records (completions, verdicts) → **JSONL**.
   Dense numeric arrays (per-token logprobs) → **compact columnar/binary** (parquet or `.npz`), keyed
   to the rollout id — NOT JSON (10× bloat, slow).
3. **Capture realized-token signals, not vocab distributions.** Per-token logprob of the *sampled*
   token under each model is cheap and sufficient for A_t + entropy; full per-token vocab
   distributions are infeasible (§6).
4. **Everything keyed by `(run_id, step, problem_id, language, sample_k)`** so JSONL ⇄ logprob arrays
   ⇄ checkpoints all join.

## 2. Artifacts written per run (under the output dir, committed to the volume)

```
<output-dir>/rollouts.jsonl      # one line per rollout: text+verdict+reward+length+step+teacher (P0-1)
<cwd>/sdpo_passk_<tag>_samples.jsonl  # eval: per-sample completion+length+verdict (P0-3)
# P0-2 metrics: KL logged natively by TRL (beta>0); length-split/diversity computed offline from rollouts.jsonl
# P1-6 per-token logprobs/entropy: regenerated OFFLINE from checkpoint-* -> logprobs/*.npz (NOT a run artifact)
```

## 3. P0 — close the exact gaps (persistence + metrics; pure-ish, cheap)

**Status:** P0-1 ✅ · P0-2 ✅ (KL native @ beta>0 + length-split offline from JSONL) · P0-3 ✅ ·
P0-4 ⏳ post-run procedure · P0-5 ❌ not in trl-1.6 (superseded by P0-1).

### P0-1 ✅  Persist every training rollout to JSONL
*Hook:* `reward_func` inside `make_feedback_reward_func` (`sdpo_feedback.py`) — it already computes,
per group, `completions[k]`, the fraction, verdict, and feedback. Add `_persist_rollout()` appending:
```json
{"step", "problem_id", "difficulty", "language", "sample_k",
 "completion", "n_tokens", "verdict", "reward_binary", "reward_fraction",
 "success", "is_teacher", "feedback", "clipped"}
```
*Notes:* `is_teacher` = selected by the SDPO gate (fraction ≥ threshold). Need the trainer's current
`global_step` in the reward fn (pass via the bus or a trainer ref). ~a few MB/run of text. **The single
highest-value item** — it is the thing we wished we had this hour.

### P0-2  Rich per-step scalar metrics
*Hook:* metrics dict assembled in the trainer subclass (where `self_distillation/*` keys are emitted).
Add, each step:
- **length distribution** p10/p50/p90/max, **split by success / fail / teacher-target** (the spiral is
  the *success & teacher* lengths shrinking — the mean hides it);
- **policy sequence entropy** (mean over rollouts) — the direct diversity signal;
- **KL-to-base** (log even when `beta=0`, as a pure diagnostic);
- within-group reward variance **per difficulty**.
*Payoff:* makes the collapse visible **live** → enables **early-stop on entropy/length floor**.

### P0-3  Eval: stop discarding completion text
*Hook:* `sdpo_passk.py` / `sdpo_eval_vllm.py` — where `r.choices[*].message.content` is judged then
dropped. Also write `eval/<tag>_samples.jsonl`: `{problem_id, difficulty, language, sample_k,
completion, n_tokens, verdict}`. Lets us review **base-vs-checkpoint quality**, not just pass@k.

### P0-4  Keep the whole checkpoint trajectory
*Hook:* `modal_sdpo.py` train flow. `save_total_limit=None` already keeps all locally, but the volume
ended iter-05 with only `checkpoint-20`. Copy **each** `checkpoint-N` to `/iteration-06/checkpoint-N/`
(not just latest), and record **seed + generation RNG** so completions are reproducible. Critical for
any trajectory / per-checkpoint re-generation later.

### P0-5  W&B completion table (quick win)
`SDPOConfig(log_completions=True, num_completions_to_print=N)` — a live sample in the W&B UI,
complementing the durable on-disk JSONL. Trivial.

## 4. P1 — token-level mechanistic capture

### P1-6  Per-token logprobs — **regenerate OFFLINE from checkpoints** (live capture dropped)
**Decision:** the live training-time capture is **dropped** — it cost **2–3 extra full forwards/step**,
which the run can't afford. Per-token logprobs are **deterministic given a checkpoint + the realized
tokens**, so nothing is lost by deferring them: after the run, load `base + checkpoint-N`, forward the
saved completion tokens (from the P0-1 rollout JSONL) under **policy** (adapter on), **base** (adapter
off), and **teacher** (adapter off + privileged reprompt), and gather per-token logp + entropy + A_t.
This moves the cost off the critical path and onto cheap post-hoc compute (GB10).

> **Hard dependency:** this only works if **every `checkpoint-N` is preserved** (P0-4). iter-05 kept
> only ckpt-20 — that would block the whole offline pass. Preserve all checkpoints + the seed.

Pure helpers for the offline script live in `src/sdpo_logprobs.py` (`realized_token_logp`,
`per_token_entropy`, `at_advantage`), unit-tested in `tests/test_observability.py`. The offline regen
script itself is **not yet built** (P1-7 territory).

*What we capture per token (and what we do NOT):*
- ✅ **Realized-token logprob** under three models, one float per generated token each:
  `logp_policy`, `logp_teacher` (teacher-with-privileged-context), `logp_base` (no context) — plus the
  token's **entropy** (from the policy distribution TRL already computes). This yields, per token:
  `A_t = logp_teacher − logp_base` (iter-04 diagnostic) and the entropy trace, **everywhere**.
- ✅ Optional **top-k** (e.g. k=20) logprobs/token for local distribution shape — ~k× storage.
- ❌ **NOT** full per-token vocab distributions (152k logprobs/token) — ~5 orders of magnitude larger,
  infeasible to store or even materialize at 20k-token sequences (it is literally the tensor that
  OOM-killed the GB10). Out of scope.

*Offline pass (post-run, GB10):* for each `checkpoint-N`, load `base + adapter`, read the saved
completion tokens from `rollouts.jsonl` (P0-1), and forward them under policy (adapter on) / base
(adapter off) / teacher (adapter off + reprompt) — gathering per-token logp + entropy via the helpers
above. No training-loop coupling, no extra cost on the run. Storage (float16, realized-token, ~4
arrays/token) is ~50–100 MB for a 20-step run.

### P1-7  Review tooling — `src/review_rollouts.py`
Reads `rollouts/*.jsonl` (+ optional logprob parquet) and renders:
- length-over-step (success / fail / teacher split) and entropy-over-step;
- **side-by-side completions** for a chosen `problem_id` across steps/checkpoints;
- A_t heatmap per token for a chosen rollout (recreates iter-04 Fig. 4 from *live* training data).
Makes "look through the completions as they shortened" **one command**.

## 5. Integration / de-risk

- P0-1/2/3 are persistence + config — low integration risk; covered by the **GB10 `--feedback` smoke**
  (JSONL writes without breaking the step). Done.
- **P1-6 stays OFF the training run entirely** — no loss-path coupling, no extra forwards. It runs
  post-hoc on the GB10, so it cannot affect the run's memory/perf/correctness.
- Writes go to the **output dir and are volume-committed each step** (survive OOM/kill; resumable).

## 6. Open items
- **P0-4 is now load-bearing:** wire the Modal flow to preserve EVERY `checkpoint-N` (+ seed) per
  iteration — the offline logprob pass is impossible without the full checkpoint trajectory.
- Build the offline regen script (load base+adapter, forward saved tokens, gather logp/entropy/A_t).
  Exact teacher-context must match the **training** SDPO reprompt (solution OR feedback, never both) so
  A_t reflects what actually trained — reuse `_LiveFeedbackBuilder.build()`'s logic.
- `review_rollouts.py` (P1-7) lives on the **GB10** (cheap, local) — pure post-hoc analysis.
