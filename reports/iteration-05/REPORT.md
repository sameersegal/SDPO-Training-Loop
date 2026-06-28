# Iteration 05 — Qwen3-8B SDPO de-risk run (logprob-gated, ≤20 steps, no token cap)

**Status: PLANNED.** First run on an **in-regime (~8B) model**, after iterations 01–03 on
Gemma-4-E2B (~2B) showed the small-scale failure the original SDPO paper predicts. The bar for this
iteration is deliberately low and concrete: **get ONE clean end-to-end run** with interpretable
telemetry, gated behind a logprob check that the self-teacher actually knows something. Beating base
is **not** required here — proving the method has signal at 8B, and that our pipeline runs on it
without collapse/OOM/hang, is the deliverable. Downstream actions are decided *after* this run.

Consolidated recommendations this executes: the **T0 list distilled from the 10
[`knowledge/`](../../knowledge/) deep reads** (cross-paper consensus on fixed teacher, demoted
distillation, collapse canaries, trace-aligned feedback). The T0 knobs are inlined in Phase 2 below.

---

## Model decision (read first)

| Candidate | Reality | Verdict for THIS run |
|---|---|---|
| "Qwen3.5-8B" | **Does not exist.** Qwen3.5 Small = 0.8B/2B/4B/**9B**, and **native multimodal** → reintroduces the gemma4 text-tower-only-LoRA + `k_norm` merge-bug hazards | ✗ |
| **Qwen3-8B** | The **SDPO paper's exact model**; **text-only** (clean all-layer LoRA, no towers); mature vLLM/TRL/PEFT support; in-regime by definition | ✅ **use this** |
| Qwen3.5-9B | Newest/strongest, but 9B, brand-new tooling, **multimodal** (LoRA complication) | Downstream upgrade once pipeline is proven (one-line model swap) |

Picking the paper's own text-only model is what makes "one clean run" achievable in the window. It
also removes the biggest infra variable (multimodal LoRA targeting) that has bitten us before.

---

## Goal & definition of "one successful run"

**Success = all of:** (1) the **logprob gate passes** (Phase 0); (2) the 20-step run **completes**
without OOM / kernel hang / judge stall; (3) the **canaries stay clean** (no length/entropy collapse,
TruncRate/RepRate sane); (4) checkpoints are **written and resumable**; (5) eval produces a
**base-vs-adapter delta** (any sign) on held-out pass@k + GSM8K. **Not required:** beating base.
That decision comes next iteration, informed by this run's telemetry.

---

## Phase 0 — The logprob gate (go/no-go BEFORE any training) ⟵ the part you asked for

**Question:** does the self-teacher, *conditioned on our judge feedback / an AC sibling*, actually
assign higher probability to the right tokens than the bare student? If not, SDPO has nothing to
distill and we **do not train**. Pure inference (forward passes + a few teacher generations) — cheap,
fast, no GPU-hours on training. New script: `src/sdpo_logprob_probe.py`.

**Probe set.** ~15–25 OJBench problems spanning easy / medium / sometimes-solvable, each with a
**known-AC reference solution** (from the judge), plus a batch of **base-model failed rollouts** with
their judge feedback text.

**Measurement A — teacher logprob gain on the correct solution.** For each problem, with `y*` a
known-correct solution and `f` the judge feedback (or an AC sibling):

```
Δ_correct = mean_t [ log q_θ(y*_t | x, f, y*_<t)  −  log π_θ(y*_t | x, y*_<t) ]
                     └ teacher: sees feedback ┘     └ student: bare prompt ┘
```

**Want `Δ_correct ≫ 0`**, sign-consistent across most problems: the feedback makes the correct
solution *more likely* under the teacher than under the unconditioned student. This is the core "is
the teacher better-informed" test from the original SDPO paper, expressed in logprobs.

**Measurement B — per-token advantage structure on the student's OWN failed rollout.** For the
student's attempt `ŷ` (a real failure) with its feedback `f`:

```
A_t = log q_θ(ŷ_t | x, f, ŷ_<t)  −  log π_θ(ŷ_t | x, ŷ_<t)      (the SDPO per-token advantage)
```

Plot `A_t` along the rollout. **Want it STRUCTURED / localized** — negative around the buggy region,
≈0 on the correct prefix (the StepAlignFB signature) — **not flat-zero** (no signal) and **not
diffuse-negative everywhere** (RefSol style-drift, the failure `feedback_alignment` warns about).

**Decision rule (explicit):**
- **GO** → `Δ_correct` clearly positive on a majority of problems **AND** `A_t` localized → the
  teacher has real, usable signal at 8B → proceed to Phase 1/2.
- **NO-GO / rethink** → `Δ_correct ≈ 0` or negative, or `A_t` is flat/diffuse → the 8B self-teacher +
  our feedback adds nothing. **That is the finding** — stop, and pivot to feedback-design
  (trace-aligned `_format_feedback`) or the SDPO+GRPO hybrid before spending training compute.

**Also verify here (cheap, same script):** the reprompt template does **not** paste the raw student
attempt into the *user* turn (the original paper's clearest footgun: entropy 0.41→0.23); and the
feedback is LeetCode/trace-shaped (error + location + failing input), the format the model has the
best in-context priors for. (`src/sdpo_feedback.py` `_LiveFeedbackBuilder`, `_format_feedback`
`src/sdpo_ojbench.py:182`.)

---

## Phase 1 — Infra bring-up & smoke (Qwen3-8B)

- **Model swap.** Point `sdpo_train.py`, the serve scripts, and `modal_sdpo.py` at `Qwen/Qwen3-8B`.
- **LoRA targets get SIMPLER.** Qwen3-8B is a standard text-only decoder → target
  `q,k,v,o,gate,up,down`_proj across **all** layers (no `language_model.*` prefix, no vision/audio
  towers, no `k_norm` merge bug). `max-lora-rank 32`, bf16, as before.
- **Thinking mode — DECIDED: ON.** Qwen3 think-ON for a stronger self-teacher on code reasoning
  (it's *why* 8B clears the bar) — consistent with the no-cap choice below, at the cost of long
  generations. Revisit only if lengths/cost explode.
- **Smoke** `sdpo_train.py --smoke --feedback --reward-mode fraction` on GB10 to validate wiring
  (loads, LoRA attaches, dense reward + feedback fire, adapter saves). Memory note: **8B + colocate
  vLLM + no token cap is H200 territory** — the smoke proves wiring; the real run is **Modal H200**.

---

## Phase 2 — The 20-step run (Modal H200) with T0 baked in

**T0 knobs (distilled from the `knowledge/` deep reads):**
- **T0-2 — Fixed teacher.** `teacher_model_kind` = **fixed/initial**, not EMA. For a *first clean
  run* this minimizes moving parts (6-paper consensus that EMA amplifies collapse). The EMA-α≈0.01
  A/B is a deliberate downstream experiment, not for run #1.
- **T0-3 — Demote distillation, verifier-dominant.** `distillation_weight` **down from 1.0** (the
  "pure" value that collapsed us at 2B). **Nuance for 8B:** the paper's 8B win used *strong*
  distillation, so don't crank to SDPG's 1e-3 either — **DECIDED: `distillation_weight=0.3`**
  (constant), reward dominant, canary-arbitrated. For only 20 steps, skip the full warmup-decay
  *schedule* (it earns its keep on long runs via end-decay). `sdpo_train.py:130`.
- **T0-4 — Two-sided canaries (kill signals).** Log per step to W&B: completion **length** (natural,
  uncapped), **TruncRate** (`finish_reason=="length"`; ≈0 by construction with no cap — that's the
  point, we *observe* natural length), **RepRate** (zlib ratio of last 10k chars > 10), **policy
  entropy**. A sharp move in length **either** direction, or an entropy crash, kills the run.

**Your constraints:**
- **No token cap initially.** Set `--max-completion-length` effectively uncapped to see the *natural*
  length distribution before choosing a sensible cap downstream. Memory: H200, `per_device_train_
  batch_size=1`, grad checkpointing — the LM-head logits tensor scales with length (the OOM gotcha),
  so watch the first steps. (This is *why* TruncRate is informative here rather than pinned at 1.0.)
- **≤20 steps.** `--max-steps 20`, **`--save-steps 2`** so the run is resumable inside the window
  (checkpoint cadence < interruption interval, per CLAUDE.md).

**Baked-in hazard mitigations (CLAUDE.md, non-negotiable):** decoupled launch
(`setsid nohup … modal run --detach … </dev/null &`), write `RUNNING_APP_ID.txt`, `enforce_eager`
(kernel hang), parallel judge with group-kill timeouts, `--resume`, no-progress watchdog.

**Data / reward (reuse iteration-03 machinery, kept minimal):** dense reward (`reward_mode=fraction`),
feedback-ON, and a **simple frontier band** for the 20 steps — easy + sometimes-solvable medium (the
"activate-the-gate" band) — rather than the full moving-frontier curriculum. Goal is *one clean run*,
not the full curriculum; that's downstream.

**KL anchor — DECIDED: OUT of run #1.** Keep the first run minimal (fewer integration variables);
revisit as T1 insurance for the longer downstream run.

---

## Phase 3 — Eval & decide

- **Eval** base vs adapter on the held-out split, **pass@k n≥8**, py & cpp × difficulty, on the same
  vLLM `--enable-lora` server; **GSM8K** regression probe. Read **all canaries** alongside.
- **Then decide downstream** (the whole point of run #1):
  - Gate passed + run clean + signs of life → **longer run**, add the warmup-decay schedule, EMA-α
    A/B, the KL anchor, then **Qwen3.5-9B** / the **SDPO+GRPO hybrid**.
  - Canaries collapsed despite 8B → diagnose with the per-token advantage probe (Phase 0 tooling).
  - (Gate failing would have stopped us at Phase 0 — no training spend.)

---

## Run order (once Phase 0 passes)

1. `python src/sdpo_logprob_probe.py --model Qwen/Qwen3-8B --feedback` → **Δ_correct + A_t plots → GO/NO-GO.**
2. `sdpo_train.py --smoke --feedback --reward-mode fraction` (GB10) → wiring green.
3. Watchdog-protected **Modal H200 pre-flight** `--max-steps 3 --save-steps 1` (~15 min, ~$) → real env + checkpoint + watchdog.
4. **The run:** Modal H200, `Qwen/Qwen3-8B`, fixed teacher, `distillation_weight≈0.3`, no token cap,
   `--max-steps 20 --save-steps 2`, feedback-ON, dense reward, canaries on, decoupled + watchdog + resume.
5. Eval (held-out pass@k n≥8 + GSM8K) base vs adapter; read canaries; **decide downstream.**

## Decisions (locked) & remaining open items
**Locked:** thinking-mode **ON**; `distillation_weight` **0.3** (constant, canary-arbitrated); **KL
anchor OUT** of run #1; teacher **fixed** (not EMA); **no token cap**; **≤20 steps**, `save-steps 2`.
**Still to settle at run time:** (c) Phase-0 probe-set size/composition and the GO threshold on
`Δ_correct` (calibrate on a few easy problems where feedback *obviously* should help); (d) data band
for the 20 steps; (f) confirm Qwen3-8B H200 memory headroom under no-cap long generations.

## Provenance (to fill on run)
- Model: `Qwen/Qwen3-8B`. Probe: `data/logprob_gate.json` (Δ_correct, A_t). Adapter: Modal
  `sdpo-outputs:/iteration-05/`. Design source of truth: [`docs/EXPERIMENT.md`](../../docs/EXPERIMENT.md).
