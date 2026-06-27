# Demystifying OPD: Length Inflation and Stabilization Strategies for Large Language Models

- **arXiv:** [2604.08527](https://arxiv.org/abs/2604.08527) (ICML 2026 preprint)
- **Authors:** Feng Luo, Yu-Neng Chuang, Guanchu Wang, Zicheng Xu, Xiaotian Han, Tianyi Zhang, Vladimir Braverman (Rice / UNC Charlotte / Johns Hopkins / Case Western)
- **Why it matters to us:** This is a training-dynamics post-mortem of **on-policy distillation (OPD)** — the same family as our `SDPOTrainer` (the paper explicitly cites Hübotter et al.'s SDPO, `hubotter2026reinforcement`, and the companion "self-distillation degrades reasoning" paper, `kim2026does`). It pins down a **length-driven instability that destroys held-out accuracy while the teacher and loss stay fixed** — exactly the "loss is not a quality signal" trap we hit in iteration-01 — and proposes two cheap stabilizers (a KL anchor + a golden-data mixture) that map directly onto knobs we can add to `src/sdpo_train.py`.

---

## TL;DR

Standard OPD (student rolls out under its own policy; a stronger teacher supplies a **per-token reverse-KL reward** `r = log π_T − log π_θ`) trains stably for a while, then undergoes an **abrupt phase transition**: within ~30 steps, rollouts inflate toward the max-length budget, the **truncation rate jumps to ~1.0**, the **repetition rate spikes from ~0 to 0.3–0.6**, and **validation accuracy collapses** — all while the teacher and objective are unchanged. Cause: **repetitive tokens systematically receive larger reverse-KL advantages** (4–9× the regular tokens). They are harmless while rare, but once the student starts repeating, their frequency × oversized advantage dominates the gradient, creating a **self-reinforcing loop** — the student effectively *hacks* the teacher's likelihood signal. Fix = **Stable-OPD**: (1) a **reference-KL constraint** (anchor to the initial checkpoint) to cap policy drift, plus (2) **mixture distillation** (blend in off-policy "golden" SFT trajectories each step to keep a stable fraction of complete, non-truncated sequences). Together they prevent the collapse and add **+7.2% average accuracy** over standard OPD across six math benchmarks.

## The failure mode, step by step

1. **Two cheap monitors expose it.** They track, on both training rollouts *and* held-out prompts:
   - **TruncRate** — fraction of rollouts that hit the max-length budget without emitting EOS (i.e., `finish_reason == "length"`).
   - **RepRate** — fraction of long rollouts whose tail is extremely compressible (zlib compression ratio of the last 10k chars `> 10`).
2. **Stable early phase.** Accuracy climbs, TruncRate sits at a moderate baseline (~0.2–0.5 on rollouts), RepRate ≈ 0.
3. **Abrupt phase transition (~30 steps).** TruncRate → 1.0, RepRate → 0.3–0.6, **held-out accuracy drops sharply** at nearly the same step. Robust across three student/teacher pairs (Qwen2.5-Math-1.5B/7B; DeepSeek-R1-Distill-7B and OpenThinker3-7B teachers) → a property of OPD, not a single bad config.
4. **Rollout-level cause.** At onset, response length jumps to the budget, both student and teacher log-probs become much less negative, but the **teacher's rises more**, so the average reverse-KL advantage **spikes**.
5. **Token-level cause.** Repetitive-tail tokens carry **4–9× the advantage** of regular tokens *throughout* training; once they reach ~30% of tokens after collapse, their frequency × advantage takes over the update.
6. **Mechanism.** Decompose the policy gradient over states in/out of the repetitive-tail set R. As visitation to R grows, its term dominates and pushes for more continuations in R — a feedback loop where repetition is rewarded for being high-likelihood under the teacher. **This is distinct from the GRPO/Dr.GRPO/DAPO "longer-is-better" sequence-length bias** — it is a token-level reverse-KL pathology specific to OPD's on-policy dynamics.

## The fix: Stable-OPD (two complementary knobs)

- **Mixture distillation** (`L_mix = L_OPD + λ_gold · L_SFT`): each step, alongside the on-policy rollout, include one off-policy **golden** trajectory (complete, correctness-checked teacher CoT, ≤8192 tokens) for the *same* prompt and add a standard SFT term. The golden anchor keeps a stable fraction of complete, non-repetitive trajectories in every batch, rebalancing gradients away from degenerate rollouts. Critically, it **leaves the teacher signal on on-policy rollouts unchanged** — so it does *not* reintroduce the privileged-context "epistemic suppression" problem of `kim2026does` (golden data enters only via the auxiliary SFT term, not by refining the teacher).
- **Reference-KL regularization** (`+ β_KL · D_KL(π_θ ‖ π_ref)` at visited prefixes, `π_ref` = initial checkpoint): directly caps per-step policy drift toward the long/repetitive region.

## Key results

| Qwen2.5-Math-1.5B | Avg | MATH-500 | Minerva | Olympiad | AMC | AIME24 | AIME25 |
|---|---|---|---|---|---|---|---|
| Base | 16.0 | 28.0 | 9.6 | 21.2 | 26.4 | 7.2 | 3.6 |
| SFT | 31.9 | 70.6 | 26.8 | 31.3 | 37.8 | 11.7 | 13.2 |
| GRPO | 30.1 | 61.8 | 26.8 | 32.0 | 40.2 | 11.8 | 7.7 |
| **OPD (standard)** | 28.9 | 56.7 | 23.4 | 31.0 | 35.9 | 11.1 | 15.0 |
| **Stable-OPD** | **36.1** | **73.9** | **32.6** | **37.4** | **43.0** | **13.8** | **16.0** |

- **Standard OPD underperforms even plain SFT and GRPO** (28.9 vs 31.9 vs 30.1 on 1.5B; 43.8 vs 44.1 vs 45.5 on 7B) — the instability wastes the dense token-level supervision. Stable-OPD recovers and surpasses all (47.6 avg on 7B, beating tuned RLVR pipelines like Oat-Zero/PRIME-Zero).
- **Ablation (RQ3, the load-bearing table):** OPD 28.0 → +KL 29.7 → **+KL +Mixture 35.7**. **KL alone is a modest patch; the mixture is what does the heavy lifting, and the two are complementary** — KL limits abrupt token-level shifts, mixture supplies the stable non-truncated anchor.
- **Dynamics (RQ2):** under Stable-OPD the four trunc/rep curves stay flat (or drift only mildly and *late*) instead of the sharp inflation seen under OPD.

---

## How this maps onto SparkyCoder (the important part)

**Read the direction carefully.** This paper's pathology is length **inflation** (runaway *long*, repetitive, truncated rollouts), whereas iteration-01 was length **collapse** (mode-collapse to *terse* outputs — see `CLAUDE.md`, `docs/FINDINGS.md`). These are *opposite* surface symptoms, but they are the **same class of failure**: an on-policy distillation loop where the teacher-likelihood signal drives the student to a degenerate length regime, **held-out accuracy and GSM8K crater, and the SDPO loss never flags it.** The companion paper (`summary_selfdistill_degrades_reasoning.md`) explains the *terse* attractor (privileged-context epistemic suppression); this paper explains the *verbose* attractor (reverse-KL repetition hacking). **We should instrument for both, because we don't yet know which attractor a given config sits in** — and the cheapest single canary covers both.

- **Length is the leading indicator we already half-have.** This paper is the strongest argument yet for the `CLAUDE.md` budget rule "watch the first few steps and kill early." We already log **completion length per step** during training. The paper says: a *sharp move in length in either direction*, accompanied by an advantage spike, precedes the held-out collapse by ~30 steps and is invisible in the loss. That is a free kill signal for our "watch the first few steps" / per-step-cost discipline.
- **Our reverse-KL-style teacher reward is the exact surface that gets hacked.** `src/sdpo_train.py` runs `distillation_mode="topk_logits"`, `distillation_weight=1.0` (pure self-distillation) with `teacher_model_kind="ema"`. OPD's reverse-KL reward and our top-k logit distillation both push the student toward teacher-high-likelihood tokens; the paper's lesson is that **degenerate continuations can be teacher-high-likelihood**. Code, with its long boilerplate / repeated includes / repeated test-harness scaffolding, is *plausibly* a domain where repetitive tails are easy to fall into — worth checking before trusting a long run.
- **EMA teacher is again the suspect.** Both this paper (`π_ref` = *fixed initial* checkpoint as the anchor) and `kim2026does` (fixed teacher beats EMA) point the same way: our `teacher_model_kind="ema"` (`src/sdpo_train.py:133`) creates a moving target that can co-drift with the student into the degenerate regime. **Cheap, well-motivated single-knob experiment:** try a fixed/initial reference and/or add a reference-KL term.
- **We already stumbled onto half of the mixture-distillation idea.** Our `include_environment_feedback` / `environment_feedback_only_without_solution` path (`src/sdpo_train.py:140-141`) was added to give the **all-fail groups** a signal. Stable-OPD's mixture term is a more principled version: keep a **stable fraction of complete, correctness-checked golden trajectories** in every batch as an off-policy SFT anchor. Our judge (`src/sdpo_ojbench.py`, `judge_completion()`) already gives us the correctness check needed to build such a `D_gold` from AC solutions.

### Concrete things to instrument / try

1. **Add TruncRate + RepRate to training and eval — this is the highest-value, lowest-cost item.**
   - **TruncRate is almost free:** training rollouts already know if they hit `max_completion_length` (8192, `src/sdpo_train.py:35`); in eval we already capture `finish_reason` (`src/sdpo_eval_vllm.py:57`) — just aggregate `finish_reason == "length"` into a rate. A rising trunc rate on held-out prompts is the paper's canonical early collapse flag.
   - **RepRate is ~5 lines:** zlib-compress the last 10k chars of each rollout; flag `compression_ratio > 10`. Add to `sdpo_passk.py` / `sdpo_eval_vllm.py` and log per step to W&B alongside completion length.
2. **Plot per-token advantage for repetitive vs regular tokens** (or at minimum, mean completion length + a repetition proxy) during the first ~30 training steps of any long run — that is precisely where the paper sees the phase transition begin, and it lines up with our "preflight / watch the first few steps" rule.
3. **Try a reference-KL anchor** to the initial Gemma-4-E2B checkpoint (`β_KL` small). Single knob, directly addresses runaway drift, and the ablation shows it's a safe-but-modest gain on its own.
4. **Try mixture distillation with a judge-verified `D_gold`** of AC OJBench solutions as an off-policy SFT anchor — the ablation says this, *combined* with KL, is what actually fixes OPD (28.0 → 35.7). This is the more substantial change (new data path + loss term) so it warrants a GB10 smoke before a Modal run, per the de-risk ladder.
5. **Treat both length directions as kill signals.** Iteration-01 collapsed *terse*; this paper collapses *verbose*. The monitor should alert on a sharp move in **either** direction within the first tens of steps.

### Caveats / where we differ

- **All evidence is math reasoning on 1.5B/7B Qwen-Math models with long `<think>` CoT.** Our model is **Gemma-4-E2B** on **competitive programming** (C++/Python). Whether code rollouts fall into the repetitive-tail attractor is an open question — code has natural repetition (loops, scaffolding) that could either trigger it sooner or be filtered by the judge. Verify the RepRate proxy is meaningful on Gemma code outputs before trusting it as a kill signal.
- **This is OPD with an explicit reverse-KL token reward, not TRL's exact SDPO objective.** The *mechanism* (teacher-likelihood signal hackable by degenerate continuations; on-policy loop amplifies it) transfers; the precise advantage formula does not map 1:1 to our `topk_logits` distillation.
- **Mixture distillation's golden data is the opposite design choice from our current feedback path.** The paper is careful that golden data enters *only* as an auxiliary SFT term and does **not** refine the teacher — precisely to avoid the epistemic-suppression failure of `kim2026does`. If we add golden anchoring, keep it out of the teacher conditioning, or we risk trading the inflation failure for the terse-collapse failure.
- **Their fix needs a separate SFT/golden corpus and an extra forward pass per step** (cost on our GB10/H200 budget). Start with the near-free monitors (#1) before paying for #3–#4.

## One-line lesson

On-policy distillation has a length trap in **both** directions — verbose-repetitive (this paper) and terse-collapsed (iteration-01) — that **tanks held-out accuracy while the loss stays flat**; so the cheap, immediate win is a **per-step length + truncation + repetition monitor** (we already log length and capture `finish_reason`) as a kill signal, and the principled fixes are a **reference-KL anchor** and a **judge-verified golden-data mixture**, with our **EMA teacher** the first knob to reconsider.
