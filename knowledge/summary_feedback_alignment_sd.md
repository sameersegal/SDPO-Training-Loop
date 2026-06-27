# The Role of Feedback Alignment in Self-Distillation

- **arXiv:** [2606.11173](https://arxiv.org/abs/2606.11173) (ICML 2026 Workshop on RL from World Feedback, RLxF)
- **Authors:** Semih Kara, Oğuzhan Ersoy (Gensyn)
- **Why it matters to us:** This paper studies **exactly the knob iteration-02 turns** — *what text you feed the SDPO self-teacher*. It builds on the same SDPO lineage as our `SDPOTrainer` (Hübotter et al. 2026; OPSD/Zhao et al. 2026) and shows that **how the feedback is structured matters as much as whether it is correct**: feedback that is *aligned to the student's own reasoning trace* turns self-distillation into implicit process supervision, while a correct-but-unaligned reference solution produces a *diffuse, partly-counterproductive* signal. Our `_format_feedback` (`src/sdpo_ojbench.py:182`) is currently **verdict/outcome feedback, not trace-aligned** — this paper is the strongest argument yet for making it align to the rollout.

---

## TL;DR

Self-distillation supplies dense, per-token credit by matching a **student** (sees only the question) to a **self-teacher** (sees question + extra context `c`). The per-token advantage `A_t = log π(ŷ_t | x, c, y_<t) − log π(ŷ_t | x, y_<t)` measures how much the context shifts the next-token prediction — so **the content and structure of `c` is the entire learning signal.** The paper compares three contexts on math reasoning (Qwen3-1.7B solver, frozen QwQ-32B critic, OpenMathReasoning):

1. **GRPO** — binary reward, no self-distillation (baseline).
2. **RefSol** — teacher conditioned on the dataset's **reference solution** (≈ OPSD / our "successful-rollout-as-teacher" path).
3. **StepAlignFB** — teacher conditioned on a **step-by-step critique aligned to the student's own trace**: copy correct steps verbatim, rewrite only the wrong step.

**StepAlignFB wins on every accuracy metric** — beating GRPO by **+16.11 Avg@12** and RefSol by **+5.27 Avg@12** (and +13.33 Majority-Vote@12), *despite never seeing the ground-truth derivation*. The mechanism: aligned feedback **concentrates the negative advantage at the error-adjacent tokens** and **reinforces the correct surrounding steps** (PRM-like), whereas the reference solution diverges in surface form even where the student was right, so it produces **diffuse negative advantage across the whole rollout** — mixing genuine error-correction with mere stylistic disagreement.

## Method / findings

**Setup.** Solver–critic loop (their Fig. 1): solver emits step-tagged traces `<step_1>...<step_N><answer>`; frozen critic grades them; solver is trained with forward-KL self-distillation using the critic's feedback as context `c`. Only the feedback `f` varies across conditions — solver, loss, divergence, all hyperparameters fixed. Group size **G=1** for SD (1 rollout), G=8 for GRPO; both trained 7 epochs (so dataset exposure isn't a confound). **Teacher is held FIXED at the initial LoRA-disabled base policy** (no EMA). Dataset deliberately filtered to the **"learnable but hard"** band — problems the 1.7B solves rarely (Avg@16 < 5/16) but the critic *can* solve — so the model gets *actionable* feedback rather than being handed a full solution (which would collapse RefSol and StepAlignFB into the same thing). 312 problems, 282 train / 30 eval.

**Headline results (per-metric best checkpoint, steps 10–70):**

| Method | Pass@12 | Maj@12 | Avg@12 |
|---|---|---|---|
| GRPO | 76.67 | 26.67 | 19.72 |
| RefSol | 86.67 | 43.33 | 30.56 |
| **StepAlignFB** | **90.00** | **56.67** | **35.83** |

Both SD variants dominate GRPO throughout training (≈8-point Avg@12 gap); StepAlignFB then beats RefSol by another ~5 points. The large Majority-Vote gain says StepAlignFB **sharpens** probability onto correct answers (not just covers them) — the regime that benefits most from test-time aggregation.

**The mechanism (per-token advantage analysis — the heart of the paper):**

- **StepAlignFB ≈ a process reward model for free.** At incorrect steps the self-teacher diverges sharply from the student → large *negative* advantages localized at the error; at correct steps (including the prefix before the error) the critic faithfully preserves the student's path → *positive* advantages. Localized credit, no PRM training, no per-step scalar labels.
- **RefSol gives a diffuse, suppressive signal.** A complete *alternative* derivation reaches the right answer via different notation/phrasing, so the teacher prefers its surface form **even on steps the student got right** → diffuse negative advantage almost everywhere. It conflates "you erred here" with "I'd have written it differently."
- **Induction-head copying governs everything (the "faithful-scribe convention").** Three regimes, all explained by `[A][B]...[A]→[B]` copy heads (Olsson et al. 2022):
  - **Full verbatim quote of the wrong step** ("here is your previous attempt … step N is wrong") → the model *copies the erroneous continuation*; the appended correction arrives too late → advantages diffusely **positive** (reinforces the error!). Matches Hübotter et al.'s finding that including the raw attempt biases the teacher and kills exploration.
  - **Omit the trace entirely** ("this step is correct") → no in-context anchor → teacher drifts to other surface forms → diffuse **negative** advantage even on correct steps.
  - **Partial verbatim up to (not including) the error, then the corrected step** → anchors the correct prefix (positive) and lets the fresh correction govern the error position (sharp negative). This is the *only* configuration that works, and the critic prompt is engineered to elicit it (4-case schema A/B/C/D; "copy correct steps verbatim, rewrite only the wrong one").

**Caveat the authors flag:** StepAlignFB needs a strong critic (QwQ-32B here) → real cost/complexity. RefSol is cheaper and *reusable across runs* (not tailored per rollout). All evidence is OpenMathReasoning only — transfer to other domains/sizes is untested. SD also benefits from **early stopping** (5–6 of 7 epochs peaked); a fixed end-of-run eval *understates* SD's ceiling, so per-checkpoint selection on a held-out set is required for a fair read.

---

## How this maps onto SparkyCoder (the important part)

This paper is about *the precise design decision iteration-02 made* — wiring live judge feedback into the teacher via `include_environment_feedback` (`src/sdpo_train.py:140`) and `_format_feedback` (`src/sdpo_ojbench.py:182`). It gives us a **theory of what makes that feedback help vs. do nothing**, plus a concrete diagnostic (per-token advantages) we don't currently look at.

- **Our feedback is OUTCOME-aligned, not TRACE-aligned — this is the gap.** `_format_feedback` emits `Verdict: WA. Passed 16/20 (80%). Failing test '...'. Input: … Expected output: … Your output: …`. That is genuinely useful *correctness* signal (and the pass-rate proximity line is a nice touch), but in the paper's terms it sits **between RefSol and StepAlignFB, leaning RefSol**: it tells the teacher *that* and *roughly how badly* the rollout failed, and hands it I/O examples, but it is **not aligned to the student's reasoning/code trace** — it doesn't say *which line/step* of the student's program is wrong, nor copy the correct prefix verbatim. By the paper's mechanism, that risks a **diffuse** advantage: the teacher writes a fresh correct solution in its own style and the gradient pushes against the student's whole rollout, not the buggy region.
- **`environment_feedback_only_without_solution=True` (`src/sdpo_train.py:141`) is well-motivated by this paper.** We deliberately give the teacher *prompt + feedback* OR *prompt + solution*, never both. The paper's "omit-the-solution" regime warns this can drift on correct steps — **but** our memory-constrained choice (fits the 80 GB H100) is defensible because our feedback already contains the failing I/O, i.e. it's not a bare "this is wrong." The actionable question is whether *adding a verbatim copy of the student's correct prefix* would convert our signal from RefSol-like to StepAlignFB-like.
- **Teacher kind: the paper uses FIXED-initial; we use EMA.** `src/sdpo_train.py:133` sets `teacher_model_kind="ema"`. **Every** SD paper we've now summarized (this one + 2603.24472) used a **fixed teacher** and explicitly argued against a moving one. This is a second independent vote for trying `teacher_model_kind="fixed"`/initial next iteration — cheap, single-knob.
- **Their dataset filter IS our "learnability frontier" plan.** They keep only problems the solver fails *but the critic can solve* (Avg@16 < 5/16, high formatting accuracy), precisely so feedback is *actionable* rather than a disguised full solution. That is exactly the iteration-02 frontier idea (`CLAUDE.md`: "easy + sometimes-solvable medium"). It also dodges our easy+medium → 0-rollout gotcha *and* the iteration-01 low-coverage overfit, because filtering to a solver-tractable hard band keeps both a learning signal and OOD-relevant difficulty.

### Concrete things to try / instrument

1. **Make feedback trace-aligned (the headline experiment).** Upgrade `_format_feedback` toward StepAlignFB for code: include the **student's own code verbatim up to the failing region**, then the verdict + the specific failing input/expected/got. For WA/RE this means pointing at *the student's code*, not just abstract I/O. Even cheaply: prepend "Your submitted code: ```...```" + "It fails on input X with Y instead of Z" so the teacher anchors on the student's tokens (induction-head copy) and concentrates the correction. This is the single change the paper most predicts will help.
2. **Don't quote a WRONG line verbatim.** The paper's sharpest practical rule: a verbatim quote of the *erroneous* span back-fires (the teacher copies the bug → positive advantage on the error). So if we add code-quoting, quote the **correct prefix** and *describe* (don't paste) the buggy tail. For our judge, "passed 16/20" near-misses are the ideal StepAlignFB case; a 1/20 rollout is their Case C/D "write a fresh solution," which is fine.
3. **Add a per-token advantage diagnostic.** The paper's whole diagnosis came from plotting `A_t = log π(ŷ|x,c,y_<t) − log π(ŷ|x,y_<t)` along a rollout. We already compute both terms inside `FeedbackSDPOTrainer` (`src/sdpo_feedback.py`). Logging a handful of these per N steps (mean |A_t|, fraction of tokens with negative A on AC rollouts, error-region vs. prefix advantage ratio) would tell us **whether our feedback is StepAlignFB-localized or RefSol-diffuse** — directly, cheaply, before pass@k confirms anything. A *diffuse-negative advantage on AC rollouts* would be the red flag that our feedback is fighting correct code.
4. **Switch the teacher to fixed/initial** and compare held-out pass@k against the EMA run. Backed by two SD papers now.
5. **Adopt their filter as the frontier selector.** Pre-score the candidate problems with base-model Avg@k and keep "solver-fails-but-judge-can-AC" — gives actionable feedback *and* avoids low-coverage collapse. Our judge (`judge_completion`) already gives us the AC oracle to build this filter.
6. **Per-checkpoint eval, early-stop on pass@k.** They peaked at 5–6 of 7 epochs and warn end-of-run eval understates SD. Reinforces our standing rule: pass@k per checkpoint, early-stop — never trust the SD loss or a single final checkpoint.

### Caveats / where we differ

- **Domain is math CoT with explicit `<step_N>` tags; ours is competitive-programming code.** Their "step alignment" assumes a step-segmented reasoning trace to align critique to. Our rollouts are *code*, not numbered reasoning steps — "trace alignment" for us likely means **line/function-level** or "correct-code-prefix" alignment, and may need the model to actually emit a reasoning-then-code structure (cf. `CP_METHOD_SYS` in `sdpo_ojbench.py`) for there to be a trace to align to. The *mechanism* (anchor correct tokens, localize the negative signal at the error) should transfer; the literal step schema won't.
- **They need a 32B critic; we have a lightweight judge, not an LLM critic.** Our "critic" is the deterministic OJBench judge — *cheaper and more reliable than QwQ-32B for verdicts*, but it produces I/O diffs, not natural-language step critiques. To get true StepAlignFB we'd either (a) template the judge output into trace-aligned form (cheap, deterministic — preferred), or (b) add an LLM critic stage (cost/complexity the paper warns about). Option (a) is the SparkyCoder-native move and probably most of the win.
- **They train a 1.7B with a fixed teacher and G=1; we run Gemma-4-E2B with EMA.** Their config (LoRA r=64/α=128 on `{q,k,v,o,gate,up,down}_proj`, lr 5e-6, forward-KL/JSD, fixed teacher) is a close cousin of ours and a useful reference point for hyperparameters.
- **This is a positive/"how to make SD work" paper; the companion (2603.24472) is the failure-mode paper.** Read together: 2603 says *don't over-compress with a too-rich teacher on low coverage*; 2606 says *if you do feed rich feedback, align it to the student's trace so the signal localizes instead of diffusing*. For us that's one coherent prescription: **broaden coverage (frontier filter), keep the teacher fixed, and make the feedback trace-aligned and localized.**

## One-line lesson

In self-distillation, **feedback alignment ≈ feedback quality**: a correct-but-unaligned reference solution diffuses the gradient across the whole rollout (even correct steps), while feedback that copies the student's correct prefix and rewrites only the wrong step turns SD into a free process-reward model — so for SparkyCoder the move is to make `_format_feedback` **trace-aligned to the student's code** (anchor the correct prefix, localize the correction), switch to a **fixed teacher**, filter to the **solver-fails/judge-can-AC frontier**, and **instrument the per-token advantage** to verify our signal is localized, not diffuse.
