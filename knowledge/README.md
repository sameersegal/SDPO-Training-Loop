# Knowledge — literature around SDPO

This repo builds on **Reinforcement Learning via Self-Distillation** (SDPO),
[arxiv 2601.20802](https://arxiv.org/abs/2601.20802) — Hübotter, Lübeck, Behric, Baumann,
Bagatella, Marta, Hakimi, Shenfeld, Kleine Buening, Guestrin & Krause (ICML 2026). SDPO is the
method our `SDPOTrainer` runs (see `CLAUDE.md` / `docs/EXPERIMENT.md` §6).

## Files
- **`citing_papers.csv`** — the original paper plus every paper Google Scholar lists as citing it.
  Columns: `arxiv_id, relationship (original|citing), related, reviewed, title, url, abstract`.
  `reviewed` (`yes`/`no`) tracks whether a human has confirmed the `related` label.
- **`src/knowledge_view.py`** — one-file web UI to browse, search, and re-classify the CSV:
  `python src/knowledge_view.py knowledge/citing_papers.csv --serve --port 7400`. Clicking a
  classification saves it to the CSV and auto-marks the paper `reviewed`; a separate "Mark reviewed"
  button confirms a correct label. `--out` writes the static read-only `papers.html` instead.
  The `abstract` column holds each paper's full arXiv abstract (pulled from the arXiv API
  2026-06-27), so the whole set is reviewable from the one file.
  - **`related`** classifies each paper's relevance to our SDPO-on-OJBench work (verified against
    each abstract, 2026-06-27):
    - `yes` — directly relevant: on-policy / self-distillation, rich-feedback RL, reasoning
      degradation / forgetting, or dense-reward LLM post-training (19 papers).
    - `tangential` — shares a method or theme but different domain/scope (recommenders, video world
      models, inference-time scaling, safety tax, curriculum) (5 papers).
    - `no` — unrelated (HCI user study, MRI reconstruction, image editing) (3 papers).
- `summary_*.md` — per-paper deep summaries produced via the `/read-arxiv-paper` skill, each with a
  "how it applies to SparkyCoder" section. One per reviewed `original`/`yes` paper:
  - `summary_sdpo_original.md` — *Reinforcement Learning via Self-Distillation* ([2601.20802](https://arxiv.org/abs/2601.20802)) — the method we run; gains are emergent at scale, hybrid SDPO+GRPO for sub-8B.
  - `summary_sd_zero_binary_to_dense.md` — *Self-Distillation Zero* ([2604.12002](https://arxiv.org/abs/2604.12002)) — reviser conditions on the *wrong* attempt → dissolves our 0-successful-rollout gotcha.
  - `summary_selfdistill_degrades_reasoning.md` — *Why Does Self-Distillation Degrade Reasoning?* ([2603.24472](https://arxiv.org/abs/2603.24472)) — iteration-01's terse-collapse = epistemic-verbalization suppression.
  - `summary_opd_length_inflation.md` — *Demystifying OPD* ([2604.08527](https://arxiv.org/abs/2604.08527)) — repetition tokens hijack reverse-KL; add truncation/repetition monitors.
  - `summary_opsdl_long_context.md` — *OPSDL* ([2604.17535](https://arxiv.org/abs/2604.17535)) — teacher on a denoised slice, not the gold solution; safe end of the richness axis.
  - `summary_opd_rock_tokens.md` — *Cornerstones or Stumbling Blocks?* ([2605.09253](https://arxiv.org/abs/2605.09253)) — ~18% "rock tokens" dominate the gradient; freeze them for ~1.5× speedup.
  - `summary_rlcsd_contrastive.md` — *RLCSD* ([2606.11709](https://arxiv.org/abs/2606.11709)) — contrast correct vs incorrect hints to cancel privilege-induced style drift.
  - `summary_self_distilled_policy_gradient.md` — *Self-Distilled Policy Gradient* ([2606.04036](https://arxiv.org/abs/2606.04036)) — our `distillation_weight=1.0` is ~1000× too strong; gate + schedule + KL anchor.
  - `summary_feedback_alignment_sd.md` — *The Role of Feedback Alignment in Self-Distillation* ([2606.11173](https://arxiv.org/abs/2606.11173)) — make feedback trace-aligned (fix only the wrong step), not outcome-aligned.
  - `summary_when_context_returns.md` — *When Context Returns* ([2606.11627](https://arxiv.org/abs/2606.11627)) — context-induced degradation; No-Context Anchoring term.
- `papers.html` — regenerable static export (gitignored).

## Source of the citing list
Pulled from Google Scholar's "cited by" cluster for SDPO (cluster id `17227046000706328640`):

> https://scholar.google.com/scholar?cites=17227046000706328640&hl=en&as_sdt=2005&sciodt=0,5

Captured **2026-06-27**. Scholar reported ~28 citing results; all 28 are recorded (paginated via
`&start=0`, `&start=10`, `&start=20`).

## Caveats
- Scholar clustering is noisy: a few entries (e.g. the MRI-reconstruction and image-editing papers)
  may be incidental rather than methodological citations.
- Most relevant to our iteration work (on-policy distillation mechanics, reasoning degradation /
  forgetting, length inflation, GRPO variants): `2603.24472`, `2604.08527`, `2604.17535`,
  `2605.09253`, `2606.04036`, `2606.13657`, `2606.11173`, `2606.00172`, `2606.11627`, `2604.12002`.
- arxiv IDs were extracted from Scholar HTML by an automated fetch — spot-verify before bulk-reading.

## Refreshing
Re-run the Scholar query above (bump `&start=` by 10 per page) and append new rows to the CSV.
