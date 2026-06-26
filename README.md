# SDPO Training Loop — Gemma-4-E2B-it on OJBench

Post-train **Gemma-4-E2B-it** with **SDPO (Self-Distillation Policy Optimization)** on
**OJBench** competitive-programming problems, with a hybrid held-out generalization set and a
GSM8K regression probe.

## Repository layout
```
README.md  CLAUDE.md          # this file + agent guidance
docs/                         # design & process docs (read EXPERIMENT.md first)
src/                          # all Python (training, eval, judge, Modal, plotting)
scripts/                      # local vLLM serve helpers
data/                         # committed inputs: ojb_splits.json, ojbench_selected.json
reports/                      # per-iteration results (committed): report + figures + data
  iteration-01/               #   REPORT.md, PROVENANCE.md, figures/, data/
  comparison/                 #   cross-iteration overlays (figures gitignored, regenerable)
runs/                         # per-iteration raw outputs (gitignored) — see runs/README.md
ojbench_data/                 # OJBench test cases (~2.7 GB, gitignored, re-fetchable)
```

## Docs (in `docs/`, read in this order)
| Doc | What it is |
|---|---|
| [`docs/EXPERIMENT.md`](./docs/EXPERIMENT.md) | **Design source of truth** — method, splits, configs, decisions, status |
| [`docs/FINDINGS.md`](./docs/FINDINGS.md) | **Results index** — links per-iteration reports; standing lessons |
| [`reports/iteration-01/REPORT.md`](./reports/iteration-01/REPORT.md) | **Iteration 01** — self-contained report with embedded graphs |
| [`docs/MODAL.md`](./docs/MODAL.md) | **Cloud scale-up** — run training/eval on Modal H100/H200 |
| [`docs/HANDOFF.md`](./docs/HANDOFF.md) | original next-steps handoff (executed — kept for history) |

## Code (in `src/`)
| File | Role |
|---|---|
| `_paths.py` | path resolver (works in local `src/` layout **and** the flat Modal container) |
| `sdpo_ojbench.py` | OJBench env adapter: prompts, public/private split, reward func, py+cpp judging |
| `ojbench_eval.py` | base judging primitives (extract code, run tests) |
| `sdpo_train.py` | SDPO training (TRL `SDPOTrainer` + LoRA + vLLM colocate) |
| `sdpo_eval_vllm.py` | held-out pass@1 by language via a vLLM endpoint (+W&B) |
| `sdpo_passk.py` | held-out **pass@k** (k=1,2,4,8) — the discriminative metric |
| `eval_runner.py` | GSM8K / math eval harness (regression probe) |
| `modal_sdpo.py` | run training + eval on Modal H100/H200 (see `docs/MODAL.md`) |
| `generate_slides.py`, `compare_iterations.py` | figures from committed `reports/*/data` |

## TL;DR of the current approach
- **Train easy-only** (easy+medium yields 0 successful rollouts cold → no SDPO signal).
- **LoRA r=32** on the text tower; **8k** training completions, **32k** eval cap.
- **Serve the adapter via vLLM `--enable-lora`** (not a merged checkpoint — merge drops weights on this multimodal model).
- **pass@k is the metric** (greedy pass@1 ±2/25 noise hides effects); **prototype on the GB10, scale on a single Modal H100**.
- Iteration 01: base has a real pass@k frontier; 20 steps null; **100 steps overfit/regressed** — see [`docs/FINDINGS.md`](./docs/FINDINGS.md).

## Setup
```bash
uv venv .venv && source .venv/bin/activate
uv pip install vllm trl peft accelerate wandb datasets openai huggingface_hub matplotlib pyyaml
# put WANDB_API_KEY in .env at the repo root (gitignored)
```

## Running
Scripts are in `src/`; run them with `src/` on the path. Put raw outputs under `runs/iteration-NN/`:
```bash
mkdir -p runs/iteration-02 && cd runs/iteration-02
PYTHONPATH=../../src python ../../src/sdpo_train.py --difficulties easy --max-steps 20 --output-dir sdpo_out
# cloud (from repo root):
.venv/bin/modal run src/modal_sdpo.py --difficulties easy --max-steps 100
```
Outputs land in the current directory; curated results + figures go to `reports/iteration-NN/`.
Full run order: [`docs/EXPERIMENT.md`](./docs/EXPERIMENT.md) §9 · cloud: [`docs/MODAL.md`](./docs/MODAL.md).
