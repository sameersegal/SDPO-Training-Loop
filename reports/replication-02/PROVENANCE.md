# Replication-02 provenance

Their artifact: github.com/lasgroup/SDPO @ `7c457fc1b1f636ae794eb0362ba37d4743b06fbc`
(verl 0.7.0.dev fork), cloned at that commit inside the Modal image build.
Image: `verlai/verl:app-verl0.5-transformers4.55.4-vllm0.10.0-mcore0.13.0-te2.2` (x86 base;
their canonical GH200/ARM image is unusable on Modal) + fork `--no-deps -e` + their pins
(transformers 4.57.1, ray 2.53.0, tensordict 0.10.0, fastapi 0.127.0, wandb 0.23.1,
math-verify 0.8.0, latex2sympy2, word2number). Full freeze: `runs/replication-02/env_pins.txt`.
Our wrapper: `src/modal_verl_repl.py` (app `sdpo-verl-repl`).

Data: their 3-script prep run locally (GB10, CPU) → 131 LCBv6 problems (Feb–May 2025 window,
`livecodebench/code_generation_lite` rev `refs/pr/6`), 50/50 public/private test split,
byte-verified CODE_PROMPT + feedback formats — see `runs/replication-02/DATA_VERIFICATION.md`.
Parquets on volume `verl-repl:/data/lcb_v6/`.

Entry: `python -m verl.trainer.main_ppo --config-name sdpo` + overrides (in
`src/modal_verl_repl.py::g2_preflight`): batch 32, rollout.n=8, mini=1, prompt/response
2048/8192, TP=2, lr 1e-6 warmup 0, topk 20 (+tail, actor default), alpha 1.0, EMA 0.01,
threshold 0.5 (default), remove_thinking (default), reprompt right-trunc 10240 (default),
rollout_is token.

## Modal apps (workspace ac-W3DwsH8kQ2eULMoI1rp1zL)

| Phase | App | Cost |
|---|---|---|
| G0 build probes (CPU) | (pennies) | ~$0.1 |
| G1 0.6B smoke, attempts 1–4 | ap-ySXI0…/ap-lpFHq…/ap-c9M9T…/ap-5cyEli3jfLsCtJKFUS196L | $3.13 |
| G2 8B preflight + resume proof | ap-Z9x5APH3xfWGNWsYNoko63, ap-imbpcl9d4Ia297rUifgmM6 | $49.55 |
| G3 SDPO arm steps 1–12 (timeout'd) | ap-Kjwl2qwhOvAkBIj69Yw0YA | $125.84 |
| G3 resume steps 13–15 + final val | ap-nLmtgQaxGX7r7pPLtganrg | $46.59 |
| G3 base anchor (1 nil step + val) | ap-xsEl8IFpDDNsvF9OfTcsdk | $31.64 |
| **Total** | | **≈ $259** |

W&B: project `sdpo-verl-repl` (runs g3-sdpo-arm, g3-base-anchor, g2-preflight, g1 `g7f094ee`).
Checkpoints: volume `verl-repl:/ckpts/g3-sdpo-arm/global_step_{2..15}` (final = 15).
Logs + app ids + recipes: `runs/replication-02/` (`RUNNING_APP_ID.txt`, `g*_*.log`).

Incidents: G2 watchdog false-kill at 2400s (silent serial val; watchdog → 7200s);
G3 Modal function-timeout at step 12 (6h decorator; → 12h) — lossless resume.
