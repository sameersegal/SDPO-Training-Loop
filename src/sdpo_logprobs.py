"""Offline per-token logprob/entropy helpers (iteration-06).

The *live* training-time capture was dropped: it cost 2–3 extra full forwards per step,
which we can't afford. Per-token logprobs are **deterministic given a checkpoint + the
realized tokens**, so we regenerate them OFFLINE from the saved `checkpoint-*` adapters
instead (which is why preserving EVERY checkpoint — OBSERVABILITY.md P0-4 — is now load-
bearing). These are the pure, model-agnostic building blocks for that offline pass:

  realized_token_logp(logits, token_ids)  log π of the actually-sampled token, per position
  per_token_entropy(logits)               H = -Σ p·log p, chunked over time (memory-safe)
  at_advantage(rec)                        A_t = log π_teacher(ŷ|x,c) − log π_base(ŷ|x)  [iter-04]

Workflow (offline): load base + a checkpoint adapter, forward the saved completion tokens
under {policy=adapter on, base=adapter off, teacher=adapter off + privileged reprompt},
gather these per-token, and store/inspect. No training-loop coupling.
"""
import numpy as np


def realized_token_logp(logits, token_ids):
    """Per-token logprob of the realized token. `logits` [B,T,V], `token_ids` [B,T].
    Uses the memory-efficient gather-after-logsumexp (no full log_softmax materialized)."""
    import torch
    logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return torch.gather(logp, -1, token_ids.unsqueeze(-1)).squeeze(-1)


def _entropy_chunked(logits, chunk=4096):
    """Per-token entropy H = -Σ p·log p, computed in time-slices so the full-vocab
    softmax tensor is never materialized for the whole (long) sequence at once."""
    import torch
    T = logits.size(1)
    out = torch.empty(logits.size(0), T, dtype=torch.float32, device=logits.device)
    for s in range(0, T, chunk):
        lp = torch.log_softmax(logits[:, s:s + chunk, :], dim=-1)
        out[:, s:s + chunk] = -(lp.exp() * lp).sum(-1)
    return out


# back-compat / readable alias
per_token_entropy = _entropy_chunked


def at_advantage(rec):
    """A_t = teacher − base per token (iteration-04). NaN/None where no teacher context.
    `rec` is a dict-like of per-token arrays (e.g. a loaded npz). Pure helper."""
    if "teacher_logp" not in rec or "base_logp" not in rec:
        return None
    return np.asarray(rec["teacher_logp"], dtype=np.float32) - np.asarray(rec["base_logp"], dtype=np.float32)
