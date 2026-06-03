"""The GRPO algorithm: group-relative advantages + the clipped PG loss + KL.

This file *is* the project. Everything else is plumbing that feeds it. Given a
batch of scored completions (their log-probs, rewards, and masks) it produces a
single scalar loss whose gradient nudges the policy toward the completions that
beat their group, and away from the ones that lagged.

The pieces, in order:

1. **Group-relative advantage.** GRPO's signature move: no critic network. For
   each group of G completions to the *same* question, the group's mean reward is
   the baseline. ``A_i = (r_i - mean) / (std + eps)``. Positive = better than your
   siblings → push up; negative = worse → push down. One scalar per completion,
   later broadcast to all its tokens.

2. **Policy ratio.** ``ratio = exp(logp_new - logp_old)`` per token — how much more
   (or less) likely the *current* policy is to produce this token than the policy
   that sampled it. (Exp of a log-difference = probability ratio.)

3. **Clipped surrogate (from PPO).** ``min(ratio*A, clip(ratio, 1-ε, 1+ε)*A)``. The
   clip stops a single update from moving the policy too far on one batch.

4. **KL penalty** against a frozen reference policy, using the low-variance, always
   non-negative **k3 estimator**: ``exp(Δ) - Δ - 1`` with ``Δ = logp_ref - logp_new``.
   A leash that keeps the model from drifting into degeneracy while chasing reward.

Final per-token loss = ``-(clipped surrogate) + β·KL`` (we minimise, so negate the
objective we want to maximise), then masked-averaged over real generated tokens.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, group_size: int, *,
                     normalize: bool = True, eps: float = 1e-4) -> torch.Tensor:
    """Group-relative advantages. ``rewards`` is [B] in group-contiguous order
    (prompt0's G samples, then prompt1's, ...). Returns [B].

    With ``normalize=True`` we divide by the group std (the original GRPO form);
    with ``normalize=False`` we only subtract the mean (the Dr. GRPO variant that
    avoids the std-based length/difficulty bias — used in a Phase 6 ablation).
    A degenerate group where every reward is equal gets advantage 0 (the ``eps``
    keeps it finite), so it contributes no gradient — which is correct: if all
    siblings tied, none was better.
    """
    assert rewards.numel() % group_size == 0, "rewards not divisible by group_size"
    groups = rewards.view(-1, group_size)                 # [P, G]
    mean = groups.mean(dim=1, keepdim=True)
    centered = groups - mean
    if normalize:
        std = groups.std(dim=1, unbiased=False, keepdim=True)
        adv = centered / (std + eps)
    else:
        adv = centered
    return adv.reshape(-1)                                 # [B]


def _masked_mean(x: torch.Tensor, mask: torch.Tensor, agg: str) -> torch.Tensor:
    """Average ``x`` over masked tokens.

    ``agg="token"``: one global mean over all real tokens (DAPO / token-level).
    ``agg="seq"``:   per-sequence mean first, then mean over sequences (original
    GRPO; longer sequences get downweighted per token). This choice is exactly the
    bias DAPO / Dr. GRPO discuss; we expose it for the Phase 6 ablation.
    """
    mask = mask.to(x.dtype)
    if agg == "token":
        return (x * mask).sum() / mask.sum().clamp(min=1.0)
    if agg == "seq":
        per_seq = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return per_seq.mean()
    raise ValueError(f"unknown agg {agg!r}")


def grpo_loss(logp_new: torch.Tensor, logp_old: torch.Tensor,
              advantages: torch.Tensor, completion_mask: torch.Tensor, *,
              logp_ref: torch.Tensor | None = None,
              clip_eps: float = 0.2, kl_beta: float = 0.0,
              loss_agg: str = "seq") -> tuple[torch.Tensor, dict]:
    """Compute the scalar GRPO loss and a metrics dict.

    Shapes: ``logp_new/logp_old/logp_ref/completion_mask`` are [B, L]; ``advantages``
    is [B] (one scalar per completion, broadcast across its L tokens). Set
    ``kl_beta=0`` or ``logp_ref=None`` to drop the KL term (no-KL ablation).
    """
    adv = advantages.unsqueeze(1)                          # [B, 1] -> broadcasts

    # (2) per-token probability ratio between the current and sampling policies.
    ratio = torch.exp(logp_new - logp_old)                # [B, L]

    # (3) clipped surrogate; we *maximise* this, so the loss negates it.
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)         # [B, L]

    per_token_loss = -surrogate                           # [B, L]

    # (4) optional KL penalty via the non-negative k3 estimator.
    if kl_beta != 0.0 and logp_ref is not None:
        delta = logp_ref - logp_new                       # [B, L]
        kl = torch.exp(delta) - delta - 1.0               # >= 0 elementwise
        per_token_loss = per_token_loss + kl_beta * kl
    else:
        kl = torch.zeros_like(ratio)

    loss = _masked_mean(per_token_loss, completion_mask, loss_agg)

    # --- metrics (reported with a plain token-level mean for interpretability) ---
    with torch.no_grad():
        m = completion_mask.to(ratio.dtype)
        denom = m.sum().clamp(min=1.0)
        clip_frac = ((ratio - 1.0).abs() > clip_eps).to(ratio.dtype)
        metrics = {
            "loss": loss.detach(),
            "mean_ratio": (ratio * m).sum() / denom,
            "mean_kl": (kl * m).sum() / denom,
            "clip_frac": (clip_frac * m).sum() / denom,
            "mean_adv": advantages.mean(),
        }
        metrics = {k: float(v) for k, v in metrics.items()}
    return loss, metrics
