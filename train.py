"""The GRPO training loop — where rollout, rewards, and the loss meet.

Each step:
    1. sample a group of completions per prompt        (rollout.generate_group)
    2. score them -> rewards                            (rewards, inside rollout)
    3. group-relative advantages                        (grpo.group_advantages)
    4. re-score tokens under the *current* policy       (rollout.compute_logprobs)
    5. (optional) reference log-probs for the KL term
    6. GRPO loss -> backward -> clip -> optimizer step   (grpo.grpo_loss)

Two run modes:
    python train.py --config tiny --overfit     # CPU smoke: full cycle, no NaNs
    python train.py --config gpu  --overfit     # GPU: reward must climb on a few problems
    python train.py --config gpu                # the real run (Phase 5)

The ``--overfit`` sanity check is the make-or-break gate: if we can't drive reward
up on a tiny fixed set of problems, the bug is upstream (Phase 2/3), not here.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import get_config, parse_overrides
from data import load_gsm8k
from grpo import grpo_loss, group_advantages
from rollout import compute_logprobs, generate_group
from utils import JsonlLogger, load_model_and_tokenizer, set_seed, trainable_parameters


def reference_logprobs(policy, ref_model, cfg, full_ids, attention_mask):
    """Log-probs under the frozen reference policy, for the KL term.

    With LoRA we get the reference "for free" by disabling the adapters — the base
    weights *are* the reference, so no second copy sits in memory. Without LoRA we
    score through a separately-held frozen copy. Always under ``no_grad``.
    """
    with torch.no_grad():
        if cfg.use_lora:
            with policy.disable_adapter():
                return compute_logprobs(policy, full_ids, attention_mask)
        return compute_logprobs(ref_model, full_ids, attention_mask)


def train(cfg, *, overfit: bool) -> None:
    set_seed(cfg.seed)
    if overfit:
        # Shrink to a small fixed pool so we test pure memorization of a few problems.
        cfg = replace(cfg, num_train_examples=min(cfg.num_train_examples, cfg.overfit_n))

    policy, tokenizer, device = load_model_and_tokenizer(cfg, for_training=True)
    policy.train()
    if cfg.gradient_checkpointing:
        policy.gradient_checkpointing_enable()

    # Reference model only needed for KL when we are NOT using LoRA (LoRA disables
    # adapters instead). Skip entirely when kl_beta == 0 (the no-KL ablation).
    ref_model = None
    use_kl = cfg.kl_beta != 0.0
    if use_kl and not cfg.use_lora:
        ref_model, _, _ = load_model_and_tokenizer(cfg, for_training=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)

    params = trainable_parameters(policy)
    optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    pool = load_gsm8k("train", n=cfg.num_train_examples, seed=cfg.seed)
    logger = JsonlLogger(cfg)
    print(f"[train] {cfg.model_name} | steps={cfg.max_steps} "
          f"prompts/step={cfg.prompts_per_step} G={cfg.group_size} "
          f"pool={len(pool)} kl={'on' if use_kl else 'off'} overfit={overfit}")

    optimizer.zero_grad()
    accum = 0
    for step in range(cfg.max_steps):
        t0 = time.time()
        # Pick this step's prompts (rotating window over the pool; for --overfit the
        # pool is tiny so we revisit the same problems repeatedly).
        start = (step * cfg.prompts_per_step) % len(pool)
        batch = [pool[(start + i) % len(pool)] for i in range(cfg.prompts_per_step)]

        # 1-3: rollout (old policy, no grad) + advantages.
        rollout = generate_group(policy, tokenizer, batch, cfg, device)
        advantages = group_advantages(
            rollout.rewards, cfg.group_size,
            normalize=cfg.normalize_advantage, eps=cfg.adv_eps,
        )
        t_rollout = time.time() - t0

        # 4-5: re-score under the current policy (WITH grad) + reference for KL.
        t1 = time.time()
        logp_new = compute_logprobs(policy, rollout.full_ids, rollout.attention_mask)
        logp_ref = (reference_logprobs(policy, ref_model, cfg,
                                       rollout.full_ids, rollout.attention_mask)
                    if use_kl else None)

        # 6: loss -> backward (scaled for grad accumulation).
        loss, m = grpo_loss(
            logp_new, rollout.logp_old, advantages, rollout.completion_mask,
            logp_ref=logp_ref, clip_eps=cfg.clip_eps, kl_beta=cfg.kl_beta,
            loss_agg=cfg.loss_agg,
        )
        (loss / cfg.grad_accum_steps).backward()
        accum += 1
        grad_norm = float("nan")
        if accum == cfg.grad_accum_steps:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
            optimizer.step()
            optimizer.zero_grad()
            accum = 0
        t_update = time.time() - t1

        # --- logging ---
        if step % cfg.log_every == 0:
            frac_correct = sum(p["is_correct"] for p in rollout.reward_parts) / len(
                rollout.reward_parts)
            logger.log(step, {
                "reward": float(rollout.rewards.mean()),
                "frac_correct": float(frac_correct),
                "comp_len": float(rollout.completion_lengths().float().mean()),
                "loss": float(loss),
                "kl": m["mean_kl"],
                "ratio": m["mean_ratio"],
                "clip_frac": m["clip_frac"],
                "grad_norm": grad_norm,
                "t_rollout": round(t_rollout, 2),
                "t_update": round(t_update, 2),
            })

    # Save the trained weights (LoRA adapters or full model) for Phase 5 eval.
    out = Path(cfg.log_dir) / cfg.name / "final"
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.close()
    print(f"[train] done. saved to {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--overfit", action="store_true",
                    help="train on a tiny fixed pool to verify reward climbs")
    args = ap.parse_args()
    cfg = get_config(args.config, **parse_overrides(args.set))
    train(cfg, overfit=args.overfit)


if __name__ == "__main__":
    main()
