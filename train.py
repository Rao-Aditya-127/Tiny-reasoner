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
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

# Reduce CUDA fragmentation on small (16 GB) cards. Must be set before torch inits CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import get_config, parse_overrides
from data import load_gsm8k
from eval import evaluate
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
        # Recompute activations in backward instead of storing them -> ~10x less
        # activation memory, the lever that lets a large batch fit. use_reentrant=False
        # is the modern variant; enable_input_require_grads is required so gradients
        # flow back to the LoRA params through the (frozen) checkpointed base layers.
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        policy.enable_input_require_grads()

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
    run_dir = Path(cfg.log_dir) / cfg.name
    print(f"[train] {cfg.model_name} | steps={cfg.max_steps} "
          f"prompts/step={cfg.prompts_per_step} G={cfg.group_size} "
          f"pool={len(pool)} kl={'on' if use_kl else 'off'} overfit={overfit}")

    # Held-out eval set, fixed across the run so the accuracy curve and the
    # qualitative samples (same questions every time) are directly comparable.
    eval_examples = load_gsm8k("test", n=cfg.eval_n, seed=cfg.seed) if cfg.eval_every else []
    best_acc = -1.0

    def run_eval(at_step: int) -> None:
        nonlocal best_acc
        if not eval_examples:
            return
        if device.type == "cuda":
            torch.cuda.empty_cache()
        policy.eval()
        with torch.no_grad():
            acc, fmt, rows = evaluate(policy, tokenizer, device, eval_examples, cfg,
                                      dump=0)
        policy.train()
        logger.log(at_step, {"eval_acc": acc, "eval_format": fmt})
        # Dump the first few completions (same questions each eval -> shows reasoning
        # growing over training, the R1-style "aha" visualization).
        with open(run_dir / f"samples_step{at_step}.txt", "w", encoding="utf-8") as fh:
            for ex, comp, parts in rows[:3]:
                fh.write(f"Q: {ex.question}\nGOLD: {ex.gold}  "
                         f"CORRECT: {parts['is_correct']:.0f}\n"
                         f"COMPLETION:\n{comp}\n{'=' * 70}\n")
        if acc > best_acc:
            best_acc = acc
            policy.save_pretrained(str(run_dir / "best"))

    optimizer.zero_grad()
    accum = 0
    last_grad_norm = float("nan")   # carried between optimizer steps for logging
    for step in range(cfg.max_steps):
        # Periodic held-out eval (step 0 gives the in-loop baseline point).
        if cfg.eval_every and step % cfg.eval_every == 0:
            run_eval(step)

        t0 = time.time()
        # Pick this step's prompts (rotating window over the pool; for --overfit the
        # pool is tiny so we revisit the same problems repeatedly).
        start = (step * cfg.prompts_per_step) % len(pool)
        batch = [pool[(start + i) % len(pool)] for i in range(cfg.prompts_per_step)]

        # 1-3: rollout (old policy, no grad) + advantages. Disable gradient
        # checkpointing during generation so the KV-cache is used (cached decoding is
        # vastly faster); re-enable it for the training forward.
        if cfg.gradient_checkpointing:
            policy.gradient_checkpointing_disable()
        rollout = generate_group(policy, tokenizer, batch, cfg, device)
        if cfg.gradient_checkpointing:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        advantages = group_advantages(
            rollout.rewards, cfg.group_size,
            normalize=cfg.normalize_advantage, eps=cfg.adv_eps,
        )
        t_rollout = time.time() - t0

        # 4-5: reference log-probs FIRST (no-grad, freed immediately) so its transient
        # logits don't overlap with the big logp_new graph, then re-score under the
        # current policy WITH grad.
        t1 = time.time()
        logp_ref = (reference_logprobs(policy, ref_model, cfg,
                                       rollout.full_ids, rollout.attention_mask)
                    if use_kl else None)
        logp_new = compute_logprobs(policy, rollout.full_ids, rollout.attention_mask)

        # 6: loss -> backward (scaled for grad accumulation).
        loss, m = grpo_loss(
            logp_new, rollout.logp_old, advantages, rollout.completion_mask,
            logp_ref=logp_ref, clip_eps=cfg.clip_eps, kl_beta=cfg.kl_beta,
            loss_agg=cfg.loss_agg,
        )
        (loss / cfg.grad_accum_steps).backward()
        accum += 1
        if accum == cfg.grad_accum_steps:
            last_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm))
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
                "loss": float(loss.detach()),
                "kl": m["mean_kl"],
                "ratio": m["mean_ratio"],
                "clip_frac": m["clip_frac"],
                "grad_norm": last_grad_norm,
                "t_rollout": round(t_rollout, 2),
                "t_update": round(t_update, 2),
            })

    # Final held-out eval (the "after" number) + save the trained weights.
    run_eval(cfg.max_steps)
    out = run_dir / "final"
    policy.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))
    logger.close()
    print(f"[train] done. saved to {out} | best eval_acc={best_acc:.3f}")


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
