"""The rollout engine: sample groups of completions and score every token.

This is the data source for RL and the single most bug-prone part of the project.
Two ideas dominate, and getting either wrong silently breaks training:

1. **The logit -> token shift (off-by-one).** A causal LM's logits at position t
   predict the token at position t+1. So the log-probability of the token sitting
   at full-sequence position (k+1) comes from the logits at position k. We work in
   the "scored frame": for a sequence of length T there are T-1 scored positions
   (we can't score position 0 — nothing precedes it). Scored index k holds
   ``log P(token[k+1] | token[0..k])``.

2. **The completion mask.** We only train on *generated* tokens — never the prompt,
   never padding, never anything after the end-of-sequence token. Every later sum or
   mean over tokens must be taken under this mask.

Conventions used everywhere downstream:
  * ``full_ids``        [B, T]    prompt (left-padded) + completion
  * ``attention_mask``  [B, T]    1 on real tokens the model may attend to
  * ``logp``            [B, T-1]  per-scored-token log-prob (the shift above)
  * ``completion_mask`` [B, T-1]  1 iff the scored token is a real generated token
  ``logp`` and ``completion_mask`` are aligned element-wise — that alignment is the
  whole game.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import get_config, parse_overrides
from data import build_prompt
from rewards import total_reward
from utils import load_model_and_tokenizer, set_seed


@dataclass
class Rollout:
    """One batch of B = num_prompts * group_size scored completions."""

    full_ids: torch.Tensor          # [B, T]   prompt + completion token ids
    attention_mask: torch.Tensor    # [B, T]
    completion_mask: torch.Tensor   # [B, T-1] 1 on real generated (scored) tokens
    logp_old: torch.Tensor          # [B, T-1] log-probs under the sampling policy
    rewards: torch.Tensor           # [B]      scalar reward per completion
    reward_parts: list[dict]        # [B]      reward breakdown per completion
    completion_texts: list[str]     # [B]      decoded completions (for logging)
    prompt_texts: list[str]         # [B]
    group_size: int
    num_prompts: int

    def completion_lengths(self) -> torch.Tensor:
        """Number of real generated tokens per completion. [B]"""
        return self.completion_mask.sum(dim=1)


def compute_logprobs(model, input_ids: torch.Tensor,
                     attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-token log-probabilities of an *already-generated* sequence.

    One teacher-forced forward pass scores the sequence. Returns ``[B, T-1]`` in the
    scored frame: index k is ``log P(input_ids[k+1] | input_ids[0..k])``.

    NOT wrapped in ``no_grad`` on purpose — Phase 3 calls this through the *current*
    policy and needs gradients. The caller decides.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    # logits[:, k] predicts token k+1, so drop the last step; targets drop the first.
    logits = logits[:, :-1, :]                # [B, T-1, V]
    targets = input_ids[:, 1:]               # [B, T-1]
    bsz, length, vocab = logits.shape
    # cross_entropy is a fused, numerically-stable log_softmax+gather. Crucially it
    # does NOT materialise a second full [B, T-1, V] tensor (the old `.float()` copy
    # did, doubling peak memory and OOM-ing the LM-head logits on a 16 GB card).
    # NLL = -log p(target), so the log-prob is its negation. We upcast only the tiny
    # [B, T-1] result to fp32 so downstream ratio/KL math stays stable.
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(bsz * length, vocab),
        targets.reshape(bsz * length),
        reduction="none",
    )
    return (-nll).reshape(bsz, length).float()   # [B, T-1]


def _generated_token_mask(gen_ids: torch.Tensor, eos_id: int | None) -> torch.Tensor:
    """Mask over the generated region: 1 up to and including the first EOS, else 0.

    After a row emits EOS, ``generate`` fills the rest with padding; those positions
    (and only those) must be masked out. We keep the EOS token itself as a real,
    trainable token. Rows that never emit EOS are valid for their full length.
    """
    bsz, gen_len = gen_ids.shape
    mask = torch.ones_like(gen_ids)
    if eos_id is None:
        return mask
    is_eos = gen_ids == eos_id                       # [B, gen_len]
    has_eos = is_eos.any(dim=1)
    first_eos = torch.argmax(is_eos.int(), dim=1)    # first True index (0 if none)
    positions = torch.arange(gen_len, device=gen_ids.device)
    # valid where position <= first_eos for rows that have an eos
    eos_cut = torch.where(has_eos, first_eos, torch.full_like(first_eos, gen_len - 1))
    mask = (positions[None, :] <= eos_cut[:, None]).int()
    # rows with no eos keep the all-ones mask
    mask[~has_eos] = 1
    return mask


@torch.no_grad()
def generate_group(model, tokenizer, examples, cfg, device) -> Rollout:
    """Sample ``group_size`` completions per example and score every token.

    ``examples`` is a list of :class:`data.Example`. Each prompt is repeated G times
    in the batch so the group shares one question. Returns a :class:`Rollout` whose
    rows are ordered group-contiguously: prompt 0's G samples, then prompt 1's, ...
    """
    G = cfg.group_size
    prompts = [build_prompt(ex.question, tokenizer) for ex in examples]
    # Repeat-interleave so rows 0..G-1 are prompt 0, G..2G-1 are prompt 1, etc.
    expanded_prompts = [p for p in prompts for _ in range(G)]
    expanded_golds = [ex.gold for ex in examples for _ in range(G)]

    enc = tokenizer(expanded_prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = enc["input_ids"].shape[1]

    gen = model.generate(
        **enc,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        pad_token_id=tokenizer.pad_token_id,
    )                                              # [B, prompt_len + gen_len]
    full_ids = gen
    bsz, total_len = full_ids.shape
    gen_ids = full_ids[:, prompt_len:]             # [B, gen_len]

    # Attention mask for the scoring pass: real prompt tokens (left-pad zeros from the
    # tokenizer) + all generated positions. Trailing post-EOS pads are harmless to the
    # log-probs we keep (a causal model's earlier positions can't see them) and are
    # masked out of the loss anyway via completion_mask.
    gen_attn = torch.ones_like(gen_ids)
    attention_mask = torch.cat([enc["attention_mask"], gen_attn], dim=1)

    # Build the real-generated-token mask over the FULL sequence, then shift into the
    # scored frame so it aligns with logp ([:, 1:] drops position 0, always prompt).
    gen_real = _generated_token_mask(gen_ids, tokenizer.eos_token_id)   # [B, gen_len]
    full_completion = torch.zeros((bsz, total_len), dtype=torch.long, device=device)
    full_completion[:, prompt_len:] = gen_real
    completion_mask = full_completion[:, 1:]                            # [B, T-1]

    logp_old = compute_logprobs(model, full_ids, attention_mask)        # [B, T-1]

    completion_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    rewards = torch.zeros(bsz, dtype=torch.float32, device=device)
    reward_parts: list[dict] = []
    for i, (text, gold) in enumerate(zip(completion_texts, expanded_golds)):
        r, parts = total_reward(text, gold, cfg)
        rewards[i] = r
        reward_parts.append(parts)

    return Rollout(
        full_ids=full_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        logp_old=logp_old,
        rewards=rewards,
        reward_parts=reward_parts,
        completion_texts=completion_texts,
        prompt_texts=expanded_prompts,
        group_size=G,
        num_prompts=len(examples),
    )


def _demo() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = get_config(args.config, **parse_overrides(args.set))
    set_seed(cfg.seed)

    from data import load_gsm8k

    model, tokenizer, device = load_model_and_tokenizer(cfg, for_training=False)
    model.eval()
    examples = load_gsm8k("train", n=1, seed=cfg.seed)
    ro = generate_group(model, tokenizer, examples, cfg, device)

    print(f"[rollout] 1 prompt x G={ro.group_size} -> {ro.full_ids.shape[0]} rows")
    print(f"[rollout] full_ids {tuple(ro.full_ids.shape)}  "
          f"logp_old {tuple(ro.logp_old.shape)}  "
          f"completion_mask {tuple(ro.completion_mask.shape)}")
    lengths = ro.completion_lengths()
    for i in range(ro.full_ids.shape[0]):
        print(f"\n--- completion {i} | reward={ro.rewards[i]:.2f} "
              f"correct={ro.reward_parts[i]['is_correct']:.0f} "
              f"len={int(lengths[i])} tokens ---")
        print(ro.completion_texts[i][:240].replace("\n", " "))
    # The key invariant: masked logp count == sum of completion lengths.
    masked = int(ro.completion_mask.sum())
    print(f"\n[rollout] masked scored tokens = {masked}  "
          f"(== sum of completion lengths = {int(lengths.sum())})")
    assert masked == int(lengths.sum())
    print("[rollout] OK")


if __name__ == "__main__":
    _demo()
