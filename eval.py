"""Baseline / checkpoint evaluation: accuracy on a held-out GSM8K split.

This produces the honest "before" number (and later the "after"). It decodes
greedily (temperature 0) by default so the result is deterministic and
reproducible — we want a number we trust, not a lucky sample.

    python eval.py --config gpu --set eval_n=200      # on the GPU box
    python eval.py --config tiny                       # CPU smoke (eval_n=8)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs import get_config, parse_overrides
from data import build_prompt, load_gsm8k
from rewards import extract_answer, total_reward
from utils import load_model_and_tokenizer, set_seed


@torch.no_grad()
def evaluate(model, tokenizer, device, examples, cfg, *, dump=3):
    """Decode examples in batches, score them, return (accuracy, format_rate, rows).

    We generate ``eval_batch_size`` prompts at once. Single-sequence decoding
    barely uses the GPU (kernel-launch bound) and is very slow; batching with
    left-padding keeps the GPU busy and prints running progress so the run is
    never mistaken for a hang. ``dump`` controls how many samples to print.
    """
    n_correct = 0
    n_format = 0
    rows = []
    greedy = cfg.eval_temperature == 0.0
    bs = max(1, cfg.eval_batch_size)
    n = len(examples)

    for start in range(0, n, bs):
        batch = examples[start:start + bs]
        prompts = [build_prompt(ex.question, tokenizer) for ex in batch]
        # Left-padding (set on the tokenizer) means every row's generation starts
        # at the same index, so we can slice off the shared prompt width below.
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else cfg.eval_temperature,
            top_p=None if greedy else cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        gen = out[:, prompt_len:]
        completions = tokenizer.batch_decode(gen, skip_special_tokens=True)

        for j, (ex, completion) in enumerate(zip(batch, completions)):
            _, parts = total_reward(completion, ex.gold, cfg)
            n_correct += int(parts["is_correct"])
            n_format += int(parts["format"] > 0)
            rows.append((ex, completion, parts))
            idx = start + j
            if idx < dump:
                print(f"\n--- sample {idx} ---")
                print("Q:", ex.question[:120].replace("\n", " "))
                print("gold:", ex.gold, "| pred:", extract_answer(completion))
                print("completion:", completion[:300].replace("\n", " "))

        done = min(start + bs, n)
        print(f"[eval] {done}/{n}  running_acc={n_correct / done:.3f}  "
              f"format_rate={n_format / done:.3f}", flush=True)

    return n_correct / n, n_format / n, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="tiny")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = get_config(args.config, **parse_overrides(args.set))
    set_seed(cfg.seed)
    model, tokenizer, device = load_model_and_tokenizer(cfg, for_training=False)
    model.eval()

    examples = load_gsm8k(args.split, n=cfg.eval_n, seed=cfg.seed)
    print(f"[eval] {cfg.model_name} on {len(examples)} {args.split} examples "
          f"(greedy={cfg.eval_temperature == 0.0})")
    acc, fmt, _ = evaluate(model, tokenizer, device, examples, cfg)
    print(f"\n[eval] accuracy = {acc:.3f}  format_rate = {fmt:.3f}")


if __name__ == "__main__":
    main()
