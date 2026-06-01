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
    """Greedy-decode each example, score it, return (accuracy, format_rate, rows).

    ``dump`` controls how many sample completions to print for eyeballing.
    """
    n_correct = 0
    n_format = 0
    rows = []
    greedy = cfg.eval_temperature == 0.0
    for i, ex in enumerate(examples):
        prompt = build_prompt(ex.question, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else cfg.eval_temperature,
            top_p=None if greedy else cfg.top_p,
            pad_token_id=tokenizer.pad_token_id,
        )
        completion = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        _, parts = total_reward(completion, ex.gold, cfg)
        n_correct += int(parts["is_correct"])
        n_format += int(parts["format"] > 0)
        rows.append((ex, completion, parts))
        if i < dump:
            print(f"\n--- sample {i} ---")
            print("Q:", ex.question[:120].replace("\n", " "))
            print("gold:", ex.gold, "| pred:", extract_answer(completion))
            print("completion:", completion[:300].replace("\n", " "))

    n = len(examples)
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
