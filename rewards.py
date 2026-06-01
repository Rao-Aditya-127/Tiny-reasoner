"""Reward functions: turn a raw completion string into a number.

This is the most important file to get right. RL optimizes whatever the reward
says — if extraction is buggy, the model will learn the bug (e.g. game the format
and ignore correctness). So we keep two well-separated signals:

  * correctness_reward — 1.0 if the final answer matches the gold, else 0.0. This
    is a *verifiable* reward: we just check the math, no learned reward model.
  * format_reward — a small bonus (default 0.1) for using the <think>/<answer>
    tags correctly. Kept << correctness so it can't dominate.

``total_reward`` combines them and returns both the scalar and a breakdown dict
for logging.
"""

from __future__ import annotations

import re
from fractions import Fraction

# Last <answer>...</answer> block. We take the *last* one so a model that restates
# the answer doesn't trip us up. DOTALL so reasoning can span newlines.
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
# A signed number with optional thousands separators, decimals, or a fraction.
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
# Full strict format: <think>...</think> immediately followed by <answer>...</answer>.
_FORMAT_RE = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)


def normalize_number(text: str) -> str | None:
    """Normalize a raw answer fragment to a canonical numeric string, or None.

    Strips $, %, commas, units and trailing punctuation, collapses fractions and
    decimals to a canonical form so '1,000', '$1000', '1000.0' all compare equal.
    Returns None if no number can be parsed.
    """
    if text is None:
        return None
    s = text.strip()
    # Grab the first number-like token in the fragment.
    m = _NUMBER_RE.search(s)
    if not m:
        return None
    token = m.group(0).replace(",", "")
    try:
        if "/" in token:
            value = Fraction(token)
        else:
            value = Fraction(token)  # Fraction parses ints and decimals too
    except (ValueError, ZeroDivisionError):
        return None
    # Canonical string: integer if whole, else reduced fraction's float repr.
    if value.denominator == 1:
        return str(value.numerator)
    return str(float(value))


def extract_answer(completion: str) -> str | None:
    """Extract the model's final answer, normalized, from a completion.

    Priority: last <answer> tag → \\boxed{} → last number anywhere. Returns the
    normalized numeric string, or None if nothing parseable is found.
    """
    tag_matches = _ANSWER_TAG_RE.findall(completion)
    if tag_matches:
        norm = normalize_number(tag_matches[-1])
        if norm is not None:
            return norm

    boxed = _BOXED_RE.findall(completion)
    if boxed:
        norm = normalize_number(boxed[-1])
        if norm is not None:
            return norm

    numbers = _NUMBER_RE.findall(completion)
    if numbers:
        return normalize_number(numbers[-1])
    return None


def correctness_reward(completion: str, gold: str, value: float = 1.0) -> float:
    """``value`` if the model's normalized answer equals the normalized gold."""
    pred = extract_answer(completion)
    target = normalize_number(gold)
    if pred is None or target is None:
        return 0.0
    return value if pred == target else 0.0


def has_format(completion: str) -> bool:
    """True if the completion uses <think>...</think><answer>...</answer>."""
    return _FORMAT_RE.search(completion) is not None


def format_reward(completion: str, value: float = 0.1) -> float:
    """Small bonus for following the required tag format."""
    return value if has_format(completion) else 0.0


def total_reward(completion: str, gold: str, cfg=None) -> tuple[float, dict]:
    """Combine correctness + format into one scalar plus a breakdown dict.

    ``cfg`` (optional) supplies ``correct_reward`` / ``format_reward`` magnitudes;
    defaults are used when it's None (handy in unit tests).
    """
    correct_w = getattr(cfg, "correct_reward", 1.0) if cfg else 1.0
    format_w = getattr(cfg, "format_reward", 0.1) if cfg else 0.1
    correct = correctness_reward(completion, gold, value=correct_w)
    fmt = format_reward(completion, value=format_w)
    total = correct + fmt
    return total, {
        "reward": total,
        "correct": correct,
        "format": fmt,
        "is_correct": float(correct > 0),
    }
