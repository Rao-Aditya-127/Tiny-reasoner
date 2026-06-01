"""GSM8K loading + prompt formatting.

GSM8K (`openai/gsm8k`, config `main`) is grade-school math word problems. Each
example has:
  * ``question`` — the word problem (a string).
  * ``answer``   — a worked solution ending in a line ``#### <number>`` where the
    text after ``####`` is the *gold* final answer.

We ask the model to think, then answer, in a strict format:
    <think> ...reasoning... </think><answer> ...final number... </answer>
so the reward function (rewards.py) can reliably find the final answer and so we
can give a small bonus for following the format.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The system prompt defines the contract: think first, then give the final answer
# inside <answer></answer>. Keep it short and strict — the reward depends on it.
SYSTEM_PROMPT = (
    "You are a careful math assistant. Solve the problem step by step. "
    "Put your reasoning inside <think> </think> tags, then give ONLY the final "
    "numeric answer inside <answer> </answer> tags. "
    "For example: <think> 2 plus 3 is 5 </think><answer> 5 </answer>"
)

_GOLD_RE = re.compile(r"####\s*(.+?)\s*$", re.DOTALL)


@dataclass
class Example:
    question: str
    gold: str          # normalized gold answer string (e.g. "72")
    raw_answer: str    # the original worked solution, for reference


def extract_gold(answer_field: str) -> str:
    """Pull the gold answer that follows '####' and normalize it.

    GSM8K golds are integers possibly written with thousands separators, e.g.
    '#### 1,072'. We strip commas and surrounding whitespace.
    """
    m = _GOLD_RE.search(answer_field)
    text = m.group(1) if m else answer_field
    return text.replace(",", "").strip()


def load_gsm8k(split: str, n: int | None = None, seed: int = 0) -> list[Example]:
    """Load a GSM8K split as a list of :class:`Example`.

    ``split`` is "train" or "test". If ``n`` is given, return a deterministic
    shuffled subset of size ``n`` (so subset runs are reproducible).
    """
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=split)
    if n is not None:
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))
    return [
        Example(question=row["question"], gold=extract_gold(row["answer"]),
                raw_answer=row["answer"])
        for row in ds
    ]


def build_prompt(question: str, tokenizer) -> str:
    """Render the chat-formatted prompt string for one question.

    Uses the model's own chat template so the special tokens / roles match what
    the model was trained on. ``add_generation_prompt=True`` appends the
    assistant turn marker so the model continues as the assistant.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
