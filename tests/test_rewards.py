"""Hand-checked tests for answer extraction and reward functions.

These are the safety net for the most bug-prone, highest-leverage code in the
project. Each case is something a real model completion might look like.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from rewards import (  # noqa: E402
    correctness_reward,
    extract_answer,
    format_reward,
    has_format,
    normalize_number,
    total_reward,
)


# --- normalize_number --------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("72", "72"),
    ("1,072", "1072"),
    ("$250", "250"),
    ("250 dollars", "250"),
    ("50%", "50"),
    ("-3", "-3"),
    ("5.0", "5"),          # whole-valued decimal collapses to int
    ("1/2", "0.5"),        # fraction -> canonical float string
    ("0.5", "0.5"),
    ("no number here", None),
    ("", None),
])
def test_normalize_number(raw, expected):
    assert normalize_number(raw) == expected


# --- extract_answer ----------------------------------------------------------

def test_extract_from_answer_tag():
    c = "<think> 60+10 </think><answer> 72 </answer>"
    assert extract_answer(c) == "72"

def test_extract_strips_commas_and_dollar():
    assert extract_answer("<answer>$1,072</answer>") == "1072"

def test_extract_takes_last_answer_when_repeated():
    # Model restates the answer; we trust the last <answer> block.
    c = "<answer>5</answer> wait, actually <answer>7</answer>"
    assert extract_answer(c) == "7"

def test_extract_negative():
    assert extract_answer("<answer>-3</answer>") == "-3"

def test_extract_boxed_fallback():
    assert extract_answer("the result is \\boxed{42}") == "42"

def test_extract_last_number_fallback():
    # No tags at all -> fall back to the last number in the text.
    assert extract_answer("first 12 then finally 15") == "15"

def test_extract_no_answer_returns_none():
    assert extract_answer("I am not sure how to solve this") is None

def test_extract_answer_with_trailing_prose():
    assert extract_answer("<answer>The answer is 250.</answer>") == "250"


# --- correctness_reward ------------------------------------------------------

def test_correct_exact():
    assert correctness_reward("<answer>72</answer>", "72") == 1.0

def test_correct_with_formatting_differences():
    assert correctness_reward("<answer>$1,000</answer>", "1000") == 1.0

def test_correct_decimal_vs_int():
    assert correctness_reward("<answer>5.0</answer>", "5") == 1.0

def test_incorrect():
    assert correctness_reward("<answer>10</answer>", "72") == 0.0

def test_no_answer_is_incorrect():
    assert correctness_reward("I don't know", "72") == 0.0


# --- format_reward / has_format ---------------------------------------------

def test_format_present():
    c = "<think> reasoning </think><answer> 5 </answer>"
    assert has_format(c) is True
    assert format_reward(c) == pytest.approx(0.1)

def test_format_missing_think():
    assert has_format("<answer>5</answer>") is False
    assert format_reward("<answer>5</answer>") == 0.0

def test_format_missing_answer():
    assert has_format("<think> reasoning </think>") is False


# --- total_reward ------------------------------------------------------------

def test_total_correct_and_formatted():
    c = "<think> 60+12 </think><answer> 72 </answer>"
    total, parts = total_reward(c, "72")
    assert total == pytest.approx(1.1)
    assert parts["is_correct"] == 1.0
    assert parts["format"] == pytest.approx(0.1)

def test_total_formatted_but_wrong():
    c = "<think> oops </think><answer> 10 </answer>"
    total, parts = total_reward(c, "72")
    assert total == pytest.approx(0.1)   # only the format bonus
    assert parts["is_correct"] == 0.0
