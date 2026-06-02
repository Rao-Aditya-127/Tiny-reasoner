"""Tests for the rollout engine's two danger zones: the logit->token shift and
the completion mask. These use a tiny stub model (no network, no real weights) so
they run in milliseconds and pin down the exact alignment math.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from rollout import _generated_token_mask, compute_logprobs  # noqa: E402


class _StubModel:
    """Returns fixed logits regardless of input — lets us check the math exactly."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids=None, attention_mask=None):
        class _Out:
            pass
        out = _Out()
        out.logits = self._logits
        return out


# --- the off-by-one shift ----------------------------------------------------

def test_logprobs_shift_and_value():
    torch.manual_seed(0)
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    model = _StubModel(logits)

    logp = compute_logprobs(model, input_ids, torch.ones(B, T))

    # Manual reference: logits at position k score the token at position k+1.
    ref_logsoftmax = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    expected = ref_logsoftmax.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    assert logp.shape == (B, T - 1)            # scored frame is one shorter
    assert torch.allclose(logp, expected, atol=1e-5)


def test_logprobs_are_log_probs():
    # A log-prob is <= 0; sanity that we didn't flip a sign or forget the partition.
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 10)
    input_ids = torch.randint(0, 10, (1, 4))
    logp = compute_logprobs(_StubModel(logits), input_ids, torch.ones(1, 4))
    assert torch.all(logp <= 0)


# --- the completion mask -----------------------------------------------------

def test_mask_cuts_after_first_eos():
    eos = 99
    gen = torch.tensor([[5, 9, eos, 0, 0]])      # eos at index 2, then padding
    mask = _generated_token_mask(gen, eos)
    assert mask.tolist() == [[1, 1, 1, 0, 0]]    # eos kept, pad after dropped


def test_mask_no_eos_is_all_ones():
    eos = 99
    gen = torch.tensor([[3, 4, 5, 6, 7]])        # never emits eos
    mask = _generated_token_mask(gen, eos)
    assert mask.tolist() == [[1, 1, 1, 1, 1]]


def test_mask_eos_at_start():
    eos = 99
    gen = torch.tensor([[eos, 1, 2]])
    mask = _generated_token_mask(gen, eos)
    assert mask.tolist() == [[1, 0, 0]]


def test_mask_per_row_independent():
    eos = 99
    gen = torch.tensor([[1, eos, 7], [1, 2, 3]])
    mask = _generated_token_mask(gen, eos)
    assert mask.tolist() == [[1, 1, 0], [1, 1, 1]]


# --- the full -> scored frame shift used in generate_group -------------------

def test_full_to_scored_frame_alignment():
    # Reproduce generate_group's mask construction on a known layout:
    # prompt_len=2, gen_len=3 -> T=5. The first generated token sits at full
    # position 2, which becomes scored index 1 after dropping position 0.
    prompt_len, gen_len = 2, 3
    bsz = 1
    gen_real = torch.tensor([[1, 1, 0]])             # last generated slot is pad
    full = torch.zeros((bsz, prompt_len + gen_len), dtype=torch.long)
    full[:, prompt_len:] = gen_real
    completion_mask = full[:, 1:]
    assert completion_mask.tolist() == [[0, 1, 1, 0]]
    # Exactly the 2 real generated tokens survive the shift.
    assert int(completion_mask.sum()) == int(gen_real.sum())
