"""Tests for the GRPO core: advantages and the clipped loss, all checked against
numbers worked out by hand. If these pass, the algorithm is correct in isolation;
Phase 4 then only has to wire it up.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
import torch  # noqa: E402

from grpo import group_advantages, grpo_loss  # noqa: E402


# --- group_advantages --------------------------------------------------------

def test_advantage_normalized_single_group():
    # rewards [3,1,1,-1]: mean=1, deviations [2,0,0,-2], population var=2, std=sqrt2.
    r = torch.tensor([3.0, 1.0, 1.0, -1.0])
    adv = group_advantages(r, group_size=4, normalize=True, eps=0.0)
    s2 = math.sqrt(2.0)
    assert torch.allclose(adv, torch.tensor([s2, 0.0, 0.0, -s2]), atol=1e-5)


def test_advantage_unnormalized_is_just_centered():
    r = torch.tensor([3.0, 1.0, 1.0, -1.0])
    adv = group_advantages(r, group_size=4, normalize=False)
    assert torch.allclose(adv, torch.tensor([2.0, 0.0, 0.0, -2.0]), atol=1e-6)


def test_advantage_two_groups_independent():
    # Two groups of 2: [1,0] -> [1,-1], [0,1] -> [-1,1] (each normalized within group).
    r = torch.tensor([1.0, 0.0, 0.0, 1.0])
    adv = group_advantages(r, group_size=2, normalize=True, eps=0.0)
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, -1.0, 1.0]), atol=1e-5)


def test_advantage_degenerate_group_is_zero():
    # All rewards equal -> no sibling was better -> zero advantage (no gradient).
    r = torch.tensor([2.0, 2.0, 2.0])
    adv = group_advantages(r, group_size=3, normalize=True, eps=1e-4)
    assert torch.allclose(adv, torch.zeros(3), atol=1e-6)


def test_advantage_mean_is_zero_per_group():
    torch.manual_seed(0)
    r = torch.randn(12)
    adv = group_advantages(r, group_size=4, normalize=True)
    # Each group of advantages must sum to ~0 (we subtracted the group mean).
    assert torch.allclose(adv.view(-1, 4).mean(dim=1), torch.zeros(3), atol=1e-5)


# --- grpo_loss: the ratio=1 identity ----------------------------------------

def test_loss_ratio_one_reduces_to_neg_mean_advantage():
    # When the current policy == the sampling policy, ratio=1, the surrogate is just
    # the advantage, and (no KL) the loss is -mean(advantage) over tokens.
    torch.manual_seed(0)
    logp = torch.randn(3, 4)
    adv = torch.tensor([1.0, -0.5, 2.0])
    mask = torch.ones(3, 4)
    loss, m = grpo_loss(logp, logp.clone(), adv, mask, loss_agg="token")
    assert loss.item() == pytest.approx(-(1.0 - 0.5 + 2.0) / 3.0, abs=1e-6)
    assert m["mean_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert m["clip_frac"] == pytest.approx(0.0, abs=1e-6)
    assert m["mean_kl"] == 0.0


# --- grpo_loss: the KL term --------------------------------------------------

def test_kl_zero_when_ref_equals_policy():
    logp = torch.zeros(1, 2)
    loss, m = grpo_loss(logp, logp.clone(), torch.tensor([1.0]), torch.ones(1, 2),
                        logp_ref=logp.clone(), kl_beta=1.0, loss_agg="token")
    assert m["mean_kl"] == pytest.approx(0.0, abs=1e-7)


def test_kl_k3_matches_hand_value():
    # delta = logp_ref - logp_new = 0.5  ->  k3 = e^0.5 - 0.5 - 1 = 0.1487212...
    logp_new = torch.zeros(1, 2)
    logp_ref = torch.full((1, 2), 0.5)
    loss, m = grpo_loss(logp_new, logp_new.clone(), torch.tensor([1.0]),
                        torch.ones(1, 2), logp_ref=logp_ref, kl_beta=1.0,
                        loss_agg="token")
    kl_expected = math.exp(0.5) - 0.5 - 1.0
    assert m["mean_kl"] == pytest.approx(kl_expected, abs=1e-6)
    # loss = -(surrogate=adv=1) + beta*kl
    assert loss.item() == pytest.approx(-1.0 + kl_expected, abs=1e-6)


def test_kl_is_nonnegative():
    torch.manual_seed(1)
    logp_new = torch.randn(2, 5)
    logp_ref = torch.randn(2, 5)
    _, m = grpo_loss(logp_new, logp_new.clone(), torch.tensor([1.0, -1.0]),
                     torch.ones(2, 5), logp_ref=logp_ref, kl_beta=1.0,
                     loss_agg="token")
    assert m["mean_kl"] >= 0.0


# --- grpo_loss: clipping is one-sided ---------------------------------------

def test_clip_binds_for_positive_advantage():
    # ratio = e^1 ~= 2.718 > 1.2. Positive advantage -> clipped (1.2) is the min.
    logp_new = torch.full((1, 1), 1.0)
    logp_old = torch.zeros(1, 1)
    loss, m = grpo_loss(logp_new, logp_old, torch.tensor([1.0]), torch.ones(1, 1),
                        clip_eps=0.2, loss_agg="token")
    assert loss.item() == pytest.approx(-1.2, abs=1e-5)
    assert m["clip_frac"] == pytest.approx(1.0, abs=1e-6)


def test_clip_does_not_bind_for_negative_advantage():
    # Same big ratio, but negative advantage -> unclipped (more negative) is the min,
    # so the clip does NOT cap it. This one-sidedness is the whole point of PPO clip.
    logp_new = torch.full((1, 1), 1.0)
    logp_old = torch.zeros(1, 1)
    loss, _ = grpo_loss(logp_new, logp_old, torch.tensor([-1.0]), torch.ones(1, 1),
                        clip_eps=0.2, loss_agg="token")
    assert loss.item() == pytest.approx(math.e, abs=1e-4)   # -(e * -1) = e


# --- grpo_loss: masking ------------------------------------------------------

def test_masked_tokens_are_ignored():
    # Position 2 is masked; give it a wild log-prob. The loss must not change.
    logp_old = torch.zeros(1, 3)
    logp_new = torch.tensor([[0.0, 0.0, 100.0]])     # huge ratio at masked slot
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    loss, _ = grpo_loss(logp_new, logp_old, torch.tensor([2.0]), mask,
                        loss_agg="token")
    # Only the 2 real tokens count, each with advantage 2 and ratio 1 -> loss = -2.
    assert loss.item() == pytest.approx(-2.0, abs=1e-5)
