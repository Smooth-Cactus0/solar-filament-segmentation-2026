"""Tests for filseg.losses and the output-head helpers in filseg.model (CPU, tiny tensors)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.losses import (  # noqa: E402
    FilamentLoss, LossConfig, boundary_band, lovasz_hinge, ordinal_targets,
)
from filseg.model import MaskSpineLoss, head_channels, head_of, mask_prob  # noqa: E402


def _batch(seed: int = 0, c_logits: int = 1, c_target: int = 1, size: int = 32):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(2, c_logits, size, size, generator=g) * 2
    target = (torch.rand(2, c_target, size, size, generator=g) > 0.8).float()
    valid = torch.ones(2, 1, size, size)
    valid[..., :4, :] = 0
    return logits, target, valid


def test_defaults_reproduce_mask_spine_loss():
    logits, target, valid = _batch()
    new = FilamentLoss(LossConfig(), 0.25, 3.0)(logits, target, valid)
    old = MaskSpineLoss(0.25, 3.0)(logits, target, valid)
    assert torch.allclose(new, old, atol=1e-6)


def test_defaults_reproduce_the_spine_variant():
    logits, target, valid = _batch(c_logits=2, c_target=2)
    target[:, 1] = torch.rand_like(target[:, 1])
    new = FilamentLoss(LossConfig(), 0.25, 3.0, 0.2, 10.0)(logits, target, valid)
    old = MaskSpineLoss(0.25, 3.0, 0.2, 10.0)(logits, target, valid)
    assert torch.allclose(new, old, atol=1e-6)


def test_config_rejects_unknown_keys_and_bad_combinations():
    assert LossConfig.from_dict(None) == LossConfig()
    with pytest.raises(ValueError, match="unknown"):
        LossConfig.from_dict({"lovasz": 1.0})
    with pytest.raises(ValueError, match="consensus"):
        LossConfig.from_dict({"head": "ordinal"})
    with pytest.raises(ValueError, match="boundary_kernel"):
        LossConfig.from_dict({"boundary_kernel": 4})
    with pytest.raises(ValueError, match="target"):
        LossConfig.from_dict({"target": "mean"})
    with pytest.raises(ValueError, match="small_weight_max"):
        LossConfig.from_dict({"small_weight_max": 0.5})
    with pytest.raises(ValueError, match="spine"):
        FilamentLoss(LossConfig(target="consensus", head="ordinal"), spine_weight=0.2)


def test_ordinal_targets_are_cumulative_agreement_levels():
    frac = torch.tensor([0.0, 1 / 3, 0.5, 2 / 3, 1.0]).view(1, 1, 1, 5)
    t = ordinal_targets(frac)[0, :, 0]                      # (3 levels, 5 pixels)
    expected = torch.tensor([[0, 0, 0], [1, 0, 0], [1, 0.5, 0], [1, 1, 0], [1, 1, 1]])
    assert torch.allclose(t.T, expected, atol=1e-6)


def test_ordinal_targets_mean_to_the_agreement_fraction():
    frac = torch.linspace(0, 1, 11).view(1, 1, 1, 11)
    mean = ordinal_targets(frac).mean(dim=1)
    assert torch.allclose(mean, frac[:, 0], atol=1e-6)


def test_mask_prob_reads_the_right_channels():
    logits = torch.zeros(1, 3, 2, 2)
    logits[:, 0], logits[:, 1], logits[:, 2] = 10.0, -10.0, -10.0
    assert torch.allclose(mask_prob(logits, "ordinal"), torch.full((1, 1, 2, 2), 1 / 3), atol=1e-3)
    assert torch.allclose(mask_prob(logits, "mask"), torch.sigmoid(logits[:, :1]))
    assert head_channels("mask") == 1 and head_channels("ordinal") == 3
    with pytest.raises(ValueError):
        head_channels("spine")
    assert head_of(torch.nn.Conv2d(1, 1, 1)) == "mask"


def test_boundary_band_marks_edges_only():
    t = torch.zeros(1, 1, 20, 20)
    t[..., 5:15, 5:15] = 1
    b = boundary_band(t, 3)
    assert b[0, 0, 10, 10] == 0 and b[0, 0, 0, 0] == 0      # deep inside, far outside
    assert b[0, 0, 5, 10] == 1 and b[0, 0, 4, 10] == 1      # inner and outer edge rows


def test_boundary_weight_raises_the_loss_of_edge_errors():
    t = torch.zeros(1, 1, 20, 20)
    t[..., 5:15, 5:15] = 1
    valid = torch.ones_like(t)
    logits = torch.zeros_like(t)                            # p = 0.5 everywhere
    plain = FilamentLoss(LossConfig(), dice_weight=0.0)(logits, t, valid)
    edged = FilamentLoss(LossConfig(boundary_weight=2.0, boundary_kernel=3),
                         dice_weight=0.0)(logits, t, valid)
    assert edged > plain


def test_lovasz_is_zero_for_confident_correct_predictions():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:5, 2:5] = 1
    valid = torch.ones_like(t)
    good = (t * 2 - 1) * 5.0
    assert lovasz_hinge(good, t, valid).item() == pytest.approx(0.0, abs=1e-6)
    assert lovasz_hinge(-good, t, valid).item() > 1.0
    assert lovasz_hinge(good, t, torch.zeros_like(t)).item() == 0.0


def test_lovasz_gradients_flow():
    t = torch.zeros(1, 1, 8, 8)
    t[..., 2:5, 2:5] = 1
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    lovasz_hinge(logits, t, torch.ones_like(t)).backward()
    assert logits.grad.abs().sum() > 0


def test_skeleton_term_rewards_covering_the_skeleton():
    t = torch.zeros(1, 1, 20, 20)
    valid = torch.ones_like(t)
    skel = torch.zeros_like(t)
    skel[..., 10, 5:15] = 1
    hit = torch.full_like(t, -10.0)
    hit[..., 10, 5:15] = 10.0
    miss = torch.full_like(t, -10.0)
    with_skel = FilamentLoss(LossConfig(skeleton_weight=1.0), dice_weight=0.0)
    plain = FilamentLoss(LossConfig(), dice_weight=0.0)
    assert (with_skel(hit, t, valid, skel=skel) - plain(hit, t, valid)).item() == pytest.approx(0.0, abs=1e-3)
    assert (with_skel(miss, t, valid, skel=skel) - plain(miss, t, valid)).item() == pytest.approx(1.0, abs=1e-3)


def test_pixel_weights_scale_the_bce():
    logits, target, valid = _batch()
    f = FilamentLoss(LossConfig(), dice_weight=0.0)
    ones = torch.ones_like(valid)
    assert torch.allclose(f(logits, target, valid, pixw=ones), f(logits, target, valid))
    assert torch.allclose(f(logits, target, valid, pixw=2 * ones), 2 * f(logits, target, valid))


def test_ordinal_loss_prefers_the_matching_levels():
    frac = torch.zeros(1, 1, 4, 4)
    frac[..., :2, :] = 2 / 3
    valid = torch.ones_like(frac)
    good = torch.full((1, 3, 4, 4), -8.0)
    good[:, 0, :2] = 8.0
    good[:, 1, :2] = 8.0
    f = FilamentLoss(LossConfig(target="consensus", head="ordinal"), dice_weight=0.0)
    assert f(good, frac, valid) < f(-good, frac, valid)
