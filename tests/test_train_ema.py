"""Tests for the weight EMA in scripts/train.py (CPU, tiny modules)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train import ema_update, make_ema  # noqa: E402


def _net(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Conv2d(4, 1, 1))


def test_make_ema_is_a_frozen_copy():
    m = _net(0)
    e = make_ema(m)
    assert not e.training
    assert all(not p.requires_grad for p in e.parameters())
    for (n1, p1), (n2, p2) in zip(m.named_parameters(), e.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2) and p1.data_ptr() != p2.data_ptr()


def test_first_step_copies_the_live_weights_almost_verbatim():
    m, e = _net(0), make_ema(_net(1))          # EMA starts from different weights
    before = [p.detach().clone() for p in e.parameters()]
    d = ema_update(e, m, decay=0.999, step=0)
    assert d == pytest.approx(0.1)             # (1 + 0) / (10 + 0)
    for p_m, p_e, p_0 in zip(m.parameters(), e.parameters(), before):
        assert torch.allclose(p_e, 0.1 * p_0 + 0.9 * p_m, atol=1e-6)
    # After many steps from the same live weights the EMA converges onto them.
    for step in range(1, 400):
        ema_update(e, m, decay=0.999, step=step)
    for p_m, p_e in zip(m.parameters(), e.parameters()):
        assert torch.allclose(p_e, p_m, atol=1e-2)


def test_decay_warms_up_to_the_target():
    m, e = _net(0), make_ema(_net(0))
    assert ema_update(e, m, 0.999, 0) == pytest.approx(0.1)
    assert ema_update(e, m, 0.999, 90) == pytest.approx(0.91)
    assert ema_update(e, m, 0.999, 100_000) == pytest.approx(0.999)


def test_ema_is_the_exact_exponential_average_of_one_scalar():
    m, e = _net(0), make_ema(_net(0))
    p_live = next(m.parameters())
    p_ema = next(e.parameters())
    with torch.no_grad():
        p_live.fill_(1.0)
        p_ema.fill_(0.0)
    d = ema_update(e, m, decay=0.5, step=10_000)   # warm-up long over, decay = 0.5
    assert d == 0.5 and torch.allclose(p_ema, torch.full_like(p_ema, 0.5))
    ema_update(e, m, decay=0.5, step=10_001)
    assert torch.allclose(p_ema, torch.full_like(p_ema, 0.75))


def test_buffers_are_copied_not_averaged():
    m, e = _net(0), make_ema(_net(0))
    m.train()
    m(torch.randn(2, 3, 8, 8))                  # moves BatchNorm running stats
    ema_update(e, m, decay=0.999, step=5_000)
    for b_m, b_e in zip(m.buffers(), e.buffers()):
        assert torch.equal(b_m, b_e)
