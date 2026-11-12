"""Tests for the SWA running mean in scripts/train.py (CPU, tiny modules)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train import swa_update, update_bn  # noqa: E402


def _net(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.BatchNorm2d(4), nn.Conv2d(4, 1, 1))


def test_swa_is_the_equal_weight_mean_of_the_snapshots():
    nets = [_net(s) for s in range(3)]
    state = None
    for i, n in enumerate(nets):
        state = swa_update(state, n, i)
    for k, v in state.items():
        if v.is_floating_point():
            expect = torch.stack([n.state_dict()[k].float() for n in nets]).mean(0)
            assert torch.allclose(v, expect, atol=1e-6), k


def test_integer_buffers_take_the_latest_value():
    a, b = _net(0), _net(1)
    a[1].num_batches_tracked.fill_(3)
    b[1].num_batches_tracked.fill_(7)
    state = swa_update(swa_update(None, a, 0), b, 1)
    assert state["1.num_batches_tracked"].item() == 7


def test_snapshot_does_not_alias_the_live_model():
    m = _net(0)
    state = swa_update(None, m, 0)
    with torch.no_grad():
        m[0].weight.add_(1.0)
    assert not torch.equal(state["0.weight"], m[0].weight)


def test_update_bn_reestimates_running_stats_from_the_loader():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4))
    net.eval()
    net[1].momentum = 0.3               # a non-default value, to check it is restored
    orig_momentum = net[1].momentum
    batches = [{"x": torch.randn(2, 3, 8, 8)} for _ in range(5)]
    with torch.no_grad():
        outs = [net[0](b["x"]) for b in batches]
    expected_mean = torch.cat(outs, dim=0).mean(dim=(0, 2, 3))

    n = update_bn(net, batches, "cpu", amp=False, amp_dtype=torch.float32)

    assert n == 5
    assert torch.allclose(net[1].running_mean, expected_mean, atol=1e-4)
    assert net[1].momentum == orig_momentum
    assert not net.training


def test_update_bn_stops_at_max_batches():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4))
    batches = [{"x": torch.randn(2, 3, 8, 8)} for _ in range(5)]
    n = update_bn(net, batches, "cpu", amp=False, amp_dtype=torch.float32, max_batches=3)
    assert n == 3


def test_update_bn_leaves_a_net_with_no_batchnorm_untouched():
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
    net.eval()
    state_before = {k: v.clone() for k, v in net.state_dict().items()}
    n = update_bn(net, [{"x": torch.randn(1, 3, 8, 8)}], "cpu", amp=False, amp_dtype=torch.float32)
    assert n == 0
    for k, v in net.state_dict().items():
        assert torch.equal(v, state_before[k])
    assert not net.training
