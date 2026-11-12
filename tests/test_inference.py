"""Tiled inference reproduces full-frame inference for a translation-equivariant model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.inference import cosine_window, tile_starts, tiled  # noqa: E402


def test_tile_starts_cover_the_frame_and_end_flush():
    assert tile_starts(2048, 1024, 0.25) == [0, 768, 1024]
    assert tile_starts(1024, 1024, 0.25) == [0] and tile_starts(100, 128, 0.25) == [0]
    s = tile_starts(2048, 512, 0.5)
    assert s[0] == 0 and s[-1] == 1536 and all(b - a <= 256 for a, b in zip(s, s[1:]))


def test_cosine_window_is_positive_and_symmetric():
    w = cosine_window(64)
    assert w.shape == (64, 64) and w.min() > 0 and np.allclose(w, w.T) and np.allclose(w, w[::-1, ::-1])
    assert w[32, 32] > 0.99


def test_tiled_identity_is_exact_and_box_filter_matches_inside():
    x = torch.rand(1, 3, 256, 256, generator=torch.Generator().manual_seed(0))
    out = tiled(lambda v: v[:, :1], x, tile=128, overlap=0.25)
    assert out.shape == (1, 1, 256, 256) and torch.allclose(out, x[:, :1], atol=1e-5)
    k = torch.ones(1, 1, 5, 5) / 25
    box = lambda v: F.conv2d(v[:, :1], k, padding=2)
    full = box(x)
    t = tiled(box, x, tile=128, overlap=0.25)
    assert (t - full).abs().max() < 0.05 and (t - full)[..., 8:-8, 8:-8].abs().mean() < 1e-3


def test_tiled_refuses_a_tile_larger_than_the_frame():
    x = torch.rand(1, 3, 64, 64)
    assert tiled(lambda v: v[:, :1], x, tile=64).shape == (1, 1, 64, 64)    # one tile, the frame
    try:
        tiled(lambda v: v[:, :1], x, tile=128)
    except ValueError as e:
        assert "128" in str(e) and "64" in str(e)
    else:
        raise AssertionError("a 128 px tile of a 64 px frame accepted")
