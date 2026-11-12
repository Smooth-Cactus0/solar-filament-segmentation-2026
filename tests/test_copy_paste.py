"""Tests for filseg.augment.paste_instances (copy-paste of annotated filaments)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.augment import paste_instances  # noqa: E402


def _frames():
    x = np.zeros((3, 64, 64), dtype=np.float32)
    x[2] = 0.25
    y = np.zeros((1, 64, 64), dtype=np.float32)
    y[0, 5:9, 5:30] = 1.0                        # existing filament, top-left
    dx = np.full((3, 64, 64), -0.5, dtype=np.float32)
    dx[2] = 0.9                                  # donor radius differs: must not be copied
    far = np.zeros((64, 64), dtype=np.float32); far[40:44, 20:50] = 1.0
    clash = np.zeros((64, 64), dtype=np.float32); clash[6:10, 10:20] = 1.0
    return x, y, dx, [far, clash]


def test_pastes_only_the_non_overlapping_instance():
    x, y, dx, masks = _frames()
    gx, gy, n = paste_instances(x.copy(), y.copy(), dx, masks, np.random.default_rng(0), max_n=3)
    assert n == 1
    assert gy[0, 40:44, 20:50].min() == 1.0          # far instance added to the target
    assert gy[0, 6:10, 10:20].sum() == y[0, 6:10, 10:20].sum()  # clashing one skipped
    assert np.all(gx[:2, 40:44, 20:50] == -0.5)       # donor pixels copied
    np.testing.assert_array_equal(gx[2], x[2])        # radius channel untouched


def test_margin_zone_is_copied_but_not_labelled():
    x, y, dx, masks = _frames()
    gx, gy, _ = paste_instances(x.copy(), y.copy(), dx, masks[:1], np.random.default_rng(0),
                                max_n=1, margin=2)
    assert gx[0, 39, 30] == -0.5 and gy[0, 39, 30] == 0.0   # 1 px outside the mask


def test_max_n_caps_the_number_pasted():
    x, y, dx, _ = _frames()
    masks = []
    for r in (30, 40, 50):
        m = np.zeros((64, 64), dtype=np.float32); m[r:r + 3, 5:15] = 1.0
        masks.append(m)
    _, gy, n = paste_instances(x.copy(), y.copy(), dx, masks, np.random.default_rng(1), max_n=2)
    assert n == 2 and gy[0, 25:64].sum() == 2 * 30


def test_nothing_to_paste_is_a_no_op():
    x, y, dx, _ = _frames()
    gx, gy, n = paste_instances(x.copy(), y.copy(), dx, [], np.random.default_rng(0), max_n=3)
    assert n == 0
    np.testing.assert_array_equal(gx, x)
    np.testing.assert_array_equal(gy, y)
