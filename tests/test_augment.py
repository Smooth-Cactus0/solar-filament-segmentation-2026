"""Tests for filseg.augment.

The golden test pins the refactor: with the default config, apply() must return
exactly what the pre-refactor dataset.augment returned, for many seeds. Every
experiment trained so far used that function, so any drift here would silently
change the recipe that the new screens are compared against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg import augment as A  # noqa: E402
from filseg.dataset import augment as dataset_augment  # noqa: E402


def legacy_augment(x, y, rng):
    """Verbatim copy of dataset.augment as of 2026-09-25 (commit c40ed36)."""
    if rng.random() < 0.5:
        x, y = x[:, ::-1], y[:, ::-1]
    if rng.random() < 0.5:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)
    if rng.random() < 0.8:
        ang = float(rng.uniform(-15, 15))
        sc = float(rng.uniform(0.95, 1.05))
        h, w = y.shape[1:]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        x = np.stack([
            cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=float(c.min()))
            for c in x
        ])
        y = np.stack([
            cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            for c in y
        ])
    if rng.random() < 0.7:
        x[:2] = x[:2] * float(rng.uniform(0.85, 1.15)) + float(rng.uniform(-0.1, 0.1))
    if rng.random() < 0.25:
        k = int(rng.choice([3, 5]))
        x[:2] = np.stack([cv2.GaussianBlur(c, (k, k), 0) for c in x[:2]])
    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def _sample(size=64, seed=0):
    r = np.random.default_rng(seed)
    x = r.normal(size=(3, size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    x[2] = (np.hypot(xx - size / 2, yy - size / 2) / (size / 2)).astype(np.float32) - 0.6
    y = np.zeros((1, size, size), dtype=np.float32)
    y[0, 20:26, 10:50] = 1.0
    return x, y


@pytest.mark.parametrize("seed", range(40))
def test_default_config_reproduces_the_legacy_function_exactly(seed):
    x, y = _sample(seed=seed)
    ex, ey = legacy_augment(x.copy(), y.copy(), np.random.default_rng(seed))
    gx, gy = A.apply(x.copy(), y.copy(), np.random.default_rng(seed), A.AugmentConfig())
    np.testing.assert_array_equal(gx, ex)
    np.testing.assert_array_equal(gy, ey)


@pytest.mark.parametrize("seed", range(10))
def test_dataset_augment_is_the_default_apply(seed):
    x, y = _sample(seed=seed)
    ex, ey = legacy_augment(x.copy(), y.copy(), np.random.default_rng(seed))
    gx, gy = dataset_augment(x.copy(), y.copy(), np.random.default_rng(seed))
    np.testing.assert_array_equal(gx, ex)
    np.testing.assert_array_equal(gy, ey)


def test_from_dict_defaults_overrides_and_typos():
    assert A.AugmentConfig.from_dict(None) == A.AugmentConfig()
    c = A.AugmentConfig.from_dict({"rot90": 0.75, "scale": [0.8, 1.25]})
    assert c.rot90 == 0.75 and c.scale == (0.8, 1.25) and c.flip_rows == 0.5
    with pytest.raises(ValueError, match="rot_90"):
        A.AugmentConfig.from_dict({"rot_90": 0.5})


def _off(**kw):
    """Every default transform off, then the given overrides."""
    base = dict(flip_rows=0.0, flip_cols=0.0, affine_p=0.0, photo_p=0.0, blur_p=0.0)
    base.update(kw)
    return A.AugmentConfig.from_dict(base)


def test_everything_off_is_identity():
    x, y = _sample()
    gx, gy = A.apply(x.copy(), y.copy(), np.random.default_rng(0), _off())
    np.testing.assert_array_equal(gx, x)
    np.testing.assert_array_equal(gy, y)


def test_rot90_rotates_image_and_target_together():
    x, y = _sample()
    seen = set()
    for s in range(30):
        gx, gy = A.apply(x.copy(), y.copy(), np.random.default_rng(s), _off(rot90=1.0))
        k = next(k for k in (1, 2, 3) if np.array_equal(gy, np.rot90(y, k, axes=(1, 2))))
        np.testing.assert_array_equal(gx, np.rot90(x, k, axes=(1, 2)))
        seen.add(k)
    assert seen == {1, 2, 3}


def test_free_rotation_uses_the_configured_range():
    x, y = _sample()
    cfg = _off(affine_p=1.0, rotate_deg=180.0, scale=(1.0, 1.0))
    gx, gy = A.apply(x.copy(), y.copy(), np.random.default_rng(3), cfg)
    assert gy.shape == y.shape and gy.sum() > 0 and not np.array_equal(gy, y)


def test_zoom_scales_the_target_area():
    x, y = _sample()
    big = _off(affine_p=1.0, rotate_deg=0.0, scale=(1.25, 1.25))
    small = _off(affine_p=1.0, rotate_deg=0.0, scale=(0.8, 0.8))
    a_big = A.apply(x.copy(), y.copy(), np.random.default_rng(0), big)[1].sum()
    a_small = A.apply(x.copy(), y.copy(), np.random.default_rng(0), small)[1].sum()
    assert a_big > y.sum() > a_small


def test_per_channel_photometric_differs_between_channels():
    x, y = _sample()
    cfg = _off(photo_p=1.0, per_channel=True, contrast=(0.5, 1.5), brightness=(-0.5, 0.5))
    gx, _ = A.apply(x.copy(), y.copy(), np.random.default_rng(1), cfg)
    r0 = np.polyfit(x[0].ravel(), gx[0].ravel(), 1)
    r1 = np.polyfit(x[1].ravel(), gx[1].ravel(), 1)
    assert not np.allclose(r0, r1)
    np.testing.assert_array_equal(gx[2], x[2])          # radius untouched


def test_noise_and_gradient_touch_only_image_channels():
    x, y = _sample()
    for cfg in (_off(noise_p=1.0, noise_sigma=0.1), _off(gradient_p=1.0, gradient_amp=0.5)):
        gx, gy = A.apply(x.copy(), y.copy(), np.random.default_rng(2), cfg)
        assert not np.allclose(gx[:2], x[:2])
        np.testing.assert_array_equal(gx[2], x[2])
        np.testing.assert_array_equal(gy, y)


def test_gradient_is_linear_across_the_frame():
    x = np.zeros((3, 32, 32), dtype=np.float32)
    y = np.zeros((1, 32, 32), dtype=np.float32)
    gx, _ = A.apply(x, y, np.random.default_rng(5), _off(gradient_p=1.0, gradient_amp=0.5))
    g = gx[0]
    assert np.abs(g).max() <= 0.5 * np.sqrt(2) + 1e-5
    # a linear field has zero second differences
    assert np.allclose(np.diff(g, 2, axis=0), 0, atol=1e-5)
    assert np.allclose(np.diff(g, 2, axis=1), 0, atol=1e-5)


def test_cutout_blanks_image_and_target_consistently():
    x, y = _sample()
    y[:] = 1.0
    gx, gy = A.apply(x.copy(), y.copy(), np.random.default_rng(4),
                     _off(cutout_p=1.0, cutout_n=2, cutout_size=(16, 16)))
    holes = gy[0] == 0
    assert holes.any()
    assert np.all(gx[0][holes] == 0) and np.all(gx[1][holes] == 0)
    np.testing.assert_array_equal(gx[2], x[2])


def test_new_transforms_off_draw_nothing_from_the_rng():
    x, y = _sample()
    r1, r2 = np.random.default_rng(9), np.random.default_rng(9)
    A.apply(x.copy(), y.copy(), r1, A.AugmentConfig())
    legacy_augment(x.copy(), y.copy(), r2)
    assert r1.random() == r2.random()


def _disk_sample(size=256, seed=0):
    """A sample whose channel 2 is a real clipped radius map: R = 0.88 * half-width,
    rr = clip(dist / R, 0, 1.2) - 0.6 (see dataset.build_input)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, size, size)).astype(np.float32)
    half = size / 2.0
    R = 0.88 * half
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dist = np.hypot(xx - half, yy - half)
    x[2] = (np.clip(dist / R, 0.0, 1.2) - 0.6).astype(np.float32)
    y = np.zeros((1, size, size), dtype=np.float32)
    return x, y, R


def test_radius_border_sky_keeps_warped_in_border_off_disk():
    """With the flag on, the affine warp's border fill for channel 2 is the sky
    value (channel max), so `valid = (gx[2] + 0.6) < 0.98` matches the warped
    disk to within pixel-discretisation noise. With the flag off (default), the
    border instead reads as disk-centre, so `valid` is inflated by the border
    ring -- this is the bug (~36% of the frame at scale 0.8, matching the
    measurement in the augment module docstring / bug report)."""
    size = 256
    x, y, R = _disk_sample(size)
    scale = 0.8
    cfg_fixed = _off(affine_p=1.0, rotate_deg=0.0, scale=(scale, scale), radius_border_sky=True)
    cfg_default = _off(affine_p=1.0, rotate_deg=0.0, scale=(scale, scale))

    gx_fixed, _ = A.apply(x.copy(), y.copy(), np.random.default_rng(0), cfg_fixed)
    gx_default, _ = A.apply(x.copy(), y.copy(), np.random.default_rng(0), cfg_default)

    valid_fixed = float(((gx_fixed[2] + 0.6) < 0.98).mean())
    valid_default = float(((gx_default[2] + 0.6) < 0.98).mean())

    # The warp shrinks the disk radius by the scale factor; the valid threshold
    # (0.98, applied in the source frame before warping) shrinks with it.
    valid_radius = R * scale * 0.98
    analytic = np.pi * valid_radius ** 2 / size ** 2

    assert abs(valid_fixed - analytic) < 0.02
    assert valid_default - valid_fixed >= 0.10


def test_radius_border_sky_only_changes_channel_2():
    size = 256
    x, y, _ = _disk_sample(size)
    scale = 0.8
    seed = 7
    cfg_on = _off(affine_p=1.0, rotate_deg=0.0, scale=(scale, scale), radius_border_sky=True)
    cfg_off = _off(affine_p=1.0, rotate_deg=0.0, scale=(scale, scale))

    gx_on, gy_on = A.apply(x.copy(), y.copy(), np.random.default_rng(seed), cfg_on)
    gx_off, gy_off = A.apply(x.copy(), y.copy(), np.random.default_rng(seed), cfg_off)

    np.testing.assert_array_equal(gx_on[:2], gx_off[:2])
    assert not np.array_equal(gx_on[2], gx_off[2])
    np.testing.assert_array_equal(gy_on, gy_off)


def test_extra_planes_are_filled_with_zero_by_the_warp():
    x = np.full((4, 64, 64), -1.0, np.float32)
    y = np.zeros((1, 64, 64), np.float32)
    cfg = A.AugmentConfig(flip_rows=0.0, flip_cols=0.0, affine_p=1.0, rotate_deg=0.0,
                          scale=(0.5, 0.5), photo_p=0.0, blur_p=0.0)
    xo, _ = A.apply(x, y, np.random.default_rng(0), cfg)
    assert xo[3, 0, 0] == 0.0 and xo[3, 32, 32] == -1.0     # extra plane: 0 outside the warped frame
    assert xo[0, 0, 0] == -1.0                              # channels 0-1: their own minimum, as before
