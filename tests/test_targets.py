"""Tests for filseg.targets and the dataset's target options (synthetic data, CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import synth_data  # noqa: E402
from filseg.dataset import FilamentDataset  # noqa: E402
from filseg.targets import consensus, small_instance_weights, tubed_skeleton  # noqa: E402

S = 256


def test_consensus_is_the_mean_mask():
    a = np.zeros((4, 4), np.float32)
    b = np.ones((4, 4), np.float32)
    assert np.allclose(consensus([a, b, b], (4, 4)), 2 / 3)
    assert consensus([], (3, 5)).shape == (3, 5) and not consensus([], (3, 5)).any()


def test_tubed_skeleton_is_a_thin_line_inside_the_mask():
    m = np.zeros((40, 80), np.float32)
    m[15:26, 10:70] = 1.0                                   # an 11-px-thick bar
    sk = tubed_skeleton(m, radius=1)
    assert sk.dtype == np.float32 and set(np.unique(sk)) <= {0.0, 1.0}
    assert sk[20, 40] == 1.0                                # on the axis
    assert sk[15, 40] == 0.0 and sk[25, 40] == 0.0          # the bar's own edges are not skeleton
    assert not sk[m == 0].any()                             # never outside the mask
    assert (sk[:, 40] > 0).sum() <= 3                       # tube width 2 * radius + 1
    assert not tubed_skeleton(np.zeros((8, 8), np.float32)).any()


def test_small_instance_weights_upweight_small_instances_only():
    m = np.zeros((60, 60), np.float32)
    m[5:15, 5:15] = 1.0                                     # area 100
    m[30:60, 30:60] = 1.0                                   # area 900
    w = small_instance_weights(m, ref_area=400.0, w_max=4.0)
    assert w[10, 10] == pytest.approx(2.0)                  # (400 / 100) ** 0.5
    assert w[40, 40] == pytest.approx(1.0)                  # large: clipped up to 1
    assert w[0, 59] == 1.0                                  # background
    assert small_instance_weights(m, ref_area=1e6, w_max=4.0)[10, 10] == pytest.approx(4.0)
    assert (small_instance_weights(m, 400.0, 1.0) == 1.0).all()


def _ds(monkeypatch, **kw):
    folds, ann, disks = synth_data.install(monkeypatch)
    return FilamentDataset(folds, ann, size=S, cache_disks=disks, **kw)


def test_pass_target_is_each_pass_own_mask(monkeypatch):
    ds = _ds(monkeypatch, train=False)
    assert ds[0]["y"][0, 127, 120].item() == pytest.approx(1.0)
    assert ds[2]["y"][0, 127, 120].item() == pytest.approx(0.0)
    assert "skel" not in ds[0] and "pixw" not in ds[0]


def test_consensus_target_is_shared_by_every_pass(monkeypatch):
    ds = _ds(monkeypatch, train=False, loss_cfg={"target": "consensus"})
    for i in range(3):
        assert ds[i]["y"][0, 127, 120].item() == pytest.approx(2 / 3, abs=1e-6)
    assert ds[0]["y"][0, 60, 60].item() == 0.0


def test_training_items_carry_skeleton_and_pixel_weights(monkeypatch):
    ds = _ds(monkeypatch, train=True, augment_cfg=synth_data.NO_AUG,
             loss_cfg={"skeleton_weight": 0.25, "small_weight_max": 4.0,
                       "small_ref_area": 64000.0})
    item = ds[0]
    assert item["skel"].shape == (1, S, S) and item["pixw"].shape == (1, S, S)
    assert item["skel"][0, 125:130, 100:150].sum() > 0
    assert item["skel"][0, :120].sum() == 0
    # 5 x 50 = 250 px at 256; ref 64000 * (256 / 2048) ** 2 = 1000 -> (1000 / 250) ** 0.5 = 2
    assert item["pixw"][0, 127, 120].item() == pytest.approx(2.0)
    assert item["pixw"][0, 60, 60].item() == 1.0


def test_exposure_weights_are_uniform_per_exposure(monkeypatch):
    """S1 has 3 passes (p1, p2, p3); adding S2 with a single pass (p4) must give S1's
    rows 1/3 each and S2's row 1, so every exposure is sampled equally often."""
    folds, ann, disks = synth_data.install(monkeypatch)
    folds = pd.concat([folds, pd.DataFrame(
        {"image_id": ["p4"], "file_stem": ["S2"], "fold": [1]})], ignore_index=True)
    disks["S2"] = synth_data.DISK
    ds = FilamentDataset(folds, ann, size=S, cache_disks=disks, loss_cfg={"target": "consensus"})

    w = ds.exposure_weights()

    assert w.dtype == np.float64
    np.testing.assert_allclose(w, [1 / 3, 1 / 3, 1 / 3, 1.0])
    sums = pd.Series(w).groupby(folds.file_stem.values).sum()
    assert sums["S1"] == pytest.approx(1.0)
    assert sums["S2"] == pytest.approx(1.0)


from filseg.targets import contrast_instance_weights  # noqa: E402


def test_contrast_weights_upweight_faint_instances_only():
    mask = np.zeros((60, 120), np.float32)
    mask[20:30, 10:50] = 1
    mask[20:30, 70:110] = 1
    flat = np.zeros((60, 120), np.float32)
    flat[20:30, 10:50] = -0.30          # strong: ring 0 minus inside -0.30 = contrast 0.30
    flat[20:30, 70:110] = -0.04         # faint: contrast 0.04
    w = contrast_instance_weights(mask, flat, ref=0.116, w_max=4.0, ring=3)
    assert w[25, 30] == 1.0                                   # 0.116 / 0.30 < 1 -> 1
    assert abs(w[25, 90] - 2.9) < 0.05                        # 0.116 / 0.04
    assert (w[mask == 0] == 1.0).all()


def test_contrast_weights_clip_at_w_max_and_handle_edge_cases():
    mask = np.zeros((40, 40), np.float32)
    mask[10:20, 10:30] = 1
    flat = np.zeros((40, 40), np.float32)
    flat[10:20, 10:30] = -0.01                                 # 11.6x -> clipped
    assert contrast_instance_weights(mask, flat, 0.116, 4.0, ring=3)[15, 15] == 4.0
    flat[10:20, 10:30] = +0.05                                 # brighter than its ring: contrast <= 0 -> w_max
    assert contrast_instance_weights(mask, flat, 0.116, 4.0, ring=3)[15, 15] == 4.0
    assert (contrast_instance_weights(np.zeros((8, 8), np.float32), flat[:8, :8], 0.116, 4.0) == 1.0).all()
    assert (contrast_instance_weights(mask, flat, 0.116, 1.0, ring=3) == 1.0).all()


def test_dataset_emits_contrast_weights_above_one_on_the_faint_rectangle(monkeypatch):
    folds, ann, disks = synth_data.install(monkeypatch)
    ds = FilamentDataset(folds, ann, size=S, train=True, cache_disks=disks, augment_cfg=synth_data.NO_AUG,
                         loss_cfg={"contrast_weight_max": 4.0})
    item = ds[0]
    w = item["pixw"][0].numpy()
    assert w.shape == (S, S) and w.max() > 1.3 and w.min() == 1.0    # the rectangle is 10/150 darker: ~1.7x
    assert w[0, 0] == 1.0 and w[127, 120] > 1.3
