"""The dataset's native-resolution crop mode (synthetic data)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import synth_data  # noqa: E402
from filseg.dataset import FilamentDataset, crop_window  # noqa: E402


def test_crop_window_is_inside_the_frame_and_deterministic_for_validation():
    d = synth_data.DISK
    y0, x0 = crop_window(d, 2048, 1024)
    assert (y0, x0) == (512, 512)                                       # centred on the disk
    assert crop_window(d, 2048, 1024, j=0, per=4) == crop_window(d, 2048, 1024, j=0, per=4)
    corners = {crop_window(d, 2048, 1024, j=j, per=4) for j in range(4)}
    assert len(corners) == 4
    rng = np.random.default_rng(3)
    for _ in range(50):
        y, x = crop_window(d, 2048, 1024, rng=rng)
        assert 0 <= y <= 1024 and 0 <= x <= 1024
    y, x = crop_window(d, 512, 256, rng=np.random.default_rng(1))       # the disk is given at 2048 px
    assert 0 <= y <= 256 and 0 <= x <= 256


def test_dataset_crop_mode_returns_crops_and_repeats_training_rows(monkeypatch):
    folds, ann, disks = synth_data.install(monkeypatch)
    ds = FilamentDataset(folds, ann, size=512, train=True, cache_disks=disks, augment_cfg=synth_data.NO_AUG,
                         crops={"size": 256, "per_exposure": 3})
    assert len(ds) == 3 * 3 and len(ds.exposure_weights()) == len(ds)
    item = ds[4]
    assert tuple(item["x"].shape) == (3, 256, 256) and tuple(item["y"].shape) == (1, 256, 256)
    assert tuple(item["valid"].shape) == (1, 256, 256) and item["image_id"] == "p2"
    va = FilamentDataset(folds, ann, size=512, train=False, cache_disks=disks, crops={"size": 256, "per_exposure": 3})
    assert len(va) == 3 * 3 and len(va.exposure_weights()) == len(va)    # per_exposure fixed crops per row
    assert tuple(va[0]["x"].shape) == (3, 256, 256) and va[3]["image_id"] == "p2"
    # item 0 is row 0's crop j = 0: half the radius out at angle 0, i.e. right of the disk centre
    y0, x0 = crop_window(synth_data.DISK, 512, 256, j=0, per=3)
    assert y0 == 128 and x0 > 128
    full = FilamentDataset(folds, ann, size=512, train=False, cache_disks=disks)
    np.testing.assert_array_equal(va[0]["x"].numpy(), full[0]["x"].numpy()[:, y0:y0 + 256, x0:x0 + 256])
    assert va[0]["y"][0].numpy().sum() > 0                                # the rectangle's right part
    crops = [va[j]["x"].numpy() for j in range(3)]                        # one exposure's crops all differ
    assert all(not np.array_equal(crops[a], crops[b]) for a in range(3) for b in range(a + 1, 3))
    np.testing.assert_array_equal(va[0]["x"].numpy(), va[0]["x"].numpy())  # and are fixed


def test_crop_mode_refuses_derived_input_planes(monkeypatch):
    import pytest
    folds, ann, disks = synth_data.install(monkeypatch)
    for train in (True, False):
        for plane in ("tophat", "clahe"):
            with pytest.raises(ValueError, match=plane):
                FilamentDataset(folds, ann, size=512, train=train, cache_disks=disks, augment_cfg=synth_data.NO_AUG,
                                crops={"size": 256}, inputs=["flat", "raw", "radius", plane])
    FilamentDataset(folds, ann, size=512, train=True, cache_disks=disks, augment_cfg=synth_data.NO_AUG,
                    crops={"size": 256}, inputs=["flat", "raw", "radius", "latitude"])   # geometric is fine
