"""Tests for filseg.inputs and the dataset's input selection (synthetic data, CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import synth_data  # noqa: E402
from filseg import inputs as finputs  # noqa: E402
from filseg.dataset import FilamentDataset, build_input, model_input  # noqa: E402

S = 256
ALL = ["flat", "raw", "radius", "latitude", "tophat", "clahe"]


def test_validate():
    assert finputs.validate(None) == finputs.DEFAULT
    assert finputs.validate(["raw", "flat"]) == ("raw", "flat")
    for bad in ([], ["flat", "flat"], ["flat", "gamma"]):
        with pytest.raises(ValueError):
            finputs.validate(bad)
    assert finputs.wants_latitude(["flat", "latitude"]) and not finputs.wants_latitude(None)


def test_default_selection_is_the_untouched_legacy_array():
    x = np.random.default_rng(0).normal(size=(3, 8, 8)).astype(np.float32)
    assert finputs.select(x, None) is x


def test_selection_follows_the_listed_order():
    x = np.random.default_rng(0).normal(size=(3, 8, 8)).astype(np.float32)
    out = finputs.select(x, ["radius", "flat"])
    assert np.array_equal(out[0], x[2]) and np.array_equal(out[1], x[0])
    with pytest.raises(ValueError, match="latitude"):
        finputs.select(x, ["flat", "latitude"])


def test_latitude_plane_is_north_positive_and_zero_off_disk():
    lat = finputs.latitude_plane(synth_data.DISK, (2048, 2048))
    assert lat[300, 1024] > 0.7 and lat[1750, 1024] < -0.7
    assert abs(lat[1024, 1024]) < 1e-6 and lat[10, 10] == 0.0


def test_tophat_responds_to_a_dark_line_only():
    flat = np.zeros((64, 64), np.float32)
    flat[30:33, 10:54] = -0.4
    th = finputs.black_tophat(flat, 1024)                   # 1024-px windows (9/17/33)
    assert th[31, 30] > 0.5 and th[5, 5] == pytest.approx(0.0)


def test_clahe_is_zero_off_disk_and_finite():
    raw = np.random.default_rng(0).normal(scale=0.2, size=(64, 64)).astype(np.float32)
    disk = np.zeros((64, 64), bool)
    disk[8:56, 8:56] = True
    c = finputs.clahe_plane(raw, disk)
    assert np.isfinite(c).all() and (c[~disk] == 0).all() and c[disk].std() > 0


def _ds(monkeypatch, inputs=None):
    folds, ann, disks = synth_data.install(monkeypatch)
    return FilamentDataset(folds, ann, size=S, train=False, cache_disks=disks, inputs=inputs)


def test_dataset_builds_every_plane(monkeypatch):
    default = _ds(monkeypatch)[0]
    item = _ds(monkeypatch, ALL)[0]
    x = item["x"].numpy()
    assert x.shape == (6, S, S)
    assert np.array_equal(x[:3], default["x"].numpy())
    assert np.array_equal(item["valid"].numpy(), default["valid"].numpy())
    assert x[3, 30, 128] > 0 and x[3, 226, 128] < 0          # latitude: north up
    assert x[4, 127, 125] > 0.5 and x[4, 60, 128] < 0.1       # top-hat lights up the filament
    assert x[4, 0, 0] == 0 and x[5, 0, 0] == 0               # derived planes are 0 off-disk


def test_dropping_planes_keeps_the_valid_mask(monkeypatch):
    default = _ds(monkeypatch)[0]
    item = _ds(monkeypatch, ["flat", "raw"])[0]
    assert item["x"].shape == (2, S, S)
    assert np.array_equal(item["x"].numpy(), default["x"].numpy()[:2])
    assert np.array_equal(item["valid"].numpy(), default["valid"].numpy())


def test_model_input_matches_the_dataset(monkeypatch):
    img = synth_data.image()
    for inputs in (None, ALL, ["raw", "radius"]):
        x_ds = _ds(monkeypatch, inputs)[0]["x"].numpy()
        assert np.array_equal(model_input(img, synth_data.DISK, S, inputs), x_ds)
    assert np.array_equal(model_input(img, synth_data.DISK, S), build_input(img, synth_data.DISK, S))
