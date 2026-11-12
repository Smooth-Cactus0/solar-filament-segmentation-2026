"""Tests for the probability-map cache.

The cache exists so post-processing can be developed without a GPU, which only
works if thresholding a cached map gives the same instances as thresholding the
live map. These tests pin the storage contract: lossless uint8 PNG, quantisation
error bounded by 1/510, a manifest that carries everything instance extraction
needs (fold, disk, size), and fold indices read from checkpoint names rather than
guessed.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cache_probs as cp  # noqa: E402
import predict as fpredict  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.instances import probs_to_instances  # noqa: E402


def _probs(size: int = 256, seed: int = 0) -> np.ndarray:
    """A smooth map with a few confident blobs and a broad faint background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    p = 0.05 * rng.random((size, size)).astype(np.float32)
    for cy, cx, h in ((60, 80, 0.95), (160, 100, 0.7), (200, 40, 0.35)):
        p += h * np.exp(-((yy - cy) ** 2 / 60.0 + (xx - cx) ** 2 / 400.0)).astype(np.float32)
    return np.clip(p, 0.0, 1.0)


# --- quantisation ---------------------------------------------------------------

def test_quantize_round_trip_is_within_half_a_level():
    p = _probs()
    back = cp.dequantize(cp.quantize(p))
    assert back.dtype == np.float32
    assert np.abs(back - p).max() <= 0.5 / cp.QUANT + 1e-6


def test_quantize_clips_out_of_range_values():
    q = cp.quantize(np.array([[-0.2, 0.0, 0.5, 1.0, 1.7]], dtype=np.float32))
    assert q.dtype == np.uint8
    assert q.tolist() == [[0, 0, 128, 255, 255]]


def test_png_round_trip_is_lossless(tmp_path):
    p = _probs()
    path = tmp_path / "oof" / "20140609195854Bh.png"
    cp.write_map(path, p)
    back = cp.read_map(path)
    assert back.shape == p.shape
    np.testing.assert_array_equal(cp.quantize(back), cp.quantize(p))


def test_read_map_rejects_a_colour_image(tmp_path):
    import cv2

    path = tmp_path / "rgb.png"
    cv2.imwrite(str(path), np.zeros((8, 8, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        cp.read_map(path)


def test_thresholding_a_cached_map_only_differs_within_quantisation(tmp_path):
    """The property the cache is for: instances from the cached map match the live map
    except where a pixel sits within half a level of the threshold."""
    p = _probs()
    path = tmp_path / "m.png"
    cp.write_map(path, p)
    back = cp.read_map(path)
    for thr in (0.25, 0.4, 0.5):
        live = p > thr
        cached = back > thr
        disagree = live ^ cached
        assert np.all(np.abs(p[disagree] - thr) <= 0.5 / cp.QUANT + 1e-6)
    # And at a threshold that is not near any level boundary, the instance lists agree.
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    thr = 0.4 + 0.5 / cp.QUANT  # half a level above a representable value
    a = probs_to_instances(p, thr, min_area=10, disk=disk, size=256)
    b = probs_to_instances(back, thr, min_area=10, disk=disk, size=256)
    assert [(i.y0, i.x0, i.area) for i in a] == [(i.y0, i.x0, i.area) for i in b]


# --- fold parsing ---------------------------------------------------------------

@pytest.mark.parametrize("name, fold", [
    ("exp2_convnext_1536_fold0.pt", 0),
    ("checkpoints/exp2_convnext_1536_fold4.pt", 4),
    ("/kaggle/input/notebooks/u/filseg-exp2-f3/exp2_convnext_1536_fold3.pt", 3),
    ("exp1_convnext_1024_fold12.pt", 12),
])
def test_fold_of_reads_the_index_from_the_filename(name, fold):
    assert cp.fold_of(name) == fold


def test_fold_of_refuses_a_name_without_a_fold():
    with pytest.raises(ValueError):
        cp.fold_of("exp2_convnext_1536_best.pt")


# --- manifest -------------------------------------------------------------------

def _cfg():
    return {"encoder": "tu-convnext_small", "size": 1536, "amp": True,
            "eval": {"tta": True}}


def test_manifest_records_what_instance_extraction_needs(tmp_path):
    m = cp.new_manifest(_cfg(), "configs/exp2_convnext_1536.yaml",
                        {1: "x/exp2_convnext_1536_fold1.pt", 0: "y/exp2_convnext_1536_fold0.pt"},
                        tta=True, scales=None)
    assert m["config"] == "exp2_convnext_1536.yaml"
    assert m["size"] == 1536 and m["quant"] == cp.QUANT
    assert m["scales"] == [1536]  # single-scale recorded explicitly
    assert list(m["checkpoints"]) == ["0", "1"]  # sorted by fold, names only
    assert m["checkpoints"]["0"] == "exp2_convnext_1536_fold0.pt"

    disk = Disk(cx=1023.5, cy=1024.0, radius=890.25)
    m["oof"]["20140609195854Bh"] = {"fold": 0, **cp.disk_record(disk)}
    m["test"]["20150125172714Mh"] = cp.disk_record(disk)
    cp.write_manifest(tmp_path, m)

    loaded = cp.load_manifest(tmp_path)
    assert loaded == json.loads(json.dumps(m))
    assert cp.disk_from_record(loaded["oof"]["20140609195854Bh"]) == disk
    assert cp.disk_from_record(loaded["test"]["20150125172714Mh"]) == disk

    disks = (tmp_path / cp.DISKS_CSV).read_text(encoding="utf-8").splitlines()
    assert disks[0] == "split,stem,cx,cy,radius,fold"
    assert "oof,20140609195854Bh,1023.5,1024.0,890.25,0" in disks
    assert "test,20150125172714Mh,1023.5,1024.0,890.25,-1" in disks


def test_manifest_keeps_multi_scale_settings():
    m = cp.new_manifest(_cfg(), "c.yaml", {0: "f_fold0.pt"}, tta=False, scales=[1280, 1536])
    assert m["scales"] == [1280, 1536] and m["tta"] is False


# --- zip ------------------------------------------------------------------------

def test_zip_cache_holds_every_file_at_its_relative_path(tmp_path):
    out = tmp_path / "exp2"
    cp.write_map(out / "oof" / "a.png", _probs(64))
    cp.write_map(out / "test" / "b.png", _probs(64, seed=1))
    (out / cp.MANIFEST).write_text("{}", encoding="utf-8")
    zp = cp.zip_cache(out)
    assert zp == tmp_path / "exp2.zip"
    with zipfile.ZipFile(zp) as zf:
        assert sorted(zf.namelist()) == ["manifest.json", "oof/a.png", "test/b.png"]
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())

# --- --scales override ----------------------------------------------------

def test_scales_override_wins_over_the_config_and_reaches_the_manifest():
    cfg = {"encoder": "e", "size": 1536, "eval": {"tta": True, "scales": [1536]}}
    assert cp.resolve_scales(cfg, [1280, 1536]) == [1280, 1536]
    assert cp.resolve_scales(cfg, None) == [1536]
    assert cp.resolve_scales({"eval": {"tta": True}}, None) is None
    m = cp.new_manifest(cfg, "x.yaml", {0: "x_fold0.pt"}, True, cp.resolve_scales(cfg, [1280, 1536]))
    assert m["scales"] == [1280, 1536]
    assert cp.new_manifest(cfg, "x.yaml", {}, True, cp.resolve_scales({"eval": {}}, None))["scales"] == [1536]


def test_manifest_records_the_cache_size_not_the_model_size():
    cfg = {"encoder": "e", "size": 2048, "eval": {}}
    m = cp.new_manifest(cfg, "c.yaml", {0: "c_fold0.pt"}, True, None, cache_size=1536)
    assert m["size"] == 1536 and m["model_size"] == 2048
    assert m["scales"] == [1536]           # single-scale: the scales describe the stored maps
    assert cp.new_manifest(cfg, "c.yaml", {0: "c_fold0.pt"}, True, None)["size"] == 2048
    ms = cp.new_manifest(cfg, "c.yaml", {0: "c_fold0.pt"}, True, [1536, 2048], cache_size=1536)
    assert ms["scales"] == [1536, 2048] and ms["model_size"] == 2048    # multi-scale TTA kept as run


class TwoHead(torch.nn.Module):
    """A fake model: channel 0 = x[0], channel 1 = x[1] (the spine), as logits.

    Not flip-equivariant on purpose: it only sees the spine in the top-left quadrant, so a
    flipped view loses the peak (an equivariant model would return it unchanged under TTA).
    """
    filseg_head = "mask"

    def forward(self, x):
        out = x[:, :2].clone()
        out[:, 1, 8:, :] = 0.0
        out[:, 1, :, 8:] = 0.0
        return out


class OneHead(torch.nn.Module):
    filseg_head = "mask"

    def forward(self, x):
        return x[:, :1]


def test_spine_for_reads_the_channel_after_the_mask_head():
    x = torch.zeros(1, 3, 16, 16)
    x[0, 1, 4, 4] = 4.0                      # a strong spine logit at one pixel
    sp = fpredict.spine_for([TwoHead()], x, tta=False, amp=False)
    assert sp.shape == (16, 16) and sp[4, 4] > 0.98 and abs(sp[0, 0] - 0.5) < 1e-6
    sp_tta = fpredict.spine_for([TwoHead()], x, tta=True, amp=False)
    assert abs(sp_tta[4, 4] - (0.98 + 3 * 0.5) / 4) < 0.02   # flips move the peak away three times out of four
    with pytest.raises(ValueError, match="spine"):
        fpredict.spine_for([OneHead()], x, tta=False, amp=False)


# --- the spine cache -------------------------------------------------

def _fake_run(tmp_path, monkeypatch):
    import pandas as pd
    cfg = {"encoder": "e", "size": 8, "amp": False, "eval": {"tta": False}}
    monkeypatch.setattr(cp, "load_config", lambda path: cfg)
    monkeypatch.setattr(cp, "load_model", lambda cfg, ckpt, device: object())
    monkeypatch.setattr(cp.fdata, "load_image", lambda stem, split="train": np.zeros((16, 16), np.uint8))
    monkeypatch.setattr(cp.fdata, "detect_disk", lambda img: Disk(cx=8.0, cy=8.0, radius=6.0))
    monkeypatch.setattr(cp.fdata, "list_stems", lambda split: ["t0"])
    monkeypatch.setattr(cp, "model_input", lambda img, disk, size, inputs=None: np.zeros((3, 8, 8), np.float32))
    monkeypatch.setattr(cp, "probs_for", lambda *a, **k: np.full((8, 8), 0.3, np.float32))
    monkeypatch.setattr(cp, "spine_for", lambda *a, **k: np.full((8, 8), 0.6, np.float32))
    folds = tmp_path / "folds.csv"
    pd.DataFrame({"file_stem": ["o0", "o1"], "fold": [0, 1]}).to_csv(folds, index=False)
    out, spine = tmp_path / "c", tmp_path / "c_spine"
    cp.main(["--config", "x.yaml", "--checkpoints", "x_fold0.pt", "x_fold1.pt", "--folds-csv", str(folds),
             "--outdir", str(out), "--spine-outdir", str(spine), "--zip"])
    return out, spine


def test_the_spine_cache_is_marked_and_zipped_beside_the_probability_cache(tmp_path, monkeypatch):
    from filseg.cache import ProbCache
    out, spine = _fake_run(tmp_path, monkeypatch)
    assert ProbCache.open(out).manifest.get("channel") != "spine"
    sc = ProbCache.open(spine, allow_spine=True)
    assert sc.manifest["channel"] == "spine" and sc.stems("oof") == ["o0", "o1"]
    np.testing.assert_allclose(sc.probs("test", "t0"), 0.6, atol=1 / cp.QUANT)
    for z in (tmp_path / "c.zip", tmp_path / "c_spine.zip"):
        with zipfile.ZipFile(z) as zf:
            assert {"manifest.json", "oof/o0.png", "test/t0.png"} <= set(zf.namelist())


@pytest.mark.parametrize("module,extra", [
    ("build_scorer_table", []),
    ("build_tree_table", ["--out", "t.csv.gz"]),
    ("build_scorer_crops", ["--table", "t.csv.gz", "--out", "crops"]),
    ("build_ignore_masks", ["--scored", "s.csv.gz", "--out", "ignore"]),
])
def test_cache_builders_refuse_a_spine_cache(tmp_path, monkeypatch, module, extra):
    import importlib
    _, spine = _fake_run(tmp_path, monkeypatch)
    monkeypatch.chdir(tmp_path)
    mod = pytest.importorskip(module)          # a builder that is not part of this checkout is skipped
    with pytest.raises(ValueError, match="spine"):
        mod.main(["--cache", str(spine), *extra])


# --- --device (the pipeline notebook's DEVICE) -----------------------------------------

def test_device_flag_reaches_the_model_loader_and_turns_amp_off_on_cpu(tmp_path, monkeypatch):
    import pandas as pd
    cfg = {"encoder": "e", "size": 8, "amp": True, "eval": {"tta": False}}
    loaded, amps = [], []
    monkeypatch.setattr(cp, "load_config", lambda path: cfg)
    monkeypatch.setattr(cp, "load_model", lambda cfg, ckpt, device: loaded.append(device) or object())
    monkeypatch.setattr(cp.fdata, "load_image", lambda stem, split="train": np.zeros((16, 16), np.uint8))
    monkeypatch.setattr(cp.fdata, "detect_disk", lambda img: Disk(cx=8.0, cy=8.0, radius=6.0))
    monkeypatch.setattr(cp.fdata, "list_stems", lambda split: ["t0", "t1"])
    monkeypatch.setattr(cp, "model_input", lambda img, disk, size, inputs=None: np.zeros((3, 8, 8), np.float32))
    monkeypatch.setattr(cp, "probs_for",
                        lambda models, x, tta, amp, *a, **k: amps.append(amp) or np.full((8, 8), 0.3, np.float32))
    folds = tmp_path / "folds.csv"
    pd.DataFrame({"file_stem": ["o0"], "fold": [0]}).to_csv(folds, index=False)
    cp.main(["--config", "x.yaml", "--checkpoints", "x_fold0.pt", "x_fold1.pt", "--folds-csv", str(folds),
             "--outdir", str(tmp_path / "c"), "--no-oof", "--limit", "1", "--device", "cpu"])
    assert loaded == ["cpu", "cpu"]          # both fold models, on the requested device
    assert amps == [False]                   # one test image (--limit 1), autocast off on the CPU
    assert sorted(p.name for p in (tmp_path / "c" / "test").iterdir()) == ["t0.png"]
