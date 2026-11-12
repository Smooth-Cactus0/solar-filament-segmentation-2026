"""Pseudo-label pretraining: GONG frames with the ensemble's soft maps as the target.

Everything is synthesised in ``tmp_path`` (a bright disk on dark sky saved as JPEG, and a soft
teacher blob written with ``cache_probs.write_map``), so no competition data is needed. The
item contract is the one ``FilamentDataset`` already honours, because the trainer's loop,
``dice_proxy`` and ``FilamentLoss`` consume both datasets unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cache_probs as cp  # noqa: E402
import train  # noqa: E402
from filseg import data as fdata  # noqa: E402
from filseg.cache import read_map  # noqa: E402
from filseg.dataset import PseudoDataset  # noqa: E402
from train import check_pseudo, pseudo_split  # noqa: E402

SIZE = 256
STEMS = ["20150101180000Bh", "20150102180000Ch"]
R2_AUGMENT = {"radius_border_sky": True, "scale": [0.8, 1.25], "cutout_p": 0.5, "cutout_n": 4,
              "cutout_size": [32, 96], "blur_p": 0.0}
FLIP_ROWS_ONLY = {"flip_rows": 1.0, "flip_cols": 0.0, "affine_p": 0.0, "photo_p": 0.0, "blur_p": 0.0}


def _frame(size: int = SIZE, seed: int = 0) -> np.ndarray:
    """A limb-darkened disk of radius 0.43 * size at the centre, on dark sky, with a dark streak."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2.0
    r = np.hypot(xx - c, yy - c) / (0.43 * size)
    img = np.where(r <= 1.0, 150.0 - 20.0 * r ** 2, 5.0) + rng.normal(0.0, 1.0, (size, size))
    img[int(0.45 * size):int(0.48 * size), int(0.35 * size):int(0.60 * size)] -= 12.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _target(size: int = SIZE) -> np.ndarray:
    """A soft teacher blob peaking at 0.9 over the streak, falling off smoothly."""
    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = 0.465 * size, 0.475 * size
    return (0.9 * np.exp(-((yy - cy) ** 2 / (0.03 * size) ** 2 + (xx - cx) ** 2 / (0.12 * size) ** 2))
            ).astype(np.float32)


def _write_frames(images: Path, targets: Path, stems, size: int = SIZE) -> None:
    images.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(stems):
        Image.fromarray(_frame(size, seed=i)).save(images / f"{s}.jpg", quality=95)
        cp.write_map(targets / f"{s}.png", _target(size))


@pytest.fixture
def gong(tmp_path):
    images, targets = tmp_path / "images", tmp_path / "targets"
    _write_frames(images, targets, STEMS)
    return images, targets


def test_item_carries_the_filament_dataset_keys(gong):
    ds = PseudoDataset(STEMS, *gong, size=SIZE, train=False)
    assert len(ds) == 2
    item = ds[1]
    assert set(item) == {"x", "y", "valid", "image_id", "file_stem"}
    assert item["image_id"] == item["file_stem"] == STEMS[1]
    assert item["x"].dtype == torch.float32 and tuple(item["x"].shape) == (3, SIZE, SIZE)
    assert item["valid"].dtype == torch.float32 and tuple(item["valid"].shape) == (1, SIZE, SIZE)


def test_x_has_one_plane_per_selected_input(gong):
    for inputs in (["flat", "raw", "radius", "latitude"], ["flat", "radius"], ["flat", "raw", "radius", "tophat"]):
        x = PseudoDataset(STEMS, *gong, size=SIZE, train=False, inputs=inputs)[0]["x"]
        assert x.shape[0] == len(inputs), inputs


def test_target_is_the_soft_teacher_map(gong):
    images, targets = gong
    y = PseudoDataset(STEMS, images, targets, size=SIZE, train=False)[0]["y"]
    assert y.dtype == torch.float32 and tuple(y.shape) == (1, SIZE, SIZE)
    assert float(y.min()) >= 0.0 and float(y.max()) <= 1.0
    assert bool(((y > 0.05) & (y < 0.95)).any()), "a soft map must keep its non-binary levels"
    np.testing.assert_array_equal(y[0].numpy(), read_map(targets / f"{STEMS[0]}.png", quant=255))


def test_target_is_area_resized_to_the_dataset_size(gong):
    images, targets = gong
    y = PseudoDataset(STEMS, images, targets, size=128, train=False)[0]["y"]
    assert tuple(y.shape) == (1, 128, 128)
    want = cv2.resize(read_map(targets / f"{STEMS[0]}.png", quant=255), (128, 128), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(y[0].numpy(), want, atol=1e-6)


def test_valid_is_zero_off_disk_and_one_at_the_centre(gong):
    v = PseudoDataset(STEMS, *gong, size=SIZE, train=False)[0]["valid"][0]
    assert float(v[SIZE // 2, SIZE // 2]) == 1.0
    assert float(v[0, 0]) == 0.0 and float(v[SIZE - 1, SIZE - 1]) == 0.0
    # The loss stops at 0.98 R: a pixel just inside the limb is already off.
    r = 0.43 * SIZE
    assert float(v[SIZE // 2, int(SIZE / 2 + 0.995 * r)]) == 0.0
    assert float(v[SIZE // 2, int(SIZE / 2 + 0.9 * r)]) == 1.0


def test_training_items_are_deterministic_for_a_fixed_seed(gong):
    a = PseudoDataset(STEMS, *gong, size=SIZE, train=True, seed=7, augment_cfg=R2_AUGMENT)
    b = PseudoDataset(STEMS, *gong, size=SIZE, train=True, seed=7, augment_cfg=R2_AUGMENT)
    for i in range(len(STEMS)):
        ia, ib = a[i], b[i]
        for k in ("x", "y", "valid"):
            assert torch.equal(ia[k], ib[k]), (i, k)


def test_training_augments_input_and_target_together(gong):
    val = PseudoDataset(STEMS, *gong, size=SIZE, train=False)[0]
    tr = PseudoDataset(STEMS, *gong, size=SIZE, train=True, augment_cfg=FLIP_ROWS_ONLY)[0]
    assert torch.equal(tr["y"], val["y"].flip(1))
    assert torch.equal(tr["x"], val["x"].flip(1))
    assert torch.equal(tr["valid"], val["valid"].flip(1))


def test_disk_is_detected_once_per_stem(gong, monkeypatch):
    calls = []
    real = fdata.detect_disk
    monkeypatch.setattr(fdata, "detect_disk", lambda img: calls.append(1) or real(img))
    ds = PseudoDataset(STEMS, *gong, size=SIZE, train=False)
    ds[0], ds[0], ds[1]
    assert len(calls) == 2


def test_loss_only_targets_follow_the_loss_config(gong):
    item = PseudoDataset(STEMS, *gong, size=SIZE, train=True, augment_cfg=FLIP_ROWS_ONLY,
                         loss_cfg={"skeleton_weight": 0.5, "contrast_weight_max": 3.0})[0]
    assert tuple(item["skel"].shape) == tuple(item["pixw"].shape) == (1, SIZE, SIZE)


def test_paste_augmentations_are_refused(gong):
    for aug in ({"copy_paste_p": 0.5}, {"bank_paste_p": 0.5}):
        with pytest.raises(ValueError, match="paste"):
            PseudoDataset(STEMS, *gong, size=SIZE, train=True, augment_cfg=aug)


def _manifest(stems, keep=None) -> pd.DataFrame:
    keep = [1] * len(stems) if keep is None else keep
    return pd.DataFrame({"stem": stems, "site": [s[14] for s in stems], "keep": keep})


def _stems(n: int) -> list[str]:
    return [f"2015{1 + i // 28:02d}{1 + i % 28:02d}180000{'BCLMTU'[i % 6]}h" for i in range(n)]


def test_pseudo_split_holds_out_the_requested_share_deterministically():
    m = _manifest(_stems(40))
    tr, va = pseudo_split(m, 0.1, seed=20260924)
    assert (len(tr), len(va)) == (36, 4)
    assert not set(tr) & set(va) and set(tr) | set(va) == set(m.stem)
    assert pseudo_split(m, 0.1, seed=20260924) == (tr, va)
    assert pseudo_split(m.iloc[::-1].reset_index(drop=True), 0.1, seed=20260924) == (tr, va)


def test_pseudo_split_uses_only_kept_frames():
    stems = _stems(40)
    m = _manifest(stems, keep=[0 if i % 4 == 0 else 1 for i in range(40)])
    tr, va = pseudo_split(m, 0.1, seed=0)
    assert set(tr) | set(va) == {s for i, s in enumerate(stems) if i % 4}
    assert (len(tr), len(va)) == (27, 3)


def test_pseudo_split_never_leaves_a_side_empty():
    tr, va = pseudo_split(_manifest(_stems(2)), 0.05, seed=0)
    assert (len(tr), len(va)) == (1, 1)
    with pytest.raises(ValueError):
        pseudo_split(_manifest(_stems(10)), 1.5, seed=0)


def test_pseudo_is_refused_with_annotation_only_options():
    base = {"pseudo": {"manifest": "m.csv", "images": "i", "targets": "t", "holdout": 0.05}}
    check_pseudo(dict(base))
    for extra in ({"crops": {"size": 512}}, {"ignore_dir": "data/ignore/x"}, {"spine_weight": 0.5},
                  {"loss": {"target": "consensus"}}):
        with pytest.raises(SystemExit, match="pseudo"):
            check_pseudo({**base, **extra})
    with pytest.raises(SystemExit, match="holdout"):
        check_pseudo({"pseudo": {"manifest": "m.csv", "images": "i", "targets": "t"}})


# --- the trainer's pseudo branch, end to end on CPU with a one-layer model -------------------

PRE_STEMS = _stems(9)
PRE_KEEP = [1, 1, 1, 1, 1, 1, 1, 0, 1]                        # PRE_STEMS[7] failed the quality rule


def _pretrain_config(tmp_path: Path, holdout: float = 0.25) -> Path:
    stems = PRE_STEMS
    images, targets = tmp_path / "gong" / "images", tmp_path / "gong" / "targets"
    _write_frames(images, targets, stems[:8], size=64)       # stems[8] is kept but never downloaded
    m = _manifest(stems, keep=PRE_KEEP)
    for col in ("date", "utc", "url", "status", "cx", "cy", "radius", "mean", "std", "reason"):
        m[col] = ""
    m.to_csv(tmp_path / "gong" / "manifest.csv", index=False)
    cfg = {"name": "pre_test", "encoder": "tiny", "size": 64, "seed": 1, "epochs": 2, "batch_size": 2,
           "grad_accum": 1, "lr": 1.0e-3, "lr_min": 1.0e-5, "weight_decay": 0.0, "grad_clip": 1.0,
           "workers": 0, "amp": False, "dice_weight": 0.25, "pos_weight": 3.0, "spine_weight": 0.0,
           "augment": R2_AUGMENT,
           "pseudo": {"manifest": str(tmp_path / "gong" / "manifest.csv"), "images": str(images),
                      "targets": str(targets), "holdout": holdout}}
    p = tmp_path / "pre_test.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _run_main(monkeypatch, tmp_path: Path, argv: list[str]) -> None:
    monkeypatch.setattr(train, "CKPT_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(train, "RESULT_DIR", tmp_path / "results")
    monkeypatch.setattr(train, "build_from_config",
                        lambda cfg, pretrained=None: nn.Conv2d(3, 1, 3, padding=1))
    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    train.main()


def test_pretraining_run_writes_an_unfolded_checkpoint_and_result(tmp_path, monkeypatch):
    cfg_path = _pretrain_config(tmp_path)
    _run_main(monkeypatch, tmp_path, ["--config", str(cfg_path)])
    assert (tmp_path / "checkpoints" / "pre_test.pt").is_file()
    res = json.loads((tmp_path / "results" / "pre_test.json").read_text(encoding="utf-8"))
    assert set(res) == {"name", "fold", "config", "best_proxy_dice", "history", "pretrain", "n_train", "n_val"}
    assert res["name"] == "pre_test" and res["fold"] == -1 and res["pretrain"] is True
    # The 8 kept frames split 6 / 2 first; the one never downloaded then drops out of its side.
    tr, va = pseudo_split(_manifest(PRE_STEMS, keep=PRE_KEEP), 0.25, seed=1)
    assert (len(tr), len(va)) == (6, 2)
    assert (res["n_train"], res["n_val"]) == (len(set(tr) - {PRE_STEMS[8]}), len(set(va) - {PRE_STEMS[8]}))
    assert res["n_train"] + res["n_val"] == 7
    assert len(res["history"]) == 2 and 0.0 <= res["best_proxy_dice"] <= 1.0
    assert not train.init_result_missing(tmp_path / "results", str(tmp_path / "checkpoints" / "pre_test.pt"))


def test_pretraining_smoke_run_keeps_the_real_names_free(tmp_path, monkeypatch):
    _run_main(monkeypatch, tmp_path, ["--config", str(_pretrain_config(tmp_path)), "--smoke"])
    res = json.loads((tmp_path / "results" / "pre_test_smoke.json").read_text(encoding="utf-8"))
    assert res["config"]["name"] == "pre_test_smoke" and res["config"]["epochs"] == 2
    assert not (tmp_path / "results" / "pre_test.json").exists()


def test_pretraining_accepts_fold_minus_one_and_refuses_a_real_fold(tmp_path, monkeypatch):
    cfg_path = _pretrain_config(tmp_path)
    with pytest.raises(SystemExit, match="fold"):
        _run_main(monkeypatch, tmp_path, ["--config", str(cfg_path), "--fold", "0"])
    _run_main(monkeypatch, tmp_path, ["--config", str(cfg_path), "--fold", "-1"])
    assert (tmp_path / "results" / "pre_test.json").is_file()


def test_pretraining_refuses_missing_teacher_maps(tmp_path, monkeypatch):
    cfg_path = _pretrain_config(tmp_path)
    (tmp_path / "gong" / "targets" / f"{PRE_STEMS[0]}.png").unlink()
    with pytest.raises(SystemExit, match="teacher"):
        _run_main(monkeypatch, tmp_path, ["--config", str(cfg_path)])


def test_fold_is_still_required_without_a_pseudo_block(tmp_path, monkeypatch):
    with pytest.raises(SystemExit) as e:
        _run_main(monkeypatch, tmp_path, ["--config", str(ROOT / "configs" / "scr_r2_base_1024.yaml")])
    assert e.value.code == 2
