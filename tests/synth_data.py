"""A synthetic exposure for dataset tests: no JPEG, no COCO file.

One exposure ``S1`` with three annotation passes: ``p1`` and ``p2`` each mark the
same dark rectangle (annotations ``a1``, ``a2``), ``p3`` marks nothing. At 256 px
the rectangle covers rows 125-129 and columns 100-149, so its consensus is 2/3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from filseg import data as fdata

FULL = 2048
DISK = fdata.Disk(cx=1024.0, cy=1024.0, radius=900.0)
RECT = (slice(1000, 1040), slice(800, 1200))                  # rows, cols at 2048
NO_AUG = {"flip_rows": 0.0, "flip_cols": 0.0, "affine_p": 0.0, "photo_p": 0.0, "blur_p": 0.0}


def image() -> np.ndarray:
    yy, xx = np.mgrid[0:FULL, 0:FULL]
    r = np.hypot(xx - DISK.cx, yy - DISK.cy) / DISK.radius
    img = np.where(r <= 1.0, 150.0 - 20.0 * r ** 2, 5.0).astype(np.float32)
    img[RECT] -= 10.0
    return np.clip(img, 0, 255).astype(np.uint8)


def _rect_mask() -> np.ndarray:
    m = np.zeros((FULL, FULL), np.uint8)
    m[RECT] = 1
    return m


def install(monkeypatch):
    img = image()
    masks = {"a1": _rect_mask(), "a2": _rect_mask()}
    empty = np.zeros((FULL, FULL), np.uint8)
    monkeypatch.setattr(fdata, "load_image", lambda stem, split="train": img)
    monkeypatch.setattr(fdata, "ann_to_mask", lambda aid, size=FULL: masks.get(aid, empty))
    folds = pd.DataFrame({"image_id": ["p1", "p2", "p3"], "file_stem": ["S1"] * 3, "fold": [1, 1, 1]})
    ann = pd.DataFrame({"image_id": ["p1", "p2"], "ann_id": ["a1", "a2"]})
    return folds, ann, {"S1": DISK}
