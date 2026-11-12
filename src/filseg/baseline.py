"""The naive limb-corrected threshold detector -- our known-score reference.

This is the strongest of the classical baselines we measured,
scoring **pooled PQ 0.166** (SQ 0.638, RQ 0.260) on held-out training images at
``ratio_threshold=0.94, min_area=800``. It traces the filaments it finds about
as well as a human does; it simply finds far too few and invents many.

Its value is as a *calibration probe*: because we know the number it should
produce, submitting it tells us whether our metric implementation and CSV
format agree with the leaderboard before any model is built on top of them.
"""
from __future__ import annotations

import cv2
import numpy as np

from .data import Disk, detect_disk, load_image
from .instances import disk_mask, label_instances
from .metric import Instance

# Tuned on 40 held-out training images.
DEFAULT_RATIO_THRESHOLD = 0.94
DEFAULT_MIN_AREA = 800

_K_OPEN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_K_CLOSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def illumination(img: np.ndarray) -> np.ndarray:
    """Smooth disk illumination estimate, for flat-fielding.

    A full-resolution median blur is far too slow. Filaments vanish at 1/8
    scale, so a median filter on a 256x256 thumbnail is a clean estimate of the
    smooth component, upsampled back.
    """
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    med = cv2.medianBlur(small, 31)
    return cv2.resize(med, img.shape[::-1], interpolation=cv2.INTER_LINEAR)


def flat_ratio(img: np.ndarray) -> np.ndarray:
    """Image divided by its own illumination: 1.0 is quiet Sun, < 1 is darker."""
    return img.astype(np.float32) / (illumination(img).astype(np.float32) + 1.0)


def detect(
    img: np.ndarray,
    disk: Disk | None = None,
    ratio_threshold: float = DEFAULT_RATIO_THRESHOLD,
    min_area: int = DEFAULT_MIN_AREA,
) -> list[Instance]:
    """Detect filament instances in one full-resolution frame."""
    if disk is None:
        disk = detect_disk(img)
    fg = (flat_ratio(img) < ratio_threshold) & disk_mask(disk, img.shape)
    m = fg.astype(np.uint8)
    # Open then close: drop single-pixel noise, then bridge one-pixel gaps
    # inside a filament. Both kernels are small -- closing hard would fuse
    # neighbouring filaments, which PQ punishes harder than splitting.
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _K_OPEN)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _K_CLOSE)
    return label_instances(m.astype(bool), min_area=min_area)


def detect_stem(stem: str, split: str = "test", **kwargs) -> list[Instance]:
    """Convenience wrapper: load an image by stem and detect."""
    return detect(load_image(stem, split), **kwargs)
