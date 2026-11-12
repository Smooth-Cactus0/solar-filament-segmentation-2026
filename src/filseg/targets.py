"""Training targets derived from the rasterised annotation masks (spec §2c).

Everything here is a function of the masks alone. The annotated spine polylines are
MAGFiLO metadata that the rules keep out of supervision, so the skeleton is
recomputed from the mask instead.
"""
from __future__ import annotations

import cv2
import numpy as np


def consensus(masks: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel mean over an exposure's annotation passes: the agreement fraction."""
    if not masks:
        return np.zeros(shape, dtype=np.float32)
    return np.mean(np.stack(masks).astype(np.float32), axis=0)


def tubed_skeleton(mask: np.ndarray, threshold: float = 0.5, radius: int = 1) -> np.ndarray:
    """Skeleton of the binarised mask, dilated to a (2 * radius + 1)-px tube, clipped to the mask."""
    from skimage.morphology import skeletonize  # lazy: keep scikit-image off filseg.dataset's import path

    b = np.asarray(mask) >= threshold
    if not b.any():
        return np.zeros(b.shape, dtype=np.float32)
    sk = skeletonize(b).astype(np.uint8)
    if radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        sk = cv2.dilate(sk, k)
    return (sk & b.astype(np.uint8)).astype(np.float32)


def small_instance_weights(mask: np.ndarray, ref_area: float, w_max: float,
                           threshold: float = 0.5, gamma: float = 0.5) -> np.ndarray:
    """Per-pixel BCE weight: ``(ref_area / area) ** gamma`` on each instance, clipped to [1, w_max].

    Instances are the 8-connected components of the binarised mask; background is 1.
    """
    b = (np.asarray(mask) >= threshold).astype(np.uint8)
    out = np.ones(b.shape, dtype=np.float32)
    if w_max <= 1.0 or not b.any():
        return out
    _, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    w = np.clip((ref_area / np.maximum(area, 1.0)) ** gamma, 1.0, w_max).astype(np.float32)
    w[0] = 1.0
    return w[lab]


def contrast_instance_weights(mask: np.ndarray, flat: np.ndarray, ref: float, w_max: float,
                              ring: int = 5, threshold: float = 0.5) -> np.ndarray:
    """Per-pixel BCE weight ``clip(ref / contrast, 1, w_max)`` on each instance; background is 1.

    ``contrast`` is the mean of ``flat`` (the flat-field ratio minus one) on a ``ring``-px ring
    around the instance minus its mean inside: the scorer's ``flat_contrast``, which the
    report's slices showed is the strongest predictor of a miss (recall 0.33 in the faintest
    sixth, 0.79 in the strongest). ``ref`` is the median contrast over all annotations, so an
    instance at or above the median keeps weight 1 and a faint one is weighted up; an
    instance no darker than its ring (contrast <= 0) gets ``w_max``.
    """
    b = (np.asarray(mask) >= threshold).astype(np.uint8)
    out = np.ones(b.shape, dtype=np.float32)
    if w_max <= 1.0 or not b.any():
        return out
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    H, W = b.shape
    for i in range(1, n):
        x, y, w, h = (int(stats[i, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                                   cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        y0, y1, x0, x1 = max(0, y - ring), min(H, y + h + ring), max(0, x - ring), min(W, x + w + ring)
        inst = lab[y0:y1, x0:x1] == i
        ring_m = cv2.dilate(inst.astype(np.uint8), k).astype(bool) & (b[y0:y1, x0:x1] == 0)
        fw = flat[y0:y1, x0:x1]
        c = float(fw[ring_m].mean() - fw[inst].mean()) if ring_m.any() else ref
        out[y0:y1, x0:x1][inst] = float(np.clip(ref / c, 1.0, w_max)) if c > 0 else float(w_max)
    return out
