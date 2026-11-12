"""Per-instance features and labels for the learned instance scorer.

The shipped pipeline decides which components become filaments with three blunt
rules -- probability threshold, minimum area, top-k per image -- and the Tier-1
leaderboard sweep showed that this decision, not the model, is where the score
moves. The fragmentation diagnostic put a number on it: 17% of our predictions
touch nothing any annotator drew, worth up to +0.04 PQ if they could all be dropped.

This module turns each candidate component into a feature vector describing how
confident the model is in it (probability statistics), what it looks like (size,
shape, elongation), where it sits on the disk, and what the rest of the frame looks
like (crowding, rank) -- everything a second-stage model needs to estimate
``P(an annotator drew this)`` and, for the ones that match, the IoU it will get.
Labels come from the annotation passes, one row per pass, because the official
scorer evaluates each annotator record separately.

All features are computed from the cached probability map and the image alone, so
the same code scores test candidates at submission time.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from .data import Disk
from .instances import disk_mask, label_instances
from .metric import IOU_GATE, Instance, iou

# Probability level above which a pixel counts as "core" of the component; the
# core fraction separates a solid detection from a smear that barely clears the
# threshold everywhere.
CORE_P = 0.7
# Ring width (px at 2048) for the "what surrounds this component" features.
RING_PX = 6
# Neighbourhood radius (px) for the crowding features.
NEIGHBOUR_PX = 100.0
SMALL_AREA = 800

FEATURE_NAMES: tuple[str, ...] = (
    # confidence
    "area", "log_area", "p_mean", "p_max", "p_p90", "p_p50", "p_std", "core_frac",
    "ring_p_mean", "p_contrast",
    # shape
    "bbox_h", "bbox_w", "bbox_fill", "bbox_diag", "skel_len", "width_mean", "elongation",
    "n_endpoints", "solidity",
    # image
    "flat_mean", "flat_min", "flat_ring", "flat_contrast",
    # position
    "r_disk", "abs_lat", "abs_lon", "limb_dist",
    # frame context
    "n_cand", "rank_p", "rank_area", "frac_rank_p", "n_nb", "n_small_nb", "nn_dist",
    "frame_area_sum", "frame_p_mean",
)

# Cross-model agreement, present only when the table builder is given the blend's
# member caches. A member "supports" a candidate when at least SUPPORT_FRAC of the
# candidate's pixels are at or above SUPPORT_P in that member's own map. Only
# aggregates are kept, so the feature list does not depend on the member count.
AGREEMENT_NAMES: tuple[str, ...] = ("agree_mean", "agree_std", "agree_min", "agree_max", "n_support")
CNN_NAMES: tuple[str, ...] = ("cnn_p", "cnn_q")    # the crop scorer's out-of-fold p and q (train_crop_scorer.py)
SUPPORT_P = 0.40
SUPPORT_FRAC = 0.5


# --- geometry --------------------------------------------------------------------

def helio(cx: float, cy: float, disk: Disk) -> tuple[float, float, float]:
    """Radial fraction and approximate heliographic latitude / longitude (degrees).

    Orthographic projection with B0 = 0 and the image north-up; the +/-7 deg B0 and
    the P angle are ignored, which is fine for a feature that only needs to say
    "near the limb" or "far from the equator".
    """
    X = (cx - disk.cx) / disk.radius
    Y = -(cy - disk.cy) / disk.radius
    r = float(np.hypot(X, Y))
    z = float(np.sqrt(max(1.0 - X * X - Y * Y, 0.0)))
    lat = float(np.degrees(np.arcsin(np.clip(Y, -1.0, 1.0))))
    lon = float(np.degrees(np.arctan2(X, z)))
    return r, lat, lon


# --- candidates ------------------------------------------------------------------

def upsample(probs: np.ndarray, size: int = 2048) -> np.ndarray:
    """The cached map at full resolution, exactly as ``probs_to_instances`` does it."""
    if probs.shape[0] == size:
        return probs
    return cv2.resize(probs, (size, size), interpolation=cv2.INTER_LINEAR)


def candidates(probs_full: np.ndarray, disk: Disk, threshold: float, min_area: int) -> list[Instance]:
    """Connected components of the thresholded full-resolution map, on-disk only.

    Same chain as ``filseg.instances.probs_to_instances`` on an already-upsampled
    map, so with the shipped (threshold, min_area) the candidates are the shipped
    instances.
    """
    fg = probs_full > threshold
    fg &= disk_mask(disk, fg.shape)
    return label_instances(fg, min_area=min_area)


# --- features --------------------------------------------------------------------

def _padded_window(arr: np.ndarray, inst: Instance, pad: int) -> tuple[np.ndarray, tuple[int, int]]:
    """``arr`` cropped around the instance with ``pad`` px of margin (clipped at the frame)."""
    y0, x0 = max(inst.y0 - pad, 0), max(inst.x0 - pad, 0)
    y1, x1 = min(inst.y1 + pad, arr.shape[0]), min(inst.x1 + pad, arr.shape[1])
    return arr[y0:y1, x0:x1], (inst.y0 - y0, inst.x0 - x0)


def _skeleton_stats(patch: np.ndarray) -> tuple[int, int]:
    """Skeleton length and endpoint count of a boolean patch."""
    if patch.size == 0 or not patch.any():
        return 0, 0
    skel = skeletonize(patch)
    n = int(skel.sum())
    if n == 0:
        return 0, 0
    # An endpoint is a skeleton pixel with exactly one skeleton neighbour (8-connected).
    k = np.ones((3, 3), dtype=np.uint8)
    nb = ndimage.convolve(skel.astype(np.uint8), k, mode="constant") - skel.astype(np.uint8)
    return n, int(((nb == 1) & skel).sum())


def _solidity(patch: np.ndarray) -> float:
    contours, _ = cv2.findContours(patch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 1.0
    pts = np.concatenate(contours, axis=0)
    hull = cv2.convexHull(pts)
    hull_area = float(cv2.contourArea(hull))
    area = float(patch.sum())
    return float(min(area / hull_area, 1.0)) if hull_area > 0 else 1.0


def instance_features(inst: Instance, probs_full: np.ndarray, disk: Disk,
                      flat: np.ndarray | None = None) -> dict[str, float]:
    """Confidence, shape, image and position features of one candidate.

    ``probs_full`` and ``flat`` (the flat-fielded image ratio, 1.0 = quiet Sun) are
    both at the instance's resolution. Frame-context features are added afterwards
    by :func:`add_frame_features`, since they need every candidate of the frame.
    """
    patch = inst.patch
    area = float(inst.area)
    vals = probs_full[inst.y0 : inst.y1, inst.x0 : inst.x1][patch]

    # Ring: dilate the patch inside a padded window and subtract the patch.
    pad = RING_PX
    win, (oy, ox) = _padded_window(probs_full, inst, pad)
    inst_mask = np.zeros(win.shape, dtype=bool)
    inst_mask[oy : oy + patch.shape[0], ox : ox + patch.shape[1]] = patch
    dil = cv2.dilate(inst_mask.astype(np.uint8), np.ones((2 * pad + 1, 2 * pad + 1), np.uint8)) > 0
    ring = dil & ~inst_mask
    ring_p = float(win[ring].mean()) if ring.any() else 0.0

    skel_len, n_end = _skeleton_stats(patch)
    h, w = patch.shape
    cy = inst.y0 + float(np.nonzero(patch)[0].mean())
    cx = inst.x0 + float(np.nonzero(patch)[1].mean())
    r, lat, lon = helio(cx, cy, disk)

    f: dict[str, float] = {
        "area": area,
        "log_area": float(np.log(area)),
        "p_mean": float(vals.mean()),
        "p_max": float(vals.max()),
        "p_p90": float(np.percentile(vals, 90)),
        "p_p50": float(np.percentile(vals, 50)),
        "p_std": float(vals.std()),
        "core_frac": float((vals > CORE_P).mean()),
        "ring_p_mean": ring_p,
        "p_contrast": float(vals.mean() - ring_p),
        "bbox_h": float(h),
        "bbox_w": float(w),
        "bbox_fill": area / float(h * w),
        "bbox_diag": float(np.hypot(h, w)),
        "skel_len": float(skel_len),
        "width_mean": area / max(float(skel_len), 1.0),
        "elongation": float(skel_len) ** 2 / area if area else 0.0,
        "n_endpoints": float(n_end),
        "solidity": _solidity(patch),
        "r_disk": r,
        "abs_lat": abs(lat),
        "abs_lon": abs(lon),
        "limb_dist": 1.0 - r,
        "_cx": cx,
        "_cy": cy,
    }
    if flat is not None:
        fwin, _ = _padded_window(flat, inst, pad)
        fv = fwin[inst_mask]
        fr = fwin[ring]
        f["flat_mean"] = float(fv.mean())
        f["flat_min"] = float(fv.min())
        f["flat_ring"] = float(fr.mean()) if ring.any() else float(fv.mean())
        f["flat_contrast"] = f["flat_ring"] - f["flat_mean"]
    else:
        f.update({"flat_mean": np.nan, "flat_min": np.nan, "flat_ring": np.nan, "flat_contrast": np.nan})
    return f


def add_frame_features(feats: list[dict[str, float]]) -> list[dict[str, float]]:
    """Crowding and rank features; mutates and returns ``feats`` (one dict per candidate)."""
    n = len(feats)
    if n == 0:
        return feats
    p = np.array([f["p_mean"] for f in feats])
    a = np.array([f["area"] for f in feats])
    cxy = np.array([[f["_cx"], f["_cy"]] for f in feats])
    order_p = np.argsort(-p)
    order_a = np.argsort(-a)
    rank_p = np.empty(n, dtype=int)
    rank_a = np.empty(n, dtype=int)
    rank_p[order_p] = np.arange(n)
    rank_a[order_a] = np.arange(n)
    if n > 1:
        d = np.sqrt(((cxy[:, None, :] - cxy[None, :, :]) ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        near = d < NEIGHBOUR_PX
        n_nb = near.sum(axis=1)
        n_small_nb = (near & (a[None, :] < SMALL_AREA)).sum(axis=1)
    else:
        nn = np.array([np.inf])
        n_nb = np.zeros(1, dtype=int)
        n_small_nb = np.zeros(1, dtype=int)
    for i, f in enumerate(feats):
        f["n_cand"] = float(n)
        f["rank_p"] = float(rank_p[i])
        f["rank_area"] = float(rank_a[i])
        f["frac_rank_p"] = float(rank_p[i]) / n
        f["n_nb"] = float(n_nb[i])
        f["n_small_nb"] = float(n_small_nb[i])
        f["nn_dist"] = float(min(nn[i], 4096.0))
        f["frame_area_sum"] = float(a.sum())
        f["frame_p_mean"] = float(p.mean())
    return feats


def agreement_features(inst: Instance, member_maps: list[np.ndarray]) -> dict[str, float]:
    """How the blend's members agree on one candidate.

    ``member_maps`` are the members' probability maps at the instance's resolution.
    Returns NaN for every name when there are no members (LightGBM handles NaN).
    """
    if not member_maps:
        return {k: float("nan") for k in AGREEMENT_NAMES}
    patch = inst.patch
    means: list[float] = []
    support = 0
    for m in member_maps:
        vals = m[inst.y0 : inst.y1, inst.x0 : inst.x1][patch]
        means.append(float(vals.mean()))
        if float((vals >= SUPPORT_P).mean()) >= SUPPORT_FRAC:
            support += 1
    arr = np.asarray(means)
    return {"agree_mean": float(arr.mean()), "agree_std": float(arr.std()),
            "agree_min": float(arr.min()), "agree_max": float(arr.max()),
            "n_support": float(support)}


def frame_features(insts: list[Instance], probs_full: np.ndarray, disk: Disk,
                   flat: np.ndarray | None = None,
                   member_maps: list[np.ndarray] | None = None) -> list[dict[str, float]]:
    """All features for every candidate of one frame, in ``insts`` order.

    With ``member_maps`` (the blend's members at the frame's resolution) every
    candidate also carries the cross-model agreement features.
    """
    feats = [instance_features(it, probs_full, disk, flat) for it in insts]
    if member_maps is not None:
        for it, f in zip(insts, feats):
            f.update(agreement_features(it, member_maps))
    return add_frame_features(feats)


# --- trained models --------------------------------------------------------------

@dataclass(frozen=True)
class Scorer:
    """The fold models of one scorer version, averaged at prediction time.

    ``train_scorer.py`` saves one ``p`` classifier and one ``q`` regressor per
    exposure fold. At submission time no test image belongs to any fold, so the five
    of each are averaged -- the same ensembling the segmentation checkpoints get.
    """

    p_models: tuple
    q_models: tuple
    features: tuple[str, ...] = FEATURE_NAMES

    @classmethod
    def load(cls, model_dir, version: str = "scorer_v1", features=None) -> "Scorer":
        """The fold models of ``version``; the feature list comes from ``features``, else
        from ``<version>.features.json`` beside the models (scorer v3), else v1/v2's."""
        import json
        from pathlib import Path

        import lightgbm as lgb

        d = Path(model_dir)
        p_files = sorted(d.glob(f"{version}_p_fold*.txt"))
        q_files = sorted(d.glob(f"{version}_q_fold*.txt"))
        if not p_files:
            raise FileNotFoundError(f"no {version}_p_fold*.txt under {d}")
        if len(q_files) != len(p_files):
            raise FileNotFoundError(f"{len(p_files)} p models but {len(q_files)} q models under {d}")
        if features is None:
            fj = d / f"{version}.features.json"
            features = json.loads(fj.read_text(encoding="utf-8")) if fj.is_file() else FEATURE_NAMES
        return cls(tuple(lgb.Booster(model_file=str(f)) for f in p_files),
                   tuple(lgb.Booster(model_file=str(f)) for f in q_files), tuple(features))

    def _matrix(self, feats: list[dict[str, float]]) -> np.ndarray:
        return np.array([[f[k] for k in self.features] for f in feats], dtype=np.float64)

    def predict(self, feats: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
        """``(p, q)`` per candidate: match probability and expected IoU if matched."""
        if not feats:
            return np.zeros(0), np.zeros(0)
        x = self._matrix(feats)
        p = np.mean([m.predict(x) for m in self.p_models], axis=0)
        q = np.clip(np.mean([m.predict(x) for m in self.q_models], axis=0), 0.0, 1.0)
        return p, q


# --- labels ----------------------------------------------------------------------

@dataclass(frozen=True)
class PassLabels:
    """How one annotation pass judges a frame's candidates."""

    n_gt: int
    matched_gt: list[int]     # per candidate: index of the best GT above the gate, or -1
    best_iou: list[float]     # per candidate: IoU with its best GT (0 if none overlaps)


def label_against_pass(insts: list[Instance], gts: list[Instance],
                       gate: float = IOU_GATE) -> PassLabels:
    """Match every candidate to one annotation pass under the official gate.

    Non-unique, like the scorer: a candidate is a TP if *any* GT passes the gate
    with it, and the GT it is credited to is the one with the highest IoU.
    """
    matched, best = [], []
    for it in insts:
        ious = [iou(g, it) for g in gts]
        if ious:
            j = int(np.argmax(ious))
            best.append(float(ious[j]))
            matched.append(j if ious[j] > gate else -1)
        else:
            best.append(0.0)
            matched.append(-1)
    return PassLabels(len(gts), matched, best)
