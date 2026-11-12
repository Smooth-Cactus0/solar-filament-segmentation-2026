"""Panoptic Quality exactly as the competition leaderboard computes it.

This reproduces the organisers' published scorer
(``refs/self-evaluation-notebook.ipynb``), which departs from textbook panoptic
evaluation in two ways that materially change how a model should be tuned:

1. **No one-to-one matching.** The scorer thresholds the full
   (n_gt x n_pred) IoU matrix and counts *every* pair above the gate as its own
   true positive. Textbook PQ forces a unique assignment; this does not. For
   mutually disjoint predictions the two agree, but they diverge as soon as
   predictions overlap each other.
2. **Counts are pooled globally.** TP/FP/FN and the summed IoU accumulate across
   every image before a single PQ is formed. It is *not* the mean of per-image
   scores.

    PQ = sum(IoU over TP pairs) / (|TP| + 0.5|FP| + 0.5|FN|),  gate: IoU > 0.5

``pq_unique`` implements the textbook variant so the two can be compared.

Note: scoring MAGFiLO ground truth against itself does not return exactly 1.0
under the official rule, because a few annotation passes contain two overlapping
outlines of the same filament and both are counted.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IOU_GATE = 0.5


@dataclass(frozen=True)
class Instance:
    """One segment, stored as a cropped patch plus its offset in the full frame.

    Full 2048x2048 boolean masks are ~4 MB each and an image may hold dozens, so
    instances are kept cropped to their bounding box throughout.
    """

    y0: int
    x0: int
    patch: np.ndarray  # bool, (h, w)

    @property
    def y1(self) -> int:
        return self.y0 + self.patch.shape[0]

    @property
    def x1(self) -> int:
        return self.x0 + self.patch.shape[1]

    @property
    def area(self) -> int:
        return int(self.patch.sum())


def from_mask(mask: np.ndarray, y0: int = 0, x0: int = 0) -> Instance | None:
    """Crop a boolean mask to its bounding box; None if it is empty."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    a, b = int(ys.min()), int(ys.max()) + 1
    c, d = int(xs.min()), int(xs.max()) + 1
    return Instance(y0 + a, x0 + c, np.ascontiguousarray(mask[a:b, c:d].astype(bool)))


def intersection(a: Instance, b: Instance) -> int:
    """Overlapping pixel count; 0 when the bounding boxes miss."""
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    if y1 <= y0 or x1 <= x0:
        return 0
    pa = a.patch[y0 - a.y0 : y1 - a.y0, x0 - a.x0 : x1 - a.x0]
    pb = b.patch[y0 - b.y0 : y1 - b.y0, x0 - b.x0 : x1 - b.x0]
    return int(np.count_nonzero(pa & pb))


def iou(a: Instance, b: Instance) -> float:
    inter = intersection(a, b)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


def image_counts(
    gts: list[Instance], preds: list[Instance], gate: float = IOU_GATE
) -> dict[str, float | list[float]]:
    """Official (non-unique) counts for a single image."""
    ious: list[float] = []
    gt_hit = [False] * len(gts)
    pred_hit = [False] * len(preds)
    for j, p in enumerate(preds):
        for i, g in enumerate(gts):
            v = iou(g, p)
            if v > gate:
                ious.append(v)
                gt_hit[i] = True
                pred_hit[j] = True
    return {
        "tp": len(ious),
        "fp": pred_hit.count(False),
        "fn": gt_hit.count(False),
        "iou_sum": float(np.sum(ious)) if ious else 0.0,
        "ious": ious,
    }


def unique_counts(
    gts: list[Instance], preds: list[Instance], gate: float = IOU_GATE
) -> dict[str, float | list[float]]:
    """Textbook counts: greedy one-to-one matching on descending IoU."""
    pairs = []
    for j, p in enumerate(preds):
        for i, g in enumerate(gts):
            v = iou(g, p)
            if v > gate:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    ious: list[float] = []
    for v, i, j in pairs:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i)
        used_pred.add(j)
        ious.append(v)
    return {
        "tp": len(ious),
        "fp": len(preds) - len(used_pred),
        "fn": len(gts) - len(used_gt),
        "iou_sum": float(np.sum(ious)) if ious else 0.0,
        "ious": ious,
    }


class PQAccumulator:
    """Pools counts across images, then forms one PQ -- the leaderboard's rule."""

    def __init__(self, unique: bool = False, gate: float = IOU_GATE) -> None:
        self._counter = unique_counts if unique else image_counts
        self.gate = gate
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.iou_sum = 0.0
        self.matched_ious: list[float] = []

    def add(self, gts: list[Instance], preds: list[Instance]) -> None:
        c = self._counter(gts, preds, self.gate)
        self.tp += int(c["tp"])
        self.fp += int(c["fp"])
        self.fn += int(c["fn"])
        self.iou_sum += float(c["iou_sum"])
        self.matched_ious.extend(c["ious"])  # type: ignore[arg-type]

    def summary(self) -> dict[str, float]:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        if denom == 0:
            # An empty prediction set against empty ground truth is perfect.
            return {"pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "fp": 0, "fn": 0}
        return {
            "pq": self.iou_sum / denom,
            "sq": (self.iou_sum / self.tp) if self.tp else 0.0,
            "rq": self.tp / denom,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
        }


def panoptic_quality(
    per_image: list[tuple[list[Instance], list[Instance]]], unique: bool = False
) -> dict[str, float]:
    """Pooled PQ over a list of (ground_truth, prediction) instance lists."""
    acc = PQAccumulator(unique=unique)
    for gts, preds in per_image:
        acc.add(gts, preds)
    return acc.summary()


def binary_dice(gts: list[Instance], preds: list[Instance], size: int = 2048) -> float:
    """Image-level Dice over the union of all instances (the secondary metric)."""
    a = np.zeros((size, size), dtype=bool)
    b = np.zeros((size, size), dtype=bool)
    for it in gts:
        a[it.y0 : it.y1, it.x0 : it.x1] |= it.patch
    for it in preds:
        b[it.y0 : it.y1, it.x0 : it.x1] |= it.patch
    tot = int(a.sum()) + int(b.sum())
    if tot == 0:
        return 1.0
    return 2.0 * int(np.count_nonzero(a & b)) / tot
