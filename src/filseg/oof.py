"""Out-of-fold bookkeeping: ground truth per exposure and how predictions relate to it.

Panoptic Quality reports TP / FP / FN, but those three numbers hide *why* an
instance was missed. A filament we found but split in two is one FN plus two FPs;
two filaments we fused into one prediction are two FNs plus one FP; a filament we
never touched is a plain FN. Each failure calls for a different fix (rejoin, split,
detect), and the competition rubric scores the first two directly as one-to-many and
many-to-one relations. This module names them.

Relations are decided per (annotation pass, prediction set) pair, because the
official scorer evaluates every annotator's record separately -- see
``filseg.metric``. The classification is pure and works on cropped
:class:`~filseg.metric.Instance` objects, so it is cheap to test on synthetic masks.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import data as fdata
from .dataset import instances_from_annotations
from .metric import IOU_GATE, Instance, intersection

# A prediction "touches" a ground-truth filament when it covers at least this share
# of the filament's pixels. Grazing contact of a few pixels between neighbouring
# filaments must not register as a relation, or every crowded frame would look
# fragmented.
TOUCH_FRAC = 0.10


# --- ground truth ----------------------------------------------------------------

@lru_cache(maxsize=1)
def _passes_by_stem() -> dict[str, list[str]]:
    img = fdata.image_table()
    return {s: sorted(g.image_id) for s, g in img.groupby("file_stem")}


@lru_cache(maxsize=1)
def _ann_ids_by_pass() -> dict[str, list[str]]:
    ann = fdata.annotation_table()[["image_id", "ann_id"]]
    return {k: list(v) for k, v in ann.groupby("image_id").ann_id}


def passes_of(stem: str) -> list[str]:
    """Annotation-pass ids (``image_id``) of one exposure, sorted."""
    return _passes_by_stem().get(stem, [])


def gt_passes(stem: str) -> list[list[Instance]]:
    """Ground-truth instances of every annotation pass of ``stem``, at 2048."""
    by_pass = _ann_ids_by_pass()
    return [instances_from_annotations(by_pass.get(pid, [])) for pid in passes_of(stem)]


# --- relations -------------------------------------------------------------------

def overlap_tables(gts: list[Instance], preds: list[Instance]):
    """Intersection, IoU and GT-coverage matrices, shape (n_gt, n_pred)."""
    inter = np.zeros((len(gts), len(preds)), dtype=np.int64)
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            inter[i, j] = intersection(g, p)
    g_area = np.array([g.area for g in gts], dtype=np.int64)[:, None]
    p_area = np.array([p.area for p in preds], dtype=np.int64)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(inter > 0, inter / (g_area + p_area - inter), 0.0)
        cover = np.where(inter > 0, inter / np.maximum(g_area, 1), 0.0)
    return inter, iou, cover


def _union_iou(g: Instance, preds: list[Instance]) -> float:
    """IoU between one ground-truth instance and the pixel union of several predictions."""
    if not preds:
        return 0.0
    y0 = min([g.y0, *(p.y0 for p in preds)])
    x0 = min([g.x0, *(p.x0 for p in preds)])
    y1 = max([g.y1, *(p.y1 for p in preds)])
    x1 = max([g.x1, *(p.x1 for p in preds)])
    gm = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    gm[g.y0 - y0 : g.y1 - y0, g.x0 - x0 : g.x1 - x0] = g.patch
    um = np.zeros_like(gm)
    for p in preds:
        um[p.y0 - y0 : p.y1 - y0, p.x0 - x0 : p.x1 - x0] |= p.patch
    inter = int(np.count_nonzero(gm & um))
    union = int(np.count_nonzero(gm | um))
    return inter / union if union else 0.0


# Why a ground-truth filament was not a true positive.
GT_MATCHED = "matched"          # some prediction passes the IoU gate
GT_FRAGMENTED = "fragmented"    # no single prediction passes, the union of those touching it does
GT_MERGED = "merged"            # its best prediction also covers another filament
GT_PARTIAL = "partial"          # touched, but too little of it -- undersized or displaced
GT_MISSED = "missed"            # nothing touches it

# Why a prediction was not a true positive.
PRED_MATCHED = "matched"
PRED_FRAGMENT = "fragment"      # a piece of a fragmented filament
PRED_MERGE = "merge"            # covers two or more filaments
PRED_PARTIAL = "partial"        # touches a filament without passing the gate
PRED_SPURIOUS = "spurious"      # touches nothing annotated


@dataclass(frozen=True)
class Relations:
    """Per-instance verdicts for one (annotation pass, prediction set) pair."""

    gt_kind: list[str]
    pred_kind: list[str]
    gt_fragments: list[int]      # per GT: how many predictions touch it
    pred_filaments: list[int]    # per pred: how many GT filaments it touches
    matched_ious: list[float]    # IoU of every (gt, pred) pair above the gate
    gt_best_iou: list[float]     # per GT: best single-prediction IoU (0 if untouched)
    gt_union_iou: list[float]    # per GT: IoU with the union of the predictions touching it
    gt_best_cover: list[float]   # per GT: share of its pixels covered by its best prediction

    @property
    def one_to_many(self) -> int:
        """Ground-truth filaments touched by two or more predictions (rubric relation)."""
        return sum(1 for n in self.gt_fragments if n >= 2)

    @property
    def many_to_one(self) -> int:
        """Predictions touching two or more ground-truth filaments (rubric relation)."""
        return sum(1 for n in self.pred_filaments if n >= 2)


def classify(gts: list[Instance], preds: list[Instance], gate: float = IOU_GATE,
             touch: float = TOUCH_FRAC) -> Relations:
    """Name every miss and every false alarm.

    A prediction *touches* a filament when it covers at least ``touch`` of the
    filament's pixels. Verdicts, in priority order:

    * GT ``matched`` if any prediction has IoU > ``gate`` with it; ``fragmented`` if
      none does but the union of the predictions touching it does; ``merged`` if the
      prediction covering most of it also touches another filament; ``partial`` if
      something touches it; else ``missed``.
    * Pred ``matched`` if it passes the gate with any filament; ``merge`` if it touches
      two or more; ``fragment`` if the filament it touches was classed fragmented;
      ``partial`` if it touches one; else ``spurious``.
    """
    n_g, n_p = len(gts), len(preds)
    inter, iou, cover = overlap_tables(gts, preds)
    touching = cover >= touch                     # (n_gt, n_pred)
    hit = iou > gate

    gt_fragments = touching.sum(axis=1).tolist()
    pred_filaments = touching.sum(axis=0).tolist()
    matched_ious = [float(v) for v in iou[hit]]

    gt_kind: list[str] = []
    gt_best_iou = [float(iou[i].max()) if n_p else 0.0 for i in range(n_g)]
    gt_best_cover = [float(cover[i].max()) if n_p else 0.0 for i in range(n_g)]
    gt_union_iou: list[float] = []
    for i in range(n_g):
        touching_preds = [preds[j] for j in np.flatnonzero(touching[i])]
        union_iou = _union_iou(gts[i], touching_preds) if touching_preds else 0.0
        gt_union_iou.append(union_iou)
        if n_p and hit[i].any():
            gt_kind.append(GT_MATCHED)
            continue
        if not touching_preds:
            gt_kind.append(GT_MISSED)
            continue
        if len(touching_preds) >= 2 and union_iou > gate:
            gt_kind.append(GT_FRAGMENTED)
            continue
        best = int(np.argmax(cover[i]))
        if touching[:, best].sum() >= 2:
            gt_kind.append(GT_MERGED)
            continue
        gt_kind.append(GT_PARTIAL)

    pred_kind: list[str] = []
    for j in range(n_p):
        if n_g and hit[:, j].any():
            pred_kind.append(PRED_MATCHED)
            continue
        touched = np.flatnonzero(touching[:, j]) if n_g else np.array([], dtype=int)
        if len(touched) >= 2:
            pred_kind.append(PRED_MERGE)
        elif len(touched) == 1 and gt_kind[int(touched[0])] == GT_FRAGMENTED:
            pred_kind.append(PRED_FRAGMENT)
        elif len(touched) == 1:
            pred_kind.append(PRED_PARTIAL)
        else:
            pred_kind.append(PRED_SPURIOUS)

    return Relations(gt_kind, pred_kind, gt_fragments, pred_filaments, matched_ious,
                     gt_best_iou, gt_union_iou, gt_best_cover)


def spurious_everywhere(preds: list[Instance], passes: list[list[Instance]]) -> list[bool]:
    """True for a prediction classed ``spurious`` in every annotation pass of its exposure
    (an exposure without any pass makes every prediction spurious). Stricter than the
    false-positive gallery's worst-case summary: a prediction that touches anything any
    annotator drew is not an annotator omission."""
    if not preds:
        return []
    if not passes:
        return [True] * len(preds)
    kinds = [classify(gts, preds).pred_kind for gts in passes]
    return [all(k[j] == PRED_SPURIOUS for k in kinds) for j in range(len(preds))]
