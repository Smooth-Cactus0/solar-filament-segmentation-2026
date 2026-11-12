"""The union gate: every member's accepted instances as one candidate set, each
candidate carrying the other members' evidence, for a second learned decision step.

Averaging maps drowns a filament only one member finds; the per-pixel maximum floods the
scorer with specks; fixed unions keep hits and misses alike. Here each member's own scorer
has already accepted its instances; the union keeps them whole, in member precedence, and
records how many members agree on each one. A LightGBM gate (``train_scorer.py``) then
decides with the v2 features on the proposer's map, the agreement block over every member's
map, and the gate block below.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Disk
from .metric import Instance, iou
from .scorer import add_frame_features, agreement_features, instance_features

GATE_NAMES: tuple[str, ...] = ("gate_src", "gate_score", "gate_n_support", "gate_iou_max")


@dataclass
class GateCand:
    """One union candidate: the proposing member, its score, and the agreement found."""

    inst: Instance
    src: int             # index of the proposing member, in precedence order
    score: float         # the proposer's own p*q (or the detector's confidence)
    n_support: int = 1   # members whose accepted instances overlap it at IoU >= gate, proposer included
    iou_max: float = 0.0  # the best such IoU from another member, 0 when none


def union_candidates(members: list[list[tuple[Instance, float]]], gate: float = 0.5) -> list[GateCand]:
    """Members in precedence order; an instance joins unless another member's kept one overlaps
    it at IoU >= gate, in which case that one gains support instead. Nothing is clipped.

    A member never absorbs its own instances (a detector's detections may overlap each other;
    the submission writer makes them disjoint by score), and each member adds at most one unit
    of support to a candidate, so ``n_support`` counts members, not detections.
    """
    kept: list[GateCand] = []
    for src, cands in enumerate(members):
        supported: set[int] = set()
        for inst, score in cands:
            best, best_iou = None, 0.0
            for k in kept:
                if k.src == src:
                    continue
                v = iou(k.inst, inst)
                if v > best_iou:
                    best, best_iou = k, v
            if best is not None and best_iou >= gate:
                if id(best) not in supported:
                    best.n_support += 1
                    supported.add(id(best))
                best.iou_max = max(best.iou_max, best_iou)
            else:
                kept.append(GateCand(inst, src, float(score)))
    return kept


def gate_features(cands: list[GateCand]) -> list[dict[str, float]]:
    return [{"gate_src": float(c.src), "gate_score": float(c.score),
             "gate_n_support": float(c.n_support), "gate_iou_max": float(c.iou_max)} for c in cands]


def candidate_features(cands: list[GateCand], maps: list[np.ndarray], disk: Disk,
                       flat: np.ndarray | None) -> list[dict[str, float]]:
    """All features of every union candidate of one frame, in ``cands`` order.

    v2 features on the proposer's own map (``maps[c.src]``), the agreement block over every
    member map, the frame features over the whole union, then the gate block.
    """
    feats = [instance_features(c.inst, maps[c.src], disk, flat) for c in cands]
    for c, f in zip(cands, feats):
        f.update(agreement_features(c.inst, maps))
    feats = add_frame_features(feats)
    for f, g in zip(feats, gate_features(cands)):
        f.update(g)
    return feats
