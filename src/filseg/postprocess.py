"""Post-processing that changes an instance's *extent* without changing the decision.

The shipped chain decides existence and extent with one number: a pixel is
filament iff its probability clears the threshold, and an instance is a connected
component of those pixels. The two are better separated:

* **Hysteresis** -- decide *which* instances exist from a strict seed threshold,
  then let each one *grow* to the looser threshold. A filament whose faint middle
  dips below 0.40 stays one instance instead of two; a faint region with no
  confident seed never becomes an instance. ``mode="merge"`` lets seeds sharing one
  loose region fuse (this is where fragments rejoin); ``mode="split"`` keeps every
  seed its own instance and divides the loose region between them by watershed on
  the probabilities, so neighbouring filaments joined by a faint bridge stay apart.
* **Dilation / erosion** -- a fixed-pixel change of every mask, for the leaderboard
  extent probes. Dilation is overlap-safe: instances are painted in priority order
  and a pixel is claimed once, because the scorer rejects overlapping masks.
* **Hole filling** -- the annotation protocol draws single polygons without holes.

Everything works on full-resolution (2048) probability maps and returns the same
cropped :class:`~filseg.metric.Instance` objects the rest of the pipeline uses.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed

from .data import Disk
from .instances import MAX_AREA, disk_mask, label_instances
from .metric import Instance, from_mask

MERGE = "merge"
SPLIT = "split"
# Rejoin only: seeds that share one loose region become one instance, but the
# instance keeps the seed pixels -- the grouping changes, the extent does not.
BRIDGE = "bridge"
MODES = (MERGE, SPLIT, BRIDGE)


def _seed_labels(probs_full: np.ndarray, disk: Disk | None, seed_thr: float,
                 min_area: int) -> tuple[np.ndarray, int]:
    """Connected-component labels of the seed mask, small ones already removed."""
    seed = probs_full > seed_thr
    if disk is not None:
        seed &= disk_mask(disk, seed.shape)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(seed.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        keep[i] = min_area <= area <= MAX_AREA
    lab = np.where(keep[lab], lab, 0)
    return lab, int(keep.sum())


def hysteresis_instances(probs_full: np.ndarray, disk: Disk | None, seed_thr: float,
                         grow_thr: float, min_area: int = 300, mode: str = MERGE,
                         max_area: int = MAX_AREA) -> list[Instance]:
    """Seed at ``seed_thr``, grow to ``grow_thr``; see the module docstring for ``mode``.

    With ``grow_thr == seed_thr`` this is exactly ``probs_to_instances``. ``min_area``
    applies to the *seeds*, so a faint region only becomes an instance if a
    sufficiently large confident core exists; the grown instance can be smaller than
    ``min_area`` only if the seed itself was (it cannot be), and larger than
    ``max_area`` instances are dropped after growing.
    """
    if grow_thr > seed_thr:
        raise ValueError(f"grow_thr ({grow_thr}) must not exceed seed_thr ({seed_thr})")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    seed_lab, n_seeds = _seed_labels(probs_full, disk, seed_thr, min_area)
    if n_seeds == 0:
        return []
    loose = probs_full > grow_thr
    if disk is not None:
        loose &= disk_mask(disk, loose.shape)
    loose |= seed_lab > 0  # seeds are always inside their own instance

    if mode in (MERGE, BRIDGE):
        n, lab = cv2.connectedComponents(loose.astype(np.uint8), connectivity=8)
        has_seed = np.zeros(n, dtype=bool)
        has_seed[np.unique(lab[seed_lab > 0])] = True
        has_seed[0] = False
        grown = np.where(has_seed[lab], lab, 0)
        if mode == BRIDGE:
            # Same grouping as merge, but only the seed pixels are kept: the two
            # halves of a filament whose middle dips below the seed threshold are
            # scored as one instance, at their own extent.
            grown = np.where(seed_lab > 0, grown, 0)
    else:
        # Watershed from the seed markers over -probability, confined to the loose
        # mask: each loose pixel goes to the seed it is most continuously connected
        # to, so two seeds joined by a faint bridge split at the bridge's weakest point.
        grown = watershed(-probs_full, markers=seed_lab, mask=loose, connectivity=2)

    out: list[Instance] = []
    objs = ndimage.find_objects(grown)
    for i, sl in enumerate(objs, 1):
        if sl is None:
            continue
        patch = grown[sl] == i
        area = int(patch.sum())
        if area == 0 or area > max_area:
            continue
        # A bridged instance is two or more disconnected pieces; from_mask keeps the
        # tight bounding box either way.
        out.append(Instance(int(sl[0].start), int(sl[1].start), np.ascontiguousarray(patch)))
    return out


def union_instances(members: list[Instance]) -> Instance:
    """One instance holding the pixel union of ``members`` (tight bounding box)."""
    if len(members) == 1:
        return members[0]
    y0 = min(m.y0 for m in members)
    x0 = min(m.x0 for m in members)
    y1 = max(m.y1 for m in members)
    x1 = max(m.x1 for m in members)
    patch = np.zeros((y1 - y0, x1 - x0), dtype=bool)
    for m in members:
        patch[m.y0 - y0 : m.y1 - y0, m.x0 - x0 : m.x1 - x0] |= m.patch
    return Instance(y0, x0, patch)


def bridge_instances(insts: list[Instance], probs_full: np.ndarray, grow_thr: float,
                     disk: Disk | None = None) -> list[Instance]:
    """Rejoin already-decided instances that share one region above ``grow_thr``.

    The counterpart of ``hysteresis_instances(mode="bridge")`` for an instance list
    that has been through the scorer: existence was decided per piece, and this only
    changes the grouping -- pieces of one filament joined by a faint bridge become
    one instance at their own extent. Pieces in different loose regions are untouched.
    """
    if len(insts) < 2:
        return list(insts)
    loose = probs_full > grow_thr
    if disk is not None:
        loose &= disk_mask(disk, loose.shape)
    for inst in insts:
        loose[inst.y0 : inst.y1, inst.x0 : inst.x1] |= inst.patch
    _, lab = cv2.connectedComponents(loose.astype(np.uint8), connectivity=8)
    groups: dict[int, list[Instance]] = {}
    for inst in insts:
        # Every pixel of an instance lies in one loose component (the instance is
        # connected and inside ``loose``), so one pixel identifies it.
        ys, xs = np.nonzero(inst.patch)
        comp = int(lab[inst.y0 + ys[0], inst.x0 + xs[0]])
        groups.setdefault(comp, []).append(inst)
    return [union_instances(members) for members in groups.values()]


def fill_holes(inst: Instance) -> Instance:
    """The instance with interior holes filled (annotators draw hole-free polygons)."""
    filled = ndimage.binary_fill_holes(inst.patch)
    if filled.sum() == inst.area:
        return inst
    return Instance(inst.y0, inst.x0, np.ascontiguousarray(filled))


def _paste(canvas: np.ndarray, inst: Instance, value: int) -> None:
    view = canvas[inst.y0 : inst.y1, inst.x0 : inst.x1]
    view[inst.patch] = value


def dilate_instances(insts: list[Instance], px: int, shape: tuple[int, int],
                     priority: list[float] | None = None,
                     valid: np.ndarray | None = None) -> list[Instance]:
    """Grow every instance by ``px`` pixels without creating overlaps.

    Instances are painted onto one label canvas in order of ``priority``
    (descending; area by default), each claiming only unclaimed pixels, so a pixel
    that two grown masks want goes to the higher-priority one. ``valid`` (e.g. the
    on-disk mask) bounds the growth. ``px == 0`` returns the instances unchanged.
    """
    if px <= 0:
        return list(insts)
    order = np.argsort(-np.asarray(priority if priority is not None
                                   else [i.area for i in insts], dtype=float))
    canvas = np.zeros(shape, dtype=np.int32)
    k = np.ones((2 * px + 1, 2 * px + 1), dtype=np.uint8)
    for rank, idx in enumerate(order, 1):
        inst = insts[idx]
        y0, x0 = max(inst.y0 - px, 0), max(inst.x0 - px, 0)
        y1, x1 = min(inst.y1 + px, shape[0]), min(inst.x1 + px, shape[1])
        win = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        win[inst.y0 - y0 : inst.y1 - y0, inst.x0 - x0 : inst.x1 - x0] = inst.patch
        grown = cv2.dilate(win, k) > 0
        free = canvas[y0:y1, x0:x1] == 0
        if valid is not None:
            free &= valid[y0:y1, x0:x1]
        grown &= free
        grown[inst.y0 - y0 : inst.y1 - y0, inst.x0 - x0 : inst.x1 - x0] |= inst.patch
        canvas[y0:y1, x0:x1][grown] = rank
    out: list[Instance | None] = [None] * len(insts)
    for rank, idx in enumerate(order, 1):
        inst = insts[idx]
        y0, x0 = max(inst.y0 - px, 0), max(inst.x0 - px, 0)
        y1, x1 = min(inst.y1 + px, shape[0]), min(inst.x1 + px, shape[1])
        out[idx] = from_mask(canvas[y0:y1, x0:x1] == rank, y0, x0)
    return [i for i in out if i is not None]


def erode_instances(insts: list[Instance], px: int, min_area: int = 1) -> list[Instance]:
    """Shrink every instance by ``px`` pixels; instances eroded below ``min_area`` vanish."""
    if px <= 0:
        return list(insts)
    k = np.ones((2 * px + 1, 2 * px + 1), dtype=np.uint8)
    out: list[Instance] = []
    for inst in insts:
        # Pad with background first: cv2.erode treats pixels outside the array as
        # foreground, so eroding the tight crop directly would leave the outer edge
        # of the instance untouched.
        padded = np.pad(inst.patch.astype(np.uint8), px)
        eroded = cv2.erode(padded, k) > 0
        if eroded.sum() < min_area:
            continue
        shrunk = from_mask(eroded, inst.y0 - px, inst.x0 - px)
        if shrunk is not None:
            out.append(shrunk)
    return out


def relabel_components(insts: list[Instance], min_area: int = 1) -> list[Instance]:
    """Split any instance that is no longer connected (e.g. after erosion) into pieces."""
    out: list[Instance] = []
    for inst in insts:
        pieces = label_instances(inst.patch, min_area=min_area)
        if len(pieces) == 1:
            out.append(inst)
            continue
        out.extend(Instance(inst.y0 + p.y0, inst.x0 + p.x0, p.patch) for p in pieces)
    return out
