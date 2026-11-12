"""Two-level candidate trees: the production candidates as parents, one child generator below.

The parked instance tree (filseg.tree) cut nested components at six thresholds over every
candidate and lost to the flat candidates (0.4477 vs 0.4496): the scorer could not rank
nested outlines of one object. Its selection rule (tree.select_nodes) was not the problem;
the child generator was. Here the parents are exactly the production candidates and
children exist only where the report's misses census says the candidate is wrong:

* ``watershed``: a candidate with two or more probability peaks separated by a saddle is
  split by marker-based watershed (crowding: recall 0.30 / 0.07 at 2 / 3+ neighbours).
* ``extent``: the largest candidates get their components at higher thresholds as nested
  children (merged lobes: the annotator outlined one lobe of a region drawn whole).
* ``spine``: a candidate holding two or more spine components (the spine head's cache)
  is split by nearest spine.

Every generator returns the subtree under one candidate as ``(nodes, level, parent)``:
``level`` counts from 1, ``parent`` indexes the subtree, -1 meaning the candidate itself.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as _watershed

from .metric import Instance, from_mask
from .tree import DUPLICATE_SHARE, Tree

GENERATORS = ("watershed", "extent", "spine")
MIN_CHILD = 50
WS_MIN_PARENT = 300
WS_SIGMA = 2.0
WS_MIN_DISTANCE = 15
WS_SADDLE = 0.9
EXTENT_LEVELS = (0.55, 0.70)
EXTENT_MIN_AREA = 4098      # 90th percentile of production-candidate areas, best blend, 2026-10-27
SPINE_THRESHOLD = 0.5
SPINE_MIN_PX = 10
EIGHT = np.ones((3, 3), dtype=int)

Subtree = tuple[list[Instance], list[int], list[int]]
EMPTY: Subtree = ([], [], [])


def window(arr: np.ndarray, inst: Instance) -> np.ndarray:
    """``arr`` over the instance's bounding box."""
    return arr[inst.y0:inst.y1, inst.x0:inst.x1]


def _components(mask: np.ndarray, inst: Instance, min_area: int) -> list[Instance]:
    """8-connected components of a window-sized mask, at least ``min_area`` px, in frame coordinates."""
    lab, n = ndi.label(mask, structure=EIGHT)
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() >= min_area:
            out.append(from_mask(m, inst.y0, inst.x0))
    return out


def saddle(height: np.ndarray, mask: np.ndarray, a, b, steps: int = 8) -> float:
    """The highest level at which ``a`` and ``b`` lie in one component of ``{height >= level} & mask``."""
    lo, hi = 0.0, float(min(height[a], height[b]))
    for _ in range(steps):
        t = (lo + hi) / 2
        lab, _ = ndi.label((height >= t) & mask, structure=EIGHT)
        if lab[a] > 0 and lab[a] == lab[b]:
            lo = t
        else:
            hi = t
    return lo


def peaks(prob_win: np.ndarray, mask: np.ndarray, sigma: float = WS_SIGMA,
          min_distance: int = WS_MIN_DISTANCE, saddle_ratio: float = WS_SADDLE):
    """Peaks of the smoothed map inside ``mask``, highest first, each separated from every kept
    peak by a saddle below ``saddle_ratio`` x the lower of the two. Returns ``(peaks, smoothed)``."""
    sm = prob_win.astype(np.float32)
    if sigma > 0:
        sm = cv2.GaussianBlur(sm, (0, 0), sigma)
    pts = peak_local_max(sm, min_distance=min_distance, labels=mask.astype(np.int32), exclude_border=False)
    kept: list[tuple[int, int]] = []
    for y, x in sorted(pts.tolist(), key=lambda q: -sm[q[0], q[1]]):
        p = (int(y), int(x))
        if all(saddle(sm, mask, p, k) < saddle_ratio * min(sm[p], sm[k]) for k in kept):
            kept.append(p)
    return kept, sm


def watershed_children(probs: np.ndarray, inst: Instance, min_area: int = MIN_CHILD,
                       min_parent: int = WS_MIN_PARENT, **kw) -> Subtree:
    """Marker-based watershed of a candidate with two or more separated peaks."""
    if inst.area < min_parent:
        return EMPTY
    win = window(probs, inst)
    pk, sm = peaks(win, inst.patch, **kw)
    if len(pk) < 2:
        return EMPTY
    markers = np.zeros(win.shape, np.int32)
    for i, (y, x) in enumerate(pk, 1):
        markers[y, x] = i
    lab = _watershed(-sm, markers, mask=inst.patch)
    kids = [c for c in (from_mask((lab == k) & inst.patch, inst.y0, inst.x0) for k in range(1, len(pk) + 1))
            if c is not None and c.area >= min_area]
    if len(kids) < 2:
        return EMPTY
    return kids, [1] * len(kids), [-1] * len(kids)


def extent_children(probs: np.ndarray, inst: Instance, levels=EXTENT_LEVELS, min_area: int = MIN_CHILD,
                    min_parent: int = EXTENT_MIN_AREA) -> Subtree:
    """Nested components of a large candidate above each level; a level-k component sits under
    the level-(k-1) component that contains it, level-1 components under the candidate."""
    if inst.area < min_parent:
        return EMPTY
    win = window(probs, inst)
    nodes: list[Instance] = []
    level: list[int] = []
    parent: list[int] = []
    prev: list[tuple[int, Instance]] = [(-1, inst)]
    for li, t in enumerate(levels, 1):
        cur = []
        for pidx, pnode in prev:
            m = np.zeros(win.shape, bool)
            m[pnode.y0 - inst.y0:pnode.y1 - inst.y0, pnode.x0 - inst.x0:pnode.x1 - inst.x0] = pnode.patch
            for c in _components(m & (win >= t), inst, min_area):
                nodes.append(c)
                level.append(li)
                parent.append(pidx)
                cur.append((len(nodes) - 1, c))
        if not cur:
            break
        prev = cur
    return nodes, level, parent


def spine_children(probs: np.ndarray, inst: Instance, spine: np.ndarray | None = None,
                   threshold: float = SPINE_THRESHOLD, min_px: int = SPINE_MIN_PX,
                   min_area: int = MIN_CHILD) -> Subtree:
    """One child per spine component inside the candidate, pixels assigned to the nearest spine."""
    if spine is None:
        raise ValueError("spine children need the spine map: split_tree(..., spine=<map>)")
    sw = window(spine, inst)
    lab, n = ndi.label((sw >= threshold) & inst.patch, structure=EIGHT)
    comps = [k for k in range(1, n + 1) if (lab == k).sum() >= min_px]
    if len(comps) < 2:
        return EMPTY
    dist = np.stack([ndi.distance_transform_edt(lab != k) for k in comps])
    assign = np.argmin(dist, axis=0) + 1
    assign[~inst.patch] = 0
    kids = [c for c in (from_mask(assign == i, inst.y0, inst.x0) for i in range(1, len(comps) + 1))
            if c is not None and c.area >= min_area]
    if len(kids) < 2:
        return EMPTY
    return kids, [1] * len(kids), [-1] * len(kids)


_GEN = {"watershed": watershed_children, "extent": extent_children, "spine": spine_children}


def default_levels(children: str) -> tuple[float, ...]:
    """The threshold the parents were cut at, then one entry per child level."""
    return (0.40, *EXTENT_LEVELS) if children == "extent" else (0.40, 1.0)


def split_tree(probs: np.ndarray, cands: list[Instance], children: str, levels=None, /,
               spine: np.ndarray | None = None, **params) -> Tree:
    """The candidates as roots (level 0) with one generator's subtree under each.

    ``levels`` (the tree's level labels) is positional-only, so a ``levels=`` keyword in
    ``params`` reaches the generator: the extent generator's own threshold list.
    """
    if children not in GENERATORS:
        raise ValueError(f"children must be one of {GENERATORS}, got {children!r}")
    gen = _GEN[children]
    levels = tuple(levels) if levels is not None else default_levels(children)
    if children == "spine":
        params = {**params, "spine": spine}
    nodes: list[Instance] = []
    level: list[int] = []
    parent: list[int] = []
    for inst in cands:
        root = len(nodes)
        nodes.append(inst)
        level.append(0)
        parent.append(-1)
        sub_nodes, sub_level, sub_parent = gen(probs, inst, **params)
        base = len(nodes)
        for n_, l_, p_ in zip(sub_nodes, sub_level, sub_parent):
            nodes.append(n_)
            level.append(l_)
            parent.append(root if p_ < 0 else base + p_)
    kids: list[list[int]] = [[] for _ in nodes]
    for i, p in enumerate(parent):
        if p >= 0:
            kids[p].append(i)
    duplicate = [False] * len(nodes)
    for i, ks in enumerate(kids):
        if len(ks) == 1 and nodes[ks[0]].area >= DUPLICATE_SHARE * nodes[i].area:
            duplicate[ks[0]] = True
    return Tree(nodes, level, parent, kids, duplicate, levels)
