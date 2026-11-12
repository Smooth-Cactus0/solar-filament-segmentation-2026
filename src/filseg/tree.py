"""The candidate tree: nested components of the probability map at several thresholds.

A single threshold decides what one filament is, and the missed-filament census
(scripts/misses_gallery.py) showed that is where most recoverable misses are lost:
pieces too small at 0.40 (partial), a neighbour swallowed (merged), a filament broken
in two (fragmented), an outline just under the IoU gate. Cutting the same map at
several thresholds gives every one of those a better alternative: thresholds are
monotone, so each component at a higher threshold sits inside exactly one component
at the next lower one, and the components form a tree. Choosing a parent grows or
joins; choosing its children splits.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import scorer as fscorer
from .data import Disk
from .instances import MAX_AREA, disk_mask
from .metric import Instance

LEVELS: tuple[float, ...] = (0.25, 0.30, 0.35, 0.40, 0.50, 0.60)
MIN_AREA = 50
DUPLICATE_SHARE = 0.95


def settings_path(table) -> "Path":
    """Sidecar beside a tree table holding the cut settings: ``x.csv.gz`` -> ``x.tree.json``."""
    t = Path(table)
    name = t.name[:-len(".csv.gz")] if t.name.endswith(".csv.gz") else t.stem
    return t.with_name(name + ".tree.json")


@dataclass
class Tree:
    """Nodes of one exposure's candidate tree, lowest threshold first."""

    nodes: list[Instance]
    level: list[int]           # index into ``levels``
    parent: list[int]          # -1 for roots
    children: list[list[int]]
    duplicate: list[bool]      # same hypothesis as its parent; never kept itself
    levels: tuple[float, ...]

    def __len__(self) -> int:
        return len(self.nodes)

    def roots(self) -> list[int]:
        return [i for i, p in enumerate(self.parent) if p < 0]

    def select(self, gain) -> list[int]:
        """Kept node indices for per-node ``gain`` (see :func:`select_nodes`)."""
        return select_nodes(self.parent, self.level, self.duplicate, gain)


def build_tree(probs: np.ndarray, disk: Disk, levels=LEVELS, min_area: int = MIN_AREA,
               clip: float = 0.98, max_area: int = MAX_AREA) -> Tree:
    """Components of ``probs`` above each level, inside ``clip`` x the disk radius, as a tree.

    8-connected, as ``filseg.instances.label_instances``. Components smaller than
    ``min_area`` or larger than ``max_area`` are not nodes; a node whose lower-level
    container was dropped for size becomes a root.
    """
    levels = tuple(sorted(levels))
    on = disk_mask(disk, probs.shape, frac=clip)
    nodes: list[Instance] = []
    level: list[int] = []
    parent: list[int] = []
    prev_lab = prev_ids = None
    for li, t in enumerate(levels):
        fg = ((probs > t) & on).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        ids = np.full(n, -1, dtype=np.int64)
        for k in range(1, n):
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            x, y = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
            w, h = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
            patch = np.ascontiguousarray(lab[y:y + h, x:x + w] == k)
            par = -1
            if prev_lab is not None:
                ys, xs = np.nonzero(patch)
                par = int(prev_ids[prev_lab[y + ys[0], x + xs[0]]])
            ids[k] = len(nodes)
            nodes.append(Instance(y, x, patch))
            level.append(li)
            parent.append(par)
        prev_lab, prev_ids = lab, ids

    children: list[list[int]] = [[] for _ in nodes]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    duplicate = [False] * len(nodes)
    for i, kids in enumerate(children):
        if len(kids) == 1 and nodes[kids[0]].area >= DUPLICATE_SHARE * nodes[i].area:
            duplicate[kids[0]] = True
    return Tree(nodes, level, parent, children, duplicate, levels)


STABLE_CHANGE = 0.15
TREE_FEATURES: tuple[str, ...] = (
    "level", "depth", "parent_area_ratio", "n_children", "child_cover", "stability",
    "fringe_p_mean", "fringe_flat_mean",
)
V3_FEATURES: tuple[str, ...] = fscorer.FEATURE_NAMES + TREE_FEATURES


def _stability(tree: Tree, i: int, area: np.ndarray) -> int:
    """Levels over which the node's lineage keeps its area within +/-15 % per level.

    Downward through parents while a parent is at most 15 % larger; upward through the
    largest child while it is at most 15 % smaller. A shape that persists across
    thresholds is usually a real filament edge (the idea behind MSER).
    """
    n, cur = 0, i
    while tree.parent[cur] >= 0 and area[tree.parent[cur]] <= (1 + STABLE_CHANGE) * area[cur]:
        n, cur = n + 1, tree.parent[cur]
    cur = i
    while tree.children[cur]:
        k = max(tree.children[cur], key=lambda c: area[c])
        if area[k] < area[cur] / (1 + STABLE_CHANGE):
            break
        n, cur = n + 1, k
    return n


def tree_features(tree: Tree, probs: np.ndarray, flat: np.ndarray | None = None) -> list[dict[str, float]]:
    """Position of each node in its tree, and what it adds beyond the next level."""
    area = np.array([n.area for n in tree.nodes], dtype=np.float64)
    out = []
    for i, node in enumerate(tree.nodes):
        par, kids, li = tree.parent[i], tree.children[i], tree.level[i]
        depth, j = 0, par
        while j >= 0:
            depth, j = depth + 1, tree.parent[j]
        fringe_p = fringe_flat = float("nan")
        if li + 1 < len(tree.levels):
            pv = probs[node.y0:node.y1, node.x0:node.x1]
            fr = node.patch & (pv <= tree.levels[li + 1])
            if fr.any():
                fringe_p = float(pv[fr].mean())
                if flat is not None:
                    fringe_flat = float(flat[node.y0:node.y1, node.x0:node.x1][fr].mean())
        out.append({
            "level": float(tree.levels[li]),
            "depth": float(depth),
            "parent_area_ratio": float(area[i] / area[par]) if par >= 0 else 1.0,
            "n_children": float(len(kids)),
            "child_cover": float(sum(area[k] for k in kids) / area[i]),
            "stability": float(_stability(tree, i, area)),
            "fringe_p_mean": fringe_p,
            "fringe_flat_mean": fringe_flat,
        })
    return out


def node_features(tree: Tree, probs: np.ndarray, disk: Disk,
                  flat: np.ndarray | None = None) -> list[dict[str, float]]:
    """Scorer v3 features of every node: v2's features plus the tree features.

    v2's frame-context features (rank, crowding) are computed among the nodes of the
    same level, so each level sees the frame the way v2 saw its single threshold.
    """
    base = [fscorer.instance_features(n, probs, disk, flat) for n in tree.nodes]
    for li in range(len(tree.levels)):
        fscorer.add_frame_features([base[i] for i in range(len(tree)) if tree.level[i] == li])
    return [{**b, **t} for b, t in zip(base, tree_features(tree, probs, flat))]


def select_nodes(parent, level, duplicate, gain) -> list[int]:
    """Kept nodes: the set with no node inside another that maximises the summed gain.

    ``best(n) = max(gain_n, sum of best(children of n), 0)``, computed from the highest
    level down to the roots; a duplicate has no gain of its own. Ties prefer the
    parent (fewer instances). Nodes on different root-to-leaf paths never overlap, so
    the result is pixel-disjoint.
    """
    n = len(parent)
    gain = np.asarray(gain, dtype=np.float64)
    if not (len(gain) == len(level) == len(duplicate) == n):
        raise ValueError(f"select_nodes: length mismatch (parent {n}, level {len(level)}, "
                         f"duplicate {len(duplicate)}, gain {len(gain)})")
    children: list[list[int]] = [[] for _ in range(n)]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    best = np.zeros(n)
    take = [None] * n
    for i in sorted(range(n), key=lambda k: -level[k]):
        kid_sum = float(sum(best[k] for k in children[i]))
        own = -np.inf if (duplicate[i] or not np.isfinite(gain[i])) else float(gain[i])
        if own > 0 and own >= kid_sum:
            best[i], take[i] = own, "self"
        elif kid_sum > 0:
            best[i], take[i] = kid_sum, "kids"
    kept, stack = [], [i for i in range(n) if parent[i] < 0]
    while stack:
        i = stack.pop()
        if take[i] == "self":
            kept.append(i)
        elif take[i] == "kids":
            stack.extend(children[i])
    return sorted(kept)
