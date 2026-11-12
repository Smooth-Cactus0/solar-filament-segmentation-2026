"""filseg.split: watershed, extent and spine children under production candidates (synthetic maps)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from filseg import split  # noqa: E402
from filseg.metric import from_mask  # noqa: E402
from test_tree import two_blobs  # noqa: E402


def full(inst, shape):
    m = np.zeros(shape, bool)
    m[inst.y0:inst.y1, inst.x0:inst.x1] = inst.patch
    return m


def test_saddle_between_two_blobs_is_the_bridge_level():
    p = two_blobs()
    mask = p > 0.25
    s = split.saddle(p, mask, (50, 50), (50, 150))
    assert abs(s - 0.3) < 0.01


def test_watershed_splits_two_peaks_and_leaves_a_ridge_whole():
    p = two_blobs()
    parent = from_mask(p > 0.25)                     # both blobs and the bridge: one candidate
    nodes, level, par = split.watershed_children(p, parent, min_area=10, min_parent=100, min_distance=5)
    assert len(nodes) == 2 and level == [1, 1] and par == [-1, -1]
    a, b = full(nodes[0], p.shape), full(nodes[1], p.shape)
    assert not (a & b).any() and not ((a | b) & ~full(parent, p.shape)).any()
    assert a[50, 50] != b[50, 50] and a[50, 150] != b[50, 150]      # one peak each
    ridge = np.zeros((60, 160), np.float32)
    ridge[20:40, 10:150] = 0.8                        # one flat plateau: no second lobe
    assert split.watershed_children(ridge, from_mask(ridge > 0.4), min_area=10, min_parent=100) == split.EMPTY
    small = from_mask(ridge > 0.4)
    assert split.watershed_children(ridge, small, min_area=10, min_parent=10_000) == split.EMPTY


def test_extent_children_are_nested_components_above_each_level():
    p = np.zeros((60, 100), np.float32)
    p[10:50, 10:90] = 0.45
    p[20:40, 20:50] = 0.8                            # one core inside the rim
    parent = from_mask(p > 0.4)
    nodes, level, par = split.extent_children(p, parent, levels=(0.55, 0.70), min_area=10, min_parent=100)
    assert level == [1, 2] and par == [-1, 0]        # the core at 0.55, the same core again at 0.70
    assert nodes[0].area == nodes[1].area == 600
    assert split.extent_children(p, parent, min_parent=10_000) == split.EMPTY


def test_spine_children_partition_the_parent_by_nearest_spine():
    p = np.zeros((40, 100), np.float32)
    p[10:30, 5:95] = 0.8
    parent = from_mask(p > 0.4)
    spine = np.zeros((40, 100), np.float32)
    spine[20, 10:40] = 1.0
    spine[20, 60:90] = 1.0
    nodes, level, par = split.spine_children(p, parent, spine=spine, min_px=5, min_area=10)
    assert len(nodes) == 2 and level == [1, 1]
    a, b = full(nodes[0], p.shape), full(nodes[1], p.shape)
    assert (a | b).sum() == parent.area and not (a & b).any()
    assert a[20, 20] and b[20, 80]
    one = np.zeros_like(spine)
    one[20, 10:90] = 1.0
    assert split.spine_children(p, parent, spine=one, min_px=5) == split.EMPTY


def test_split_tree_has_parents_at_level_zero_and_marks_duplicates():
    p = two_blobs()
    cands = [from_mask(p > 0.25)]
    t = split.split_tree(p, cands, "watershed", min_area=10, min_parent=100, min_distance=5)
    assert len(t) == 3 and t.level == [0, 1, 1] and t.parent == [-1, 0, 0]
    assert t.children == [[1, 2], [], []] and t.duplicate == [False] * 3 and t.levels == (0.40, 1.0)
    q = np.zeros((60, 100), np.float32)
    q[10:50, 10:90] = 0.45
    q[20:40, 20:50] = 0.8
    t2 = split.split_tree(q, [from_mask(q > 0.4)], "extent", min_area=10, min_parent=100)
    assert t2.level == [0, 1, 2] and t2.duplicate == [False, False, True] and t2.levels == (0.40, 0.55, 0.70)
    try:
        split.split_tree(q, [], "nope")
    except ValueError as e:
        assert "children" in str(e)
    else:
        raise AssertionError("unknown generator accepted")


def test_split_tree_passes_the_extent_generator_its_own_levels():
    q = np.zeros((60, 100), np.float32)
    q[10:50, 10:90] = 0.45
    q[20:40, 20:50] = 0.8
    t = split.split_tree(q, [from_mask(q > 0.4)], "extent", None, levels=[0.55], min_area=10, min_parent=100)
    assert t.level == [0, 1] and t.duplicate == [False, False]
    assert t.levels == (0.40, 0.55, 0.70)            # the tree's own levels are the positional argument
