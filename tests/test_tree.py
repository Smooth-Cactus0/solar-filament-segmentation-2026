"""Tests for filseg.tree: the candidate tree, its features and node selection (synthetic maps)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.data import Disk  # noqa: E402
from filseg.tree import build_tree  # noqa: E402

BIG_DISK = Disk(cx=100.0, cy=60.0, radius=1000.0)     # everything is on the disk
LEVELS3 = (0.25, 0.40, 0.60)


def two_blobs() -> np.ndarray:
    """Blobs A and B (rim 0.45, core 0.8) joined by a faint 0.3 bridge."""
    p = np.zeros((120, 200), np.float32)
    p[40:60, 20:80] = 0.45
    p[45:55, 30:70] = 0.8
    p[40:60, 120:180] = 0.45
    p[45:55, 130:170] = 0.8
    p[48:52, 80:120] = 0.3
    return p


def full(inst, shape=(120, 200)) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[inst.y0:inst.y1, inst.x0:inst.x1] = inst.patch
    return m


def test_levels_nest_into_a_tree():
    t = build_tree(two_blobs(), BIG_DISK, LEVELS3, min_area=10)
    assert len(t) == 5 and t.levels == LEVELS3
    assert t.level == [0, 1, 1, 2, 2]
    assert t.parent == [-1, 0, 0, 1, 2]
    assert t.children == [[1, 2], [3], [4], [], []]
    assert t.roots() == [0]
    assert [n.area for n in t.nodes] == [2560, 1200, 1200, 400, 400]
    assert t.duplicate == [False] * 5
    for i, p in enumerate(t.parent):
        if p >= 0:
            assert not (full(t.nodes[i]) & ~full(t.nodes[p])).any()   # child inside parent


def test_near_identical_child_is_a_duplicate():
    p = np.zeros((60, 60), np.float32)
    p[10:30, 10:40] = 0.8                     # the same blob at every level
    t = build_tree(p, BIG_DISK, LEVELS3, min_area=10)
    assert len(t) == 3 and t.parent == [-1, 0, 1]
    assert t.duplicate == [False, True, True]


def test_min_area_clip_and_max_area():
    p = two_blobs()
    p[5:8, 5:8] = 0.9                          # 9 px speck
    assert all(n.area >= 10 for n in build_tree(p, BIG_DISK, LEVELS3, min_area=10).nodes)
    small = Disk(cx=50.0, cy=50.0, radius=60.0)   # blob B lies outside the disk
    t = build_tree(two_blobs(), small, LEVELS3, min_area=10)
    assert all(n.x1 <= 110 for n in t.nodes)
    t_big = build_tree(two_blobs(), BIG_DISK, LEVELS3, min_area=10, max_area=2000)
    assert [n.area for n in t_big.nodes] == [1200, 1200, 400, 400]   # the 2560 px root is dropped
    assert t_big.roots() == [0, 1]


def test_clip_one_keeps_the_last_two_percent_of_the_radius():
    p = np.zeros((200, 200), np.float32)
    p[97:103, 186:199] = 0.9                   # straddles 0.98 R of a radius-98 disk centred at x=100
    disk = Disk(cx=100.0, cy=100.0, radius=98.0)
    a98 = sum(n.area for n in build_tree(p, disk, (0.5,), min_area=5, clip=0.98).nodes)
    a100 = sum(n.area for n in build_tree(p, disk, (0.5,), min_area=5, clip=1.00).nodes)
    assert a100 > a98


def test_empty_map_gives_an_empty_tree():
    t = build_tree(np.zeros((50, 50), np.float32), BIG_DISK, LEVELS3, min_area=10)
    assert len(t) == 0 and t.roots() == []


from filseg.tree import (  # noqa: E402
    TREE_FEATURES, V3_FEATURES, node_features, select_nodes, tree_features,
)


def test_tree_features_on_two_blobs():
    p = two_blobs()
    t = build_tree(p, BIG_DISK, LEVELS3, min_area=10)
    f = tree_features(t, p)
    assert set(f[0]) == set(TREE_FEATURES)
    assert f[0]["level"] == 0.25 and f[3]["level"] == 0.60
    assert [x["depth"] for x in f] == [0, 1, 1, 2, 2]
    assert f[1]["parent_area_ratio"] == pytest.approx(1200 / 2560)
    assert f[0]["n_children"] == 2 and f[3]["n_children"] == 0
    assert f[0]["child_cover"] == pytest.approx(2400 / 2560)
    assert f[1]["stability"] == 0                              # grows 2.1x down, shrinks 3x up
    assert f[1]["fringe_p_mean"] == pytest.approx(0.45)        # rim pixels, at or below 0.60
    assert f[0]["fringe_p_mean"] == pytest.approx(0.30)        # the bridge, at or below 0.40
    assert np.isnan(f[3]["fringe_p_mean"]) and np.isnan(f[3]["fringe_flat_mean"])  # top level
    assert np.isnan(f[1]["fringe_flat_mean"])                  # no flat image given


def test_stability_counts_levels_of_unchanged_area():
    p = np.zeros((60, 60), np.float32)
    p[10:30, 10:40] = 0.8
    t = build_tree(p, BIG_DISK, LEVELS3, min_area=10)
    assert [x["stability"] for x in tree_features(t, p)] == [2, 2, 2]


def test_node_features_cover_v3_and_rank_within_a_level():
    p = two_blobs()
    t = build_tree(p, BIG_DISK, LEVELS3, min_area=10)
    flat = np.where(p > 0.25, 0.8, 1.0).astype(np.float32)
    f = node_features(t, p, BIG_DISK, flat)
    assert len(f) == 5 and all(set(V3_FEATURES) <= set(x) for x in f)
    assert f[0]["n_cand"] == 1 and f[1]["n_cand"] == 2 and f[3]["n_cand"] == 2
    assert f[1]["fringe_flat_mean"] == pytest.approx(0.8)


PARENT, LEVEL = [-1, 0, 0, 1, 2], [0, 1, 1, 2, 2]


@pytest.mark.parametrize("gain, kept", [
    ([0.9, 0.3, 0.3, 0.1, 0.1], [0]),          # the parent beats its children
    ([0.5, 0.3, 0.4, 0.1, 0.1], [1, 2]),       # splitting pays
    ([-1, -1, 0.2, 0.5, -0.1], [2, 3]),        # mixed depths
    ([-1, -1, -1, -1, -1], []),                # nothing pays
])
def test_select_nodes_picks_the_best_antichain(gain, kept):
    assert select_nodes(PARENT, LEVEL, [False] * 5, np.array(gain, float)) == kept


def test_duplicates_are_never_kept_themselves():
    dup = [False, True, False, False, False]
    gain = np.array([-1, 5.0, -1, 0.2, -1])
    assert select_nodes(PARENT, LEVEL, dup, gain) == [3]


def test_selected_nodes_never_overlap():
    p = two_blobs()
    t = build_tree(p, BIG_DISK, LEVELS3, min_area=10)
    rng = np.random.default_rng(0)
    for _ in range(20):
        kept = t.select(rng.normal(size=len(t)))
        cover = sum(full(t.nodes[i]).astype(int) for i in kept) if kept else np.zeros(1)
        assert cover.max() <= 1


def test_select_nodes_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="length"):
        select_nodes(PARENT, LEVEL, [False] * 5, np.zeros(4))
    with pytest.raises(ValueError, match="length"):
        select_nodes(PARENT, LEVEL, [False] * 4, np.zeros(5))
    with pytest.raises(ValueError, match="length"):
        select_nodes(PARENT, LEVEL[:-1], [False] * 5, np.zeros(5))
