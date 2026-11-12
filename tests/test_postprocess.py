"""Tests for ``filseg.postprocess`` on hand-built probability maps."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg import postprocess as pp  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.instances import probs_to_instances  # noqa: E402
from filseg.metric import from_mask  # noqa: E402

DISK = Disk(cx=128.0, cy=128.0, radius=120.0)


def _areas(insts):
    return sorted(i.area for i in insts)


def _gapped_bar():
    """A 6 x 100 bar at 0.9 whose middle 8 columns dip to 0.3: two seeds at 0.4, one at 0.2."""
    p = np.zeros((256, 256), dtype=np.float32)
    p[60:66, 40:140] = 0.9
    p[60:66, 86:94] = 0.3
    return p


def test_equal_thresholds_reproduce_the_shipped_chain():
    p = _gapped_bar()
    a = probs_to_instances(p, 0.4, min_area=10, disk=DISK, size=256)
    b = pp.hysteresis_instances(p, DISK, seed_thr=0.4, grow_thr=0.4, min_area=10)
    assert _areas(a) == _areas(b) == [46 * 6, 46 * 6]


def test_merge_mode_rejoins_a_bar_through_its_faint_middle():
    p = _gapped_bar()
    insts = pp.hysteresis_instances(p, DISK, seed_thr=0.4, grow_thr=0.2, min_area=10, mode=pp.MERGE)
    assert _areas(insts) == [600]                     # one instance, full 6 x 100


def test_split_mode_keeps_the_two_seeds_but_grows_them():
    p = _gapped_bar()
    insts = pp.hysteresis_instances(p, DISK, seed_thr=0.4, grow_thr=0.2, min_area=10, mode=pp.SPLIT)
    assert len(insts) == 2 and sum(i.area for i in insts) == 600   # the gap is shared out


def test_bridge_mode_rejoins_without_growing():
    p = _gapped_bar()
    insts = pp.hysteresis_instances(p, DISK, seed_thr=0.4, grow_thr=0.2, min_area=10, mode=pp.BRIDGE)
    assert len(insts) == 1
    assert insts[0].area == 2 * 46 * 6                # the two seed halves, gap left out
    assert (insts[0].y0, insts[0].x0, insts[0].y1, insts[0].x1) == (60, 40, 66, 140)
    # Two seeds NOT connected at the grow threshold stay separate.
    p[60:66, 86:94] = 0.1
    assert len(pp.hysteresis_instances(p, DISK, 0.4, 0.2, min_area=10, mode=pp.BRIDGE)) == 2


def test_bridge_instances_regroups_pieces_without_changing_pixels():
    p = _gapped_bar()
    pieces = probs_to_instances(p, 0.4, min_area=10, disk=DISK, size=256)
    assert len(pieces) == 2
    joined = pp.bridge_instances(pieces, p, grow_thr=0.2, disk=DISK)
    assert len(joined) == 1 and joined[0].area == sum(i.area for i in pieces)
    assert (joined[0].y0, joined[0].x0, joined[0].y1, joined[0].x1) == (60, 40, 66, 140)
    # Same pieces, bridge too faint: nothing changes (same objects back).
    p[60:66, 86:94] = 0.1
    assert pp.bridge_instances(pieces, p, grow_thr=0.2, disk=DISK) == pieces
    # A third instance in its own region stays separate.
    p[150:156, 40:140] = 0.9
    three = probs_to_instances(p, 0.4, min_area=10, disk=DISK, size=256)
    assert len(three) == 3
    p[60:66, 86:94] = 0.3
    out = pp.bridge_instances(three, p, grow_thr=0.2, disk=DISK)
    assert sorted(i.area for i in out) == [46 * 6 * 2, 600]
    assert pp.bridge_instances([], p, 0.2) == [] and pp.bridge_instances(three[:1], p, 0.2) == three[:1]


def test_union_instances():
    a = from_mask(np.pad(np.ones((2, 3), bool), ((5, 0), (5, 0))))     # rows 5-6, cols 5-7
    b = from_mask(np.pad(np.ones((2, 2), bool), ((9, 0), (10, 0))))    # rows 9-10, cols 10-11
    u = pp.union_instances([a, b])
    assert (u.y0, u.x0, u.y1, u.x1) == (5, 5, 11, 12) and u.area == 6 + 4
    assert pp.union_instances([a]) is a


def test_faint_region_without_a_seed_never_becomes_an_instance():
    p = np.zeros((256, 256), dtype=np.float32)
    p[60:66, 40:140] = 0.9        # seeded
    p[150:156, 40:140] = 0.3      # faint only
    insts = pp.hysteresis_instances(p, DISK, seed_thr=0.4, grow_thr=0.2, min_area=10)
    assert _areas(insts) == [600]


def test_min_area_applies_to_the_seed():
    p = np.zeros((256, 256), dtype=np.float32)
    p[60:66, 40:140] = 0.3        # big faint body ...
    p[62:64, 88:92] = 0.9         # ... with an 8 px confident core
    assert pp.hysteresis_instances(p, DISK, 0.4, 0.2, min_area=10) == []
    grown = pp.hysteresis_instances(p, DISK, 0.4, 0.2, min_area=5)
    assert _areas(grown) == [600]


def test_grow_threshold_above_seed_is_refused():
    with pytest.raises(ValueError):
        pp.hysteresis_instances(_gapped_bar(), DISK, seed_thr=0.4, grow_thr=0.5)
    with pytest.raises(ValueError):
        pp.hysteresis_instances(_gapped_bar(), DISK, 0.4, 0.2, mode="fuse")


def test_fill_holes():
    m = np.zeros((20, 20), dtype=bool)
    m[2:18, 2:18] = True
    m[8:12, 8:12] = False
    inst = from_mask(m)
    filled = pp.fill_holes(inst)
    assert filled.area == 16 * 16 and (filled.y0, filled.x0) == (2, 2)
    solid = from_mask(np.ones((5, 5), dtype=bool))
    assert pp.fill_holes(solid) is solid


def test_dilate_is_overlap_safe_and_bounded():
    m1 = np.zeros((64, 64), dtype=bool); m1[10:20, 10:30] = True    # 200 px
    m2 = np.zeros((64, 64), dtype=bool); m2[23:33, 10:30] = True    # 200 px, 3 px below
    a, b = from_mask(m1), from_mask(m2)
    grown = pp.dilate_instances([a, b], px=2, shape=(64, 64))
    assert len(grown) == 2
    canvas = np.zeros((64, 64), dtype=int)
    for i, g in enumerate(grown, 1):
        canvas[g.y0 : g.y1, g.x0 : g.x1][g.patch] += 1
    assert canvas.max() == 1                            # no pixel claimed twice
    assert all(g.area > 200 for g in grown)
    # The higher-priority (first-painted) instance takes the contested rows.
    prio = pp.dilate_instances([a, b], px=2, shape=(64, 64), priority=[1.0, 0.0])
    assert prio[0].area >= prio[1].area
    # A validity mask bounds the growth.
    valid = np.zeros((64, 64), dtype=bool); valid[:, :25] = True
    bounded = pp.dilate_instances([a], px=2, shape=(64, 64), valid=valid)
    assert bounded[0].x1 <= 30 and bounded[0].area < grown[0].area


def test_dilate_zero_is_identity_and_erode_undoes_dilate_on_a_block():
    m = np.zeros((64, 64), dtype=bool); m[10:30, 10:40] = True
    inst = from_mask(m)
    assert pp.dilate_instances([inst], 0, (64, 64)) == [inst]
    grown = pp.dilate_instances([inst], 2, (64, 64))[0]
    assert grown.area == 24 * 34
    back = pp.erode_instances([grown], 2)[0]
    assert back.area == inst.area and (back.y0, back.x0) == (inst.y0, inst.x0)


def test_erode_drops_vanished_instances_and_relabel_splits_pieces():
    thin = np.zeros((64, 64), dtype=bool); thin[10:12, 10:40] = True      # 2 px thick
    assert pp.erode_instances([from_mask(thin)], 1) == []
    dumbbell = np.zeros((64, 64), dtype=bool)
    dumbbell[10:20, 10:20] = True; dumbbell[10:20, 30:40] = True; dumbbell[14:16, 20:30] = True
    inst = from_mask(dumbbell)
    eroded = pp.erode_instances([inst], 1)
    pieces = pp.relabel_components(eroded)
    assert len(pieces) == 2 and all(p.area == 8 * 8 for p in pieces)
