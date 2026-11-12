"""Tests for ``filseg.scorer``: candidate extraction, features and labels.

Synthetic maps with known geometry, so every feature can be checked against a
number worked out by hand rather than against a snapshot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg import scorer  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.metric import from_mask  # noqa: E402

DISK = Disk(cx=128.0, cy=128.0, radius=120.0)


def _map(size=256):
    """A confident horizontal bar, a fainter vertical bar, and a tiny blob."""
    p = np.zeros((size, size), dtype=np.float32)
    p[60:66, 40:140] = 0.9      # 6 x 100 = 600 px, confident
    p[120:160, 200:206] = 0.5   # 40 x 6 = 240 px, fainter
    p[200:204, 100:104] = 0.95  # 16 px blob
    return p


def test_candidates_match_the_shipped_chain():
    p = _map()
    c = scorer.candidates(p, DISK, threshold=0.4, min_area=100)
    areas = sorted(i.area for i in c)
    assert areas == [240, 600]                       # the 16 px blob is under min_area
    c50 = scorer.candidates(p, DISK, threshold=0.4, min_area=10)
    assert sorted(i.area for i in c50) == [16, 240, 600]
    assert scorer.candidates(p, DISK, threshold=0.7, min_area=10) and \
        sorted(i.area for i in scorer.candidates(p, DISK, threshold=0.7, min_area=10)) == [16, 600]


def test_upsample_is_identity_at_full_size_and_bilinear_otherwise():
    p = _map(256)
    assert scorer.upsample(p, 256) is p
    up = scorer.upsample(p, 512)
    assert up.shape == (512, 512) and up.dtype == np.float32
    assert up.max() <= 0.95 + 1e-6


def test_helio_centre_and_limb():
    r, lat, lon = scorer.helio(DISK.cx, DISK.cy, DISK)
    assert (r, lat, lon) == (0.0, 0.0, 0.0)
    r, lat, lon = scorer.helio(DISK.cx + DISK.radius, DISK.cy, DISK)   # west limb
    assert r == pytest.approx(1.0) and lat == pytest.approx(0.0) and lon == pytest.approx(90.0)
    r, lat, lon = scorer.helio(DISK.cx, DISK.cy - 0.5 * DISK.radius, DISK)  # north
    assert lat == pytest.approx(30.0) and lon == pytest.approx(0.0)


def test_instance_features_of_a_clean_bar():
    p = _map()
    bar = [i for i in scorer.candidates(p, DISK, 0.4, 100) if i.area == 600][0]
    f = scorer.instance_features(bar, p, DISK)
    assert f["area"] == 600 and f["log_area"] == pytest.approx(np.log(600))
    assert f["p_mean"] == pytest.approx(0.9) and f["p_max"] == pytest.approx(0.9)
    assert f["p_std"] == pytest.approx(0.0) and f["core_frac"] == 1.0
    assert f["ring_p_mean"] == 0.0 and f["p_contrast"] == pytest.approx(0.9)
    assert (f["bbox_h"], f["bbox_w"]) == (6, 100) and f["bbox_fill"] == 1.0
    assert 90 <= f["skel_len"] <= 100 and 5.5 <= f["width_mean"] <= 7.0
    assert f["n_endpoints"] == 2                     # a straight line has two ends
    assert f["solidity"] == pytest.approx(1.0, abs=0.05)
    # centroid at (x=89.5, y=62.5): west of centre, north of centre
    assert f["r_disk"] == pytest.approx(np.hypot(89.5 - 128, 62.5 - 128) / 120, abs=1e-6)
    assert f["abs_lat"] > 0 and f["abs_lon"] > 0 and f["limb_dist"] == pytest.approx(1 - f["r_disk"])
    assert np.isnan(f["flat_mean"])                  # no image given


def test_image_features_read_the_flat_field():
    p = _map()
    bar = [i for i in scorer.candidates(p, DISK, 0.4, 100) if i.area == 600][0]
    flat = np.ones_like(p)
    flat[60:66, 40:140] = 0.92                       # 8% darker than the quiet Sun around it
    f = scorer.instance_features(bar, p, DISK, flat)
    assert f["flat_mean"] == pytest.approx(0.92) and f["flat_min"] == pytest.approx(0.92)
    assert f["flat_ring"] == pytest.approx(1.0) and f["flat_contrast"] == pytest.approx(0.08)


def test_frame_features_rank_and_count_neighbours():
    p = _map()
    insts = scorer.candidates(p, DISK, 0.4, 10)     # three candidates
    feats = scorer.frame_features(insts, p, DISK)
    assert len(feats) == 3 and all(set(scorer.FEATURE_NAMES) <= set(f) for f in feats)
    by_area = {int(f["area"]): f for f in feats}
    assert by_area[600]["rank_area"] == 0 and by_area[16]["rank_area"] == 2
    assert by_area[16]["rank_p"] == 0                # 0.95 blob is the most confident
    assert by_area[240]["rank_p"] == 2 and by_area[240]["frac_rank_p"] == pytest.approx(2 / 3)
    assert all(f["n_cand"] == 3 for f in feats)
    # bar centroid (89.5, 62.5), blob (101.5, 201.5), vertical bar (202.5, 139.5): none within 100 px
    assert all(f["n_nb"] == 0 for f in feats)
    assert by_area[600]["nn_dist"] == pytest.approx(np.hypot(202.5 - 89.5, 139.5 - 62.5), abs=1e-3)
    assert by_area[600]["frame_area_sum"] == 856


def test_frame_features_on_a_single_candidate():
    p = np.zeros((256, 256), dtype=np.float32)
    p[60:66, 40:140] = 0.9
    feats = scorer.frame_features(scorer.candidates(p, DISK, 0.4, 10), p, DISK)
    assert len(feats) == 1
    assert feats[0]["n_nb"] == 0 and feats[0]["nn_dist"] == 4096.0 and feats[0]["frac_rank_p"] == 0.0


def test_small_neighbours_are_counted_within_the_radius():
    p = np.zeros((256, 256), dtype=np.float32)
    p[100:110, 20:120] = 0.9                         # 1000 px bar (not "small"), centroid (69.5, 104.5)
    p[120:130, 60:80] = 0.6                          # 200 px blob 20 px below it
    p[230:234, 128:132] = 0.6                        # 16 px blob far away (r = 0.85, on-disk)
    insts = scorer.candidates(p, DISK, 0.4, 10)
    assert len(insts) == 3
    feats = {int(f["area"]): f for f in scorer.frame_features(insts, p, DISK)}
    assert feats[1000]["n_nb"] == 1 and feats[1000]["n_small_nb"] == 1
    assert feats[200]["n_nb"] == 1 and feats[200]["n_small_nb"] == 0   # its neighbour is big
    assert feats[16]["n_nb"] == 0 and feats[16]["n_small_nb"] == 0


def test_labels_against_a_pass():
    p = _map()
    insts = scorer.candidates(p, DISK, 0.4, 10)     # areas 16, 240, 600 in some order
    # Two GT instances: the 0.9 bar and the 0.95 blob; the 0.5 bar is unannotated.
    g1 = np.zeros_like(p, dtype=bool); g1[60:66, 40:140] = True
    g2 = np.zeros_like(p, dtype=bool); g2[200:204, 100:104] = True
    gts = [from_mask(g1), from_mask(g2)]
    lab = scorer.label_against_pass(insts, gts)
    assert lab.n_gt == 2
    by_area = {i.area: k for k, i in enumerate(insts)}
    assert lab.matched_gt[by_area[600]] == 0 and lab.best_iou[by_area[600]] == pytest.approx(1.0)
    assert lab.matched_gt[by_area[16]] == 1 and lab.best_iou[by_area[16]] == pytest.approx(1.0)
    assert lab.matched_gt[by_area[240]] == -1 and lab.best_iou[by_area[240]] == 0.0
    empty = scorer.label_against_pass(insts, [])
    assert empty.n_gt == 0 and empty.matched_gt == [-1, -1, -1] and empty.best_iou == [0.0, 0.0, 0.0]


# --- cross-model agreement -------------------------------------------------

def _bar_and_members():
    p = _map()
    insts = scorer.candidates(p, DISK, threshold=0.4, min_area=100)
    bar = next(i for i in insts if i.area == 600)          # rows 60:66, cols 40:140 at 0.9
    full = _map()                                          # agrees everywhere
    half = np.zeros_like(p)
    half[60:66, 40:90] = 0.9                               # covers exactly half of the bar
    none = np.zeros_like(p)
    return bar, [full, half, none]


def test_agreement_features_aggregate_member_support():
    bar, members = _bar_and_members()
    f = scorer.agreement_features(bar, members)
    assert set(f) == set(scorer.AGREEMENT_NAMES)
    assert f["agree_max"] == pytest.approx(0.9)
    assert f["agree_min"] == 0.0
    assert f["agree_mean"] == pytest.approx((0.9 + 0.45 + 0.0) / 3)
    assert f["n_support"] == 2            # full and half (50 % counts) support it; none does not


def test_agreement_feature_names_do_not_depend_on_member_count():
    bar, members = _bar_and_members()
    two = scorer.agreement_features(bar, members[:2])
    five = scorer.agreement_features(bar, members + members[:2])
    assert list(two) == list(five) == list(scorer.AGREEMENT_NAMES)
    nan = scorer.agreement_features(bar, [])
    assert all(np.isnan(v) for v in nan.values())


def test_frame_features_without_members_is_unchanged():
    p = _map()
    insts = scorer.candidates(p, DISK, threshold=0.4, min_area=100)
    plain = scorer.frame_features(insts, p, DISK)
    assert not any(k in plain[0] for k in scorer.AGREEMENT_NAMES)
    with_members = scorer.frame_features(insts, p, DISK, member_maps=[p, np.zeros_like(p)])
    assert with_members[0]["n_support"] == 1.0
    assert {k: v for k, v in with_members[0].items() if k in scorer.FEATURE_NAMES} == \
        {k: v for k, v in plain[0].items() if k in scorer.FEATURE_NAMES}
