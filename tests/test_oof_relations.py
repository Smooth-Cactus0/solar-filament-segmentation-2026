"""Tests for ``filseg.oof.classify`` on hand-built masks.

Each case is the smallest picture of one failure mode, so a wrong verdict points
straight at the rule that produced it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg import oof  # noqa: E402
from filseg.metric import Instance, from_mask  # noqa: E402


def bar(y0, x0, h, w, size=128) -> Instance:
    m = np.zeros((size, size), dtype=bool)
    m[y0 : y0 + h, x0 : x0 + w] = True
    return from_mask(m)


def test_exact_match_is_matched_on_both_sides():
    g = bar(10, 10, 6, 60)
    r = oof.classify([g], [g])
    assert r.gt_kind == [oof.GT_MATCHED] and r.pred_kind == [oof.PRED_MATCHED]
    assert r.matched_ious == [1.0]
    assert r.one_to_many == 0 and r.many_to_one == 0


def test_a_filament_split_in_two_is_fragmented_and_its_pieces_are_fragments():
    g = bar(10, 10, 6, 60)
    left, right = bar(10, 10, 6, 28), bar(10, 42, 6, 28)   # 4 px gap in the middle
    r = oof.classify([g], [left, right])
    assert r.gt_kind == [oof.GT_FRAGMENTED]
    assert r.pred_kind == [oof.PRED_FRAGMENT, oof.PRED_FRAGMENT]
    assert r.gt_fragments == [2] and r.one_to_many == 1
    assert r.matched_ious == []
    # 56 of the 60 columns are covered once the pieces are united; each piece alone is 28/60.
    assert r.gt_union_iou == pytest.approx([56 / 60])
    assert r.gt_best_iou == pytest.approx([28 / 60]) and r.gt_best_cover == pytest.approx([28 / 60])


def test_two_filaments_fused_into_one_prediction_are_merged():
    g1, g2 = bar(10, 10, 6, 30), bar(10, 44, 6, 30)        # collinear neighbours
    fused = bar(10, 10, 6, 64)                              # one prediction spanning both
    r = oof.classify([g1, g2], [fused])
    assert r.gt_kind == [oof.GT_MERGED, oof.GT_MERGED]
    assert r.pred_kind == [oof.PRED_MERGE]
    assert r.pred_filaments == [2] and r.many_to_one == 1


def test_a_big_fused_prediction_still_matching_one_filament_is_matched():
    # IoU with the big filament stays above the gate, so the metric counts a TP;
    # the small neighbour it swallowed is what gets reported as merged.
    big, small = bar(10, 10, 6, 100), bar(10, 112, 6, 8)
    fused = bar(10, 10, 6, 110)
    r = oof.classify([big, small], [fused])
    assert r.gt_kind == [oof.GT_MATCHED, oof.GT_MERGED]
    assert r.pred_kind == [oof.PRED_MATCHED]
    assert r.many_to_one == 1


def test_an_undersized_prediction_is_partial_on_both_sides():
    g = bar(10, 10, 6, 60)
    stub = bar(10, 10, 6, 15)                               # covers a quarter of it
    r = oof.classify([g], [stub])
    assert r.gt_kind == [oof.GT_PARTIAL] and r.pred_kind == [oof.PRED_PARTIAL]


def test_grazing_contact_does_not_count_as_touching():
    g = bar(10, 10, 6, 60)
    graze = bar(15, 60, 6, 40)                              # overlaps 1 row x 10 px = 10 of 360 px
    r = oof.classify([g], [graze])
    assert r.gt_kind == [oof.GT_MISSED] and r.pred_kind == [oof.PRED_SPURIOUS]
    assert r.gt_fragments == [0] and r.pred_filaments == [0]


def test_nothing_predicted_and_nothing_annotated():
    g = bar(10, 10, 6, 60)
    r = oof.classify([g], [])
    assert r.gt_kind == [oof.GT_MISSED] and r.pred_kind == []
    r = oof.classify([], [g])
    assert r.gt_kind == [] and r.pred_kind == [oof.PRED_SPURIOUS]
    r = oof.classify([], [])
    assert r.gt_kind == [] and r.pred_kind == [] and r.matched_ious == []


@pytest.mark.parametrize("touch", [0.05, 0.10, 0.25])
def test_touch_threshold_is_respected(touch):
    g = bar(10, 10, 6, 100)                                 # 600 px
    piece = bar(10, 10, 6, 15)                              # covers 15% of it
    r = oof.classify([g], [piece], touch=touch)
    expect_touch = 0.15 >= touch
    assert (r.gt_fragments == [1]) is expect_touch
    assert (r.pred_kind == [oof.PRED_PARTIAL]) is expect_touch
    assert (r.pred_kind == [oof.PRED_SPURIOUS]) is (not expect_touch)


def test_overlap_tables_shapes_and_values():
    g = bar(0, 0, 10, 10)
    p_same, p_half, p_far = bar(0, 0, 10, 10), bar(0, 0, 10, 5), bar(50, 50, 10, 10)
    inter, iou, cover = oof.overlap_tables([g], [p_same, p_half, p_far])
    assert inter.shape == iou.shape == cover.shape == (1, 3)
    assert inter.tolist() == [[100, 50, 0]]
    assert iou[0].tolist() == pytest.approx([1.0, 0.5, 0.0])
    assert cover[0].tolist() == pytest.approx([1.0, 0.5, 0.0])
