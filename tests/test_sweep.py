"""Tests for the operating-point sweep.

The sweep re-thresholds one set of probability maps at many operating points.
The property that matters is EQUIVALENCE: a CSV emitted by the sweep at some
(threshold, min_area) must match what predict.py would have produced at that same
setting. If that fails, every leaderboard number the sweep produces is measuring
something other than the setting we think it is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from filseg.data import Disk  # noqa: E402
from filseg.instances import mean_prob, probs_to_instances  # noqa: E402
from filseg.metric import Instance  # noqa: E402


def test_sweep_module_imports():
    import sweep_submissions  # noqa: F401


# (y0, y1, x0, x1), peak probability. The faint blob sits between 0.3 and 0.5 so
# that it is present at one tested threshold and gone at the next.
_BLOBS = [
    ((40, 70, 40, 120), 0.95),    # big, confident
    ((150, 170, 60, 130), 0.75),  # medium
    ((200, 210, 30, 60), 0.35),   # faint: present at thr 0.3, gone at thr 0.5
]
_BLOB_RAMP = 0.01  # each blob spans [peak - _BLOB_RAMP, peak]


def _draw_blobs(blobs, size):
    """Paint rectangular blobs, each carrying a shallow diagonal ramp up to its peak.

    The ramp matters: real probability maps are continuous, but a constant-valued
    blob puts every high quantile exactly on the map's maximum, so the last-resort
    arm's ``probs > cut`` is empty by construction and cannot be exercised at all.
    The ramp is diagonal rather than axis-aligned so its level sets are not whole
    rows or columns, which would reintroduce the same tie in a different disguise.
    Background stays exactly zero, so no spurious components appear.
    """
    p = np.zeros((size, size), dtype=np.float32)
    for (y0, y1, x0, x1), peak in blobs:
        yy, xx = np.mgrid[0 : y1 - y0, 0 : x1 - x0]
        ramp = (yy / (y1 - y0 - 1) + xx / (x1 - x0 - 1)) / 2.0  # 0..1
        p[y0:y1, x0:x1] = peak - _BLOB_RAMP * (1.0 - ramp)
    return p


def _synthetic_probs(size=256):
    """Two clear blobs and one faint blob, so thresholds actually change the answer."""
    return _draw_blobs(_BLOBS, size)


# A deliberately adversarial layout for top_k: confidence, area and raster order
# all disagree. The most confident blob is the SMALLEST and comes SECOND in scan
# order, so `insts[:top_k]` and any size-based ranking both pick the other one.
_TOPK_BLOBS = [
    ((40, 70, 40, 120), 0.55),   # large, first in raster order, least confident
    ((150, 165, 60, 90), 0.95),  # small, second in raster order, most confident
]
_TOPK_WINNER = (150, 60)  # (y0, x0) of the only instance top_k=1 may keep


def _topk_probs(size=256):
    """Fixture where the most confident instance is neither the first nor the largest."""
    return _draw_blobs(_TOPK_BLOBS, size)


def test_instances_at_matches_probs_to_instances():
    """The sweep must reproduce probs_to_instances exactly at a given setting."""
    from sweep_submissions import instances_at

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)

    for thr in (0.30, 0.50, 0.80):
        for ma in (10, 100):
            expected = probs_to_instances(probs, thr, ma, disk, size=256)
            got = instances_at(probs, thr, ma, disk, size=256).instances
            assert len(got) == len(expected), f"count differs at thr={thr} ma={ma}"
            for a, b in zip(got, expected):
                assert a.area == b.area
                assert (a.y0, a.x0) == (b.y0, b.x0)
                assert np.array_equal(a.patch, b.patch)


def test_threshold_actually_changes_results():
    """Guard against a sweep that silently ignores its threshold argument."""
    from sweep_submissions import instances_at

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    low = instances_at(probs, 0.30, 10, disk, size=256).instances
    high = instances_at(probs, 0.50, 10, disk, size=256).instances
    assert len(low) > len(high), "the faint blob should vanish above 0.5"


def test_topk_keeps_most_confident():
    """top_k must rank by mean probability -- not by scan order, not by size.

    top_k is a lever we intend to submit: it tests whether false positives drive
    the CV-to-leaderboard gap. An implementation that quietly kept the first k, or
    the biggest k, would still produce a plausible-looking score and would answer
    that question wrongly, which is worse than failing outright. So the fixture
    makes confidence, area and raster order point at three different instances.
    """
    from sweep_submissions import instances_at

    probs = _topk_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    all_inst = instances_at(probs, 0.30, 10, disk, size=256).instances
    assert len(all_inst) == 2

    # Guard the fixture itself: if these stop holding, the test below is vacuous.
    first, second = all_inst
    assert (second.y0, second.x0) == _TOPK_WINNER, "winner must not come first in scan order"
    assert second.area < first.area, "winner must not be the largest instance"

    kept = instances_at(probs, 0.30, 10, disk, size=256, top_k=1).instances
    assert len(kept) == 1
    assert (kept[0].y0, kept[0].x0) == _TOPK_WINNER


def test_never_returns_empty():
    """Coverage is mandatory: an impossible threshold must still yield an instance."""
    from sweep_submissions import instances_at

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    got = instances_at(probs, 0.999, 100000, disk, size=256, guarantee=True)
    assert len(got.instances) >= 1
    assert got.arm == "last_resort"


# --- fixture whose blobs survive a relaxed ladder rung but not the operating point
_LADDER_MIN_AREA = 300  # a realistic submission setting, unlike the 100000 above


def test_guarantee_still_honours_topk():
    """top_k must cap the guarantee's output, not be discarded by it.

    guarantee=True is mandatory for every real submission and top_k is most
    interesting at aggressive thresholds -- exactly when frames fall through to a
    fallback. If the cap ran first the guarantee would replace the capped list
    wholesale, and a CSV labelled top_k=1 would hold two instances on precisely
    those frames, mixing two settings into one leaderboard number.
    """
    from sweep_submissions import instances_at

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    res = instances_at(probs, 0.99, _LADDER_MIN_AREA, disk, size=256,
                       top_k=1, guarantee=True)
    assert res.arm == "ladder", "fixture must fall through to the ladder"
    assert len(res.instances) == 1, "the cap must survive the guarantee"


def _same_instances(got, expected):
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        assert a.area == b.area
        assert (a.y0, a.x0) == (b.y0, b.x0)
        assert np.array_equal(a.patch, b.patch)


def _rung_area(min_area):
    from predict import FALLBACK_AREA_FACTOR, FALLBACK_MIN_AREA

    return max(int(min_area * 1.0 * FALLBACK_AREA_FACTOR), FALLBACK_MIN_AREA)


def test_relaxed_instances_resolves_at_the_operating_point_when_it_can():
    from predict import PRIMARY, relaxed_instances

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    insts, arm = relaxed_instances(probs, 0.30, 10, disk, size=256)
    assert arm == PRIMARY
    _same_instances(insts, probs_to_instances(probs, 0.30, 10, disk, size=256))


def test_relaxed_instances_stops_at_the_first_productive_rung(monkeypatch):
    """Rung order is the ladder's contract: the least-relaxed setting that finds
    anything wins. Taking a later rung would add instances the submission never
    had, on exactly the frames the arm tally already marks as suspect."""
    import predict

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    # 0.99 finds nothing; x0.9 finds only the confident blob; x0.5 adds the medium one.
    monkeypatch.setattr(predict, "FALLBACK", [(0.9, 1.0), (0.5, 1.0)])
    first = probs_to_instances(probs, 0.99 * 0.9, _rung_area(_LADDER_MIN_AREA), disk, size=256)
    second = probs_to_instances(probs, 0.99 * 0.5, _rung_area(_LADDER_MIN_AREA), disk, size=256)
    assert first and len(first) != len(second), "fixture must tell the rungs apart"

    insts, arm = predict.relaxed_instances(probs, 0.99, _LADDER_MIN_AREA, disk, size=256)
    assert arm == predict.LADDER
    _same_instances(insts, first)


def test_the_sweep_follows_predict_ladder_rather_than_a_copy(monkeypatch):
    """If predict.py gains, drops or reorders a rung, the sweep must follow.

    A sweep with its own copy of the ladder would label CSVs with the submission's
    fallback behaviour while applying a different one. Patching ``predict.FALLBACK``
    at runtime reaches only code that calls into predict for the ladder; a copied
    loop over an import-time binding never sees the change, so this test fails.
    """
    import predict
    from sweep_submissions import instances_at

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    before = instances_at(probs, 0.99, _LADDER_MIN_AREA, disk, size=256, guarantee=True)

    monkeypatch.setattr(predict, "FALLBACK", [(0.3, 1.0)])
    after = instances_at(probs, 0.99, _LADDER_MIN_AREA, disk, size=256, guarantee=True)
    assert after.arm == predict.LADDER
    assert len(after.instances) != len(before.instances), "the new rung must change the answer"
    _same_instances(after.instances,
                    probs_to_instances(probs, 0.99 * 0.3, _rung_area(_LADDER_MIN_AREA),
                                       disk, size=256))


@pytest.mark.parametrize("bad", [0, -1])
def test_topk_below_one_is_rejected(bad):
    """-1 is a plausible "no cap" idiom; silently dropping an instance is worse."""
    from sweep_submissions import instances_at

    probs = _synthetic_probs()
    disk = Disk(cx=128.0, cy=128.0, radius=127.0)
    with pytest.raises(ValueError):
        instances_at(probs, 0.30, 10, disk, size=256, top_k=bad)


def test_mean_prob():
    """Directly: the shared confidence measure averages over the mask only."""
    probs = np.array([[0.1, 0.9], [0.5, 0.7]], dtype=np.float32)
    patch = np.array([[True, False], [False, True]])
    assert mean_prob(Instance(0, 0, patch), probs) == pytest.approx(0.4)  # (0.1+0.7)/2

    # offset instances read the right window
    big = np.zeros((4, 4), dtype=np.float32)
    big[2:4, 2:4] = probs
    assert mean_prob(Instance(2, 2, patch), big) == pytest.approx(0.4)

    assert mean_prob(Instance(0, 0, np.zeros((2, 2), dtype=bool)), probs) == 0.0


# --- driver logic ----------------------------------------------------------
# The driver itself needs a GPU and 719 MB of data, so only its pure parts are
# tested here: which operating points a run expands to, what each one's CSV is
# called, and which arguments are rejected before a model is ever loaded.


def test_build_operating_points_is_the_full_grid():
    """Every threshold is crossed with every min_area, and nothing else."""
    from sweep_submissions import build_operating_points

    pts = build_operating_points([0.30, 0.50], [100, 300])
    assert len(pts) == 4
    assert {(p.threshold, p.min_area, p.top_k) for p in pts} == {
        (0.30, 100, None), (0.30, 300, None),
        (0.50, 100, None), (0.50, 300, None),
    }


def test_topk_variants_are_pinned_to_the_baseline():
    """A top-k point must vary ONLY the cap.

    top_k answers "do false positives cost us leaderboard PQ?". Crossing it with
    the whole threshold grid would confound the cap with the threshold and blow
    up the number of submissions; pinning it to the operating point we already
    submitted makes the leaderboard delta attributable to the cap alone.
    """
    from sweep_submissions import build_operating_points

    pts = build_operating_points([0.30, 0.50], [100], top_k=[5, 8],
                                 baseline_threshold=0.40, baseline_min_area=300)
    capped = [p for p in pts if p.top_k is not None]
    assert len(capped) == 2
    assert all((p.threshold, p.min_area) == (0.40, 300) for p in capped)
    assert sorted(p.top_k for p in capped) == [5, 8]


def test_a_realistic_sweep_emits_every_point_under_its_own_name():
    """Count first, uniqueness second.

    Uniqueness alone is nearly vacuous -- the tag is the dict key the points are
    collected under, so duplicates cannot survive into the return value. The
    property with teeth is that de-duplication swallowed nothing: a point missing
    from a sweep is only noticed when the CSV we planned to submit is not on
    disk, after the kernel has already run.
    """
    from sweep_submissions import build_operating_points

    thresholds, min_areas, caps = [0.25, 0.30, 0.40, 0.50, 0.60], [100, 300, 600], [3, 5, 8]
    pts = build_operating_points(thresholds, min_areas, top_k=caps)
    assert len(pts) == len(thresholds) * len(min_areas) + len(caps)
    assert len({p.filename for p in pts}) == len(pts)


def test_tags_that_collide_after_rounding_are_rejected():
    """Tags round the threshold to 2 dp, so 0.40 and 0.401 name the same file."""
    from sweep_submissions import build_operating_points

    with pytest.raises(ValueError, match="tag"):
        build_operating_points([0.40, 0.401], [300])


def test_repeated_points_collapse():
    """A repeated argument is a typo, not a request for two identical runs."""
    from sweep_submissions import build_operating_points

    pts = build_operating_points([0.30, 0.30], [300, 300], top_k=[5, 5])
    assert len(pts) == 2  # one grid point, one capped point


def _args(tmp_path, **over):
    """A namespace that validates cleanly, so each test can break one field."""
    import argparse

    cfg = tmp_path / "c.yaml"
    cfg.write_text("size: 1536\n", encoding="utf-8")
    ckpt = tmp_path / "f0.pt"
    ckpt.write_bytes(b"")
    base = dict(config=str(cfg), checkpoints=[str(ckpt)], thresholds=[0.30],
                min_areas=[300], top_k=None, baseline_threshold=0.40,
                baseline_min_area=300, limit=0, scales=None,
                outdir=str(tmp_path / "out"))
    base.update(over)
    return argparse.Namespace(**base)


def test_validate_args_accepts_a_sane_run(tmp_path):
    from sweep_submissions import validate_args

    assert validate_args(_args(tmp_path)) == []


@pytest.mark.parametrize("bad", [0, -1])
def test_validate_args_rejects_topk_below_one(tmp_path, bad):
    """Caught here rather than 15 minutes into a kernel by instances_at."""
    from sweep_submissions import validate_args

    problems = validate_args(_args(tmp_path, top_k=[5, bad]))
    assert any("top-k" in p for p in problems), problems


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_validate_args_rejects_thresholds_outside_the_unit_interval(tmp_path, bad):
    from sweep_submissions import validate_args

    problems = validate_args(_args(tmp_path, thresholds=[0.3, bad]))
    assert any("threshold" in p for p in problems), problems


@pytest.mark.parametrize("bad", [0, -5])
def test_validate_args_rejects_nonpositive_min_area(tmp_path, bad):
    from sweep_submissions import validate_args

    problems = validate_args(_args(tmp_path, min_areas=[bad]))
    assert any("min-area" in p for p in problems), problems


def test_validate_args_rejects_missing_files(tmp_path):
    """Both are read long after the run starts; existence is cheap to check now."""
    from sweep_submissions import validate_args

    problems = validate_args(_args(tmp_path, config=str(tmp_path / "nope.yaml")))
    assert any("config" in p for p in problems), problems
    problems = validate_args(_args(tmp_path, checkpoints=[str(tmp_path / "nope.pt")]))
    assert any("checkpoint" in p for p in problems), problems


def test_validate_args_rejects_a_baseline_the_topk_points_would_inherit(tmp_path):
    """The baseline is silently copied into every capped point, so it needs the
    same checks the swept values get -- otherwise a bad --baseline-threshold
    only surfaces as a ValueError from instances_at mid-run."""
    from sweep_submissions import validate_args

    problems = validate_args(_args(tmp_path, top_k=[5], baseline_threshold=1.4))
    assert any("baseline" in p for p in problems), problems


def test_validate_args_reports_every_problem_at_once(tmp_path):
    """One round trip per run: a kernel that dies twice on argument errors has
    burned two 15-minute slots."""
    from sweep_submissions import validate_args

    problems = validate_args(_args(tmp_path, thresholds=[2.0], min_areas=[0], top_k=[0]))
    # Each by its own prefix: a count would pass if one check fired three times.
    for prefix in ("threshold", "min-area", "top-k"):
        assert any(p.startswith(prefix) for p in problems), (prefix, problems)


def test_non_primary_fraction():
    """The share the fallback warning is judged on; empty tallies are not a crash."""
    from sweep_submissions import PRIMARY, LADDER, LAST_RESORT, non_primary_fraction

    assert non_primary_fraction({PRIMARY: 180}) == 0.0
    mixed = {PRIMARY: 60, LADDER: 100, LAST_RESORT: 20}
    assert non_primary_fraction(mixed) == pytest.approx(120 / 180)
    assert non_primary_fraction({}) == 0.0


def test_warning_wording_scales_with_the_contamination(capsys):
    """A banner that shouts the same words at 11% and at 70% teaches us to ignore it.

    The fallback warning is the only thing standing between a contaminated CSV
    and a leaderboard number read as if it described its nominal operating point,
    so its job is to make that number distrusted by the RIGHT amount.
    """
    from sweep_submissions import (
        LADDER, OperatingPoint, PRIMARY, warn_if_fallback_dominated)

    point = OperatingPoint(0.40, 300)

    assert not warn_if_fallback_dominated(point, {PRIMARY: 180})
    assert capsys.readouterr().out == "", "a clean point must print nothing"

    # 10.6%: over the warning line, but 89% of frames are still the real point.
    assert warn_if_fallback_dominated(point, {PRIMARY: 161, LADDER: 19})
    mild = capsys.readouterr().out
    assert "10.6%" in mild
    assert "mostly" not in mild.lower()

    assert warn_if_fallback_dominated(point, {PRIMARY: 54, LADDER: 126})
    severe = capsys.readouterr().out
    assert "MOSTLY" in severe


# --- disk_mask cache ---------------------------------------------------------
# Lives here rather than in a test file of its own because the cache exists for
# the sweep: one image's disk is rebuilt once per operating point otherwise.
_CACHE_DISK = Disk(cx=128.0, cy=128.0, radius=100.0)


def test_cached_disk_mask_cannot_be_written():
    """A cached mutable mask is a poisoning hazard: one caller writing into it
    would silently corrupt the disk of every later image with the same geometry."""
    from filseg.instances import disk_mask

    m = disk_mask(_CACHE_DISK, (256, 256))
    with pytest.raises(ValueError):
        m[0, 0] = True
    with pytest.raises(ValueError):
        m &= False  # the in-place idiom callers actually use, pointed the wrong way


def test_disk_mask_is_actually_cached():
    """Equal geometry from a distinct Disk object must hit the cache, or the
    speed-up silently disappears while every other test still passes."""
    from filseg.instances import disk_mask

    a = disk_mask(_CACHE_DISK, (256, 256))
    b = disk_mask(Disk(cx=128.0, cy=128.0, radius=100.0), (256, 256))
    assert a is b


@pytest.mark.parametrize("disk, shape, frac", [
    (_CACHE_DISK, (256, 256), 0.98),                               # the primed key itself
    (Disk(cx=140.0, cy=128.0, radius=100.0), (256, 256), 0.98),    # cx differs
    (Disk(cx=128.0, cy=140.0, radius=100.0), (256, 256), 0.98),    # cy differs
    (Disk(cx=128.0, cy=128.0, radius=90.0), (256, 256), 0.98),     # radius differs
    (_CACHE_DISK, (256, 300), 0.98),                               # shape differs
    (_CACHE_DISK, (256, 256), 0.90),                               # frac differs
])
def test_cached_disk_mask_equals_uncached(disk, shape, frac):
    """Bit-identical to the uncached computation, for every field of the key.

    The cache is primed with a DIFFERENT key first, so a key that omitted any one
    field would hand back the primed mask and fail here.
    """
    from filseg.instances import _compute_disk_mask, disk_mask

    disk_mask(_CACHE_DISK, (256, 256), 0.98)
    got = disk_mask(disk, shape, frac)
    assert np.array_equal(got, _compute_disk_mask(disk.cx, disk.cy, disk.radius, shape, frac))


# --- coverage gate and writing -------------------------------------------------
_STEM_A = "20110120105534Ch"
_STEM_B = "20110130110334Ch"


def _one_inst():
    return [Instance(0, 0, np.ones((2, 2), dtype=bool))]


def test_coverage_passes_a_complete_sweep():
    from sweep_submissions import OperatingPoint, check_coverage

    p = OperatingPoint(0.40, 300)
    rows = {p: [(_STEM_A, _one_inst()), (_STEM_B, _one_inst())]}
    assert check_coverage([p], rows, [_STEM_A, _STEM_B]) == {}


def test_coverage_fails_every_point_when_there_are_no_images():
    """0 == 0 and all([]) are both true: without this, a mount-path problem that
    lists no test images passes the gate and writes header-only CSVs."""
    from sweep_submissions import OperatingPoint, check_coverage

    p, q = OperatingPoint(0.40, 300), OperatingPoint(0.50, 300)
    failures = check_coverage([p, q], {p: [], q: []}, [])
    assert set(failures) == {p, q}


def test_coverage_names_the_empty_frame():
    """The stem is the only thing that makes the failure diagnosable from a Kaggle log."""
    from sweep_submissions import OperatingPoint, check_coverage

    p = OperatingPoint(0.40, 300)
    rows = {p: [(_STEM_A, _one_inst()), (_STEM_B, [])]}
    failures = check_coverage([p], rows, [_STEM_A, _STEM_B])
    assert _STEM_B in failures[p]
    assert _STEM_A not in failures[p]


def test_coverage_names_a_missing_image():
    from sweep_submissions import OperatingPoint, check_coverage

    p = OperatingPoint(0.40, 300)
    failures = check_coverage([p], {p: [(_STEM_A, _one_inst())]}, [_STEM_A, _STEM_B])
    assert _STEM_B in failures[p]


def test_coverage_catches_a_duplicated_image():
    """Every image present, one of them twice.

    main() cannot produce this -- rows come from one loop over a sorted listing --
    but check_coverage is a gate with its own contract, and a CSV with one frame's
    rows written twice would duplicate predictions the metric counts as false
    positives.
    """
    from sweep_submissions import OperatingPoint, check_coverage

    p = OperatingPoint(0.40, 300)
    rows = {p: [(_STEM_A, _one_inst()), (_STEM_B, _one_inst()), (_STEM_A, _one_inst())]}
    reason = check_coverage([p], rows, [_STEM_A, _STEM_B])[p]
    assert _STEM_A in reason
    assert "more than once" in reason
    assert "unexpected" not in reason, reason


def test_coverage_names_an_unexpected_image():
    """An entry for a stem the test listing does not contain.

    This used to fall into the duplicate branch and print
    ``3 image entries for 2 test images; duplicated: `` -- a count, then an empty
    list naming nothing, which is useless in exactly the log that has to explain it.
    """
    from sweep_submissions import OperatingPoint, check_coverage

    p = OperatingPoint(0.40, 300)
    stray = "20990101000000Zh"
    rows = {p: [(_STEM_A, _one_inst()), (_STEM_B, _one_inst()), (stray, _one_inst())]}
    reason = check_coverage([p], rows, [_STEM_A, _STEM_B])[p]
    assert stray in reason
    assert "unexpected" in reason
    assert "more than once" not in reason, reason
    assert not reason.rstrip().endswith(":"), reason


def test_one_bad_point_does_not_block_the_others(tmp_path):
    """A dropped point is recoverable; fifteen minutes of GPU time thrown away is not.

    The failing point must write nothing at all -- not even a temp file -- and
    every passing point must still land under its real name.
    """
    from sweep_submissions import PRIMARY, OperatingPoint, write_sweep

    good, bad = OperatingPoint(0.40, 300), OperatingPoint(0.50, 300)
    stems = [_STEM_A, _STEM_B]
    rows = {
        good: [(_STEM_A, _one_inst()), (_STEM_B, _one_inst())],
        bad: [(_STEM_A, _one_inst()), (_STEM_B, [])],
    }
    arms = {good: {PRIMARY: 2}, bad: {PRIMARY: 2}}
    written, failures = write_sweep([good, bad], rows, stems, tmp_path, arms=arms)

    assert list(failures) == [bad]
    assert [r.point for r in written] == [good]
    assert written[0].n_rows == 2
    assert sorted(f.name for f in tmp_path.iterdir()) == [good.filename]


def test_each_wrote_line_carries_its_arm_tally(tmp_path, capsys):
    """The tally is logged as each file lands, not only in the summary afterwards.

    The fallback warning and the summary table run after write_sweep returns. If a
    later write dies, the files already on disk would otherwise have no record of
    how many of their frames came from a fallback -- the one number that says
    whether a file's score can be read at face value.
    """
    from sweep_submissions import LADDER, PRIMARY, OperatingPoint, write_sweep

    p = OperatingPoint(0.40, 300)
    rows = {p: [(_STEM_A, _one_inst()), (_STEM_B, _one_inst())]}
    write_sweep([p], rows, [_STEM_A, _STEM_B], tmp_path, arms={p: {PRIMARY: 1, LADDER: 1}})
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "wrote" in ln]
    assert len(lines) == 1
    assert "1 primary" in lines[0]
    assert "1 ladder" in lines[0]
    assert "0 last-resort" in lines[0]


def test_write_is_atomic(tmp_path, monkeypatch):
    """A write that dies part-way must not leave a truncated file under a real name,
    nor clobber a complete file an earlier run left there."""
    import sweep_submissions
    from sweep_submissions import write_atomically

    final = tmp_path / "submission_thr0.40_ma300.csv"
    final.write_text("previous complete run\n", encoding="utf-8")

    def dies_mid_write(per_image, path, *a, **k):
        Path(path).write_text("filament_id,segmentation_rle\nhalf", encoding="utf-8")
        raise OSError("No space left on device")

    monkeypatch.setattr(sweep_submissions.rle, "write_submission", dies_mid_write)
    with pytest.raises(OSError):
        write_atomically([(_STEM_A, _one_inst())], final)

    assert final.read_text(encoding="utf-8") == "previous complete run\n"
    assert [f.name for f in tmp_path.iterdir()] == [final.name], "temp file must not linger"


def test_limit_runs_are_named_as_partial():
    """The filename is what gets uploaded; a 2-image CSV named like a real one is
    one careless submission away from wasting a slot."""
    from sweep_submissions import OperatingPoint, output_name

    p = OperatingPoint(0.40, 300)
    assert output_name(p, 0) == p.filename
    assert output_name(p, 20) == f"PARTIAL_limit20_{p.filename}"


def test_validate_args_rejects_an_unwritable_outdir(tmp_path, monkeypatch):
    """A path under read-only /kaggle/input otherwise fails only after inference."""
    import sweep_submissions
    from sweep_submissions import validate_args

    monkeypatch.setattr(sweep_submissions.os, "access", lambda path, mode: False)
    problems = validate_args(_args(tmp_path))
    assert any(p.startswith("outdir") for p in problems), problems


def test_validate_args_checks_the_nearest_existing_parent(tmp_path):
    """A not-yet-created outdir is fine; one whose nearest existing ancestor is a
    file can never be created."""
    from sweep_submissions import validate_args

    assert validate_args(_args(tmp_path, outdir=str(tmp_path / "a" / "b" / "c"))) == []
    blocker = tmp_path / "blocker"
    blocker.write_text("", encoding="utf-8")
    problems = validate_args(_args(tmp_path, outdir=str(blocker / "sub")))
    assert any(p.startswith("outdir") for p in problems), problems


def test_validate_args_rejects_nonpositive_scales(tmp_path):
    from sweep_submissions import validate_args

    problems = validate_args(_args(tmp_path, scales=[1536, 0]))
    assert any(p.startswith("scales") for p in problems), problems
