"""pipeline_nb.py: the glue of notebooks/pipeline.ipynb.

The argument builders are exercised through the scripts they feed (cache_probs,
blend_caches, submission_from_cache), so a renamed flag fails here and
not halfway through a notebook run. The checks are tested on submissions built to break
each rule the organisers' safeguards look at.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import blend_caches as bc  # noqa: E402
import cache_probs as cp  # noqa: E402
import pipeline_nb as nb  # noqa: E402
import submission_from_cache as sfc  # noqa: E402
from filseg import rle  # noqa: E402
from filseg.cache import ProbCache  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.metric import from_mask  # noqa: E402

FULL = 2048
DISK = {"cx": 1024.0, "cy": 1024.0, "radius": 900.0}


def _blob_map(size: int, centres, value: float = 0.9) -> np.ndarray:
    """A map of square blobs (centre given in 2048 px coordinates) on a faint background."""
    p = np.full((size, size), 0.02, np.float32)
    s = size / FULL
    for cy, cx in centres:
        y, x, h = int(cy * s), int(cx * s), max(int(40 * s), 2)
        p[y - h:y + h, x - h:x + h] = value
    return p


def _cache(root: Path, value_maps: dict[str, np.ndarray], oof: dict[str, np.ndarray] | None = None) -> Path:
    """A cache in cache_probs's format: test maps (and optionally out-of-fold maps)."""
    size = next(iter(value_maps.values())).shape[0]
    m = {"config": f"{root.name}.yaml", "encoder": "e", "size": size, "model_size": size, "quant": cp.QUANT,
         "tta": True, "scales": [size], "checkpoints": {}, "oof": {}, "test": {}}
    for stem, p in value_maps.items():
        cp.write_map(root / "test" / f"{stem}.png", p)
        m["test"][stem] = dict(DISK)
    for i, (stem, p) in enumerate((oof or {}).items()):
        cp.write_map(root / "oof" / f"{stem}.png", p)
        m["oof"][stem] = {"fold": i % 2, **DISK}
    cp.write_manifest(root, m)
    return root


def _box(y0, y1, x0, x1):
    m = np.zeros((FULL, FULL), bool)
    m[y0:y1, x0:x1] = True
    return from_mask(m)


def _csv(path: Path, per_image) -> Path:
    rle.write_submission(per_image, path)
    return path


# --- argument builders ------------------------------------------------------------------

def test_partial_mirrors_the_submission_writers_limit_naming(tmp_path):
    cache = _cache(tmp_path / "c", {"t0": _blob_map(64, [(1000, 1000)]), "t1": _blob_map(64, [(900, 900)])})
    out = tmp_path / "subs" / "x.csv"
    sfc.main(["--cache", str(cache), "--threshold", "0.4", "--min-area", "50", "--out", str(out), "--limit", "1"])
    assert nb.partial(out, 1).is_file() and not out.exists()
    assert nb.partial(out, None) == out


def test_find_checkpoints_returns_the_five_folds_in_order_or_names_the_missing(tmp_path):
    for k in (3, 0, 1, 4, 2):
        (tmp_path / f"fam_fold{k}.pt").write_bytes(b"")
    assert [p.name for p in nb.find_checkpoints(tmp_path, "fam")] == [f"fam_fold{k}.pt" for k in range(5)]
    (tmp_path / "fam_fold2.pt").unlink()
    with pytest.raises(FileNotFoundError, match="fam_fold2.pt"):
        nb.find_checkpoints(tmp_path, "fam")


def test_cache_argv_runs_cache_probs_for_test_maps_only(tmp_path, monkeypatch):
    cfg = {"encoder": "e", "size": 8, "amp": True, "eval": {"tta": True, "cache_size": 16}}
    devices = []
    monkeypatch.setattr(cp, "load_config", lambda path: cfg)
    monkeypatch.setattr(cp, "load_model", lambda cfg, ckpt, device: devices.append(device) or object())
    monkeypatch.setattr(cp.fdata, "load_image", lambda stem, split="train": np.zeros((16, 16), np.uint8))
    monkeypatch.setattr(cp.fdata, "detect_disk", lambda img: Disk(cx=8.0, cy=8.0, radius=6.0))
    monkeypatch.setattr(cp.fdata, "list_stems", lambda split: ["t0", "t1", "t2"])
    monkeypatch.setattr(cp, "model_input", lambda img, disk, size, inputs=None: np.zeros((3, 8, 8), np.float32))
    monkeypatch.setattr(cp, "probs_for", lambda *a, **k: np.full((8, 8), 0.3, np.float32))
    out = tmp_path / "cache" / "fam"
    cp.main(nb.cache_argv("x.yaml", ["a/fam_fold0.pt", "a/fam_fold1.pt"], out, tmp_path / "absent.csv",
                          "cpu", limit=2))
    c = ProbCache.open(out)
    assert devices == ["cpu", "cpu"] and c.stems("test") == ["t0", "t1"] and c.stems("oof") == []
    assert c.size == 16 and c.verify() == []          # stored at eval.cache_size, nothing missing
    assert nb.missing_maps(tmp_path / "cache", ["fam"], ["t0", "t1", "t2"]) == ["fam/t2"]
    assert nb.missing_maps(tmp_path / "cache", ["other"], ["t0"]) == ["other/t0"]


def test_staged_caches_blend_like_the_full_ones(tmp_path):
    a = _cache(tmp_path / "full" / "a", {"t0": np.full((8, 8), 0.2, np.float32), "t1": np.full((8, 8), 0.4, np.float32)},
               oof={"o0": np.zeros((8, 8), np.float32), "o1": np.zeros((8, 8), np.float32)})
    b = _cache(tmp_path / "full" / "b", {"t0": np.full((8, 8), 0.8, np.float32), "t1": np.full((8, 8), 0.6, np.float32)},
               oof={"o0": np.zeros((8, 8), np.float32), "o1": np.zeros((8, 8), np.float32)})
    work = tmp_path / "work"
    assert nb.stage_test_maps(a, work / "a") == 2
    assert nb.stage_test_maps(b, work / "b", stems=["t0"]) == 1
    sa = ProbCache.open(work / "a")
    assert sa.stems("oof") == [] and sa.stems("test") == ["t0", "t1"] and sa.verify() == []
    assert sa.disk("test", "t1") == Disk(**DISK)
    with pytest.raises(KeyError):
        nb.stage_test_maps(a, work / "x", stems=["nope"])
    nb.stage_test_maps(b, work / "b")
    bc.main(nb.blend_argv([work / "a", work / "b"], work / "ab"))
    blend = ProbCache.open(work / "ab")
    np.testing.assert_allclose(blend.probs("test", "t0"), 0.5, atol=1 / cp.QUANT)
    assert [m["weight"] for m in blend.manifest["blend"]] == pytest.approx([0.5, 0.5])


def test_rule_argv_written_and_scores_paths(tmp_path):
    rule = nb.Rule(out="sub.csv", cache="blend", scorer_version="v2", threshold=0.40, min_area=50,
                   candidate_bridge=0.20, accept_pq=0.16)
    argv = rule.argv(tmp_path / "cache", tmp_path / "scorer", tmp_path / "subs")
    flags = dict(zip(argv[::2], argv[1::2]))
    assert flags == {"--cache": str(tmp_path / "cache" / "blend"), "--threshold": "0.4", "--min-area": "50",
                     "--candidate-bridge": "0.2", "--scorer-dir": str(tmp_path / "scorer"),
                     "--scorer-version": "v2", "--accept-pq": "0.16", "--out": str(tmp_path / "subs" / "sub.csv")}
    assert rule.argv("c", "s", "o", limit=3)[-2:] == ["--limit", "3"]
    assert rule.written(tmp_path, 3).name == "PARTIAL_limit3_sub.csv"
    assert rule.scores(tmp_path) == tmp_path / "sub.scores.csv"


def test_most_rejected_picks_the_image_with_the_most_turned_down_candidates(tmp_path):
    scores = tmp_path / "s.scores.csv"
    pd.DataFrame({"stem": ["a", "a", "b", "b", "b"], "cand": [0, 1, 0, 1, 2],
                  "kept": [False, True, False, False, True]}).to_csv(scores, index=False)
    assert nb.most_rejected(scores) == "b"
    pd.DataFrame({"stem": ["a"], "cand": [0], "kept": [True]}).to_csv(scores, index=False)
    assert nb.most_rejected(scores) is None


# --- checks -------------------------------------------------------------------------------

def test_check_submission_flags_each_broken_rule(tmp_path):
    good = _csv(tmp_path / "good.csv", [("s1", [_box(0, 10, 0, 10), _box(20, 30, 0, 10)]),
                                         ("s2", [_box(0, 10, 0, 10)])])
    chk = nb.check_submission(good, ["s1", "s2"])
    assert chk.ok and chk.rows == 3 and chk.frames == 2 and chk.summary()["inst/img"] == 1.5

    overlap = _csv(tmp_path / "overlap.csv", [("s1", [_box(0, 10, 0, 10), _box(5, 15, 5, 15)])])
    chk = nb.check_submission(overlap, ["s1", "s3"])
    assert not chk.ok and chk.overlapping == ["s1"] and chk.missing == ["s3"]

    rows = nb.read_rows(good)
    dup = tmp_path / "dup.csv"
    rows.assign(filament_id="s1_1")[["filament_id", "segmentation_rle"]].to_csv(dup, index=False)
    chk = nb.check_submission(dup, ["s1"])
    assert chk.duplicate_ids == 2 and chk.unexpected == [] and chk.overlapping == ["s1"]
    assert nb.check_submission(good, ["s1"]).unexpected == ["s2"]


def test_compare_submissions_separates_bytes_rows_and_instances(tmp_path):
    a, b, c = _box(0, 10, 0, 10), _box(20, 30, 0, 10), _box(40, 50, 0, 10)
    ref = _csv(tmp_path / "ref.csv", [("s1", [a, b]), ("s2", [c])])
    same = _csv(tmp_path / "same.csv", [("s1", [a, b]), ("s2", [c])])
    swapped = _csv(tmp_path / "swapped.csv", [("s1", [b, a]), ("s2", [c])])
    moved = _csv(tmp_path / "moved.csv", [("s1", [a, b]), ("s2", [_box(41, 50, 0, 10)])])

    r = nb.compare_submissions(same, ref)
    assert r.byte_identical and r.rows_identical and r.same_instances and r.images == 2
    r = nb.compare_submissions(swapped, ref)
    assert not r.byte_identical and not r.rows_identical and r.same_instances
    r = nb.compare_submissions(moved, ref)
    assert not r.same_instances and r.differing == ["s2"]
    r = nb.compare_submissions(moved, ref, stems=["s1"])           # a limited run: only s1 compared
    assert r.byte_identical is None and r.rows_identical and r.same_instances and r.rows_reference == 2


def test_compare_maps_counts_levels_and_reports_missing_references(tmp_path, capsys):
    p = np.full((16, 16), 0.4, np.float32)          # 102 levels: no rounding tie
    q = p.copy()
    q[0, 0] += 3 / 255
    _cache(tmp_path / "a" / "fam", {"t0": p, "t1": p})
    _cache(tmp_path / "b" / "fam", {"t0": q})
    df = nb.compare_maps(tmp_path / "a", tmp_path / "b", ["fam"])
    assert df.stem.tolist() == ["t0", "t1"]
    assert df.max_levels.iloc[0] == 3 and df.frac_px_differ.iloc[0] == pytest.approx(1 / 256)
    assert np.isnan(df.max_levels.iloc[1])
    assert nb.main(["compare-maps", "--a", str(tmp_path / "a"), "--b", str(tmp_path / "b"), "--caches", "fam"]) == 1
    assert nb.main(["compare-maps", "--a", str(tmp_path / "b"), "--b", str(tmp_path / "a"), "--caches", "fam"]) == 0
    assert "per cache" in capsys.readouterr().out


def test_stage_maps_command_copies_each_cache(tmp_path, capsys):
    _cache(tmp_path / "src" / "fam", {"t0": np.zeros((8, 8), np.float32)},
           oof={"o0": np.zeros((8, 8), np.float32)})
    assert nb.main(["stage-maps", "--src", str(tmp_path / "src"), "--dst", str(tmp_path / "dst"),
                    "--caches", "fam"]) == 0
    assert ProbCache.open(tmp_path / "dst" / "fam").verify() == []
    assert "fam: 1 test maps" in capsys.readouterr().out


# --- the per-image view and the figures -------------------------------------------------------

def test_frame_view_lines_up_candidates_verdicts_and_written_instances(tmp_path, monkeypatch):
    import filseg.data as fdata

    monkeypatch.setattr(fdata, "load_image", lambda stem, split="train": np.full((FULL, FULL), 100, np.uint8))
    cache = _cache(tmp_path / "blend", {"t0": _blob_map(256, [(800, 800), (1200, 1300)])})
    rule = nb.Rule(out="s.csv", cache="blend", scorer_version="v", threshold=0.4, min_area=50,
                   candidate_bridge=0.2, accept_pq=0.16)
    scores = tmp_path / "s.scores.csv"
    pd.DataFrame({"stem": ["t0", "t0", "zz"], "cand": [1, 0, 0], "area": [1, 1, 1], "p": [0.9, 0.2, 0.5],
                  "q": [0.5, 0.5, 0.5], "kept": [True, False, True]}).to_csv(scores, index=False)
    view0 = nb.frame_view(cache, "t0", rule, scores, _csv(tmp_path / "empty.csv", [("t1", [_box(0, 5, 0, 5)])]))
    assert len(view0.candidates) == 2 and view0.scores.cand.tolist() == [0, 1]
    assert view0.scores.pq.tolist() == pytest.approx([0.1, 0.45]) and view0.written == []
    kept = [view0.candidates[1]]
    sub = _csv(tmp_path / "s.csv", [("t0", kept)])
    view = nb.frame_view(cache, "t0", rule, scores, sub)
    assert len(view.written) == 1 and np.array_equal(view.written[0].patch, kept[0].patch)
    assert view.probs.shape == (FULL, FULL)


def test_figures_draw_without_error(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import filseg.data as fdata

    monkeypatch.setattr(fdata, "load_image", lambda stem, split="train": np.full((FULL, FULL), 100, np.uint8))
    img = np.full((FULL, FULL), 100, np.uint8)
    disk = Disk(**DISK)
    _cache(tmp_path / "cache" / "fam", {"t0": _blob_map(64, [(1000, 1000)])})
    a, b = _box(1000, 1040, 1000, 1100), _box(1200, 1230, 900, 960)
    figs = [nb.show_planes(img, disk, np.zeros((3, 32, 32), np.float32)),
            nb.show_maps(tmp_path / "cache", ["fam"], "t0")]
    rule = nb.Rule("s.csv", "fam", "v", 0.4, 50, 0.2, 0.16)
    scores = tmp_path / "s.scores.csv"
    pd.DataFrame({"stem": ["t0"], "cand": [0], "area": [1], "p": [0.9], "q": [0.5], "kept": [True]}).to_csv(scores, index=False)
    view = nb.frame_view(tmp_path / "cache" / "fam", "t0", rule, scores, _csv(tmp_path / "s.csv", [("t0", [a])]))
    figs.append(nb.show_decision(view, 0.4))
    assert all(isinstance(f, plt.Figure) for f in figs)
    lab = nb.label_image([a, b])
    assert lab.max() == 2 and lab.mask.sum() == FULL * FULL - a.area - b.area
    assert nb.crop_box([], disk) == (116, 1932, 116, 1932)
    plt.close("all")
