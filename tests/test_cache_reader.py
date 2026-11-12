"""Tests for ``filseg.cache``: the read side of the probability-map cache.

Built against a cache written by ``scripts/cache_probs.py`` itself, so the two
halves cannot drift apart without a test noticing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cache_probs as cp  # noqa: E402
from filseg.cache import OOF, TEST, ProbCache  # noqa: E402
from filseg.data import Disk  # noqa: E402


def _cfg():
    return {"encoder": "tu-convnext_small", "size": 256, "amp": True, "eval": {"tta": True}}


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    m = cp.new_manifest(_cfg(), "configs/exp2_convnext_1536.yaml",
                        {0: "exp2_fold0.pt", 1: "exp2_fold1.pt"}, tta=True, scales=None)
    disk = Disk(cx=128.0, cy=127.5, radius=110.25)
    for stem, fold in (("20140609195854Bh", 0), ("20150125172714Mh", 1)):
        cp.write_map(tmp_path / OOF / f"{stem}.png", rng.random((256, 256), dtype=np.float32))
        m[OOF][stem] = {"fold": fold, **cp.disk_record(disk)}
    cp.write_map(tmp_path / TEST / "20110120105534Ch.png", rng.random((256, 256), dtype=np.float32))
    m[TEST]["20110120105534Ch"] = cp.disk_record(disk)
    cp.write_manifest(tmp_path, m)
    return tmp_path


def test_open_reads_the_manifest(cache_dir):
    c = ProbCache.open(cache_dir)
    assert c.size == 256 and c.quant == cp.QUANT
    assert c.config == "exp2_convnext_1536.yaml"
    assert c.stems(OOF) == ["20140609195854Bh", "20150125172714Mh"]
    assert c.stems(TEST) == ["20110120105534Ch"]
    assert c.fold("20150125172714Mh") == 1


def test_disk_round_trips_through_the_manifest(cache_dir):
    c = ProbCache.open(cache_dir)
    assert c.disk(OOF, "20140609195854Bh") == Disk(cx=128.0, cy=127.5, radius=110.25)
    assert c.disk(TEST, "20110120105534Ch") == Disk(cx=128.0, cy=127.5, radius=110.25)


def test_probs_match_what_the_writer_stored(cache_dir):
    c = ProbCache.open(cache_dir)
    p = c.probs(TEST, "20110120105534Ch")
    assert p.dtype == np.float32 and p.shape == (256, 256)
    assert 0.0 <= p.min() and p.max() <= 1.0
    # The reader and the writer agree on the quantisation grid.
    np.testing.assert_array_equal(cp.quantize(p), cp.quantize(cp.read_map(c.path(TEST, "20110120105534Ch"))))


def test_verify_lists_missing_maps(cache_dir):
    c = ProbCache.open(cache_dir)
    assert c.verify() == []
    c.path(OOF, "20140609195854Bh").unlink()
    assert c.verify() == ["oof/20140609195854Bh"]


def test_unknown_split_is_refused(cache_dir):
    c = ProbCache.open(cache_dir)
    with pytest.raises(ValueError):
        c.stems("train")
    with pytest.raises(ValueError):
        c.probs("val", "x")


def test_a_spine_cache_opens_only_when_asked(cache_dir):
    import json
    m = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    m["channel"] = "spine"
    (cache_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="spine"):
        ProbCache.open(cache_dir)
    assert ProbCache.open(cache_dir, allow_spine=True).manifest["channel"] == "spine"
