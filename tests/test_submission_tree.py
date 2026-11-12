"""submission_from_cache.tree_rows with a stub cache and a stub scorer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import submission_from_cache as sfc  # noqa: E402
from filseg import data as fdata  # noqa: E402
from filseg.data import Disk  # noqa: E402


class StubCache:
    def __init__(self, maps):
        self.maps = maps

    def stems(self, split):
        return sorted(self.maps)

    def probs(self, split, stem):
        return self.maps[stem]

    def disk(self, split, stem):
        return Disk(cx=1024.0, cy=1024.0, radius=1000.0)


class StubModel:
    def __init__(self, value):
        self.value = value

    def predict(self, feats):
        n = len(feats)
        return np.full(n, self.value), np.ones(n)


def _map():
    p = np.zeros((2048, 2048), np.float32)
    p[900:960, 800:1000] = 0.45
    p[915:945, 830:970] = 0.8
    p[1200:1260, 800:1000] = 0.45
    return p


NESTED = {"levels": (0.25, 0.40, 0.60), "min_area": 50, "clip": 0.98, "children": None}


def _covered(insts):
    cover = np.zeros((2048, 2048), np.int16)
    for it in insts:
        cover[it.y0:it.y1, it.x0:it.x1] += it.patch
    return cover


def test_tree_rows_keep_disjoint_nodes(monkeypatch):
    monkeypatch.setattr(fdata, "load_image", lambda stem, split="test": np.full((2048, 2048), 150, np.uint8))
    rows, arms, scores = sfc.tree_rows(StubCache({"T1": _map()}), StubModel(0.9), 0.2,
                                       NESTED, 0.40)
    (stem, insts), = rows
    assert stem == "T1" and len(insts) >= 2
    assert _covered(insts).max() == 1
    assert arms[sfc.PRIMARY] == 1 and len(scores) > 0


def test_tree_rows_guarantee_one_instance(monkeypatch):
    monkeypatch.setattr(fdata, "load_image", lambda stem, split="test": np.full((2048, 2048), 150, np.uint8))
    rows, arms, _ = sfc.tree_rows(StubCache({"T1": _map()}), StubModel(0.0), 0.2,
                                  NESTED, 0.40)
    assert len(rows[0][1]) == 1 and arms[sfc.SCORER_GUARANTEE] == 1


def _sidecar(tmp_path, **over):
    import json
    s = {"levels": [0.25, 0.4, 0.6], "clip": 0.98, "min_area": 50, "table": "t.csv.gz", **over}
    (tmp_path / "v3.tree.json").write_text(json.dumps(s), encoding="utf-8")


def test_tree_settings_come_from_the_sidecar(tmp_path):
    _sidecar(tmp_path)
    got = sfc.tree_settings(tmp_path, "v3", None, None, 50)
    assert got == {"levels": (0.25, 0.4, 0.6), "clip": 0.98, "min_area": 50, "children": None,
                   "candidates": None, "params": {}, "spine_cache": None}
    # explicit flags that agree are fine (any order)
    assert sfc.tree_settings(tmp_path, "v3", [0.6, 0.25, 0.4], 0.98, 50)["levels"] == (0.25, 0.4, 0.6)


def test_tree_settings_mismatch_is_an_error(tmp_path):
    import pytest
    _sidecar(tmp_path)
    for args in (([0.3], None, 50), (None, 0.9, 50), (None, None, 300)):
        with pytest.raises(SystemExit) as e:
            sfc.tree_settings(tmp_path, "v3", *args)
        assert "sidecar" in str(e.value) or "tree.json" in str(e.value)


def test_tree_settings_missing_sidecar_is_an_error(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="tree.json"):
        sfc.tree_settings(tmp_path, "v3", None, None, 50)


def test_tree_rows_cuts_a_split_tree_when_the_settings_say_so(monkeypatch):
    monkeypatch.setattr(fdata, "load_image", lambda stem, split="test": np.full((2048, 2048), 150, np.uint8))
    p = np.zeros((2048, 2048), np.float32)               # one 0.45 component holding two 0.8 cores
    p[900:960, 800:1100] = 0.45
    p[915:945, 830:900] = 0.8
    p[915:945, 1000:1070] = 0.8
    cache = StubCache({"S": p})
    settings = {"levels": [0.40, 1.0], "min_area": 50, "clip": 0.98, "children": "watershed",
                "candidates": {"threshold": 0.25, "min_area": 50, "bridge": 0.0},
                "params": {"min_area": 50, "min_parent": 300, "min_distance": 5}}
    rows, arms, scores = sfc.tree_rows(cache, StubModel(0.9), 0.1, settings, 0.40)
    assert [s["level"] for s in scores][0] == 0.40            # the first node is a production candidate
    assert any(s["level"] == 1.0 for s in scores)             # and it has watershed children
