"""--members on the scorer path: member maps reach the features the scorer sees."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_scorer_table as bst  # noqa: E402
import submission_from_cache as sfc  # noqa: E402
from filseg import data as fdata  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.scorer import AGREEMENT_NAMES, FEATURE_NAMES  # noqa: E402


class StubCache:
    def __init__(self, maps, folds=None):
        self.maps = maps
        self.folds = folds or {}

    def stems(self, split):
        return sorted(self.maps)

    def probs(self, split, stem):
        return self.maps[stem]

    def disk(self, split, stem):
        return Disk(cx=1024.0, cy=1024.0, radius=1000.0)

    def fold(self, stem):
        return self.folds.get(stem, 0)


class AgreeModel:
    """Keeps exactly the candidates at least one member supports."""
    features = (*FEATURE_NAMES, *AGREEMENT_NAMES)

    def predict(self, feats):
        p = np.array([1.0 if f["n_support"] >= 1 else 0.0 for f in feats])
        return p, np.ones(len(p))


def _map():
    p = np.zeros((2048, 2048), np.float32)
    p[900:960, 800:1000] = 0.8
    p[1200:1260, 800:1000] = 0.8
    return p


def _member_top_only():
    m = np.zeros((2048, 2048), np.float32)
    m[900:960, 800:1000] = 0.9
    return m


def test_scored_rows_passes_member_maps_to_the_features(monkeypatch):
    monkeypatch.setattr(fdata, "load_image", lambda stem, split="test": np.full((2048, 2048), 150, np.uint8))
    cache = StubCache({"T1": _map()})
    members = [StubCache({"T1": _member_top_only()})]
    rows, arms, scores = sfc.scored_rows(cache, AgreeModel(), 0.40, 50, None, 0.5,
                                         candidate_bridge=0.20, members=members)
    (stem, insts), = rows
    assert len(insts) == 1 and insts[0].y0 == 900         # only the supported rectangle survives
    assert sum(s["kept"] for s in scores) == 1


def test_feature_names_grow_only_with_members():
    assert bst.feature_names([]) == FEATURE_NAMES
    assert bst.feature_names([object()]) == (*FEATURE_NAMES, *AGREEMENT_NAMES)


def test_open_members_requires_the_same_stems(tmp_path):
    base = StubCache({"A": _map(), "B": _map()})
    good = StubCache({"A": _map(), "B": _map()})
    bad = StubCache({"A": _map()})
    assert bst.check_members([good], base, "oof") == [good]
    with pytest.raises(SystemExit, match="stems"):
        bst.check_members([bad], base, "oof")


# --- tying an agreement scorer to its member caches ---------------------------------

class NamedStub(StubCache):
    """A stub with what a real cache's identity is made of: a root, a manifest, a size."""

    def __init__(self, root, folds=None, blend=None, scales=(1536,)):
        super().__init__({"A": None, "B": None}, folds)
        self.root = Path(root)
        self.size = 1536
        self.manifest = {"scales": list(scales), **({"blend": blend} if blend else {})}


def test_a_member_with_other_folds_is_refused():
    base = StubCache({"A": _map(), "B": _map()}, folds={"A": 0, "B": 1})
    same = StubCache({"A": _map(), "B": _map()}, folds={"A": 0, "B": 1})
    other = StubCache({"A": _map(), "B": _map()}, folds={"A": 0, "B": 3})
    assert bst.check_members([same], base, "test") == [same]
    with pytest.raises(SystemExit, match="OOF stem B is fold 3"):
        bst.check_members([other], base, "oof")
    with pytest.raises(SystemExit, match="fold"):
        bst.check_members([other], base, "test")


def test_members_must_be_the_caches_the_blend_was_made_from(tmp_path):
    # blend_caches.py stores the path it was given: Windows separators on the laptop.
    blend = [{"cache": "cache\\m1", "config": "m1.yaml", "weight": 0.5},
             {"cache": "cache/m2", "config": "m2.yaml", "weight": 0.5}]
    cache = NamedStub(tmp_path / "blend", blend=blend)
    m1, m2, m3 = (NamedStub(tmp_path / n) for n in ("m1", "m2", "m3"))
    assert bst.check_members([m2, m1], cache, "oof") == [m2, m1]
    with pytest.raises(SystemExit, match="blended from"):
        bst.check_members([m1, m3], cache, "oof")
    with pytest.raises(SystemExit, match="blended from"):
        bst.check_members([m1], cache, "oof")
    assert bst.check_members([], cache, "oof") == []          # no members: nothing to tie


def test_members_sidecar_is_written_and_a_stale_one_dropped(tmp_path):
    table = tmp_path / "oof_thr0.40_ma50_bridge0.20_agree.csv.gz"
    side = bst.write_members_sidecar(table, NamedStub(tmp_path / "blend"),
                                     [NamedStub(tmp_path / "m1"), NamedStub(tmp_path / "m2_ms", scales=(1280, 1536))])
    assert side == tmp_path / "oof_thr0.40_ma50_bridge0.20_agree.members.json"
    rec = json.loads(side.read_text(encoding="utf-8"))
    assert rec["cache"] == "blend"
    assert rec["members"] == [{"name": "m1", "size": 1536, "scales": [1536]},
                              {"name": "m2_ms", "size": 1536, "scales": [1280, 1536]}]
    assert bst.write_members_sidecar(table, NamedStub(tmp_path / "blend"), []) is None
    assert not side.exists()


def test_default_table_name_marks_agreement_tables():
    assert bst.default_table_name(0.40, 50, 0.20, False) == "oof_thr0.40_ma50_bridge0.20.csv.gz"
    assert bst.default_table_name(0.40, 50, 0.20, True) == "oof_thr0.40_ma50_bridge0.20_agree.csv.gz"
    assert bst.default_table_name(0.40, 300, 0.0, False) == "oof_thr0.40_ma300.csv.gz"


def test_submission_refuses_members_other_than_the_scorers(tmp_path):
    (tmp_path / "v_agree.members.json").write_text(json.dumps(
        {"cache": "blend", "members": [{"name": "m1", "size": 1536, "scales": [1536]},
                                       {"name": "m2", "size": 1536, "scales": [1536]}]}), encoding="utf-8")
    m1, m2, m3 = (NamedStub(tmp_path / n) for n in ("m1", "m2", "m3"))
    sfc.check_scorer_members(tmp_path, "v_agree", [m2, m1])                    # same set: fine
    with pytest.raises(SystemExit, match="trained") as e:
        sfc.check_scorer_members(tmp_path, "v_agree", [m1, m3])
    assert "m3" in str(e.value) and "m2" in str(e.value)                      # both lists shown
    with pytest.raises(SystemExit, match="trained"):                          # a re-cache, same name
        sfc.check_scorer_members(tmp_path, "v_agree", [m1, NamedStub(tmp_path / "m2", scales=(1280, 1536))])
    sfc.check_scorer_members(tmp_path, "v_plain", [m1, m3])                   # no sidecar: no check
