"""Tests for ``filseg.scorer.Scorer``: loading fold models and averaging them."""
from __future__ import annotations

import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.scorer import FEATURE_NAMES, Scorer  # noqa: E402

PARAMS = dict(objective="binary", num_leaves=4, min_data_in_leaf=5, verbose=-1, seed=1)
REG = dict(objective="regression", num_leaves=4, min_data_in_leaf=5, verbose=-1, seed=1)


def _feats(n: int, rng) -> list[dict[str, float]]:
    return [{k: float(v) for k, v in zip(FEATURE_NAMES, rng.random(len(FEATURE_NAMES)))}
            for _ in range(n)]


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    x = rng.random((200, len(FEATURE_NAMES)))
    y = (x[:, 0] > 0.5).astype(int)             # "area" decides, so predictions are non-trivial
    for k in range(3):
        lgb.train(PARAMS, lgb.Dataset(x, y), num_boost_round=20).save_model(
            str(tmp_path / f"scorer_v1_p_fold{k}.txt"))
        lgb.train(REG, lgb.Dataset(x, 0.5 + 0.4 * x[:, 0]), num_boost_round=20).save_model(
            str(tmp_path / f"scorer_v1_q_fold{k}.txt"))
    return tmp_path


def test_load_finds_every_fold(model_dir):
    s = Scorer.load(model_dir)
    assert len(s.p_models) == 3 and len(s.q_models) == 3
    assert s.features == FEATURE_NAMES


def test_predict_averages_folds_and_clips_q(model_dir):
    s = Scorer.load(model_dir)
    rng = np.random.default_rng(1)
    feats = _feats(10, rng)
    p, q = s.predict(feats)
    assert p.shape == q.shape == (10,)
    assert np.all((0 <= p) & (p <= 1)) and np.all((0 <= q) & (q <= 1))
    x = np.array([[f[k] for k in FEATURE_NAMES] for f in feats])
    expected = np.mean([m.predict(x) for m in s.p_models], axis=0)
    np.testing.assert_allclose(p, expected)
    # The learned rule tracks the planted signal: high "area" -> high p.
    hi, lo = dict(feats[0]), dict(feats[0])
    hi["area"], lo["area"] = 0.95, 0.05
    ph, _ = s.predict([hi, lo])
    assert ph[0] > ph[1]


def test_predict_on_no_candidates(model_dir):
    p, q = Scorer.load(model_dir).predict([])
    assert p.shape == (0,) and q.shape == (0,)


def test_load_refuses_missing_or_unbalanced_models(tmp_path, model_dir):
    with pytest.raises(FileNotFoundError):
        Scorer.load(tmp_path / "nothing_here")
    (model_dir / "scorer_v1_q_fold2.txt").unlink()
    with pytest.raises(FileNotFoundError):
        Scorer.load(model_dir)


def test_load_reads_the_feature_list_written_beside_the_models(model_dir):
    import json
    names = list(FEATURE_NAMES)
    (model_dir / "scorer_v1.features.json").write_text(json.dumps(names), encoding="utf-8")
    assert Scorer.load(model_dir).features == tuple(names)
    assert Scorer.load(model_dir, features=names[:3]).features == tuple(names[:3])
