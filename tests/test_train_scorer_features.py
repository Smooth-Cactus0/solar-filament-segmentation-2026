"""train_scorer learns the agreement columns when the table has them, and records its list."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_scorer_table as bst  # noqa: E402
import train_scorer as ts  # noqa: E402
from filseg.scorer import AGREEMENT_NAMES, FEATURE_NAMES  # noqa: E402


def test_table_features_follow_the_columns():
    plain = ["stem", "fold", "label", *FEATURE_NAMES]
    assert ts.table_features(plain) == list(FEATURE_NAMES)
    assert ts.table_features(plain + list(AGREEMENT_NAMES)) == [*FEATURE_NAMES, *AGREEMENT_NAMES]
    # a partial set is a corrupt table, not a feature choice
    try:
        ts.table_features(plain + ["agree_mean"])
    except ValueError as e:
        assert "agreement" in str(e)
    else:
        raise AssertionError("partial agreement columns must be refused")


# --- per-fold PQ (the fold rule reads it) --------------------------------------------

def _rows():
    # Folds interleaved on purpose: per-fold slicing must not mix row labels and positions.
    return pd.DataFrame({
        "stem": ["b", "a", "a"], "fold": [1, 0, 0], "pass_id": [0, 0, 0], "cand": [0, 0, 1],
        "n_gt": [1, 2, 2], "matched_gt": [-1, 0, -1], "best_iou": [0.0, 0.8, 0.0]})


def test_per_fold_pq_pools_within_each_fold():
    accept = np.array([True, True, False])
    # fold 0: TP 1 (IoU 0.8), FN 1 -> 0.8 / 1.5; fold 1: FP 1, FN 1 -> 0;
    # fold 2: passes but no candidate at all -> every filament an FN -> 0.
    got = ts.per_fold_pq(_rows(), accept, {0: 2, 1: 1, 2: 4})
    assert list(got) == ["0", "1", "2"]
    assert got["0"] == pytest.approx(0.8 / 1.5) and got["1"] == 0.0 and got["2"] == 0.0
    # Without the passes file the GT count comes from the rows, as in pooled_pq.
    assert ts.per_fold_pq(_rows(), accept) == pytest.approx({"0": 0.8 / 1.5, "1": 0.0})
    # One fold alone is the pooled number.
    one = _rows()[_rows().fold == 0].reset_index(drop=True)
    assert ts.per_fold_pq(one, accept[1:])["0"] == ts.pooled_pq(one, accept[1:])["pq"]


def _synthetic_table(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    rows, passes = [], []
    for k in range(5):
        for j in range(30):
            stem = f"s{k}_{j}"
            passes.append({"stem": stem, "fold": k, "pass_id": 0, "n_gt": 3})
            for c in range(6):
                hit = c < 3 and rng.random() < 0.6
                f = {n: float(rng.normal()) for n in FEATURE_NAMES}
                f["area"] = float(rng.integers(50, 2000))
                f["p_mean"] = float(rng.uniform(0.5, 0.9) if hit else rng.uniform(0.3, 0.6))
                rows.append({"stem": stem, "fold": k, "pass_id": 0, "n_passes": 1, "cand": c,
                             "n_cand_frame": 6, "n_gt": 3, "matched_gt": c if hit else -1,
                             "best_iou": float(rng.uniform(0.5, 0.95)) if hit else 0.0,
                             "label": int(hit), **f})
    passes.append({"stem": "empty", "fold": 2, "pass_id": 0, "n_gt": 2})   # no candidate at all
    table = tmp_path / "t_agree.csv.gz"
    pd.DataFrame(rows, columns=[*bst.ID_COLUMNS, *FEATURE_NAMES]).to_csv(table, index=False)
    pd.DataFrame(passes).to_csv(bst.passes_path(table), index=False)
    return table


def test_main_records_per_fold_pq_and_the_members(tmp_path):
    table = _synthetic_table(tmp_path)
    record = {"cache": "blend", "members": [{"name": "m1", "size": 1536, "scales": [1536]}]}
    bst.members_path(table).write_text(json.dumps(record), encoding="utf-8")
    out, models = tmp_path / "t_v.json", tmp_path / "models"
    args = ["--table", str(table), "--rounds", "20", "--version", "t_v", "--out", str(out),
            "--model-dir", str(models)]
    ts.main(args)
    res = json.loads(out.read_text(encoding="utf-8"))
    for rule, key in (("best_pq_product", "c"), ("best_p_threshold", "tau")):
        r = res[rule]
        assert {key, "pq", "sq", "rq", "tp", "fp", "fn", "n_accepted"} <= set(r)   # unchanged keys
        assert list(r["per_fold_pq"]) == ["0", "1", "2", "3", "4"]
        assert all(0.0 <= v <= 1.0 for v in r["per_fold_pq"].values())
    assert res["members"] == record["members"]
    assert json.loads((models / "t_v.members.json").read_text(encoding="utf-8")) == record

    # Retrained from a table without a sidecar: the stale copy must not survive.
    bst.members_path(table).unlink()
    ts.main(args)
    assert json.loads(out.read_text(encoding="utf-8"))["members"] is None
    assert not (models / "t_v.members.json").exists()
