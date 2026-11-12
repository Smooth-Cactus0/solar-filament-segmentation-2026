"""A queued job whose result already exists must not retrain."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from train import already_done, init_result_missing  # noqa: E402


def test_already_done(tmp_path):
    assert not already_done(tmp_path, "scr_r0_base_1024_fold0")
    (tmp_path / "scr_r0_base_1024_fold0.json").write_text("{}", encoding="utf-8")
    assert already_done(tmp_path, "scr_r0_base_1024_fold0")
    assert not already_done(tmp_path, "scr_r0_base_1024_fold1")


def test_an_empty_result_file_does_not_count(tmp_path):
    (tmp_path / "x_fold0.json").write_text("", encoding="utf-8")
    assert not already_done(tmp_path, "x_fold0")


def test_init_result_missing_when_no_results_json(tmp_path):
    assert init_result_missing(tmp_path, "checkpoints/scr_r0_base_1024_fold0.pt")


def test_init_result_missing_is_false_once_stage1_finished(tmp_path):
    (tmp_path / "scr_r0_base_1024_fold0.json").write_text("{}", encoding="utf-8")
    assert not init_result_missing(tmp_path, "checkpoints/scr_r0_base_1024_fold0.pt")


def test_init_result_missing_true_for_empty_results_json(tmp_path):
    (tmp_path / "scr_r0_base_1024_fold0.json").write_text("", encoding="utf-8")
    assert init_result_missing(tmp_path, "checkpoints/scr_r0_base_1024_fold0.pt")


def test_run_name_smoke_never_matches_a_real_result(tmp_path):
    from train import run_name
    assert run_name("x", 2, smoke=False) == "x_fold2"
    assert run_name("x", 2, smoke=True) == "x_smoke_fold2"
    (tmp_path / "x_fold2.json").write_text("{}", encoding="utf-8")
    assert not already_done(tmp_path, run_name("x", 2, smoke=True))
    (tmp_path / "y_smoke_fold2.json").write_text("{}", encoding="utf-8")
    assert not already_done(tmp_path, run_name("y", 2))   # a smoke result never certifies the real run
