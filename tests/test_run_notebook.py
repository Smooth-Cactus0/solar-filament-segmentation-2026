"""run_notebook.py executes a notebook's code cells in one namespace, without Jupyter."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_notebook as rn  # noqa: E402


def _notebook(path: Path, cells: list[tuple[str, str, list[str]]]) -> Path:
    """Write a minimal nbformat-4 notebook of (cell_type, source, tags) cells."""
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {}, "cells": [
        {"cell_type": kind, "metadata": {"tags": tags} if tags else {},
         "source": src.splitlines(True), **({"outputs": [], "execution_count": None} if kind == "code" else {})}
        for kind, src, tags in cells]}
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


def _tiny(tmp_path: Path, extra: list[tuple[str, str, list[str]]] = ()) -> Path:
    return _notebook(tmp_path / "tiny.ipynb", [
        ("markdown", "# A tiny notebook", []),
        ("code", "import os\nBASE = 2", []),
        ("code", "LIMIT = None\nSCALE, NAME = 3, 'a'", ["parameters"]),
        ("code", "RESULT = BASE * SCALE\nCWD = os.getcwd()", []),
        *extra,
    ])


def test_code_cells_run_in_order_in_one_namespace_from_the_notebook_folder(tmp_path):
    ns = rn.run(_tiny(tmp_path), log=lambda m: None)
    assert ns["RESULT"] == 6 and ns["LIMIT"] is None and ns["NAME"] == "a"
    assert Path(ns["CWD"]).resolve() == tmp_path.resolve()


def test_overrides_apply_right_after_the_parameters_cell(tmp_path):
    ns = rn.run(_tiny(tmp_path), {"SCALE": 5, "LIMIT": 3}, log=lambda m: None)
    assert ns["RESULT"] == 10 and ns["LIMIT"] == 3


def test_an_override_must_name_a_variable_of_the_parameters_cell(tmp_path):
    with pytest.raises(ValueError, match="BASE"):
        rn.run(_tiny(tmp_path), {"BASE": 9}, log=lambda m: None)   # defined, but by another cell


def test_overrides_need_a_parameters_cell(tmp_path):
    nb = _notebook(tmp_path / "n.ipynb", [("code", "A = 1", [])])
    with pytest.raises(ValueError, match="parameters"):
        rn.run(nb, {"A": 2}, log=lambda m: None)


def test_parse_overrides_reads_literals_and_keeps_paths_as_strings():
    got = rn.parse_overrides(["LIMIT=3", "FROM_MAPS=True", "REF=None", "DEVICE='cuda'",
                              "WORK_DIR=C:/tmp/a b", "STEMS=['x', 'y']"])
    assert got == {"LIMIT": 3, "FROM_MAPS": True, "REF": None, "DEVICE": "cuda",
                   "WORK_DIR": "C:/tmp/a b", "STEMS": ["x", "y"]}
    for bad in ("=3", "LIMIT", "2X=1"):
        with pytest.raises(ValueError):
            rn.parse_overrides([bad])


def test_assigned_names_sees_tuple_and_annotated_targets():
    assert rn.assigned_names("A = 1\nB, (C, D) = 2, (3, 4)\nE: int = 5\nprint(F)") == {"A", "B", "C", "D", "E"}


def test_main_reports_the_failing_cell_and_its_source_line(tmp_path, capsys):
    nb = _tiny(tmp_path, [("code", "x = 1\nraise RuntimeError('boom')", [])])
    assert rn.main([str(nb), "--set", "SCALE=4"]) == 1
    out, err = capsys.readouterr()
    assert "overrides: SCALE=4" in out
    assert "[cell  4] FAILED" in out
    assert "raise RuntimeError('boom')" in err and "RuntimeError: boom" in err


def test_figures_are_saved_per_cell_and_closed(tmp_path, capsys):
    nb = _tiny(tmp_path, [("code", "import matplotlib.pyplot as plt\nplt.plot([0, 1])\nplt.show()", [])])
    figs = tmp_path / "figs"
    assert rn.main([str(nb), "--figures", str(figs)]) == 0
    assert [p.name for p in figs.iterdir()] == ["cell04_1.png"]
    import matplotlib.pyplot as plt
    assert plt.get_fignums() == []
    assert "all 4 code cells ran" in capsys.readouterr().out
