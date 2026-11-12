"""Tests for filseg.config: `base:` inheritance for experiment configs."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.config import deep_merge, load_config  # noqa: E402


def _w(p: Path, text: str) -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_plain_config_loads_unchanged(tmp_path):
    p = _w(tmp_path / "a.yaml", "name: a\nlr: 0.001\neval:\n  tta: true\n")
    assert load_config(p) == {"name": "a", "lr": 0.001, "eval": {"tta": True}}


def test_child_overrides_scalars_and_merges_dicts(tmp_path):
    _w(tmp_path / "base.yaml", "name: base\nlr: 0.001\neval:\n  tta: true\n  thresholds: [0.3, 0.5]\n")
    child = _w(tmp_path / "child.yaml",
               "base: base.yaml\nname: child\neval:\n  thresholds: [0.4]\naugment:\n  rot90: 0.75\n")
    cfg = load_config(child)
    assert cfg == {"name": "child", "lr": 0.001,
                   "eval": {"tta": True, "thresholds": [0.4]},   # list replaced, dict merged
                   "augment": {"rot90": 0.75}}
    assert "base" not in cfg


def test_chains_resolve_relative_to_each_file(tmp_path):
    (tmp_path / "sub").mkdir()
    _w(tmp_path / "root.yaml", "a: 1\nb: 1\nc: 1\n")
    _w(tmp_path / "sub" / "mid.yaml", "base: ../root.yaml\nb: 2\n")
    leaf = _w(tmp_path / "sub" / "leaf.yaml", "base: mid.yaml\nc: 3\n")
    assert load_config(leaf) == {"a": 1, "b": 2, "c": 3}


def test_cycles_are_refused(tmp_path):
    _w(tmp_path / "x.yaml", "base: y.yaml\n")
    _w(tmp_path / "y.yaml", "base: x.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "x.yaml")


def test_missing_base_names_the_file(tmp_path):
    p = _w(tmp_path / "c.yaml", "base: nope.yaml\n")
    with pytest.raises(FileNotFoundError, match="nope.yaml"):
        load_config(p)


def test_deep_merge_does_not_mutate_inputs():
    a = {"x": {"y": 1}}
    b = {"x": {"z": 2}}
    out = deep_merge(a, b)
    assert out == {"x": {"y": 1, "z": 2}} and a == {"x": {"y": 1}} and b == {"x": {"z": 2}}


def test_every_repo_config_still_loads():
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        cfg = load_config(p)
        assert isinstance(cfg, dict) and cfg.get("name"), p.name
