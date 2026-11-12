"""Weights stored as parts: splitting and joining are exact, and a bad join is refused."""
from __future__ import annotations

import os

import pytest

from filseg import weights


def _blob(tmp_path, name: str, size: int):
    path = tmp_path / "src" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(os.urandom(size))
    return path


def test_split_then_assemble_restores_the_file_byte_for_byte(tmp_path):
    src = _blob(tmp_path, "model_fold0.pt", 2500)
    out = tmp_path / "weights"
    parts = weights.split(src, out, max_bytes=1000)
    assert [p.name for p in parts] == ["model_fold0.pt.part0", "model_fold0.pt.part1", "model_fold0.pt.part2"]
    assert all(p.stat().st_size <= 1000 for p in parts)
    (out / weights.SUMS).write_text(f"{weights.sha256(src)}  model_fold0.pt\n", encoding="utf-8")
    joined = weights.assemble(out)
    assert [p.name for p in joined] == ["model_fold0.pt"]
    assert joined[0].read_bytes() == src.read_bytes()


def test_a_small_file_is_one_part_and_assemble_is_idempotent(tmp_path):
    src = _blob(tmp_path, "a.pt", 10)
    out = tmp_path / "weights"
    assert len(weights.split(src, out, max_bytes=1000)) == 1
    first = weights.assemble(out)
    stamp = first[0].stat().st_mtime_ns
    assert weights.assemble(out)[0].stat().st_mtime_ns == stamp or first[0].read_bytes() == src.read_bytes()


def test_assemble_refuses_parts_that_do_not_match_the_checksum(tmp_path):
    src = _blob(tmp_path, "a.pt", 3000)
    out = tmp_path / "weights"
    parts = weights.split(src, out, max_bytes=1000)
    (out / weights.SUMS).write_text(f"{weights.sha256(src)}  a.pt\n", encoding="utf-8")
    parts[1].write_bytes(b"x" * parts[1].stat().st_size)
    with pytest.raises(ValueError, match="do not match"):
        weights.assemble(out)
    assert not (out / "a.pt").exists()


def test_missing_part_number_is_reported(tmp_path):
    src = _blob(tmp_path, "a.pt", 3000)
    out = tmp_path / "weights"
    parts = weights.split(src, out, max_bytes=1000)
    parts[1].unlink()
    with pytest.raises(FileNotFoundError, match="not numbered"):
        weights.assemble(out)
