"""blend_caches.py averages compatible caches and refuses incompatible ones."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import blend_caches as bc  # noqa: E402
import cache_probs as cp  # noqa: E402
from filseg.cache import ProbCache  # noqa: E402

DISK = {"cx": 4.0, "cy": 4.0, "radius": 3.0}


def _cache(root: Path, value: float, scales=(8,), folds=(0, 1)) -> Path:
    m = {"config": f"{root.name}.yaml", "size": 8, "quant": cp.QUANT, "scales": list(scales),
         "oof": {}, "test": {}}
    for i, f in enumerate(folds):
        cp.write_map(root / "oof" / f"o{i}.png", np.full((8, 8), value, np.float32))
        m["oof"][f"o{i}"] = {"fold": f, **DISK}
    cp.write_map(root / "test" / "t0.png", np.full((8, 8), value, np.float32))
    m["test"]["t0"] = dict(DISK)
    cp.write_manifest(root, m)
    return root


def test_blend_averages_maps_and_records_its_members(tmp_path):
    a = _cache(tmp_path / "a", 0.2)
    b = _cache(tmp_path / "b", 0.8)
    out = tmp_path / "blend"
    bc.main(["--caches", str(a), str(b), "--weights", "3", "1", "--out", str(out)])
    c = ProbCache.open(out)
    assert c.stems("oof") == ["o0", "o1"] and c.stems("test") == ["t0"]
    assert c.manifest["scales"] == [8]
    np.testing.assert_allclose(c.probs("test", "t0"), 0.35, atol=1 / cp.QUANT)
    blend = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["blend"]
    assert [Path(e["cache"]).name for e in blend] == ["a", "b"]
    assert [e["weight"] for e in blend] == [0.75, 0.25]


def test_mixed_scales_are_refused_with_both_sets_named(tmp_path):
    a = _cache(tmp_path / "a", 0.2, scales=(8,))
    b = _cache(tmp_path / "b_ms", 0.8, scales=(6, 8))
    with pytest.raises(SystemExit, match="scales") as e:
        bc.main(["--caches", str(a), str(b), "--out", str(tmp_path / "blend")])
    assert "[8]" in str(e.value) and "[6, 8]" in str(e.value)
    assert not (tmp_path / "blend").exists()


def test_fold_disagreement_is_refused(tmp_path):
    a = _cache(tmp_path / "a", 0.2, folds=(0, 1))
    b = _cache(tmp_path / "b", 0.8, folds=(0, 2))
    with pytest.raises(SystemExit, match="fold"):
        bc.main(["--caches", str(a), str(b), "--out", str(tmp_path / "blend")])


def _manifest_cache(size: int, scales, folds=(0, 1)) -> ProbCache:
    """A cache known only by its manifest (check_compatible never reads a map)."""
    m = {"config": "m.yaml", "size": size, "quant": cp.QUANT, "scales": list(scales),
         "oof": {f"o{i}": {"fold": f, **DISK} for i, f in enumerate(folds)}, "test": {"t0": dict(DISK)}}
    return ProbCache(root=Path(f"c{size}_{'_'.join(map(str, scales))}"), manifest=m)


def test_single_scale_members_with_the_same_map_size_are_compatible():
    # scr_r2_coatnet1_1280 runs at 1280 and is cached at 1536 beside members that run at 1536.
    bc.check_compatible([_manifest_cache(1536, [1280]), _manifest_cache(1536, [1536])])


def test_members_with_different_map_sizes_are_refused():
    with pytest.raises(SystemExit, match="size") as e:
        bc.check_compatible([_manifest_cache(1536, [1536]), _manifest_cache(1024, [1024])])
    assert "1536" in str(e.value) and "1024" in str(e.value)


def test_a_multi_scale_member_beside_a_single_scale_one_is_refused():
    with pytest.raises(SystemExit, match="scales"):
        bc.check_compatible([_manifest_cache(1536, [1280, 1536]), _manifest_cache(1536, [1536])])
    bc.check_compatible([_manifest_cache(1536, [1280, 1536]), _manifest_cache(1536, [1024, 1536])])
