"""build_tree_table.tree_rows on a synthetic exposure (no cache, no images)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

from build_tree_table import ID_COLUMNS, tree_rows  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.metric import from_mask  # noqa: E402
from filseg.tree import V3_FEATURES, build_tree  # noqa: E402
from test_tree import two_blobs  # noqa: E402

DISK = Disk(cx=100.0, cy=60.0, radius=1000.0)


def test_rows_label_every_node_against_every_pass():
    p = two_blobs()
    a = np.zeros(p.shape, bool)
    a[40:60, 20:80] = True                    # pass 1 drew blob A at its 0.40 extent
    gts = [[from_mask(a)], []]                # pass 2 drew nothing
    tree = build_tree(p, DISK, (0.25, 0.40, 0.60), 10, 0.98)
    rows = tree_rows("S", 3, tree, p, DISK, None, ["p1", "p2"], gts)
    assert len(rows) == 5 * 2
    assert set(ID_COLUMNS) | set(V3_FEATURES) <= set(rows[0])
    r1 = {r["cand"]: r for r in rows if r["pass_id"] == "p1"}
    assert r1[1]["label"] == 1 and r1[1]["best_iou"] == 1.0       # node 1 is blob A at 0.40
    assert r1[0]["label"] == 0 and r1[2]["label"] == 0
    assert r1[3]["label"] == 0                                     # the 0.60 core: IoU 1/3
    assert r1[1]["parent"] == 0 and r1[0]["parent"] == -1
    assert all(r["label"] == 0 for r in rows if r["pass_id"] == "p2")
    assert all(r["n_nodes"] == 5 and r["fold"] == 3 and r["n_passes"] == 2 for r in rows)


def test_exposure_without_nodes_gives_no_rows():
    z = np.zeros((50, 50), np.float32)
    rows = tree_rows("S", 0, build_tree(z, DISK, (0.25, 0.40), 10, 0.98), z, DISK, None, ["p1"], [[]])
    assert rows == []


def test_make_tree_builds_a_split_tree_under_production_candidates():
    from build_tree_table import make_tree
    p = two_blobs()
    opts = {"children": "watershed", "candidates": {"threshold": 0.25, "min_area": 10, "bridge": 0.0},
            "params": {"min_area": 10, "min_parent": 100, "min_distance": 5}}
    t = make_tree(p, DISK, opts)
    assert t.level == [0, 1, 1] and t.parent == [-1, 0, 0]
    flat = make_tree(p, DISK, {"levels": (0.25, 0.40, 0.60), "min_area": 10, "clip": 0.98})
    assert len(flat) == 5


def test_generator_params_record_the_effective_constants_and_drive_make_tree():
    from build_tree_table import generator_params, make_tree
    from filseg import split
    assert generator_params("watershed", 50, 4098) == {
        "min_parent": split.WS_MIN_PARENT, "sigma": split.WS_SIGMA, "min_distance": split.WS_MIN_DISTANCE,
        "saddle_ratio": split.WS_SADDLE, "min_area": 50}
    assert generator_params("extent", 50, 4098) == {"levels": list(split.EXTENT_LEVELS), "min_parent": 4098,
                                                    "min_area": 50}
    assert generator_params("spine", 50, 4098) == {"threshold": split.SPINE_THRESHOLD,
                                                   "min_px": split.SPINE_MIN_PX, "min_area": 50}
    q = np.zeros((60, 100), np.float32)
    q[10:50, 10:90] = 0.45
    q[20:40, 20:50] = 0.8
    opts = {"children": "extent", "levels": list(split.default_levels("extent")),
            "candidates": {"threshold": 0.40, "min_area": 10, "bridge": 0.0},
            "params": generator_params("extent", 10, 100)}
    assert make_tree(q, DISK, opts).level == [0, 1, 2]
    recorded = {**opts, "params": {**opts["params"], "levels": [0.55]}}    # the sidecar's value wins
    assert make_tree(q, DISK, recorded).level == [0, 1]
