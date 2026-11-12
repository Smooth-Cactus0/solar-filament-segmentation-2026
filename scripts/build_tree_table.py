"""Build the (tree node x annotation pass) table scorer v3 learns from.

For every out-of-fold exposure in the cache: build the candidate tree
(``filseg.tree``), compute every node's features, and label each node against every
annotation pass (IoU > 0.5, non-unique). One row per (node, pass), as
``build_scorer_table.py`` does for v2's candidates; the node index is stored as
``cand`` so the v2 PQ bookkeeping (``train_scorer.pooled_pq``) applies unchanged, and
``parent`` / ``level_idx`` / ``duplicate`` let the evaluator rebuild each tree.

Usage:
    PYTHONPATH=src python scripts/build_tree_table.py --cache cache/blend_exp2_exp8 \\
        --out data/scorer/tree_blend_full.csv.gz
    PYTHONPATH=src python scripts/build_tree_table.py --cache cache/blend_exp2_exp8 \\
        --levels 0.40 --out data/scorer/tree_blend_l040.csv.gz
    PYTHONPATH=src python scripts/build_tree_table.py --cache cache/blend_exp2_exp8 \
        --children watershed --out data/scorer/tree_blend_ws.csv.gz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_scorer_table import cut_candidates, passes_path  # noqa: E402
from filseg import data as fdata, oof, scorer  # noqa: E402
from filseg.baseline import flat_ratio  # noqa: E402
from filseg.cache import OOF, ProbCache  # noqa: E402
from filseg.paths import IMAGE_SIZE  # noqa: E402
from filseg import split as fsplit  # noqa: E402
from filseg.split import EXTENT_MIN_AREA, GENERATORS, default_levels, split_tree  # noqa: E402
from filseg.tree import LEVELS, MIN_AREA, V3_FEATURES, Tree, build_tree, node_features, settings_path  # noqa: E402

ID_COLUMNS = ("stem", "fold", "pass_id", "n_passes", "cand", "level_idx", "parent", "duplicate",
              "n_nodes", "n_gt", "matched_gt", "best_iou", "label")

_CACHE: ProbCache | None = None
_SPINE: ProbCache | None = None
_OPTS: dict = {}


def log(m: str) -> None:
    print(m, flush=True)


def generator_params(children: str, min_area: int, extent_min_area: int) -> dict:
    """Every constant the child generator runs with, written into the sidecar's ``params`` and
    read back by :func:`make_tree`, so a later change to a code default cannot silently change
    the children a trained scorer sees at submission."""
    if children == "watershed":
        return {"min_parent": fsplit.WS_MIN_PARENT, "sigma": fsplit.WS_SIGMA,
                "min_distance": fsplit.WS_MIN_DISTANCE, "saddle_ratio": fsplit.WS_SADDLE, "min_area": min_area}
    if children == "extent":
        return {"levels": list(fsplit.EXTENT_LEVELS), "min_parent": extent_min_area, "min_area": min_area}
    if children == "spine":
        return {"threshold": fsplit.SPINE_THRESHOLD, "min_px": fsplit.SPINE_MIN_PX, "min_area": min_area}
    raise ValueError(f"children must be one of {GENERATORS}, got {children!r}")


def make_tree(probs_full, disk, opts: dict, spine=None) -> Tree:
    """The exposure's tree: a split tree under the production candidates when ``opts['children']``
    is set, else the nested-threshold tree of filseg.tree. ``opts['params']`` go to the child
    generator as keywords (see :func:`generator_params`)."""
    if opts.get("children"):
        c = opts["candidates"]
        cands = cut_candidates(probs_full, disk, c["threshold"], c["min_area"], c["bridge"])
        return split_tree(probs_full, cands, opts["children"], opts.get("levels"), spine=spine,
                          **opts.get("params", {}))
    return build_tree(probs_full, disk, opts["levels"], opts["min_area"], opts["clip"])


def tree_rows(stem, fold, tree: Tree, probs_full, disk, flat, pass_ids, gts_passes) -> list[dict]:
    """All (node, pass) rows of one exposure's tree."""
    if len(tree) == 0:
        return []
    feats = node_features(tree, probs_full, disk, flat)
    rows = []
    for pass_id, gts in zip(pass_ids, gts_passes):
        lab = scorer.label_against_pass(tree.nodes, gts)
        for c, f in enumerate(feats):
            rows.append({
                "stem": stem, "fold": fold, "pass_id": pass_id, "n_passes": len(pass_ids),
                "cand": c, "level_idx": tree.level[c], "parent": tree.parent[c],
                "duplicate": tree.duplicate[c], "n_nodes": len(tree), "n_gt": lab.n_gt,
                "matched_gt": lab.matched_gt[c], "best_iou": lab.best_iou[c],
                "label": int(lab.matched_gt[c] >= 0),
                **{k: f[k] for k in V3_FEATURES},
            })
    return rows


def _init(cache_root: str, opts: dict) -> None:
    global _CACHE, _SPINE, _OPTS
    _CACHE, _OPTS = ProbCache.open(cache_root), opts
    _SPINE = ProbCache.open(opts["spine_cache"], allow_spine=True) if opts.get("spine_cache") else None


def _exposure(stem: str) -> tuple[list[dict], list[dict]]:
    probs_full = scorer.upsample(_CACHE.probs(OOF, stem), IMAGE_SIZE)
    disk = _CACHE.disk(OOF, stem)
    flat = flat_ratio(fdata.load_image(stem, "train"))
    spine = scorer.upsample(_SPINE.probs(OOF, stem), IMAGE_SIZE) if _SPINE is not None else None
    tree = make_tree(probs_full, disk, _OPTS, spine)
    pass_ids, gts = oof.passes_of(stem), oof.gt_passes(stem)
    rows = tree_rows(stem, _CACHE.fold(stem), tree, probs_full, disk, flat, pass_ids, gts)
    passes = [{"stem": stem, "fold": _CACHE.fold(stem), "pass_id": p, "n_gt": len(g)}
              for p, g in zip(pass_ids, gts)]
    return rows, passes


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--levels", type=float, nargs="+", default=None,
                    help="nested-threshold levels (default filseg.tree.LEVELS; ignored with --children)")
    ap.add_argument("--min-area", type=int, default=MIN_AREA, help="smallest node (and child) in px")
    ap.add_argument("--clip", type=float, default=0.98)
    ap.add_argument("--children", choices=GENERATORS, default=None,
                    help="a two-level tree: the production candidates as parents, this generator's children")
    ap.add_argument("--threshold", type=float, default=0.40, help="production candidates (with --children)")
    ap.add_argument("--cand-min-area", type=int, default=50)
    ap.add_argument("--bridge", type=float, default=0.20)
    ap.add_argument("--extent-min-area", type=int, default=EXTENT_MIN_AREA,
                    help="--children extent: parents at or above this area get children")
    ap.add_argument("--spine-cache", default=None, help="--children spine: the spine-map cache")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.children == "spine" and not args.spine_cache:
        raise SystemExit("--children spine needs --spine-cache")

    stems = ProbCache.open(args.cache).stems(OOF)[: args.limit or None]
    if args.children:
        params = generator_params(args.children, args.min_area, args.extent_min_area)
        opts = {"children": args.children, "levels": list(default_levels(args.children)),
                "candidates": {"threshold": args.threshold, "min_area": args.cand_min_area, "bridge": args.bridge},
                "params": params, "spine_cache": args.spine_cache, "min_area": args.min_area, "clip": args.clip}
    else:
        opts = {"levels": tuple(sorted(args.levels or LEVELS)), "min_area": args.min_area, "clip": args.clip}
    log(f"{len(stems)} exposures | {json.dumps({k: v for k, v in opts.items() if k != 'spine_cache'})}")
    t0 = time.time()
    rows, passes = [], []
    with Pool(args.workers, initializer=_init, initargs=(args.cache, opts)) as pool:
        for i, (r, p) in enumerate(pool.imap_unordered(_exposure, stems, chunksize=4), 1):
            rows.extend(r)
            passes.extend(p)
            if i % 50 == 0 or i == len(stems):
                log(f"  {i}/{len(stems)}  {len(rows)} rows  ({time.time() - t0:.0f}s)")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=[*ID_COLUMNS, *V3_FEATURES]).sort_values(["stem", "pass_id", "cand"])
    df.to_csv(out, index=False)
    pd.DataFrame(passes, columns=["stem", "fold", "pass_id", "n_gt"]).sort_values(
        ["stem", "pass_id"]).to_csv(passes_path(out), index=False)
    settings = {**opts, "levels": list(opts["levels"])}
    settings_path(out).write_text(json.dumps(settings), encoding="utf-8")
    nodes = df.drop_duplicates(["stem", "cand"])
    log(f"\nwrote {out}: {len(df)} rows, {len(nodes)} nodes ({int((nodes.level_idx > 0).sum())} children, "
        f"{int((~nodes.duplicate.astype(bool)).sum())} non-duplicate), "
        f"{df.label.mean():.1%} positive rows, {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
