"""Write a submission CSV from cached test probability maps -- no GPU involved.

The sweep kernel re-ran inference for every operating point it emitted; with the
probability cache that cost is gone. This script thresholds the cached test maps
through ``sweep_submissions.instances_at`` (so a CSV written here is what the
sweep would have written at the same setting, up to the cache's quantisation) and
applies the same coverage guarantee, so every test image gets at least one row.

With ``--scorer-dir`` the fixed rules give way to the learned instance scorer:
candidates are cut at (``--threshold``, ``--min-area``) -- use the pair the scorer
was trained on -- and kept when ``p >= --accept-p`` or ``p*q >= --accept-pq``. A
``.scores.csv`` beside the submission records every candidate's p, q and verdict.

It is the delivery path for the post-processing experiments in the strategic
review -- extent probes, hysteresis, the instance scorer -- each of which adds its
own switch here rather than its own writer.

Usage:
    PYTHONPATH=src python scripts/submission_from_cache.py \\
        --cache cache/exp2_convnext_1536 --threshold 0.40 --min-area 300 --top-k 8 \\
        --out submissions/from_cache_thr0.40_ma300_top8.csv
    PYTHONPATH=src python scripts/submission_from_cache.py \\
        --cache cache/exp2_convnext_1536 --threshold 0.40 --min-area 50 \\
        --scorer-dir checkpoints/scorer --accept-p 0.35 --out submissions/scorer_v1_p0.35.csv
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filseg import data as fdata, postprocess as pp, rle, scorer as fscorer  # noqa: E402
from filseg.baseline import flat_ratio  # noqa: E402
from filseg.cache import TEST, ProbCache  # noqa: E402
from filseg.paths import IMAGE_SIZE  # noqa: E402
from filseg.scorer import CNN_NAMES  # noqa: E402
from filseg import tree as ftree  # noqa: E402
from build_tree_table import make_tree  # noqa: E402
from build_scorer_table import cut_candidates, member_record, open_members  # noqa: E402
from predict import PRIMARY, log, relaxed_instances  # noqa: E402
from sweep_submissions import instances_at  # noqa: E402

# When the scorer accepts nothing on a frame, the single most probable candidate is
# kept: coverage is mandatory, and a miss costs more than a false alarm.
SCORER_GUARANTEE = "scorer_guarantee"


def build_rows(cache: ProbCache, threshold: float, min_area: int, top_k: int | None,
               limit: int = 0):
    """``(stem, instances)`` for every cached test map, plus the fallback-arm tally."""
    stems = cache.stems(TEST)
    if limit:
        stems = stems[:limit]
    rows, arms = [], Counter()
    for stem in stems:
        res = instances_at(cache.probs(TEST, stem), threshold, min_area,
                           cache.disk(TEST, stem), size=IMAGE_SIZE, top_k=top_k,
                           guarantee=True)
        rows.append((stem, res.instances))
        arms[res.arm] += 1
    return rows, arms


def attach_extra(feats: list[dict], stem: str, cand_ids, extra: pd.DataFrame) -> None:
    """Add the CNN columns of (stem, cand) to each candidate's feature dict, in place.

    The CSV must hold exactly one row per candidate of this cut: a different count means the
    crops came from another cut (or another run was appended), and ids alone could still match.
    """
    sub = extra[extra.stem == stem].set_index("cand")
    if len(sub) != len(cand_ids):
        raise SystemExit(f"{stem}: --extra-features holds {len(sub)} rows, this cut has {len(cand_ids)} "
                         "candidates: rebuild the test crops from this cache")
    for f, c in zip(feats, cand_ids):
        if c not in sub.index:
            raise SystemExit(f"--extra-features has no row for {stem} cand {c}: rebuild the test crops from this cache")
        for k in CNN_NAMES:
            f[k] = float(sub.loc[c, k])


def check_scorer_cnn(model_dir, version: str, extra_csv: str | None) -> pd.DataFrame | None:
    """A scorer that learned the CNN block needs ``--extra-features`` from the same CNN run: the
    CSV's ``version`` must equal the stamped ``<version>@<run id>`` the scorer's ``.cnn.json``
    records. A scorer without the block ignores the option. Returns the loaded frame or None."""
    d = Path(model_dir)
    fj = d / f"{version}.features.json"
    feats = json.loads(fj.read_text(encoding="utf-8")) if fj.is_file() else []
    uses = all(k in feats for k in CNN_NAMES)
    if not uses:
        return None
    if not extra_csv:
        raise SystemExit(f"scorer {version} learned the CNN columns {CNN_NAMES}: pass --extra-features <test csv>")
    side = json.loads((d / f"{version}.cnn.json").read_text(encoding="utf-8"))
    if not side.get("version"):
        raise SystemExit(f"{version}.cnn.json names no CNN version: rebuild the table with --extra-features "
                         "from a stamped train_crop_scorer.py run and retrain the scorer")
    extra = pd.read_csv(extra_csv)
    got = sorted(set(extra.version.astype(str))) if "version" in extra else ["?"]
    if got != [str(side["version"])]:
        raise SystemExit(f"--extra-features is CNN version {got}, the scorer was trained with {side['version']!r}")
    return extra


def scored_rows(cache: ProbCache, model: fscorer.Scorer, threshold: float, min_area: int,
                accept_p: float | None, accept_pq: float | None, limit: int = 0,
                bridge: float = 0.0, candidate_bridge: float = 0.0, members=(),
                extra: pd.DataFrame | None = None):
    """``(stem, instances)`` kept by the learned scorer, plus the arm tally and scores.

    Candidates are cut at (``threshold``, ``min_area``) -- the table the scorer was
    trained on used the same pair, so features are in-distribution -- and kept when
    ``p >= accept_p`` (or ``p * q >= accept_pq``). A frame where nothing passes keeps
    its single most probable candidate. With ``bridge > 0`` the accepted pieces that
    share one region above that probability are rejoined into one instance;
    ``candidate_bridge > 0`` instead rejoins *before* scoring (the scorer must then
    have been trained on a table built with the same ``--bridge``).
    """
    if (accept_p is None) == (accept_pq is None):
        raise ValueError("give exactly one of accept_p / accept_pq")
    stems = cache.stems(TEST)
    if limit:
        stems = stems[:limit]
    rows, arms, scores = [], Counter(), []
    for stem in stems:
        probs = cache.probs(TEST, stem)
        probs_full = fscorer.upsample(probs, IMAGE_SIZE)
        disk = cache.disk(TEST, stem)
        insts = cut_candidates(probs_full, disk, threshold, min_area, candidate_bridge)
        if not insts:
            # Nothing clears the candidate threshold on this frame (the sweep saw 3 such
            # frames in 180). The scorer has nothing to rank, so predict.py's fallback
            # ladder covers the frame exactly as every earlier submission did.
            insts, arm = relaxed_instances(probs, threshold, min_area, disk, size=IMAGE_SIZE)
            arms[arm] += 1
            rows.append((stem, insts))
            continue
        flat = flat_ratio(fdata.load_image(stem, "test"))
        member_maps = [fscorer.upsample(m.probs(TEST, stem), IMAGE_SIZE) for m in members] or None
        feats = fscorer.frame_features(insts, probs_full, disk, flat, member_maps)
        if extra is not None:
            attach_extra(feats, stem, range(len(insts)), extra)
        p, q = model.predict(feats)
        score = p if accept_p is not None else p * q
        cut = accept_p if accept_p is not None else accept_pq
        keep = score >= cut
        if not keep.any():
            keep[int(np.argmax(score))] = True
            arms[SCORER_GUARANTEE] += 1
        else:
            arms[PRIMARY] += 1
        kept = [it for it, k in zip(insts, keep) if k]
        if bridge > 0:
            kept = pp.bridge_instances(kept, probs_full, bridge, disk)
        rows.append((stem, kept))
        scores.extend({"stem": stem, "cand": i, "area": it.area, "p": float(p[i]),
                       "q": float(q[i]), "kept": bool(keep[i])}
                      for i, it in enumerate(insts))
    return rows, arms, scores


def tree_rows(cache: ProbCache, model: fscorer.Scorer, c: float, settings: dict, threshold: float,
              limit: int = 0, spine_cache: ProbCache | None = None):
    """``(stem, instances)`` chosen from each test map's candidate tree by scorer v3.

    Every node gets p and q; the kept set maximises the summed ``p*q - c`` with no
    node inside another (``filseg.tree.select_nodes``), so it is pixel-disjoint. A
    frame where nothing pays keeps its single best non-duplicate node; a frame with
    no node at all falls back to predict.py's ladder, as the scorer path does.
    """
    stems = cache.stems(TEST)
    if limit:
        stems = stems[:limit]
    rows, arms, scores = [], Counter(), []
    for stem in stems:
        probs = cache.probs(TEST, stem)
        probs_full = fscorer.upsample(probs, IMAGE_SIZE)
        disk = cache.disk(TEST, stem)
        spine = fscorer.upsample(spine_cache.probs(TEST, stem), IMAGE_SIZE) if spine_cache is not None else None
        tree = make_tree(probs_full, disk, settings, spine)
        if len(tree) == 0:
            insts, arm = relaxed_instances(probs, threshold, settings["min_area"], disk, size=IMAGE_SIZE)
            arms[arm] += 1
            rows.append((stem, insts))
            continue
        feats = ftree.node_features(tree, probs_full, disk, flat_ratio(fdata.load_image(stem, "test")))
        p, q = model.predict(feats)
        gain = p * q - c
        gain[np.asarray(tree.duplicate)] = -np.inf
        kept = tree.select(gain)
        if kept:
            arms[PRIMARY] += 1
        else:
            live = [i for i in range(len(tree)) if not tree.duplicate[i]]
            kept = [max(live, key=lambda i: p[i] * q[i])]
            arms[SCORER_GUARANTEE] += 1
        insts = [tree.nodes[i] for i in kept]
        rows.append((stem, insts))
        scores.extend({"stem": stem, "node": i, "level": tree.levels[tree.level[i]],
                       "area": tree.nodes[i].area, "p": float(p[i]), "q": float(q[i]),
                       "kept": i in kept} for i in range(len(tree)))
    return rows, arms, scores


def check_scorer_members(model_dir, version: str, members) -> None:
    """A scorer trained on an agreement table names its member caches in
    ``<version>.members.json`` (copied there by train_scorer.py); the ``--members`` given
    here must be the same caches -- same names, map sizes and TTA scales -- or the
    agreement features would describe models the scorer never saw."""
    path = Path(model_dir) / f"{version}.members.json"
    if not path.is_file():
        return

    def key(r: dict) -> tuple:
        return r["name"], r.get("size"), json.dumps(r.get("scales"))

    want = sorted(json.loads(path.read_text(encoding="utf-8"))["members"], key=key)
    got = sorted((member_record(m) for m in members), key=key)
    if [key(r) for r in got] != [key(r) for r in want]:
        raise SystemExit(f"--members do not match the caches scorer {version} was trained with "
                         f"({path.name}):\n  given:   {got}\n  trained: {want}")


def tree_settings(model_dir, version: str, levels, clip, min_area: int) -> dict:
    """The tree-cut settings of the scorer's table; explicit flags must agree with them."""
    path = Path(model_dir) / f"{version}.tree.json"
    if not path.is_file():
        raise SystemExit(f"{path} is missing: retrain the scorer with train_scorer_v3.py "
                         "(it copies the table's tree-cut settings there)")
    side = json.loads(path.read_text(encoding="utf-8"))
    got = {"levels": tuple(sorted(side["levels"])), "clip": side["clip"], "min_area": side["min_area"],
           "children": side.get("children"), "candidates": side.get("candidates"),
           "params": side.get("params", {}), "spine_cache": side.get("spine_cache")}
    given = {"levels": None if levels is None else tuple(sorted(levels)), "clip": clip,
             "min_area": min_area}
    for k, v in given.items():
        if v is not None and v != got[k]:
            raise SystemExit(f"--tree {k} {v} differs from the scorer's tree.json ({path}): {got[k]}; "
                             "the scorer's features were built with the sidecar value")
    return got


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="cache root holding manifest.json")
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--min-area", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--scorer-dir", default=None,
                    help="directory of train_scorer.py fold models; enables the learned rule")
    ap.add_argument("--scorer-version", default="scorer_v1")
    ap.add_argument("--extra-features", default=None,
                    help="the crop scorer's test CSV (stem, cand, cnn_p, cnn_q, version), for a scorer trained with the CNN block")
    ap.add_argument("--accept-p", type=float, default=None, help="keep candidates with p >= this")
    ap.add_argument("--accept-pq", type=float, default=None, help="keep candidates with p*q >= this")
    ap.add_argument("--bridge", type=float, default=0.0,
                    help="scorer path: rejoin accepted pieces sharing a region above this probability")
    ap.add_argument("--candidate-bridge", type=float, default=0.0,
                    help="scorer path: rejoin BEFORE scoring, matching a table built with --bridge")
    ap.add_argument("--tree", action="store_true",
                    help="choose instances from the candidate tree with scorer v3 (needs --scorer-dir)")
    ap.add_argument("--tree-c", type=float, default=None, help="tree mode: keep nodes where p*q - c pays")
    ap.add_argument("--tree-levels", type=float, nargs="+", default=None,
                    help="default: the scorer's tree.json")
    ap.add_argument("--tree-clip", type=float, default=None,
                    help="default: the scorer's tree.json")
    ap.add_argument("--spine-cache", default=None,
                    help="--tree with a spine child generator: the spine-map cache")
    ap.add_argument("--members", nargs="*", default=[],
                    help="scorer path: the blend's member caches, for a scorer trained with agreement features")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="debug only; the CSV is then partial")
    args = ap.parse_args(argv)

    cache = ProbCache.open(args.cache)
    missing = cache.verify()
    if missing:
        raise SystemExit(f"cache is incomplete: {len(missing)} map(s) missing, e.g. {missing[:3]}")
    log(f"cache {cache.root} | {cache.config} | size {cache.size} | {len(cache.stems(TEST))} test maps")

    t0 = time.time()
    if args.tree:
        if not args.scorer_dir or args.tree_c is None:
            raise SystemExit("--tree needs --scorer-dir and --tree-c")
        if any(v is not None for v in (args.accept_p, args.accept_pq, args.top_k)) or args.bridge or args.candidate_bridge:
            raise SystemExit("--tree replaces --accept-p/--accept-pq/--top-k/--bridge/--candidate-bridge")
        model = fscorer.Scorer.load(args.scorer_dir, args.scorer_version)
        ts = tree_settings(args.scorer_dir, args.scorer_version, args.tree_levels, args.tree_clip,
                           args.min_area)
        log(f"tree mode: scorer {args.scorer_version} ({len(model.features)} features, "
            f"{len(model.p_models)} fold models) | levels {ts['levels']} | clip {ts['clip']} "
            f"| min_area {ts['min_area']} | c {args.tree_c}")
        if ts["children"]:
            log(f"split tree: children {ts['children']} | candidates {ts['candidates']}")
        if ts["children"] == "spine" and not args.spine_cache:
            raise SystemExit("this scorer's tree uses spine children: pass --spine-cache")
        spine = ProbCache.open(args.spine_cache, allow_spine=True) if args.spine_cache else None
        rows, arms, scores = tree_rows(cache, model, args.tree_c, ts, args.threshold, args.limit, spine)
        scores_out = Path(args.out).with_suffix(".scores.csv")
        pd.DataFrame(scores).to_csv(scores_out, index=False)
        log(f"node scores -> {scores_out} ({sum(s['kept'] for s in scores)}/{len(scores)} kept)")
    elif args.scorer_dir:
        if args.top_k is not None:
            raise SystemExit("--top-k and --scorer-dir are different rules; give one")
        model = fscorer.Scorer.load(args.scorer_dir, args.scorer_version)
        members = open_members(args.members, cache, TEST)
        check_scorer_members(args.scorer_dir, args.scorer_version, members)
        extra = check_scorer_cnn(args.scorer_dir, args.scorer_version, args.extra_features)
        if args.extra_features and extra is None:
            log("note: --extra-features given but this scorer has no CNN columns; ignored")
        needs = any(k in model.features for k in fscorer.AGREEMENT_NAMES)
        if needs and not members:
            raise SystemExit(f"scorer {args.scorer_version} was trained with agreement features: pass --members")
        if members and not needs:
            log("note: --members given but this scorer has no agreement features; they are ignored")
        rule = (f"p >= {args.accept_p}" if args.accept_p is not None else f"p*q >= {args.accept_pq}")
        if args.bridge > 0:
            rule += f", then bridge at {args.bridge}"
        cand = f"thr {args.threshold} / min_area {args.min_area}" + (
            f" / bridge {args.candidate_bridge}" if args.candidate_bridge > 0 else "")
        log(f"scorer {args.scorer_version}: {len(model.p_models)} fold models | candidates at "
            f"{cand} | keep {rule} | members {len(members)}")
        rows, arms, scores = scored_rows(cache, model, args.threshold, args.min_area,
                                         args.accept_p, args.accept_pq, args.limit,
                                         bridge=args.bridge, candidate_bridge=args.candidate_bridge,
                                         members=members, extra=extra)
        scores_out = Path(args.out).with_suffix(".scores.csv")
        pd.DataFrame(scores).to_csv(scores_out, index=False)
        log(f"candidate scores -> {scores_out} ({sum(s['kept'] for s in scores)}/{len(scores)} kept)")
    else:
        rows, arms = build_rows(cache, args.threshold, args.min_area, args.top_k, args.limit)
    empty = [stem for stem, insts in rows if not insts]
    if empty:
        raise SystemExit(f"{len(empty)} image(s) have no instances even after the fallback "
                         f"ladder, e.g. {empty[:3]}; refusing to write an incomplete CSV")

    out = Path(args.out)
    if args.limit:
        out = out.with_name(f"PARTIAL_limit{args.limit}_{out.name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = rle.write_submission(rows, out)
    log(f"wrote {out}: {n} rows over {len(rows)} images "
        f"({n / len(rows):.2f} inst/img; train GT 7.11) in {time.time() - t0:.0f}s")
    log(f"arms: {arms.get(PRIMARY, 0)} primary, {sum(arms.values()) - arms.get(PRIMARY, 0)} fallback")


if __name__ == "__main__":
    main()
