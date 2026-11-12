"""Build the (candidate x annotation pass) table the instance scorer learns from.

For every out-of-fold exposure in the probability cache: cut candidate components at
a permissive operating point, compute their features (``filseg.scorer``), and label
each one against every annotation pass of the exposure. One row per (candidate,
pass), because the official scorer judges each annotator's record separately -- a
filament drawn by one annotator in three is a positive in one row and a negative in
two, which is exactly the frequency the scorer should learn.

The candidate operating point is deliberately looser than the shipped one
(threshold 0.40 / min_area 300): with ``--min-area 50`` the table also holds the
small components the shipped rule drops, so the learned rule can decide both ways.
At the shipped settings the candidates are the shipped instances, which is how the
training script checks that its bookkeeping reproduces the known OOF PQ.

With ``--members`` (the member caches of a blend cache) the table also carries the
cross-model agreement features, and a ``<table>.members.json`` sidecar names the
member caches; ``train_scorer.py`` copies it beside the scorer so the submission
writer can insist on the same members.

Usage:
    PYTHONPATH=src python scripts/build_scorer_table.py --cache cache/exp2_convnext_1536 \\
        --threshold 0.40 --min-area 50 --out data/scorer/oof_thr0.40_ma50.csv.gz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path, PureWindowsPath

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import data as fdata, oof, postprocess as pp, scorer  # noqa: E402
from filseg.baseline import flat_ratio  # noqa: E402
from filseg.cache import OOF, ProbCache  # noqa: E402
from filseg.scorer import CNN_NAMES  # noqa: E402
from filseg.paths import IMAGE_SIZE, PROJECT_ROOT  # noqa: E402

ID_COLUMNS = ("stem", "fold", "pass_id", "n_passes", "cand", "n_cand_frame", "n_gt",
              "matched_gt", "best_iou", "label")


def log(m: str) -> None:
    print(m, flush=True)


def cut_candidates(probs_full, disk, threshold: float, min_area: int, bridge: float = 0.0):
    """Candidates at the operating point; with ``bridge > 0`` pieces sharing a region
    above that probability are one candidate (``hysteresis_instances`` in bridge mode).

    One definition for the table builder and the submission writer, so the scorer
    is trained and applied on identically cut candidates.
    """
    if bridge > 0:
        return pp.hysteresis_instances(probs_full, disk, threshold, bridge, min_area=min_area,
                                       mode=pp.BRIDGE)
    return scorer.candidates(probs_full, disk, threshold, min_area)


def feature_names(members) -> tuple[str, ...]:
    """The table's feature columns: v2's, plus the agreement block when members are given."""
    return (*scorer.FEATURE_NAMES, *scorer.AGREEMENT_NAMES) if members else scorer.FEATURE_NAMES


def cache_name(path) -> str:
    """A cache's directory name from a path string written on any OS (a blend manifest
    stores the path given to blend_caches.py, with Windows separators on the laptop)."""
    return PureWindowsPath(str(path)).name


def root_name(c) -> str:
    """The directory name of an opened cache."""
    return Path(c.root).resolve().name


def member_record(m) -> dict:
    """What identifies a member cache: its name, map size and TTA scales."""
    return {"name": root_name(m), "size": int(m.size), "scales": m.manifest.get("scales")}


def check_members(members, cache, split: str):
    """Members must be the blend's own caches: the cache's stems of ``split``, the cache's
    fold for every OOF stem, and -- when the cache's manifest lists its blend -- exactly
    the caches it was blended from."""
    want = set(cache.stems(split))
    oof_stems = cache.stems(OOF)
    for m in members:
        name = getattr(m, "root", m)
        if set(m.stems(split)) != want:
            raise SystemExit(f"member {name} does not have the same {split} stems "
                             f"as the blend cache")
        for stem in oof_stems:
            try:
                fold = m.fold(stem)
            except KeyError:
                raise SystemExit(f"member {name} has no OOF record of {stem}") from None
            if fold != cache.fold(stem):
                raise SystemExit(f"member {name}: OOF stem {stem} is fold {fold} there but fold "
                                 f"{cache.fold(stem)} in the blend cache -- not the same folds.csv")
    blend = (getattr(cache, "manifest", None) or {}).get("blend")
    if members and blend:
        listed = {cache_name(e["cache"]) for e in blend}
        given = {root_name(m) for m in members}
        if given != listed:
            raise SystemExit(f"--members {sorted(given)} are not the caches {cache.root} was blended "
                             f"from: {sorted(listed)} (its manifest.json)")
    return list(members)


def open_members(paths, cache, split: str) -> list[ProbCache]:
    """The member caches named on the command line, checked against the blend cache."""
    return check_members([ProbCache.open(p) for p in paths], cache, split)


def rows_for_exposure(cache: ProbCache, stem: str, threshold: float, min_area: int,
                      with_image: bool = True, bridge: float = 0.0, members=()) -> list[dict]:
    """All (candidate, pass) rows of one out-of-fold exposure."""
    probs_full = scorer.upsample(cache.probs(OOF, stem), IMAGE_SIZE)
    disk = cache.disk(OOF, stem)
    flat = flat_ratio(fdata.load_image(stem, "train")) if with_image else None
    insts = cut_candidates(probs_full, disk, threshold, min_area, bridge)
    member_maps = [scorer.upsample(m.probs(OOF, stem), IMAGE_SIZE) for m in members] or None
    feats = scorer.frame_features(insts, probs_full, disk, flat, member_maps)
    names = feature_names(members)
    pass_ids = oof.passes_of(stem)
    gts_passes = oof.gt_passes(stem)
    rows = []
    for pass_id, gts in zip(pass_ids, gts_passes):
        lab = scorer.label_against_pass(insts, gts)
        for c, f in enumerate(feats):
            rows.append({
                "stem": stem, "fold": cache.fold(stem), "pass_id": pass_id,
                "n_passes": len(pass_ids), "cand": c, "n_cand_frame": len(insts),
                "n_gt": lab.n_gt, "matched_gt": lab.matched_gt[c], "best_iou": lab.best_iou[c],
                "label": int(lab.matched_gt[c] >= 0),
                **{k: f[k] for k in names},
            })
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--threshold", type=float, default=0.40)
    ap.add_argument("--min-area", type=int, default=50)
    ap.add_argument("--bridge", type=float, default=0.0,
                    help="cut candidates in bridge mode: pieces sharing a region above this "
                         "probability are one candidate (0 = plain components)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-image", action="store_true", help="skip the flat-field features")
    ap.add_argument("--members", nargs="*", default=[],
                    help="member caches of a blend: adds the cross-model agreement features")
    ap.add_argument("--extra-features", default=None,
                    help="CSV of per-candidate columns to stack (the crop scorer's stem, cand, cnn_p, cnn_q, version)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    cache = ProbCache.open(args.cache)
    members = open_members(args.members, cache, OOF)
    stems = cache.stems(OOF)
    if args.limit:
        stems = stems[: args.limit]
    out = Path(args.out) if args.out else PROJECT_ROOT / "data" / "scorer" / default_table_name(
        args.threshold, args.min_area, args.bridge, bool(members))
    if not args.out and args.extra_features:
        out = out.with_name(out.name.replace('.csv.gz', '_cnn.csv.gz'))
    log(f"{len(stems)} exposures | candidates at thr {args.threshold} / min_area {args.min_area}"
        f"{f' / bridge {args.bridge}' if args.bridge > 0 else ''} | image features {not args.no_image} | members {len(members)}")

    rows: list[dict] = []
    passes: list[dict] = []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        rows.extend(rows_for_exposure(cache, stem, args.threshold, args.min_area,
                                      with_image=not args.no_image, bridge=args.bridge,
                                      members=members))
        # Every pass, including those of exposures with no candidate at all: their
        # filaments are false negatives under any rule, and a PQ computed from the
        # candidate rows alone would silently forget them.
        for pass_id, gts in zip(oof.passes_of(stem), oof.gt_passes(stem)):
            passes.append({"stem": stem, "fold": cache.fold(stem), "pass_id": pass_id,
                           "n_gt": len(gts)})
        if i % 50 == 0 or i == len(stems):
            log(f"  {i}/{len(stems)}  {len(rows)} rows  ({time.time() - t0:.0f}s)")

    df = pd.DataFrame(rows, columns=[*ID_COLUMNS, *feature_names(members)])
    cnn_side = None
    if args.extra_features:
        extra = pd.read_csv(args.extra_features)
        df = merge_extra(df, extra)[[*ID_COLUMNS, *feature_names(members), *CNN_NAMES]]
        cnn_side = cnn_path(out)
        cnn_side.write_text(json.dumps({"version": str(extra.version.iloc[0]) if "version" in extra else None,
                                        "source": args.extra_features, "n": int(len(extra))}), encoding="utf-8")
    else:
        cnn_path(out).unlink(missing_ok=True)    # a plain table must not inherit a stale CNN sidecar
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    passes_out = passes_path(out)
    pd.DataFrame(passes, columns=["stem", "fold", "pass_id", "n_gt"]).to_csv(passes_out, index=False)
    side = write_members_sidecar(out, cache, members)
    n_cand = df.drop_duplicates(["stem", "cand"]).shape[0]
    log(f"\nwrote {out}: {len(df)} rows, {n_cand} candidates, "
        f"{df.label.mean():.1%} positive rows, {(time.time() - t0) / 60:.1f} min")
    log(f"wrote {passes_out}: {len(passes)} annotation passes "
        f"({sum(p['n_gt'] for p in passes)} ground-truth filaments)")
    if cnn_side is not None:
        log(f"wrote {cnn_side}: CNN columns {list(CNN_NAMES)} from {args.extra_features}")
    if side is not None:
        log(f"wrote {side}: members {[root_name(m) for m in members]}")


def default_table_name(threshold: float, min_area: int, bridge: float, with_members: bool) -> str:
    """``oof_thr<t>_ma<a>[_bridge<b>][_agree].csv.gz``: an agreement table never takes the
    name of the plain one."""
    tag = f"oof_thr{threshold:.2f}_ma{min_area}" + (f"_bridge{bridge:.2f}" if bridge > 0 else "")
    return f"{tag}{'_agree' if with_members else ''}.csv.gz"


def _companion(table: Path, suffix_out: str) -> Path:
    name = table.name
    for suffix in (".csv.gz", ".csv"):
        if name.endswith(suffix):
            return table.with_name(name[: -len(suffix)] + suffix_out)
    return table.with_name(name + suffix_out)


def cnn_path(table) -> Path:
    """The companion file naming the CNN run behind a table with stacked CNN columns."""
    return _companion(Path(table), ".cnn.json")


def merge_extra(df: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    """The CNN columns joined on (stem, cand).

    The CSV must cover exactly this table's candidates, once each, from one run: a missing
    candidate, an extra one (another cut), a duplicated key (two runs appended) or two
    ``version`` values all mean the columns would not describe this table.
    """
    missing_cols = [c for c in ("stem", "cand", *CNN_NAMES) if c not in extra.columns]
    if missing_cols:
        raise SystemExit(f"--extra-features lacks column(s) {missing_cols}")
    if "version" in extra.columns and extra.version.nunique(dropna=False) > 1:
        raise SystemExit(f"--extra-features mixes CNN versions {sorted(extra.version.astype(str).unique())}")
    dup = extra[extra.duplicated(["stem", "cand"], keep=False)]
    if len(dup):
        raise SystemExit(f"--extra-features has {len(dup)} rows on duplicate (stem, cand) keys, e.g. "
                         f"{dup.stem.iloc[0]} cand {int(dup.cand.iloc[0])}: one row per candidate, from one run")
    want = set(zip(df.stem, df.cand.astype(int)))
    have = set(zip(extra.stem, extra.cand.astype(int)))
    gap = sorted(want - have)
    if gap:
        raise SystemExit(f"{len(gap)} candidates have no extra features, e.g. {gap[0][0]} cand {gap[0][1]}: "
                         "the CNN crops must come from this table's cut")
    surplus = sorted(have - want)
    if surplus:
        raise SystemExit(f"--extra-features has {len(surplus)} candidates this table lacks, e.g. "
                         f"{surplus[0][0]} cand {surplus[0][1]}: the CNN crops must come from this table's cut")
    return df.merge(extra[["stem", "cand", *CNN_NAMES]], on=["stem", "cand"], how="left", validate="many_to_one")


def passes_path(table: Path) -> Path:
    """The companion file listing every annotation pass behind ``table``."""
    return _companion(table, ".passes.csv")


def members_path(table: Path) -> Path:
    """The companion file naming the member caches behind an agreement table."""
    return _companion(table, ".members.json")


def write_members_sidecar(table: Path, cache, members) -> Path | None:
    """Name the member caches beside ``table``; without members, drop a stale sidecar so a
    plain table written to the same path is not mistaken for an agreement one."""
    side = members_path(Path(table))
    if not members:
        side.unlink(missing_ok=True)
        return None
    record = {"cache": root_name(cache), "members": [member_record(m) for m in members]}
    side.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    return side


if __name__ == "__main__":
    main()
