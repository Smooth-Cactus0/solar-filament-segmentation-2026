"""Train the instance scorer out-of-fold and measure what its acceptance rule buys.

Two gradient-boosted models on the table from ``build_scorer_table.py``:

* a classifier for ``p`` = P(a given annotator's record contains a filament this
  candidate matches at IoU > 0.5);
* a regressor for ``q`` = E[IoU | matched], fitted on the matched rows only.

Both are trained fold-by-fold on the exposure folds the segmentation model itself
was trained with, so every candidate's ``p`` and ``q`` come from a model that never
saw its exposure -- the acceptance PQ below is therefore an honest out-of-fold
number, comparable with the 0.4158 of the shipped rule.

The acceptance rule follows from the metric: adding a candidate adds 0.5 to the PQ
denominator whether it matches (TP +1, FN -1) or not (FP +0.5), and ``p * q`` to the
numerator in expectation, so it pays iff ``p * q > 0.5 * PQ``. The constant is swept
here rather than fixed, and the plain ``p > tau`` rule is reported alongside.

Writes ``docs/eda_stats/scorer_v1.json`` and the fold models under
``checkpoints/scorer/``. The best ``p*q >= c`` and ``p >= tau`` rules also carry their
PQ per exposure fold (``per_fold_pq``), which the blend ladder's fold rule reads. An
agreement table's ``.members.json`` sidecar is copied to
``checkpoints/scorer/<version>.members.json``, so the submission writer can insist on
the same member caches.

Usage:
    PYTHONPATH=src python scripts/train_scorer.py --table data/scorer/oof_thr0.40_ma50.csv.gz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_scorer_table import cnn_path, members_path, passes_path  # noqa: E402
from filseg.paths import EDA_STATS, PROJECT_ROOT  # noqa: E402
from filseg.gate import GATE_NAMES  # noqa: E402
from filseg.scorer import AGREEMENT_NAMES, CNN_NAMES, FEATURE_NAMES  # noqa: E402


def table_features(columns) -> list[str]:
    """v2's features, plus the agreement block and the CNN block when the table carries all of each."""
    cols = set(columns)
    out = list(FEATURE_NAMES)
    for block, label, hint in ((AGREEMENT_NAMES, "agreement", "--members"), (CNN_NAMES, "cnn", "--extra-features"),
                               (GATE_NAMES, "gate", "build_gate_table.py")):
        have = [c for c in block if c in cols]
        if have and len(have) != len(block):
            raise ValueError(f"table has a partial {label} block {have}: rebuild it with {hint}")
        if have:
            out += list(block)
    return out

SHIPPED_MIN_AREA = 300
SHIPPED_TOP_K = 8

# num_threads is pinned: with the default (all cores) one fold of the 2026-09-23 run
# stalled for ~10 minutes on this Windows laptop (OpenMP oversubscription), where the
# other folds took a second each.
CLF_PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=4, verbose=-1, seed=20260923)
REG_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=40,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=4, verbose=-1, seed=20260923)


def log(m: str) -> None:
    print(m, flush=True)


# --- pooled PQ from the table ------------------------------------------------------

def pooled_pq(rows: pd.DataFrame, accept: np.ndarray, n_gt_total: int | None = None) -> dict:
    """Official pooled PQ when the candidates flagged in ``accept`` (per row) are kept.

    Non-unique matching, as the scorer does it: every accepted row with a match is a
    TP; a GT is an FN when no accepted candidate of its pass matched it.
    ``n_gt_total`` is the number of ground-truth filaments over *all* passes, from the
    companion passes file; without it, passes whose exposure produced no candidate
    are invisible here and their filaments would not be counted as FN.
    """
    acc = rows[accept]
    tp = int((acc.matched_gt >= 0).sum())
    fp = int((acc.matched_gt < 0).sum())
    iou_sum = float(acc.best_iou[acc.matched_gt >= 0].sum())
    n_gt = n_gt_total if n_gt_total is not None else rows.groupby(["stem", "pass_id"]).n_gt.first().sum()
    matched = acc[acc.matched_gt >= 0].groupby(["stem", "pass_id"]).matched_gt.nunique().sum()
    fn = int(n_gt - matched)
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"pq": iou_sum / denom if denom else 1.0, "sq": iou_sum / tp if tp else 0.0,
            "rq": tp / denom if denom else 1.0, "tp": tp, "fp": fp, "fn": fn,
            "n_accepted": int(accept[rows.drop_duplicates(["stem", "cand"]).index].sum())}


def per_fold_pq(rows: pd.DataFrame, accept: np.ndarray,
                n_gt_by_fold: dict[int, int] | None = None) -> dict[str, float]:
    """``pooled_pq`` within each exposure fold: ``{"<fold>": pq}``.

    ``n_gt_by_fold`` is each fold's ground-truth count over all its passes (the passes
    file's ``fold`` column), for the same reason ``pooled_pq`` takes ``n_gt_total``.
    """
    accept = np.asarray(accept, dtype=bool)
    folds = {int(f) for f in rows.fold.unique()} | set(n_gt_by_fold or {})
    out = {}
    for k in sorted(folds):
        m = (rows.fold == k).to_numpy()
        n_gt = None if n_gt_by_fold is None else int(n_gt_by_fold.get(k, 0))
        out[str(k)] = pooled_pq(rows[m].reset_index(drop=True), accept[m], n_gt)["pq"]
    return out


def copy_cnn_sidecar(table: Path, model_dir: Path, version: str) -> dict | None:
    """Copy the table's ``.cnn.json`` beside the scorer as ``<version>.cnn.json``; without one,
    remove a stale copy and return None."""
    src = cnn_path(Path(table))
    dst = Path(model_dir) / f"{version}.cnn.json"
    if not src.is_file():
        dst.unlink(missing_ok=True)
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text(encoding="utf-8")
    dst.write_text(text, encoding="utf-8")
    return json.loads(text)


def copy_members_sidecar(table: Path, model_dir: Path, version: str) -> list[dict] | None:
    """Copy the table's ``.members.json`` beside the scorer as ``<version>.members.json``
    and return its member list; without one, remove a stale copy from an earlier run of
    the same version and return None."""
    src = members_path(table)
    dst = model_dir / f"{version}.members.json"
    if not src.is_file():
        dst.unlink(missing_ok=True)
        return None
    text = src.read_text(encoding="utf-8")
    dst.write_text(text, encoding="utf-8")
    return json.loads(text)["members"]


def top_k_mask(cands: pd.DataFrame, k: int, area_min: int) -> pd.Series:
    """Per-candidate: kept by the shipped top-k rule (area gate, then k best by mean prob)."""
    ok = cands.area >= area_min
    rank = cands[ok].groupby("stem").p_mean.rank(ascending=False, method="first")
    keep = pd.Series(False, index=cands.index)
    keep[rank.index] = rank <= k
    return keep


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--rounds", type=int, default=600)
    ap.add_argument("--version", default="scorer_v1",
                    help="model file prefix; Scorer.load(model_dir, version) finds them by it")
    ap.add_argument("--out", default=None, help="results JSON (default docs/eda_stats/<version>.json)")
    ap.add_argument("--model-dir", default=str(PROJECT_ROOT / "checkpoints" / "scorer"))
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = str(EDA_STATS / f"{args.version}.json")

    rows = pd.read_csv(args.table)
    feats = table_features(rows.columns)
    cands = rows.drop_duplicates(["stem", "cand"]).copy()
    passes_file = passes_path(Path(args.table))
    n_gt_by_fold = None
    if passes_file.is_file():
        passes = pd.read_csv(passes_file)
        n_gt_total = int(passes.n_gt.sum())
        if "fold" in passes.columns:
            n_gt_by_fold = {int(k): int(v) for k, v in passes.groupby("fold").n_gt.sum().items()}
        log(f"ground truth: {n_gt_total} filaments over all passes ({passes_file.name})")
    else:
        n_gt_total = None
        log(f"!! {passes_file.name} missing: passes without candidates will not count as FN")
    log(f"{len(rows)} rows, {len(cands)} candidates, {rows.label.mean():.1%} positive rows, "
        f"folds {sorted(int(f) for f in rows.fold.unique())}")

    def pq(accept) -> dict:
        return pooled_pq(rows, np.asarray(accept, dtype=bool), n_gt_total)

    # --- out-of-fold p and q per candidate --------------------------------------
    p_hat = pd.Series(np.nan, index=cands.index)
    q_hat = pd.Series(np.nan, index=cands.index)
    importance = np.zeros(len(feats))
    t0 = time.time()
    for k in sorted(rows.fold.unique()):
        tr = rows[rows.fold != k]
        te = cands[cands.fold == k]
        clf = lgb.train(CLF_PARAMS, lgb.Dataset(tr[feats], tr.label), num_boost_round=args.rounds)
        p_hat[te.index] = clf.predict(te[feats])
        pos = tr[tr.label == 1]
        reg = lgb.train(REG_PARAMS, lgb.Dataset(pos[feats], pos.best_iou), num_boost_round=args.rounds)
        q_hat[te.index] = np.clip(reg.predict(te[feats]), 0.0, 1.0)
        importance += clf.feature_importance("gain")
        Path(args.model_dir).mkdir(parents=True, exist_ok=True)
        clf.save_model(str(Path(args.model_dir) / f"{args.version}_p_fold{k}.txt"))
        reg.save_model(str(Path(args.model_dir) / f"{args.version}_q_fold{k}.txt"))
        log(f"  fold {k}: trained on {len(tr)} rows, scored {len(te)} candidates ({time.time() - t0:.0f}s)")
    (Path(args.model_dir) / f"{args.version}.features.json").write_text(
        json.dumps(feats), encoding="utf-8")
    members = copy_members_sidecar(Path(args.table), Path(args.model_dir), args.version)
    cnn_side = copy_cnn_sidecar(Path(args.table), Path(args.model_dir), args.version)
    if members is None and any(f in AGREEMENT_NAMES for f in feats):
        log(f"!! {members_path(Path(args.table)).name} missing: the submission writer cannot "
            "check --members against this scorer's member caches")
    cands["p_hat"] = p_hat
    cands["q_hat"] = q_hat

    # Broadcast candidate-level scores to the pass-level rows.
    key = ["stem", "cand"]
    rows = rows.merge(cands[key + ["p_hat", "q_hat"]], on=key, how="left")

    # Calibration of p: mean label vs mean p_hat, in deciles.
    rows["p_bin"] = pd.qcut(rows.p_hat, 10, labels=False, duplicates="drop")
    calib = rows.groupby("p_bin").agg(p_hat=("p_hat", "mean"), label=("label", "mean"),
                                      n=("label", "size")).reset_index(drop=True)

    # --- acceptance rules -------------------------------------------------------
    results: dict[str, dict] = {}
    results["shipped_thr0.40_ma300"] = pq(rows.area >= SHIPPED_MIN_AREA)
    keep8 = top_k_mask(cands, SHIPPED_TOP_K, SHIPPED_MIN_AREA)
    rows_keep8 = rows.merge(keep8.rename("keep").to_frame().join(cands[key]), on=key, how="left").keep
    results["shipped_top8"] = pq(rows_keep8)
    results["all_candidates"] = pq(np.ones(len(rows), dtype=bool))

    p_sweep, p_masks = {}, {}
    for tau in np.round(np.arange(0.10, 0.80, 0.05), 2):
        p_masks[f"{tau:.2f}"] = np.asarray(rows.p_hat >= tau, dtype=bool)
        p_sweep[f"{tau:.2f}"] = pq(p_masks[f"{tau:.2f}"])
    pq_sweep, pq_masks = {}, {}
    for c in np.round(np.arange(0.06, 0.40, 0.02), 2):
        pq_masks[f"{c:.2f}"] = np.asarray(rows.p_hat * rows.q_hat >= c, dtype=bool)
        pq_sweep[f"{c:.2f}"] = pq(pq_masks[f"{c:.2f}"])
    best_p = max(p_sweep.items(), key=lambda kv: kv[1]["pq"])
    best_pq = max(pq_sweep.items(), key=lambda kv: kv[1]["pq"])
    # The same rule, fold by fold: what the fold rule for fitted blend weights reads.
    best_p_folds = per_fold_pq(rows, p_masks[best_p[0]], n_gt_by_fold)
    best_pq_folds = per_fold_pq(rows, pq_masks[best_pq[0]], n_gt_by_fold)

    # Oracle: keep exactly the candidates that match in that pass (upper bound on any scorer).
    oracle = pq(rows.matched_gt >= 0)

    imp = sorted(zip(feats, importance / importance.sum()), key=lambda t: -t[1])
    res = {
        "version": args.version,
        "table": str(args.table), "n_rows": int(len(rows)), "n_candidates": int(len(cands)),
        "positive_row_frac": float(rows.label.mean()), "rounds": args.rounds,
        "features": feats,
        "clf_params": CLF_PARAMS, "reg_params": REG_PARAMS,
        "rules": results,
        "p_threshold_sweep": p_sweep,
        "best_p_threshold": {"tau": best_p[0], **best_p[1], "per_fold_pq": best_p_folds},
        "pq_product_sweep": pq_sweep,
        "best_pq_product": {"c": best_pq[0], **best_pq[1], "per_fold_pq": best_pq_folds},
        "members": members,
        "cnn": cnn_side,
        "oracle_perfect_scorer": oracle,
        "calibration_deciles": calib.to_dict(orient="records"),
        "feature_importance_gain": [{"feature": f, "share": float(s)} for f, s in imp],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    cands.to_csv(Path(args.table).with_name(Path(args.table).name.replace(".csv.gz", "_scored.csv.gz")),
                 index=False)

    bar = "=" * 72
    log("")
    log(bar)
    for name in ("shipped_thr0.40_ma300", "shipped_top8", "all_candidates"):
        r = results[name]
        log(f"  {name:<28} PQ {r['pq']:.4f}  SQ {r['sq']:.3f} RQ {r['rq']:.3f}  "
            f"TP {r['tp']} FP {r['fp']} FN {r['fn']}  kept {r['n_accepted']}")
    log("-" * 72)
    log(f"  best p >= tau   (tau {best_p[0]}):  PQ {best_p[1]['pq']:.4f}  "
        f"TP {best_p[1]['tp']} FP {best_p[1]['fp']} FN {best_p[1]['fn']}  kept {best_p[1]['n_accepted']}")
    log(f"  best p*q >= c   (c {best_pq[0]}):    PQ {best_pq[1]['pq']:.4f}  "
        f"TP {best_pq[1]['tp']} FP {best_pq[1]['fp']} FN {best_pq[1]['fn']}  kept {best_pq[1]['n_accepted']}")
    log(f"  oracle (perfect scorer):        PQ {oracle['pq']:.4f}")
    log("  best p*q per fold: " + "  ".join(f"{k}:{v:.4f}" for k, v in best_pq_folds.items()))
    log("-" * 72)
    log("  p >= tau sweep: " + "  ".join(f"{t}:{v['pq']:.4f}" for t, v in p_sweep.items()))
    log("  p*q >= c sweep: " + "  ".join(f"{c}:{v['pq']:.4f}" for c, v in pq_sweep.items()))
    log("-" * 72)
    log("  calibration (decile mean p_hat -> mean label): " +
        "  ".join(f"{r['p_hat']:.2f}->{r['label']:.2f}" for r in res["calibration_deciles"]))
    log("  top features: " + ", ".join(f"{f} {s:.0%}" for f, s in imp[:12]))
    log(bar)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
