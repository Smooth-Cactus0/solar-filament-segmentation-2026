"""Helpers behind ``notebooks/pipeline.ipynb``, the competition's one-notebook pipeline.

The notebook reproduces the final submission from the 180 test images. Its cells stay
short because every stage is the project's own command-line script called through
``main(argv)``: ``cache_probs.py`` (fold-ensemble test maps), ``blend_caches.py`` (the
equal-weight blend) and ``submission_from_cache.py`` (candidates, scorer, keep rule, CSV).
The notebook therefore runs the exact code that wrote the submitted file. What lives here is
the glue that is more than a line: the argument lists, the per-frame views behind the
figures, and the checks on the written CSV.

A small command line serves the validation of the notebook outside Jupyter::

    # the FROM_MAPS path: copy finished caches' test maps into the notebook's work folder
    PYTHONPATH=src python scripts/pipeline_nb.py stage-maps --src cache --dst <work>/cache \\
        --caches scr_r2_base_1536 exp8b_convnextv2_1536 scr_r2_coatnet1_1280
    # freshly computed test maps against reference ones, per cache (uint8 levels)
    PYTHONPATH=src python scripts/pipeline_nb.py compare-maps --a <work>/cache --b cache \\
        --caches scr_r2_base_1536 exp8b_convnextv2_1536 scr_r2_coatnet1_1280
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filseg import rle  # noqa: E402
from filseg.cache import MANIFEST, OOF, TEST, ProbCache  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.metric import Instance, from_mask  # noqa: E402
from filseg.paths import IMAGE_SIZE  # noqa: E402

N_FOLDS = 5


# --- arguments for the pipeline scripts -------------------------------------------------

def partial(path: str | Path, limit: int | None) -> Path:
    """Where a CSV of a ``limit``-image run lands: ``submission_from_cache.py --limit N``
    renames its output ``PARTIAL_limit<N>_<name>`` so a partial file is never mistaken for
    a submission."""
    path = Path(path)
    return path.with_name(f"PARTIAL_limit{limit}_{path.name}") if limit else path


def find_checkpoints(weights_dir: str | Path, family: str, n_folds: int = N_FOLDS) -> list[Path]:
    """The fold checkpoints ``<family>_fold<k>.pt`` for k = 0..n_folds-1, in fold order.

    Raises:
        FileNotFoundError: naming every missing fold; an ensemble short of a fold would
            silently be a different model from the one that made the submission.
    """
    paths = [Path(weights_dir) / f"{family}_fold{k}.pt" for k in range(n_folds)]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} checkpoint(s) of {family} missing under {weights_dir}: "
                                + ", ".join(missing))
    return paths


def cache_argv(config: str | Path, checkpoints: Sequence[str | Path], outdir: str | Path,
               folds_csv: str | Path, device: str, limit: int | None = None) -> list[str]:
    """``cache_probs.py`` arguments for one family's fold-ensemble test maps.

    ``--no-oof`` skips the out-of-fold maps of the training images (they serve model and
    scorer development, not the submission). ``--folds-csv`` is still required by the
    script's parser but is not read on this path.
    """
    argv = ["--config", str(config), "--checkpoints", *map(str, checkpoints),
            "--folds-csv", str(folds_csv), "--outdir", str(outdir), "--no-oof", "--device", device]
    return argv + (["--limit", str(limit)] if limit else [])


def blend_argv(members: Sequence[str | Path], out: str | Path) -> list[str]:
    """``blend_caches.py`` arguments for the equal-weight blend of ``members`` (in order)."""
    return ["--caches", *map(str, members), "--out", str(out)]


@dataclass(frozen=True)
class Rule:
    """How one submission is cut from a probability cache.

    Candidates are the 8-connected components of the cache's map (upsampled to 2048 px)
    above ``threshold`` with at least ``min_area`` pixels, pieces sharing one region above
    ``candidate_bridge`` joined into one. The LightGBM scorer ``scorer_version`` gives each
    candidate ``p`` (an annotator drew it) and ``q`` (its IoU if so); it is kept when
    ``p * q >= accept_pq``. ``out`` is the CSV's file name.
    """

    out: str
    cache: str
    scorer_version: str
    threshold: float
    min_area: int
    candidate_bridge: float
    accept_pq: float

    def argv(self, cache_root: str | Path, scorer_dir: str | Path, sub_dir: str | Path,
             limit: int | None = None) -> list[str]:
        """``submission_from_cache.py`` arguments for this rule."""
        argv = ["--cache", str(Path(cache_root) / self.cache),
                "--threshold", str(self.threshold), "--min-area", str(self.min_area),
                "--candidate-bridge", str(self.candidate_bridge),
                "--scorer-dir", str(scorer_dir), "--scorer-version", self.scorer_version,
                "--accept-pq", str(self.accept_pq), "--out", str(Path(sub_dir) / self.out)]
        return argv + (["--limit", str(limit)] if limit else [])

    def written(self, sub_dir: str | Path, limit: int | None = None) -> Path:
        """The CSV the script writes (``PARTIAL_`` prefixed on a limited run)."""
        return partial(Path(sub_dir) / self.out, limit)

    def scores(self, sub_dir: str | Path) -> Path:
        """The per-candidate ``p``, ``q`` and verdict table written beside the CSV."""
        return (Path(sub_dir) / self.out).with_suffix(".scores.csv")


# --- caches ------------------------------------------------------------------------------

def stage_test_maps(src: str | Path, dst: str | Path, stems: Iterable[str] | None = None) -> int:
    """Copy a finished cache's test maps into ``dst`` as a test-only cache; returns the count.

    The manifest keeps everything but the out-of-fold records, whose maps are not copied
    (``ProbCache.verify`` and the blend would otherwise look for them). This is the
    ``FROM_MAPS`` path: the notebook then starts from these maps instead of the checkpoints.
    """
    import cache_probs as cp

    src, dst = Path(src), Path(dst)
    manifest = json.loads((src / MANIFEST).read_text(encoding="utf-8"))
    keep = sorted(manifest[TEST]) if stems is None else list(stems)
    unknown = [s for s in keep if s not in manifest[TEST]]
    if unknown:
        raise KeyError(f"{src} has no test map for {unknown[:3]}")
    (dst / TEST).mkdir(parents=True, exist_ok=True)
    for stem in keep:
        shutil.copy2(src / TEST / f"{stem}.png", dst / TEST / f"{stem}.png")
    cp.write_manifest(dst, {**manifest, OOF: {}, TEST: {s: manifest[TEST][s] for s in keep}})
    return len(keep)


def missing_maps(cache_root: str | Path, caches: Iterable[str], stems: Iterable[str]) -> list[str]:
    """``<cache>/<stem>`` for every expected test map that is absent (manifest or PNG)."""
    out = []
    stems = list(stems)
    for name in caches:
        root = Path(cache_root) / name
        recorded = ProbCache.open(root).manifest[TEST] if (root / MANIFEST).is_file() else {}
        out += [f"{name}/{s}" for s in stems if s not in recorded or not (root / TEST / f"{s}.png").is_file()]
    return out


def compare_maps(a_root: str | Path, b_root: str | Path, caches: Iterable[str],
                 stems: Iterable[str] | None = None) -> pd.DataFrame:
    """Per cache and test stem, how far two caches' maps are apart, in uint8 levels.

    A level is 1/255 of probability. Maps of the same checkpoints on another GPU differ by
    floating-point noise, a level or two at a few pixels. ``stems`` defaults to the maps
    present under ``a_root``; a stem with no map under ``b_root`` is reported with NaNs.
    """
    rows = []
    for name in caches:
        a, b = Path(a_root) / name / TEST, Path(b_root) / name / TEST
        names = sorted(p.stem for p in a.glob("*.png")) if stems is None else list(stems)
        for stem in names:
            la = cv2.imread(str(a / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            lb = cv2.imread(str(b / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
            if la is None or lb is None:
                rows.append({"cache": name, "stem": stem, "max_levels": np.nan, "mean_levels": np.nan,
                             "frac_px_differ": np.nan, "max_abs_prob": np.nan})
                continue
            d = np.abs(la.astype(np.int16) - lb.astype(np.int16))
            rows.append({"cache": name, "stem": stem, "max_levels": int(d.max()),
                         "mean_levels": float(d.mean()), "frac_px_differ": float((d > 0).mean()),
                         "max_abs_prob": float(d.max()) / 255.0})
    return pd.DataFrame(rows, columns=["cache", "stem", "max_levels", "mean_levels", "frac_px_differ",
                                       "max_abs_prob"])


# --- submission files ----------------------------------------------------------------------

def read_rows(path: str | Path) -> pd.DataFrame:
    """A submission CSV as text (no NaN or number parsing), plus its image ``stem``."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    return df.assign(stem=df.filament_id.str.rsplit("_", n=1).str[0])


def instances_of(path: str | Path, stem: str) -> list[Instance]:
    """The decoded instances of one image in a submission CSV, in file order."""
    rows = read_rows(path)
    insts = (from_mask(rle.counts_to_mask(c)) for c in rows.segmentation_rle[rows.stem == stem])
    return [i for i in insts if i is not None]


def rows_per_stem(path: str | Path) -> pd.Series:
    """Instance count per image."""
    return read_rows(path).groupby("stem").size()


def most_rejected(scores_csv: str | Path) -> str | None:
    """The image where the scorer turned down the most candidates (its decision shows best)."""
    scores = pd.read_csv(scores_csv)
    rejected = scores[~scores.kept.astype(bool)].groupby("stem").size()
    return str(rejected.idxmax()) if len(rejected) else None


@dataclass
class SubmissionCheck:
    """What the organisers' safeguards look at: one row per instance with a unique id,
    at least one row per test image, no foreign image, and no pixel in two instances."""

    name: str
    rows: int
    frames: int
    missing: list[str] = field(default_factory=list)       # expected images with no row
    unexpected: list[str] = field(default_factory=list)    # rows for images not expected
    overlapping: list[str] = field(default_factory=list)   # images where two instances share a pixel
    duplicate_ids: int = 0
    empty_masks: int = 0

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.overlapping
                    or self.duplicate_ids or self.empty_masks)

    def summary(self) -> dict:
        return {"file": self.name, "rows": self.rows, "images": self.frames,
                "inst/img": round(self.rows / max(self.frames, 1), 2),
                "missing images": len(self.missing), "unexpected images": len(self.unexpected),
                "images with overlap": len(self.overlapping), "duplicate ids": self.duplicate_ids,
                "empty masks": self.empty_masks, "ok": self.ok}


def check_submission(path: str | Path, stems: Iterable[str], size: int = IMAGE_SIZE) -> SubmissionCheck:
    """Decode every row of a submission and check it against the expected images ``stems``."""
    rows = read_rows(path)
    expected = set(stems)
    got = set(rows.stem)
    chk = SubmissionCheck(name=Path(path).name, rows=len(rows), frames=len(got),
                          missing=sorted(expected - got), unexpected=sorted(got - expected),
                          duplicate_ids=int(rows.filament_id.duplicated().sum()))
    for stem, grp in rows.groupby("stem"):
        cover = np.zeros((size, size), np.uint8)
        for counts in grp.segmentation_rle:
            m = rle.counts_to_mask(counts, size)
            chk.empty_masks += int(not m.any())
            cover += m
        if cover.max() > 1:
            chk.overlapping.append(str(stem))
    return chk


@dataclass
class Comparison:
    """Our CSV against a reference one, on the images both are compared on.

    ``rows_identical``: the same (id, RLE) rows in the same order. ``same_instances``: per
    image the same multiset of RLE strings, ids and order aside. A pycocotools RLE string
    is a canonical encoding of its mask, so equal strings mean equal pixels.
    ``byte_identical`` is only defined for a full comparison (``None`` on a subset).
    """

    name: str
    reference: str
    images: int
    rows_ours: int
    rows_reference: int
    byte_identical: bool | None
    rows_identical: bool
    same_instances: bool
    differing: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {"file": self.name, "images compared": self.images, "rows (ours)": self.rows_ours,
                "rows (reference)": self.rows_reference, "byte-identical": self.byte_identical,
                "rows identical": self.rows_identical, "same instances": self.same_instances,
                "differing images": len(self.differing)}


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare_submissions(ours: str | Path, reference: str | Path,
                        stems: Iterable[str] | None = None) -> Comparison:
    """Compare two submission CSVs, on ``stems`` only when given (a limited run)."""
    a, b = read_rows(ours), read_rows(reference)
    if stems is not None:
        keep = set(stems)
        a, b = a[a.stem.isin(keep)], b[b.stem.isin(keep)]
    images = sorted(set(a.stem) | set(b.stem))
    ga = a.groupby("stem").segmentation_rle.apply(sorted).to_dict()
    gb = b.groupby("stem").segmentation_rle.apply(sorted).to_dict()
    differing = [s for s in images if ga.get(s) != gb.get(s)]
    rows_identical = (list(zip(a.filament_id, a.segmentation_rle))
                      == list(zip(b.filament_id, b.segmentation_rle)))
    return Comparison(name=Path(ours).name, reference=str(reference), images=len(images),
                      rows_ours=len(a), rows_reference=len(b),
                      byte_identical=(sha256(ours) == sha256(reference)) if stems is None else None,
                      rows_identical=rows_identical, same_instances=not differing,
                      differing=differing)


# --- figures ----------------------------------------------------------------------------------

@dataclass
class FrameView:
    """One test image as the decision stage sees it."""

    stem: str
    image: np.ndarray            # the 2048 px H-alpha frame
    disk: Disk
    probs: np.ndarray            # the cache's map, upsampled to 2048 px
    candidates: list[Instance]   # in the scorer's order (``cand`` in the scores table)
    scores: pd.DataFrame         # cand, area, p, q, pq, kept
    written: list[Instance]      # the instances in the submission CSV


def frame_view(cache_dir: str | Path, stem: str, rule: Rule, scores_csv: str | Path,
               submission_csv: str | Path) -> FrameView:
    """Rebuild one image's candidates exactly as the submission writer cut them, beside the
    scorer's verdicts (from the ``.scores.csv``) and the instances that reached the CSV."""
    from build_scorer_table import cut_candidates
    from filseg import data as fdata, scorer as fscorer

    cache = ProbCache.open(cache_dir)
    probs = fscorer.upsample(cache.probs(TEST, stem), IMAGE_SIZE)
    disk = cache.disk(TEST, stem)
    cands = cut_candidates(probs, disk, rule.threshold, rule.min_area, rule.candidate_bridge)
    scores = pd.read_csv(scores_csv)
    scores = scores[scores.stem == stem].sort_values("cand").reset_index(drop=True)
    scores = scores.assign(pq=scores.p * scores.q)[["cand", "area", "p", "q", "pq", "kept"]]
    return FrameView(stem, fdata.load_image(stem, "test"), disk, probs, cands, scores,
                     instances_of(submission_csv, stem))


def stretch(img: np.ndarray, disk: Disk) -> np.ndarray:
    """The frame contrast-stretched on the disk (0.5 to 99.8 percentile), for display."""
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    on = (xx - disk.cx) ** 2 + (yy - disk.cy) ** 2 <= (0.98 * disk.radius) ** 2
    lo, hi = np.percentile(img[on], [0.5, 99.8])
    return np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1.0), 0.0, 1.0)


def crop_box(insts: Sequence[Instance], disk: Disk, pad: int = 96,
             size: int = IMAGE_SIZE) -> tuple[int, int, int, int]:
    """``(y0, y1, x0, x1)`` around all instances (with ``pad``), else around the disk."""
    if insts:
        y0, y1 = min(i.y0 for i in insts) - pad, max(i.y1 for i in insts) + pad
        x0, x1 = min(i.x0 for i in insts) - pad, max(i.x1 for i in insts) + pad
    else:
        r = int(disk.radius) + 8
        y0, y1, x0, x1 = int(disk.cy) - r, int(disk.cy) + r, int(disk.cx) - r, int(disk.cx) + r
    return max(y0, 0), min(y1, size), max(x0, 0), min(x1, size)


def draw_outlines(ax, insts: Sequence[Instance], box, color, lw: float = 1.2, ls: str = "-") -> None:
    """Outline each instance on an axis that shows the crop ``box`` of the frame."""
    y0, _, x0, _ = box
    for inst in insts:
        contours, _ = cv2.findContours(inst.patch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            pts = np.vstack([c[:, 0, :], c[:1, 0, :]]).astype(float)
            ax.plot(pts[:, 0] + inst.x0 - x0, pts[:, 1] + inst.y0 - y0, color=color, lw=lw, ls=ls)


def label_image(insts: Sequence[Instance], size: int = IMAGE_SIZE) -> np.ma.MaskedArray:
    """Instances painted as labels 1..n on a zero background (a masked array)."""
    lab = np.zeros((size, size), np.int32)
    for k, inst in enumerate(insts, 1):
        win = lab[inst.y0:inst.y1, inst.x0:inst.x1]
        win[inst.patch] = k
    return np.ma.masked_equal(lab, 0)


def show_planes(img: np.ndarray, disk: Disk, x: np.ndarray,
                names: Sequence[str] = ("flat-field ratio", "raw, normalised", "radius r/R")):
    """The frame with its fitted disk, then each input plane the network sees."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1 + len(x), figsize=(4.2 * (1 + len(x)), 4.4))
    axes[0].imshow(img, cmap="gray")
    axes[0].add_patch(plt.Circle((disk.cx, disk.cy), disk.radius, fill=False, color="tab:orange", lw=1.2))
    axes[0].set_title(f"frame and fitted disk (r = {disk.radius:.0f} px)")
    for ax, plane, name in zip(axes[1:], x, names):
        im = ax.imshow(plane, cmap="gray")
        ax.set_title(f"{name}  [{plane.shape[0]} px]")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig


def show_maps(cache_root: str | Path, caches: Sequence[str], stem: str, level: float = 0.4,
              pad: int = 48):
    """One image's test map from each cache, side by side (probability 0 to 1).

    Off-disc pixels are masked: the loss ignores them, so the network's output there means
    nothing, and candidates are only cut on the disc. The view is cropped to where any map
    exceeds ``level`` on the disc (with ``pad`` px), or to the disc when none does.
    """
    import matplotlib.pyplot as plt
    from filseg.instances import disk_mask

    opened = [ProbCache.open(Path(cache_root) / name) for name in caches]
    maps = [c.probs(TEST, stem) for c in opened]
    size = maps[0].shape[0]
    d = opened[0].disk(TEST, stem)
    s = size / IMAGE_SIZE
    on = disk_mask(Disk(d.cx * s, d.cy * s, d.radius * s), (size, size))
    hot = on & np.any([m > level for m in maps], axis=0)
    ys, xs = np.nonzero(hot if hot.any() else on)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + 1 + pad, size)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + 1 + pad, size)
    fig, axes = plt.subplots(1, len(caches), figsize=(4.2 * len(caches), 4.4), squeeze=False)
    for ax, name, c, m in zip(axes[0], caches, opened, maps):
        ax.imshow(np.ma.masked_where(~on, m)[y0:y1, x0:x1], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"{name} (ran at {c.manifest.get('model_size', c.size)} px)", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"{stem}: fold-ensemble test maps at {size} px, on the disc, "
                 f"cropped to where a map exceeds {level}")
    fig.tight_layout()
    return fig


def show_decision(view: FrameView, threshold: float):
    """Three panels: the map over the frame, the candidates with the scorer's verdict
    (kept solid green, rejected dashed red, labelled with p*q), and the written instances."""
    import matplotlib.pyplot as plt

    box = crop_box(view.candidates or view.written, view.disk)
    y0, y1, x0, x1 = box
    frame = stretch(view.image, view.disk)[y0:y1, x0:x1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    from filseg.instances import disk_mask

    axes[0].imshow(frame, cmap="gray")
    off = ~disk_mask(view.disk, view.probs.shape)          # the loss ignores off-disc output
    p = np.ma.masked_where((view.probs < 0.05) | off, view.probs)[y0:y1, x0:x1]
    im = axes[0].imshow(p, cmap="magma", vmin=0, vmax=1, alpha=0.75)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    axes[0].set_title("blended probability map")
    axes[1].imshow(frame, cmap="gray")
    kept = dict(zip(view.scores.cand, view.scores.kept))
    pq = dict(zip(view.scores.cand, view.scores.pq))
    for k, inst in enumerate(view.candidates):
        ok = bool(kept.get(k, False))
        draw_outlines(axes[1], [inst], box, "lime" if ok else "red", ls="-" if ok else "--")
        if k in pq:
            axes[1].text(inst.x1 - x0, inst.y0 - y0, f"{pq[k]:.2f}", color="lime" if ok else "red",
                         fontsize=9, va="bottom", bbox={"facecolor": "black", "alpha": 0.6, "pad": 1, "lw": 0})
    axes[1].set_title(f"{len(view.candidates)} candidates (p > {threshold}), label = p*q")
    axes[2].imshow(frame, cmap="gray")
    axes[2].imshow(label_image(view.written)[y0:y1, x0:x1], cmap="tab20", alpha=0.6,
                   interpolation="nearest")
    axes[2].set_title(f"{len(view.written)} instances written")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(view.stem)
    fig.tight_layout()
    return fig


# --- command line ---------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stage-maps", help="copy finished caches' test maps into a work folder")
    st.add_argument("--src", required=True, help="folder holding the finished caches")
    st.add_argument("--dst", required=True, help="the notebook's WORK_DIR/cache")
    st.add_argument("--caches", nargs="+", required=True)
    cm = sub.add_parser("compare-maps", help="level difference per cache and image")
    cm.add_argument("--a", required=True, help="the freshly computed caches (e.g. WORK_DIR/cache)")
    cm.add_argument("--b", required=True, help="the reference caches")
    cm.add_argument("--caches", nargs="+", required=True)
    cm.add_argument("--csv", default=None, help="also write the per-image table here")
    args = ap.parse_args(argv)

    if args.cmd == "stage-maps":
        for name in args.caches:
            n = stage_test_maps(Path(args.src) / name, Path(args.dst) / name)
            print(f"{name}: {n} test maps -> {Path(args.dst) / name}", flush=True)
        return 0
    df = compare_maps(args.a, args.b, args.caches)
    if args.csv:
        df.to_csv(args.csv, index=False)
    print(df.to_string(index=False))
    per = df.groupby("cache").agg(images=("stem", "size"),
                                  no_reference=("max_levels", lambda s: int(s.isna().sum())),
                                  max_levels=("max_levels", "max"), max_abs_prob=("max_abs_prob", "max"),
                                  mean_levels=("mean_levels", "mean"))
    print("\nper cache:\n" + per.to_string(), flush=True)
    return 0 if len(df) and not df.max_levels.isna().any() else 1


if __name__ == "__main__":
    raise SystemExit(main())
