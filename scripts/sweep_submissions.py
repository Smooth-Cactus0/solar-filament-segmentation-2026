"""Emit one submission CSV per operating point from a single inference pass.

Every operating point we have submitted so far was chosen by maximising PQ on
out-of-fold predictions. That signal is biased: OOF reads 0.4173 against a
leaderboard of 0.35, consistently, and scripts/diagnose_oof.py ruled out both a
train/inference mismatch and an annotator-weighting artefact. This script exists to
tune against the real objective instead.

Inference runs once; the probability map for each image is then thresholded at
every requested operating point before being discarded. That keeps peak memory at
one image and avoids writing a 424 MB probability cache whose download would cost
more than simply re-running this kernel.

Usage:
    PYTHONPATH=src python scripts/sweep_submissions.py \\
        --config configs/exp2_convnext_1536.yaml \\
        --checkpoints checkpoints/exp2_convnext_1536_fold0.pt \\
        --thresholds 0.25 0.30 0.40 0.50 0.60 \\
        --min-areas 100 300 600 \\
        --top-k 5 8 \\
        --outdir submissions/sweep

One CSV is written per operating point, named ``submission_thr0.40_ma300.csv``
(plus ``_top5`` when a per-image cap applies). ``--top-k`` values are pinned to
``--baseline-threshold``/``--baseline-min-area`` so the cap is the only variable.
``--limit N`` truncates the test set for smoke runs; its CSVs are prefixed
``PARTIAL_limitN_`` because they are incomplete by construction and must never be
submitted. ``--scales`` overrides the config's ``eval.scales`` exactly as in
predict.py.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, NamedTuple, Sequence

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filseg import data as fdata, rle  # noqa: E402
from filseg.config import load_config  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.dataset import model_input  # noqa: E402
from filseg.instances import mean_prob, probs_to_instances  # noqa: E402
from filseg.metric import Instance  # noqa: E402
from filseg.model import build_from_config  # noqa: E402
from filseg.paths import IMAGE_SIZE, output_root  # noqa: E402
# The fallback ladder and its arm names are imported, never re-expressed: a sweep
# whose ladder drifted from predict.py's would label its CSVs with an operating
# point it did not actually apply, which is the one failure this whole script
# exists to rule out. tests/test_sweep.py patches predict.FALLBACK to prove it.
from predict import (  # noqa: E402
    LADDER,
    LAST_RESORT,
    PRIMARY,
    log,
    probs_for,
    relaxed_instances,
)

# The operating point of every submission we have made so far. top_k variants are
# pinned here so that a leaderboard delta is attributable to the cap alone rather
# than to a cap-and-threshold change made at the same time.
BASELINE_THRESHOLD = 0.40
BASELINE_MIN_AREA = 300

# Above this share of frames rescued by a fallback arm, the CSV is no longer a
# measurement of its own operating point (see warn_if_fallback_dominated).
NON_PRIMARY_WARN_FRAC = 0.10

# Mean instances per image per annotator pass in the training ground truth. Our
# single best sanity signal for a sweep point that is wildly over- or
# under-predicting, and the same reference predict.py prints.
GT_MEAN_INSTANCES = 7.11


@dataclass(frozen=True)
class SweepResult:
    """Instances for one operating point, plus which arm produced them.

    A sweep point where 120 of 180 frames fell through to a fallback is not
    measuring that operating point at all, so the driver has to be able to say so
    in the log before a submission slot is spent on the CSV -- the same reason
    predict.py reports `relaxed-threshold frames`. Reporting the arm here rather
    than re-deriving the fallback condition at the call site keeps that judgement
    in one place.
    """

    instances: list[Instance]
    arm: str


def instances_at(
    probs: np.ndarray,
    threshold: float,
    min_area: int,
    disk: Disk,
    *,
    size: int = 2048,
    top_k: int | None = None,
    guarantee: bool = False,
) -> SweepResult:
    """Instances for one probability map at one operating point.

    Thresholding is delegated to ``probs_to_instances`` rather than reimplemented,
    so that with ``top_k=None`` and ``guarantee=False`` this returns exactly what
    predict.py would have written at the same ``(threshold, min_area)``.

    Args:
        probs: probability map at any resolution. It is upsampled to ``size``
            *before* thresholding, never after.
        threshold: foreground probability cut -- half of the operating point.
        min_area: smallest component kept, in ``size`` pixels -- the other half.
        disk: solar disk, in ``size`` coordinates.
        size: side length the instances are produced in; 2048 for submissions.
            It must match the frame ``disk`` was detected in, because the on-disk
            mask is built at ``size``; a mismatch masks the wrong region silently
            rather than raising.
        top_k: keep at most this many instances, those with the highest mean
            probability. ``None`` keeps every instance. Values below 1 are
            rejected: ``-1`` is a plausible "no cap" idiom that would otherwise
            quietly drop an instance, and ``0`` would leave the frame uncovered.
        guarantee: when the operating point finds nothing, relax down predict.py's
            FALLBACK ladder and then fall back to its last-resort arm. Note this
            can return MORE instances than the operating point itself found, so a
            frame rescued this way is no longer a measurement of that operating
            point -- read ``SweepResult.arm`` to see which frames those are.

    Returns:
        ``SweepResult`` with ``arm`` one of ``PRIMARY``, ``LADDER``, ``LAST_RESORT``.

    Raises:
        ValueError: if ``top_k`` is not ``None`` and is below 1.
    """
    if top_k is not None and top_k < 1:
        raise ValueError(f"top_k must be >= 1 or None, got {top_k!r}")

    # The guarantee runs BEFORE the cap, and that order is load-bearing. Guarantee
    # establishes "at least one"; the cap establishes "at most k"; composing them
    # this way leaves both invariants intact and yields between 1 and k instances.
    # The guarantee is predict.relaxed_instances, which returns a list of its own.
    # Were the cap applied first, it would only ever see the operating point's
    # result -- empty on exactly the frames where the fallback fires -- and the
    # fallback's instances would reach the CSV uncapped, so a file labelled
    # top_k=k would hold more than k on those frames. They are common at the
    # aggressive thresholds where top_k is worth sweeping at all.
    if guarantee:
        insts, arm = relaxed_instances(probs, threshold, min_area, disk, size=size)
    else:
        insts, arm = probs_to_instances(probs, threshold, min_area, disk, size=size), PRIMARY

    if top_k is not None and len(insts) > top_k:
        # probs_to_instances returns instances in ``size`` coordinates, so the map
        # has to be brought to that scale before instance slices can index it.
        probs_full = probs
        if probs_full.shape[0] != size:
            probs_full = cv2.resize(probs, (size, size), interpolation=cv2.INTER_LINEAR)
        insts = sorted(insts, key=lambda it: mean_prob(it, probs_full), reverse=True)
        insts = insts[:top_k]

    return SweepResult(insts, arm)


@dataclass(frozen=True)
class OperatingPoint:
    """One (threshold, min_area, top_k) triple and the CSV it writes.

    Frozen so it can key the per-point accumulators, and so the tag a file is
    named after cannot drift from the settings actually applied to that file.
    """

    threshold: float
    min_area: int
    top_k: int | None = None

    @property
    def tag(self) -> str:
        return f"thr{self.threshold:.2f}_ma{self.min_area}" + (
            f"_top{self.top_k}" if self.top_k is not None else ""
        )

    @property
    def filename(self) -> str:
        return f"submission_{self.tag}.csv"


def build_operating_points(
    thresholds: Sequence[float],
    min_areas: Sequence[int],
    top_k: Sequence[int] | None = None,
    *,
    baseline_threshold: float = BASELINE_THRESHOLD,
    baseline_min_area: int = BASELINE_MIN_AREA,
) -> list[OperatingPoint]:
    """Expand the CLI arguments into the points one run will emit.

    The threshold x min_area grid is crossed in full; ``top_k`` is not crossed
    with it but pinned to the baseline, because crossing it would confound the
    cap with the threshold and multiply the number of submission slots a sweep
    consumes -- and slots, not compute, are the scarce resource here.

    Raises:
        ValueError: if two distinct points would name the same CSV. Tags round
            the threshold to two decimals, so ``0.40`` and ``0.401`` collide;
            silently overwriting would leave us submitting a file whose name
            does not describe its contents, which is unrecoverable after the fact.
    """
    points: list[OperatingPoint] = [
        OperatingPoint(float(t), int(a)) for t in thresholds for a in min_areas
    ]
    for k in top_k or ():
        points.append(
            OperatingPoint(float(baseline_threshold), int(baseline_min_area), int(k))
        )

    by_tag: dict[str, OperatingPoint] = {}
    for p in points:
        seen = by_tag.get(p.tag)
        if seen is None:
            by_tag[p.tag] = p
        elif seen != p:
            raise ValueError(
                f"operating points {seen} and {p} share the tag {p.tag!r}; "
                "one CSV would overwrite the other"
            )
        # An exactly repeated point is a duplicated argument, not a request to
        # run the same setting twice, so it collapses silently.
    return list(by_tag.values())


def resolve_outdir(outdir: str | None) -> Path:
    """Where the CSVs land: ``--outdir`` if given, else the output root's sweep folder."""
    return Path(outdir) if outdir else output_root() / "submissions" / "sweep"


def validate_args(args: argparse.Namespace) -> list[str]:
    """Return every problem with the parsed arguments, as human-readable lines.

    Checked up front rather than where each value is first used: a Kaggle kernel
    that runs 15 minutes of inference and then dies writing a CSV has burned a
    session, and everything here is knowable before the first model loads. Every
    problem is reported at once for the same reason -- fixing them one kernel
    launch at a time is the expensive failure mode.
    """
    problems: list[str] = []

    if not Path(args.config).is_file():
        problems.append(f"config not found: {args.config}")
    for c in args.checkpoints:
        if not Path(c).is_file():
            problems.append(f"checkpoint not found: {c}")

    for t in args.thresholds:
        if not 0.0 < t < 1.0:
            problems.append(f"threshold must be in (0, 1), got {t}")
    for a in args.min_areas:
        if a <= 0:
            problems.append(f"min-area must be positive, got {a}")
    for k in args.top_k or ():
        # instances_at rejects this too, but only once it is reached -- which is
        # after the whole inference pass, on the last operating point of a frame.
        if k < 1:
            problems.append(f"top-k must be >= 1, got {k}")

    # The baseline is copied into every capped point rather than swept, so it
    # escapes the loops above; unchecked, a bad value only surfaces mid-run.
    if args.top_k:
        if not 0.0 < args.baseline_threshold < 1.0:
            problems.append(
                f"baseline-threshold must be in (0, 1), got {args.baseline_threshold}"
            )
        if args.baseline_min_area <= 0:
            problems.append(
                f"baseline-min-area must be positive, got {args.baseline_min_area}"
            )

    for s in args.scales or ():
        if s <= 0:
            problems.append(f"scales must be positive input sizes, got {s}")

    if args.limit < 0:
        problems.append(f"limit must be >= 0 (0 means all images), got {args.limit}")

    # The outdir is first touched after the whole inference pass, so a path under
    # read-only /kaggle/input would otherwise cost a full run to discover. It
    # usually does not exist yet, so what must be a writable directory is its
    # nearest existing ancestor.
    outdir = resolve_outdir(args.outdir)
    anchor = next((a for a in (outdir, *outdir.parents) if a.exists()), None)
    if anchor is None or not anchor.is_dir():
        problems.append(
            f"outdir cannot be created: {outdir} (nearest existing path {anchor} "
            "is not a directory)"
        )
    elif not os.access(anchor, os.W_OK):
        problems.append(f"outdir is not writable: {outdir} ({anchor} is read-only)")

    return problems


def non_primary_fraction(arms: Mapping[str, int]) -> float:
    """Share of frames a point did NOT resolve at its own operating point."""
    total = sum(arms.values())
    if not total:
        return 0.0
    return (total - arms.get(PRIMARY, 0)) / total


def warn_if_fallback_dominated(point: OperatingPoint, arms: Mapping[str, int]) -> bool:
    """Shout when a CSV is mostly the fallback ladder's output, not the point's.

    This is the one way the sweep can lie. ``guarantee=True`` is mandatory --
    every test image needs a row -- but it means a threshold so aggressive that
    it finds nothing still yields a full CSV, filled by relaxed rungs. Reading
    that CSV's score as "threshold 0.70 scores X" is flatly wrong: the number
    describes the ladder, at whatever thresholds the ladder happened to reach.
    Since this sweep exists to decide whether false positives explain the 0.067
    CV-to-leaderboard gap, a point that quietly measured something else would
    answer that question backwards, and nothing downstream could detect it.

    Returns True when the warning fired, so the summary table can flag the row.
    """
    frac = non_primary_fraction(arms)
    if frac <= NON_PRIMARY_WARN_FRAC:
        return False
    total = sum(arms.values())
    non_primary = total - arms.get(PRIMARY, 0)
    bar = "!" * 78
    log(bar)
    log(f"!! WARNING  {point.filename}")
    log(f"!!   {non_primary}/{total} frames ({frac:.1%}) came from a fallback arm "
        f"({arms.get(LADDER, 0)} ladder, {arms.get(LAST_RESORT, 0)} last-resort)")
    # The wording tracks the fraction rather than being fixed at the loudest
    # phrasing: the banner exists to make a number distrusted by the right
    # amount, and calling an 11%-contaminated CSV "mostly fallback" would spend
    # the same words as a 70% one, which costs the warning its meaning where it
    # matters most.
    scale = "MOSTLY measures" if frac > 0.5 else "is contaminated by"
    log(f"!!   this CSV {scale} the fallback ladder, NOT threshold "
        f"{point.threshold:.2f} / min_area {point.min_area}")
    log(f"!!   its leaderboard score describes that operating point on only "
        f"{1 - frac:.1%} of frames")
    log(bar)
    return True


def output_name(point: OperatingPoint, limit: int) -> str:
    """The CSV's filename, marked when it covers only part of the test set.

    The filename is the thing that gets uploaded. A smoke-test CSV named like a
    real one is one careless submission away from spending a slot on a score
    computed over two images, so ``--limit`` output carries the limit in its name.
    """
    return f"PARTIAL_limit{limit}_{point.filename}" if limit else point.filename


def check_coverage(
    points: Sequence[OperatingPoint],
    rows: Mapping[OperatingPoint, Sequence[tuple[str, Sequence[Instance]]]],
    stems: Sequence[str],
) -> dict[OperatingPoint, str]:
    """Every point whose CSV the organisers would reject, with a reason naming an image.

    Coverage is the script's most important invariant, so it is an explicit check
    rather than an ``assert`` -- ``python -O`` strips asserts. Each reason names
    a stem because "an image has no instances" cannot be diagnosed from a Kaggle
    log and "20140609195854Bh has no instances" can.

    ``guarantee=True`` makes an empty frame rare, not impossible:
    ``guaranteed_instances`` returns ``[]`` when all of its quantile rungs fail or
    the disk mask comes back empty.
    """
    failures: dict[OperatingPoint, str] = {}
    for p in points:
        if not stems:
            # 0 == 0 and all([]) would wave this through every check below.
            failures[p] = "no test images were listed"
            continue
        point_rows = rows.get(p, ())
        counts = Counter(stem for stem, _ in point_rows)
        listed = set(stems)
        missing = [s for s in stems if s not in counts]
        # main() can produce neither of the next two -- rows come from one loop over
        # a sorted listing -- but the gate keeps its own contract, and a reason that
        # names the offending stems costs nothing. Each problem gets its own clause
        # so one never masquerades as another: an extra stem is not a duplicate.
        unexpected = sorted(s for s in counts if s not in listed)
        duplicated = sorted(s for s, n in counts.items() if n > 1)
        empty = [stem for stem, insts in point_rows if not insts]
        reasons = []
        if missing:
            reasons.append(f"{len(missing)}/{len(stems)} images have no rows, "
                           f"e.g. {', '.join(missing[:3])}")
        if unexpected:
            reasons.append(f"{len(unexpected)} unexpected image(s) not in the test "
                           f"listing, e.g. {', '.join(unexpected[:3])}")
        if duplicated:
            reasons.append(f"{len(duplicated)} image(s) listed more than once, "
                           f"e.g. {', '.join(duplicated[:3])}")
        if empty:
            reasons.append(f"{len(empty)} image(s) have no instances, "
                           f"e.g. {', '.join(empty[:3])}")
        if reasons:
            failures[p] = "; ".join(reasons)
    return failures


def write_atomically(
    per_image: Sequence[tuple[str, Sequence[Instance]]], path: Path
) -> int:
    """Write one CSV such that a failure part-way never leaves a real-looking file.

    A disk-full error mid-write would otherwise leave a truncated CSV under its
    final name, indistinguishable from a complete one until it scored badly.
    Writing a sibling and ``os.replace``-ing it into place is atomic on one
    filesystem, so the final name only ever holds a complete file -- including a
    complete one an earlier run left there, which survives a failed overwrite.
    """
    tmp = path.with_name(path.name + ".tmp")
    try:
        n_rows = rle.write_submission(per_image, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n_rows


class Written(NamedTuple):
    point: OperatingPoint
    path: Path
    n_rows: int


def write_sweep(
    points: Sequence[OperatingPoint],
    rows: Mapping[OperatingPoint, Sequence[tuple[str, Sequence[Instance]]]],
    stems: Sequence[str],
    outdir: Path,
    *,
    arms: Mapping[OperatingPoint, Mapping[str, int]],
    limit: int = 0,
) -> tuple[list[Written], dict[OperatingPoint, str]]:
    """Write every point that passes the coverage gate; write nothing for the rest.

    Gated per point, because one frame the ladder could not cover invalidates that
    point's CSV and nothing else. A dropped point is recoverable; throwing away the
    other fourteen would throw away the inference that produced them. The caller
    turns the returned failures into an exit status.

    Each ``wrote`` line carries that point's arm tally, logged as the file lands.
    The fallback warning and the summary table run only after every write has
    returned, so if a later write dies, the files already on disk would otherwise
    have no record of how many of their frames came from a fallback -- the one
    number that says whether a file's score can be read at face value.
    """
    failures = check_coverage(points, rows, stems)
    written: list[Written] = []
    for p in points:
        if p in failures:
            continue
        path = outdir / output_name(p, limit)
        n_rows = write_atomically(rows[p], path)
        a = arms[p]
        log(f"  wrote {path.name} ({n_rows} rows; {a.get(PRIMARY, 0)} primary, "
            f"{a.get(LADDER, 0)} ladder, {a.get(LAST_RESORT, 0)} last-resort)")
        written.append(Written(p, path, n_rows))
    return written, failures


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.25, 0.30, 0.40, 0.50, 0.60])
    ap.add_argument("--min-areas", type=int, nargs="+", default=[100, 300, 600])
    ap.add_argument("--top-k", type=int, nargs="+", default=None,
                    help="also emit per-image caps, e.g. --top-k 5 8")
    ap.add_argument("--baseline-threshold", type=float, default=BASELINE_THRESHOLD,
                    help="operating point the --top-k variants are pinned to")
    ap.add_argument("--baseline-min-area", type=int, default=BASELINE_MIN_AREA,
                    help="operating point the --top-k variants are pinned to")
    ap.add_argument("--scales", type=int, nargs="+", default=None,
                    help="multi-scale TTA input sizes, e.g. --scales 1536 2048; "
                         "overrides the config's eval.scales, as in predict.py")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="debug: only the first N images. Files are prefixed "
                         "PARTIAL_limitN_ and must never be submitted")
    args = ap.parse_args()

    # Everything that can be known without touching the GPU is checked here,
    # before a single weight is loaded.
    problems = validate_args(args)
    if problems:
        for problem in problems:
            log(f"argument error: {problem}")
        raise SystemExit(2)
    try:
        points = build_operating_points(
            args.thresholds, args.min_areas, args.top_k,
            baseline_threshold=args.baseline_threshold,
            baseline_min_area=args.baseline_min_area)
    except ValueError as exc:
        log(f"argument error: {exc}")
        raise SystemExit(2)
    outdir = resolve_outdir(args.outdir)

    t_start = time.time()
    # Listed before any model loads, for the same reason as the argument checks:
    # a data mount that yields no images is knowable in a second.
    stems = fdata.list_stems("test")
    if not stems:
        log("error: no test images found -- check the data mount "
            "(FILSEG_DATA_ROOT, or /kaggle/input/competitions/... on Kaggle)")
        raise SystemExit(2)
    n_total = len(stems)
    if args.limit:
        stems = stems[: args.limit]

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["amp"]) and device == "cuda"
    tta = bool(cfg["eval"]["tta"])
    # predict.py's precedence, so a sweep runs the same probability map as the
    # submission it is compared against unless explicitly told otherwise.
    scales = args.scales or cfg["eval"].get("scales")

    models = []
    for c in args.checkpoints:
        m = build_from_config(cfg, pretrained=False).to(device)
        m.load_state_dict(torch.load(c, map_location=device)["model"])
        m.eval()
        models.append(m)
    log(f"loaded {len(models)} checkpoint(s) on {device} (amp={amp})")
    # Every equivalence test covers post-processing only. The probability map is
    # the one remaining way a sweep can measure something other than the
    # submission it is compared against, so the log records what actually ran.
    scales_source = ("--scales" if args.scales
                     else "config eval.scales" if cfg["eval"].get("scales")
                     else "config size, single-scale")
    log(f"inference: input {cfg['size']}, tta={tta}, "
        f"scales={list(scales) if scales else [cfg['size']]} ({scales_source})")

    log(f"{len(points)} operating points: {', '.join(p.tag for p in points)}")
    log(f"{len(stems)} of {n_total} test images")
    if args.limit:
        log(f"LIMIT {args.limit}: files will be prefixed PARTIAL_limit{args.limit}_ "
            "-- smoke-test artefacts, never submissions")

    # The progress line follows the submitted operating point when it is in the
    # sweep, since that is the point whose normal numbers we know by heart.
    watch = next(
        (p for p in points if p.top_k is None
         and (p.threshold, p.min_area) == (args.baseline_threshold, args.baseline_min_area)),
        points[0],
    )

    # Instances for every point are held until the end so the coverage gate judges
    # complete CSVs before any is written. Instances are bbox crops, not full
    # frames, so this is tens of MB per point -- but it grows linearly in the
    # number of points, which is the limit on how wide one sweep can go before it
    # has to stream rows to disk instead.
    rows: dict[OperatingPoint, list[tuple[str, list[Instance]]]] = {p: [] for p in points}
    arms: dict[OperatingPoint, Counter] = {p: Counter() for p in points}

    t0 = time.time()
    t_post = 0.0
    for i, stem in enumerate(stems, 1):
        img = fdata.load_image(stem, "test")
        disk = fdata.detect_disk(img)
        x = torch.from_numpy(model_input(img, disk, cfg["size"], cfg.get("inputs"))).unsqueeze(0).to(device)
        # One forward pass per image feeds every operating point below, so the GPU
        # cost is one predict.py run however many points are swept. Post-processing
        # is not shared: it runs once per point per image, so its share of wall
        # time grows linearly with the number of points -- the reason disk_mask is
        # cached, since rebuilding it per point dominated that term.
        probs = probs_for(models, x, tta, amp, scales, tile=cfg["eval"].get("tile"),
                          overlap=cfg["eval"].get("tile_overlap", 0.25))
        t_pp = time.time()
        for p in points:
            # guarantee=True always: the organisers reject a submission that
            # leaves any test image without a row, so coverage outranks purity of
            # the operating point -- warn_if_fallback_dominated is what keeps the
            # resulting impurity visible instead of silent.
            res = instances_at(probs, p.threshold, p.min_area, disk,
                               size=IMAGE_SIZE, top_k=p.top_k, guarantee=True)
            rows[p].append((stem, res.instances))
            arms[p][res.arm] += 1
        t_post += time.time() - t_pp
        if i % 30 == 0 or i == len(stems):
            # The two numbers that go visibly wrong first, so a bad run can be
            # killed at minute 2 instead of discovered at minute 15.
            n_inst = sum(len(insts) for _, insts in rows[watch])
            log(f"  {i}/{len(stems)}  ({time.time() - t0:.0f}s)  {watch.tag}: "
                f"{n_inst / i:.2f} inst/img, {i - arms[watch][PRIMARY]}/{i} non-primary")

    log("")
    written, failures = write_sweep(points, rows, stems, outdir,
                                    arms=arms, limit=args.limit)
    flagged = {w.point for w in written if warn_if_fallback_dominated(w.point, arms[w.point])}

    if written:
        name_w = max(40, *(len(w.path.name) + 2 for w in written))
        rule = "=" * (name_w + 52)
        log("")
        log(rule)
        log(f"{'file':<{name_w}}{'rows':>7}{'imgs':>7}{'inst/img':>10}"
            f"{'primary':>9}{'ladder':>8}{'last':>6}{'':>5}")
        for w in written:
            a = arms[w.point]
            flag = "  <!!" if w.point in flagged else ""
            log(f"{w.path.name:<{name_w}}{w.n_rows:>7}{len(stems):>7}"
                f"{w.n_rows / len(stems):>10.2f}"
                f"{a.get(PRIMARY, 0):>9}{a.get(LADDER, 0):>8}{a.get(LAST_RESORT, 0):>6}"
                f"{flag:>5}")
        log(rule)
    log(f"reference: train ground truth is {GT_MEAN_INSTANCES} instances/image per "
        "annotator pass")
    if flagged:
        log(f"rows flagged <!! have >{NON_PRIMARY_WARN_FRAC:.0%} of frames from a fallback "
            "arm; their leaderboard score does not describe their nominal operating point")
    if args.limit:
        log(f"PARTIAL: --limit {args.limit} covered {len(stems)} of {n_total} test images; "
            "these files are smoke-test artefacts and must never be submitted")
    log(f"wrote {len(written)} of {len(points)} CSVs to {outdir}")
    log(f"total wall time: {time.time() - t_start:.0f}s "
        f"(post-processing {t_post:.0f}s of it, across {len(points)} points)")

    if failures:
        bar = "!" * 78
        log(bar)
        log(f"!! {len(failures)} operating point(s) DROPPED -- no CSV was written for:")
        for p, reason in failures.items():
            log(f"!!   {p.tag}: {reason}")
        log(bar)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
