"""Predict on the test set and write a submission.

Averages sigmoid probabilities across whatever fold checkpoints are available,
then applies the operating point (threshold, min_area) chosen by the
out-of-fold sweep in train.py.

Probability averaging across folds happens *before* thresholding, for the same
reason the map is upsampled before thresholding: binarising early throws away
the sub-pixel and cross-model agreement that a median filament -- which
tolerates only ~4 px of positional error -- depends on.

Usage:
    PYTHONPATH=src python scripts/predict.py --config configs/exp1_convnext_1024.yaml \
        --checkpoints checkpoints/exp1_convnext_1024_fold0.pt \
        --threshold 0.30 --min-area 300 --out submissions/exp1.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import data as fdata, rle  # noqa: E402
from filseg.config import load_config  # noqa: E402
from filseg.dataset import model_input  # noqa: E402
from filseg.inference import tiled  # noqa: E402
from filseg.instances import probs_to_instances  # noqa: E402
from filseg.model import build_from_config, head_channels, head_of, mask_prob  # noqa: E402
from filseg.paths import output_root  # noqa: E402

# Frames where the operating point finds nothing still need a row: the
# organisers list rejecting incomplete submissions among their safeguards, and
# a miss costs more than a false alarm anyway.
FALLBACK = [(0.7, 1.0), (0.5, 1.0), (0.35, 1.0)]  # (threshold multiplier, area multiplier)
# Relaxing the threshold alone is not enough: a filament that only appears at a
# lower threshold appears *smaller*, so the area gate has to give way with it.
# The ladder is walked in exactly one place, relaxed_instances below.
FALLBACK_AREA_FACTOR = 0.3
FALLBACK_MIN_AREA = 40
# Quiet-Sun frames really can have no filament the model believes in: on the
# 2026-08-18 fold-0 run, two of 180 test frames stayed empty even at
# threshold*0.35. Coverage is not negotiable, so the last resort takes the
# single most-confident blob whatever its probability.
LAST_RESORT_QUANTILE = 0.99995  # ~200 px of a 2048^2 frame
LAST_RESORT_MIN_AREA = 20

# Which arm of the ladder produced a frame's instances. Defined beside the ladder
# so the sweep reports fallbacks in the same terms this script counts them in.
PRIMARY = "primary"          # the operating point itself found something
LADDER = "ladder"            # a relaxed FALLBACK rung found something
LAST_RESORT = "last_resort"  # the frame's own high quantile was the only source


def guaranteed_instances(probs, disk, size: int = 2048):
    """Never return empty: threshold at a high quantile of this frame's own probs."""
    from filseg.instances import disk_mask, label_instances, mean_prob
    import cv2 as _cv2
    if probs.shape[0] != size:
        probs = _cv2.resize(probs, (size, size), interpolation=_cv2.INTER_LINEAR)
    on = probs[disk_mask(disk, probs.shape)]
    if on.size == 0:
        return []
    for q in (LAST_RESORT_QUANTILE, 0.9999, 0.9995):
        cut = float(np.quantile(on, q))
        fg = (probs > cut) & disk_mask(disk, probs.shape)
        insts = label_instances(fg, min_area=LAST_RESORT_MIN_AREA)
        if insts:
            # Keep only the most confident component; this frame is a guess.
            best = max(insts, key=lambda it: mean_prob(it, probs))
            return [best]
    return []


def relaxed_instances(probs, threshold: float, min_area: int, disk, *, size: int = 2048):
    """Instances at an operating point, relaxing down the ladder until the frame is covered.

    The single definition of the fallback ladder. This script's submission path
    and scripts/sweep_submissions.py both call it, so a rung added, dropped or
    reordered here reaches both at once. A sweep that labelled its CSVs with this
    ladder while applying a stale copy would be tuning a different submission
    from the one it claims to.

    ``FALLBACK`` is read at call time, not bound at definition, which is also
    what lets tests/test_sweep.py prove the sweep has no copy of its own.

    Returns:
        ``(instances, arm)``, ``arm`` one of ``PRIMARY``, ``LADDER``, ``LAST_RESORT``.
        The last-resort arm can still return ``[]`` -- see ``guaranteed_instances``
        -- so callers that need coverage must check for it.
    """
    insts = probs_to_instances(probs, threshold, min_area, disk, size=size)
    if insts:
        return insts, PRIMARY
    for tm, am in FALLBACK:
        insts = probs_to_instances(
            probs, threshold * tm,
            max(int(min_area * am * FALLBACK_AREA_FACTOR), FALLBACK_MIN_AREA),
            disk, size=size)
        if insts:
            return insts, LADDER
    return guaranteed_instances(probs, disk, size=size), LAST_RESORT


def log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def probs_for(models, x, tta: bool, amp: bool, scales=None, tile: int | None = None,
              overlap: float = 0.25) -> np.ndarray:
    """Mean sigmoid probability over models x flip symmetries x input scales.

    Scale averaging exploits the one lever that has measurably worked on this
    task -- 1024 to 1536 was worth +0.022 PQ. The encoder is fully convolutional,
    so it can be evaluated at sizes it was never trained at, and averaging in
    probability space (before any threshold) keeps the sub-pixel agreement that a
    4 px translation budget depends on.

    ``mask_prob`` reads the mask head's channels only: a spine channel is a
    training signal that must never reach instance extraction.
    """
    if tile:
        if scales:
            raise ValueError("tiled inference and multi-scale TTA are exclusive")
        return tiled(lambda v: torch.from_numpy(probs_for(models, v, tta, amp))[None, None].to(v.device),
                     x, tile, overlap)[0, 0].cpu().numpy()
    size = x.shape[-1]
    sizes = [size] if not scales else list(scales)
    views = [(x, None)]
    if tta:
        views += [(torch.flip(x, d), d) for d in ([2], [3], [2, 3])]

    acc, n = None, 0
    for m in models:
        for s in sizes:
            for v, dims in views:
                vi = v if s == size else F.interpolate(
                    v, size=(s, s), mode="bilinear", align_corners=False)
                with torch.autocast("cuda", enabled=amp):
                    p = mask_prob(m(vi), head_of(m))
                p = p.float()
                if s != size:
                    p = F.interpolate(p, size=(size, size), mode="bilinear",
                                      align_corners=False)
                if dims is not None:
                    p = torch.flip(p, dims)
                acc = p if acc is None else acc + p
                n += 1
    return (acc / n)[0, 0].cpu().numpy()


@torch.no_grad()
def spine_for(models, x, tta: bool, amp: bool) -> np.ndarray:
    """Mean sigmoid of the spine channel (the one after the mask head) over models x flips.

    The spine head was built as a regulariser (exp4) and never reaches instance extraction;
    the split-tree generator ``spine`` reads this map to cut crowded candidates apart.
    """
    views = [(x, None)]
    if tta:
        views += [(torch.flip(x, d), d) for d in ([2], [3], [2, 3])]
    acc, n = None, 0
    for m in models:
        nh = head_channels(head_of(m))
        for v, dims in views:
            with torch.autocast("cuda", enabled=amp):
                out = m(v)
            if out.shape[1] <= nh:
                raise ValueError(f"the model has {out.shape[1]} output channel(s) and no spine channel")
            p = torch.sigmoid(out[:, nh:nh + 1].float())
            if dims is not None:
                p = torch.flip(p, dims)
            acc = p if acc is None else acc + p
            n += 1
    return (acc / n)[0, 0].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--min-area", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0, help="debug: only N images")
    ap.add_argument("--scales", type=int, nargs="+", default=None,
                    help="multi-scale TTA input sizes, e.g. --scales 1536 2048")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["amp"]) and device == "cuda"

    models = []
    for c in args.checkpoints:
        m = build_from_config(cfg, pretrained=False).to(device)
        m.load_state_dict(torch.load(c, map_location=device)["model"])
        m.eval()
        models.append(m)
    log(f"loaded {len(models)} checkpoint(s) on {device} (amp={amp})")

    stems = fdata.list_stems("test")
    if args.limit:
        stems = stems[: args.limit]
    log(f"predicting {len(stems)} test images @ threshold {args.threshold} "
        f"min_area {args.min_area}")

    per_image, counts, n_fallback = [], [], 0
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        img = fdata.load_image(stem, "test")
        disk = fdata.detect_disk(img)
        x = torch.from_numpy(model_input(img, disk, cfg["size"], cfg.get("inputs"))).unsqueeze(0).to(device)
        probs = probs_for(models, x, cfg["eval"]["tta"], amp,
                          args.scales or cfg["eval"].get("scales"),
                          tile=cfg["eval"].get("tile"),
                          overlap=cfg["eval"].get("tile_overlap", 0.25))

        insts, arm = relaxed_instances(probs, args.threshold, args.min_area, disk, size=2048)
        if arm != PRIMARY:
            n_fallback += 1
        per_image.append((stem, insts))
        counts.append(len(insts))
        if i % 30 == 0:
            log(f"  {i}/{len(stems)}  ({time.time() - t0:.0f}s)")

    out = Path(args.out) if args.out else output_root() / "submissions" / "exp1.csv"
    rows = rle.write_submission(per_image, out)
    log(f"\n  wrote {out}")
    log(f"  {rows} rows over {len(stems)} images")
    log(f"  mean {np.mean(counts):.2f} instances/image (train ground truth: 7.11)")
    log(f"  relaxed-threshold frames: {n_fallback}")
    log(f"  frames still empty:       {sum(1 for c in counts if c == 0)}")


if __name__ == "__main__":
    main()
