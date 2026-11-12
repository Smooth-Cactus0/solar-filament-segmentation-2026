"""Cache the model's probability maps so post-processing can be developed on a CPU.

Every post-processing idea so far -- the operating-point sweep, the top-k cap, and
next the hysteresis threshold, fragment rejoining and a learned instance scorer --
has needed a GPU run just to get probability maps to work on. This script runs
inference once and keeps the maps:

* ``oof/{stem}.png``  -- each fold's checkpoint over its OWN held-out exposures,
  so together the 707 maps are a genuine out-of-fold prediction for every training
  exposure;
* ``test/{stem}.png`` -- the full fold ensemble over the 180 test images, i.e. the
  same maps ``predict.py`` and ``sweep_submissions.py`` threshold for a submission.

Both paths go through ``predict.probs_for``, so a map here is the map a submission
would have been cut from, at the model's working resolution (``cfg["size"]``).

Maps are stored as **uint8 PNG** (lossless, ``round(p * 255)``). At 1536 px a
mostly-zero map compresses to a few hundred kB, so the whole cache is well under a
gigabyte, against 3.3 GB as float16. The cost is quantisation: a probability
survives to within 1/510, so thresholding a cached map reproduces thresholding the
live map except at pixels within 0.002 of the threshold -- the ``NEAR`` class of
``verify_sweep_equivalence.py``, never a structural difference.

``manifest.json`` records the config, size, checkpoints, quantisation and, per
stem, the fold and the fitted disk (``cx, cy, radius``), because instance
extraction at 2048 needs the disk and re-fitting it means re-reading the image.

Usage:
    PYTHONPATH=src python scripts/cache_probs.py \\
        --config configs/exp2_convnext_1536.yaml \\
        --checkpoints checkpoints/exp2_convnext_1536_fold*.pt \\
        --folds-csv data/folds.csv --outdir cache/exp2 --zip
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from filseg import data as fdata  # noqa: E402
from filseg.cache import MANIFEST, OOF, TEST  # noqa: E402
from filseg.cache import dequantize as _dequantize  # noqa: E402
from filseg.cache import read_map as _read_map  # noqa: E402
from filseg.config import load_config  # noqa: E402
from filseg.data import Disk  # noqa: E402
from filseg.dataset import model_input  # noqa: E402
from filseg.model import build_from_config  # noqa: E402
from filseg.paths import output_root  # noqa: E402
from predict import log, probs_for, spine_for  # noqa: E402

# uint8 quantisation: p -> round(p * QUANT). 255 keeps the file a plain 8-bit PNG.
# The reader side (filseg.cache) takes quant from the manifest, so this is the one
# place the writer's constant is fixed.
QUANT = 255
OOF_DIR = OOF
TEST_DIR = TEST
DISKS_CSV = "disks.csv"

_FOLD_RE = re.compile(r"fold(\d+)")


def fold_of(checkpoint: str | Path) -> int:
    """The fold index encoded in a checkpoint filename such as ``..._fold3.pt``.

    Raises:
        ValueError: if the name carries no ``fold<k>``; guessing would silently
            score a fold's checkpoint against exposures it was trained on.
    """
    m = _FOLD_RE.search(Path(checkpoint).stem)
    if m is None:
        raise ValueError(f"cannot read a fold index from {Path(checkpoint).name!r}")
    return int(m.group(1))


def quantize(probs: np.ndarray) -> np.ndarray:
    """Probabilities in [0, 1] to uint8 levels."""
    return np.round(np.clip(probs, 0.0, 1.0) * QUANT).astype(np.uint8)


def dequantize(levels: np.ndarray) -> np.ndarray:
    """uint8 levels back to float32 probabilities (the reader's function, at QUANT)."""
    return _dequantize(levels, QUANT)


def write_map(path: Path, probs: np.ndarray) -> None:
    """Write one probability map as a lossless uint8 PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), quantize(probs), [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise OSError(f"cv2 could not write {path}")


def read_map(path: Path) -> np.ndarray:
    """Read a map written by :func:`write_map` back to float32 probabilities."""
    return _read_map(path, QUANT)


def disk_record(disk: Disk) -> dict[str, float]:
    return {"cx": float(disk.cx), "cy": float(disk.cy), "radius": float(disk.radius)}


def disk_from_record(rec: dict) -> Disk:
    return Disk(cx=float(rec["cx"]), cy=float(rec["cy"]), radius=float(rec["radius"]))


def load_manifest(outdir: Path) -> dict:
    return json.loads((Path(outdir) / MANIFEST).read_text(encoding="utf-8"))


def new_manifest(cfg: dict, config_path: str, ckpt_by_fold: dict[int, str],
                 tta: bool, scales, cache_size: int | None = None) -> dict:
    """The manifest skeleton; ``oof``/``test`` are filled per stem as maps are written.

    ``scales`` keeps a multi-scale TTA set as run; a single-scale cache records the size its
    maps are stored at (``[cache_size]`` when given), so it describes the maps a blend averages.
    ``model_size`` keeps the size the model ran at.
    """
    if scales and len(scales) > 1:
        recorded = [int(s) for s in scales]
    else:
        recorded = [int(cache_size or (scales[0] if scales else cfg["size"]))]
    return {
        "config": Path(config_path).name,
        "encoder": cfg["encoder"],
        "size": int(cache_size or cfg["size"]),
        "model_size": int(cfg["size"]),
        "quant": QUANT,
        "format": "uint8 PNG, level = round(p * quant)",
        "tta": bool(tta),
        "scales": recorded,
        "checkpoints": {str(k): Path(v).name for k, v in sorted(ckpt_by_fold.items())},
        "oof": {},
        "test": {},
    }


def spine_manifest(manifest: dict) -> dict:
    """The spine cache's manifest: the probability cache's, marked ``"channel": "spine"`` so the
    tools that cut candidates refuse it (``filseg.cache.ProbCache.open``)."""
    return {**manifest, "channel": "spine"}


def write_manifest(outdir: Path, manifest: dict) -> None:
    (outdir / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    rows = [{"split": split, "stem": stem, **{k: v for k, v in rec.items() if k != "fold"},
             "fold": rec.get("fold", -1)}
            for split in (OOF_DIR, TEST_DIR) for stem, rec in manifest[split].items()]
    pd.DataFrame(rows, columns=["split", "stem", "cx", "cy", "radius", "fold"]).to_csv(
        outdir / DISKS_CSV, index=False)


def zip_cache(outdir: Path, zip_path: Path | None = None) -> Path:
    """Bundle the cache into one archive, so a Kaggle output is one download, not 900.

    PNGs are already compressed, so the entries are stored, not deflated.
    """
    zip_path = zip_path or outdir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for p in sorted(outdir.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(outdir).as_posix())
    return zip_path


def load_model(cfg: dict, checkpoint: str, device: str):
    m = build_from_config(cfg, pretrained=False).to(device)
    m.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
    m.eval()
    return m


def resolve_scales(cfg: dict, override) -> list[int] | None:
    """The TTA scale set: the command line wins over ``eval.scales``; None = the model's size only."""
    return [int(s) for s in override] if override else cfg["eval"].get("scales")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="fold checkpoints; the fold index is read from each filename")
    ap.add_argument("--folds-csv", required=True)
    ap.add_argument("--outdir", default=None,
                    help="cache root (default <output root>/cache/<config name>)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N exposures per split")
    ap.add_argument("--no-test", action="store_true", help="skip the test-set ensemble maps")
    ap.add_argument("--no-oof", action="store_true", help="skip the out-of-fold maps")
    ap.add_argument("--zip", action="store_true", help="also write <outdir>.zip (and <spine-outdir>.zip)")
    ap.add_argument("--cache-size", type=int, default=None,
                    help="store maps at this size (default eval.cache_size, else the model's size)")
    ap.add_argument("--spine-outdir", default=None,
                    help="also cache the spine channel's maps here (a second ProbCache with the same manifest)")
    ap.add_argument("--scales", type=int, nargs="+", default=None,
                    help="multi-scale TTA input sizes; overrides eval.scales in the config")
    ap.add_argument("--device", default=None,
                    help="torch device to run on (default: cuda when available, else cpu)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(cfg["amp"]) and device.startswith("cuda")
    tta = bool(cfg["eval"]["tta"])
    scales = resolve_scales(cfg, args.scales)
    size = int(cfg["size"])
    cs = int(args.cache_size or cfg["eval"].get("cache_size") or size)
    tile, overlap = cfg["eval"].get("tile"), cfg["eval"].get("tile_overlap", 0.25)

    if args.spine_outdir and tile:
        raise SystemExit("the spine cache does not tile: this config sets eval.tile")
    ckpt_by_fold = {fold_of(c): c for c in args.checkpoints}
    if len(ckpt_by_fold) != len(args.checkpoints):
        raise SystemExit("two checkpoints claim the same fold: " + ", ".join(args.checkpoints))
    outdir = Path(args.outdir) if args.outdir else (
        output_root() / "cache" / Path(args.config).stem)
    outdir.mkdir(parents=True, exist_ok=True)
    spine_out = Path(args.spine_outdir) if args.spine_outdir else None
    if spine_out is not None:
        spine_out.mkdir(parents=True, exist_ok=True)
    manifest = new_manifest(cfg, args.config, ckpt_by_fold, tta, scales, cache_size=cs)

    log(f"device {device} (amp={amp}) | size {size} | cache size {cs} | tile {tile} | tta {tta} | scales {manifest['scales']}")
    log("checkpoints: " + ", ".join(f"{k}:{Path(v).name}" for k, v in sorted(ckpt_by_fold.items())))
    log(f"cache -> {outdir}")
    t0 = time.time()

    if not args.no_oof:
        folds = pd.read_csv(args.folds_csv)
        for fold in sorted(ckpt_by_fold):
            model = load_model(cfg, ckpt_by_fold[fold], device)
            stems = sorted(folds[folds.fold == fold].file_stem.unique())
            if args.limit:
                stems = stems[: args.limit]
            log(f"fold {fold}: {len(stems)} held-out exposures")
            for i, stem in enumerate(stems, 1):
                img = fdata.load_image(stem, "train")
                disk = fdata.detect_disk(img)
                x = torch.from_numpy(model_input(img, disk, size, cfg.get("inputs"))).unsqueeze(0).to(device)
                probs = probs_for([model], x, tta, amp, scales, tile=tile, overlap=overlap)
                if cs != size:
                    probs = cv2.resize(probs, (cs, cs), interpolation=cv2.INTER_AREA)
                write_map(outdir / OOF_DIR / f"{stem}.png", probs)
                if spine_out is not None:
                    sp = spine_for([model], x, tta, amp)
                    if cs != size:
                        sp = cv2.resize(sp, (cs, cs), interpolation=cv2.INTER_AREA)
                    write_map(spine_out / OOF_DIR / f"{stem}.png", sp)
                manifest[OOF_DIR][stem] = {"fold": fold, **disk_record(disk)}
                if i % 25 == 0 or i == len(stems):
                    log(f"    {i}/{len(stems)}  ({time.time() - t0:.0f}s)")
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        write_manifest(outdir, manifest)
        if spine_out is not None:
            write_manifest(spine_out, spine_manifest(manifest))

    if not args.no_test:
        models = [load_model(cfg, ckpt_by_fold[f], device) for f in sorted(ckpt_by_fold)]
        stems = fdata.list_stems("test")
        if args.limit:
            stems = stems[: args.limit]
        log(f"test: {len(stems)} images through the {len(models)}-fold ensemble")
        for i, stem in enumerate(stems, 1):
            img = fdata.load_image(stem, "test")
            disk = fdata.detect_disk(img)
            x = torch.from_numpy(model_input(img, disk, size, cfg.get("inputs"))).unsqueeze(0).to(device)
            probs = probs_for(models, x, tta, amp, scales, tile=tile, overlap=overlap)
            if cs != size:
                probs = cv2.resize(probs, (cs, cs), interpolation=cv2.INTER_AREA)
            write_map(outdir / TEST_DIR / f"{stem}.png", probs)
            if spine_out is not None:
                sp = spine_for(models, x, tta, amp)
                if cs != size:
                    sp = cv2.resize(sp, (cs, cs), interpolation=cv2.INTER_AREA)
                write_map(spine_out / TEST_DIR / f"{stem}.png", sp)
            manifest[TEST_DIR][stem] = disk_record(disk)
            if i % 30 == 0 or i == len(stems):
                log(f"    {i}/{len(stems)}  ({time.time() - t0:.0f}s)")
        write_manifest(outdir, manifest)
        if spine_out is not None:
            write_manifest(spine_out, spine_manifest(manifest))

    n_oof, n_test = len(manifest[OOF_DIR]), len(manifest[TEST_DIR])
    size_mb = sum(p.stat().st_size for p in outdir.rglob("*.png")) / 1e6
    log(f"\nwrote {n_oof} oof + {n_test} test maps, {size_mb:.0f} MB of PNG, "
        f"in {(time.time() - t0) / 60:.1f} min")
    if args.zip:
        for d in (outdir, spine_out):
            if d is not None:
                zp = zip_cache(d)
                log(f"zipped -> {zp} ({zp.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
