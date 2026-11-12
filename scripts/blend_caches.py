"""Blend two or more probability caches into one, map by map.

Each input cache holds one model family's maps (out-of-fold for training exposures,
fold-ensemble for test). Averaging them pixel-wise is the probability-space
ensemble every submission already uses across folds, now across model families.
The members must store their maps at one size (a model run at another resolution is
cached at the blend's size with ``cache_probs.py --cache-size``), and the disks and
folds must agree (they do when every cache was built from the same folds.csv). A
blend of a multi-scale TTA cache with a single-scale one would be silently neither,
so it is refused; single-scale members run at different sizes are fine.

The output is an ordinary cache, so the scorer table builder, the scorer and the
submission writer run on it unchanged.

Usage:
    PYTHONPATH=src python scripts/blend_caches.py \\
        --caches cache/exp2_convnext_1536 cache/exp8b_convnextv2_1536 \\
        --weights 0.5 0.5 --out cache/blend_exp2_exp8
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cache_probs as cp  # noqa: E402
from filseg.cache import OOF, TEST, ProbCache  # noqa: E402


def log(m: str) -> None:
    print(m, flush=True)


def blend(caches: list[ProbCache], weights: list[float], split: str, stem: str) -> np.ndarray:
    size = caches[0].size
    acc = np.zeros((size, size), dtype=np.float32)
    for c, w in zip(caches, weights):
        p = c.probs(split, stem)
        if p.shape[0] != size:
            p = cv2.resize(p, (size, size), interpolation=cv2.INTER_LINEAR)
        acc += w * p
    return acc


def check_compatible(caches: list[ProbCache], splits=(OOF, TEST)) -> None:
    """Refuse caches that cannot be blended: different map sizes, a mix of multi-scale and
    single-scale TTA, different stems or folds.

    A single ``scales`` entry is the size the model ran at, which may differ between
    members (a 1280 px model cached at 1536 beside 1536 px ones); the map ``size`` is
    what the pixels are averaged at, so it must match.
    """
    sizes = [c.size for c in caches]
    if any(s != sizes[0] for s in sizes):
        raise SystemExit("caches disagree on their map size (cache them at one size with "
                         "cache_probs.py --cache-size): " +
                         ", ".join(f"{c.root}: {s}" for c, s in zip(caches, sizes)))
    scales = [c.manifest.get("scales") for c in caches]
    multi = [len(s or []) > 1 for s in scales]
    if any(multi) and not all(multi):
        raise SystemExit("caches mix multi-scale and single-scale TTA scales (a mixed-scale blend): " +
                         ", ".join(f"{c.root}: {s}" for c, s in zip(caches, scales)))
    for split in splits:
        sets = [set(c.stems(split)) for c in caches]
        if any(s != sets[0] for s in sets):
            raise SystemExit(f"caches disagree on the {split} stems")
    for stem in caches[0].stems(OOF):
        folds = {c.fold(stem) for c in caches}
        if len(folds) != 1:
            raise SystemExit(f"{stem}: caches disagree on its fold {folds} -- not the same folds.csv")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="+", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    caches = [ProbCache.open(c) for c in args.caches]
    weights = args.weights or [1.0 / len(caches)] * len(caches)
    if len(weights) != len(caches):
        raise SystemExit("--weights must match --caches")
    weights = [w / sum(weights) for w in weights]
    check_compatible(caches)

    out = Path(args.out)
    base = caches[0].manifest
    manifest = {**{k: v for k, v in base.items() if k not in (OOF, TEST)},
                "config": "blend(" + ", ".join(c.config for c in caches) + ")",
                "blend": [{"cache": str(c.root), "config": c.config, "weight": w}
                          for c, w in zip(caches, weights)],
                OOF: dict(base[OOF]), TEST: dict(base[TEST])}
    t0 = time.time()
    for split in (OOF, TEST):
        for i, stem in enumerate(caches[0].stems(split), 1):
            cp.write_map(out / split / f"{stem}.png", blend(caches, weights, split, stem))
            if i % 200 == 0:
                log(f"  {split} {i}  ({time.time() - t0:.0f}s)")
    cp.write_manifest(out, manifest)
    log(f"wrote {out}: {len(manifest[OOF])} oof + {len(manifest[TEST])} test maps, weights "
        f"{[round(w, 3) for w in weights]} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
