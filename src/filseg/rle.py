"""Run-length encoding and submission writing for the competition CSV.

The required format is one row per predicted filament::

    filament_id,segmentation_rle
    20150125172714Mh_1,^Vj02jo16I5O2O1...

with two easily-missed details:

* ``segmentation_rle`` is the pycocotools **counts string only** -- no size
  field, no surrounding quotes. Mask size is always 2048x2048.
* pycocotools returns ``counts`` as **bytes**, so it needs an explicit
  ``.decode("utf-8")`` before it reaches the CSV. Writing the bytes object
  directly yields rows like ``b'^Vj02...'`` which score as garbage.

Round-trip losslessness was verified on 300/300 real annotation masks.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from pycocotools import mask as mask_utils

from .metric import Instance
from .paths import IMAGE_SIZE


def mask_to_counts(mask: np.ndarray) -> str:
    """Encode a full-frame boolean mask to a pycocotools counts string."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    # pycocotools hands back bytes; the CSV needs text.
    return counts.decode("utf-8") if isinstance(counts, bytes) else counts


def counts_to_mask(counts: str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Decode a counts string back to a boolean mask (inverse of the above)."""
    rle = {"size": [size, size], "counts": counts.encode("utf-8")}
    return mask_utils.decode(rle).astype(bool)


def instance_to_counts(inst: Instance, size: int = IMAGE_SIZE) -> str:
    """Paint a cropped instance into a full frame and encode it."""
    full = np.zeros((size, size), dtype=np.uint8)
    full[inst.y0 : inst.y1, inst.x0 : inst.x1] = inst.patch
    return mask_to_counts(full)


def write_submission(
    per_image: Iterable[tuple[str, Sequence[Instance]]],
    path: Path | str,
    size: int = IMAGE_SIZE,
) -> int:
    """Write the submission CSV; returns the number of rows written.

    ``per_image`` yields ``(image_stem, instances)``. The ``_n`` suffix only has
    to make rows unique, so instances are numbered from 1 in the order given.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["filament_id", "segmentation_rle"])
        for stem, insts in per_image:
            for n, inst in enumerate(insts, 1):
                writer.writerow([f"{stem}_{n}", instance_to_counts(inst, size)])
                rows += 1
    return rows


def verify_roundtrip(masks: Iterable[np.ndarray]) -> tuple[int, int]:
    """Encode/decode each mask and count exact matches. Returns (ok, total)."""
    ok = total = 0
    for m in masks:
        total += 1
        if np.array_equal(counts_to_mask(mask_to_counts(m), m.shape[0]), m.astype(bool)):
            ok += 1
    return ok, total
