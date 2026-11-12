"""Model input planes, selected by the ``inputs:`` list of an experiment config (spec §2d).

``dataset.build_input`` always builds the three base planes -- 0 flat-fielded, 1 raw
normalised, 2 normalised radius -- because augmentation (photometric transforms touch
0-1, the radius border fill touches 2) and the dataset's on-disk ``valid`` mask rely
on those positions. ``latitude`` is geometric: when requested it is built at index 3,
*before* augmentation, and moves with every geometric transform. ``tophat`` and
``clahe`` are computed *after* augmentation from the augmented flat and raw planes,
so cutout, copy-paste, photometric jitter and blur reach them exactly as they reach
their source plane. ``select`` then returns the requested planes in the listed
order; that array is what the model sees, at training and at inference alike.

The default list is the three-plane input every model through recipe r1 was trained
with, and ``select`` returns that array untouched.
"""
from __future__ import annotations

import cv2
import numpy as np

BASE = ("flat", "raw", "radius")
GEOMETRIC = ("latitude",)
DERIVED = ("tophat", "clahe")
NAMES = BASE + GEOMETRIC + DERIVED
DEFAULT = BASE

# Black top-hat window sizes at 1024 px (scaled with the working size): about 1x, 2x
# and 4x a median filament's width, so both thin and wide filaments light up.
TOPHAT_KERNELS_1024 = (9, 17, 33)
TOPHAT_GAIN = 3.0          # a median filament (-0.33 in the flat plane) reads ~1
CLAHE_CLIP = 2.0
CLAHE_TILES = 8


def validate(inputs) -> tuple[str, ...]:
    """The input list as a tuple; ``None`` means the default three planes."""
    if inputs is None:
        return DEFAULT
    names = tuple(inputs)
    if not names:
        raise ValueError("inputs: must list at least one plane")
    unknown = [n for n in names if n not in NAMES]
    if unknown:
        raise ValueError(f"unknown input plane(s): {', '.join(unknown)}; known: {', '.join(NAMES)}")
    if len(set(names)) != len(names):
        raise ValueError(f"inputs: lists a plane twice: {list(names)}")
    return names


def wants_latitude(inputs) -> bool:
    return "latitude" in validate(inputs)


def latitude_plane(disk, shape: tuple[int, int]) -> np.ndarray:
    """sin(heliographic latitude) on the disk, 0 off it; north (image up) positive.

    Orthographic projection with the solar B0 angle taken as 0. The true B0 swings
    by +-7.25 degrees over a year, and reading it would need the timestamp, which is
    metadata; the error is a small smooth shift of the equator.
    """
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx - disk.cx, yy - disk.cy) / disk.radius
    s = np.clip((disk.cy - yy) / disk.radius, -1.0, 1.0)
    return np.where(r <= 1.0, s, 0.0).astype(np.float32)


def on_disk(x: np.ndarray) -> np.ndarray:
    """On-disk pixels from the radius plane (built as clip(r / R, 0, 1.2) - 0.6)."""
    return x[2] < 0.4


def black_tophat(flat: np.ndarray, size: int) -> np.ndarray:
    """Multi-scale black top-hat of the flat plane: how much darker than its surroundings.

    Square windows: OpenCV runs rectangular morphology separably, which keeps a
    33-px window affordable inside the data loader.
    """
    out = np.zeros_like(flat, dtype=np.float32)
    for k in TOPHAT_KERNELS_1024:
        kk = max(3, int(round(k * size / 1024)) | 1)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (kk, kk))
        np.maximum(out, cv2.morphologyEx(flat.astype(np.float32), cv2.MORPH_BLACKHAT, se), out=out)
    return np.clip(out * TOPHAT_GAIN, 0.0, 3.0).astype(np.float32)


def clahe_plane(raw: np.ndarray, disk_mask: np.ndarray) -> np.ndarray:
    """CLAHE of the raw plane, centred on its on-disk median, 0 off-disk."""
    u8 = np.clip((raw + 1.5) / 4.5 * 255.0, 0, 255).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=CLAHE_CLIP,
                        tileGridSize=(CLAHE_TILES, CLAHE_TILES)).apply(u8).astype(np.float32)
    med = float(np.median(c[disk_mask])) if disk_mask.any() else 0.0
    return np.where(disk_mask, (c - med) / 64.0, 0.0).astype(np.float32)


def select(x: np.ndarray, inputs) -> np.ndarray:
    """The model's input: the requested planes of ``x`` (from build_input), in order."""
    names = validate(inputs)
    if names == DEFAULT and x.shape[0] == 3:
        return x
    planes = {"flat": x[0], "raw": x[1], "radius": x[2]}
    if x.shape[0] > 3:
        planes["latitude"] = x[3]
    disk = None
    out = []
    for n in names:
        if n in planes:
            out.append(planes[n])
            continue
        if n == "latitude":
            raise ValueError("inputs: 'latitude' requested but the latitude plane was not built "
                             "(build_input(..., latitude=True))")
        disk = on_disk(x) if disk is None else disk
        if n == "tophat":
            out.append(black_tophat(x[0], x.shape[-1]) * disk)
        else:
            out.append(clahe_plane(x[1], disk))
    return np.ascontiguousarray(np.stack(out).astype(np.float32, copy=False))
