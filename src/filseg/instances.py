"""Turning a probability or foreground map into filament instances.

The EDA supports doing this with plain connected components rather than a
box- or query-based instance model: only 2.9% of filaments have a neighbour
within 10 px and 7.2% within 25 px, so components approximate ground-truth
instances directly.

The component step is deliberately biased toward **splitting**. Over-merging is
the more expensive failure under Panoptic Quality, and 43% of close filament
pairs are near-collinear -- exactly the configuration that fuses -- so no
aggressive morphological closing happens before labelling.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from .data import Disk
from .metric import Instance

# Nothing in MAGFiLO approaches this (largest annotation is 37,739 px); it only
# stops a runaway blob from a badly-thresholded frame becoming one huge instance.
MAX_AREA = 120_000


def disk_mask(disk: Disk, shape: tuple[int, int], frac: float = 0.98) -> np.ndarray:
    """Boolean mask of the on-disk region, shrunk slightly inside the limb.

    The limb itself is a steep brightness gradient that any detector will fire
    on, so the last 2% of the radius is excluded.

    The mask is cached and returned **read-only**. Building one at 2048^2 costs
    ~200 ms, and the operating-point sweep asks for the same disk once per point
    per image -- 2,700 times in a default sweep, where it was most of the
    post-processing time. Sharing one array is only safe because it cannot be
    written: a caller that wrote into a cached mask would silently corrupt the
    disk of every later request with the same geometry, so that now raises
    instead. Existing callers only read it (``fg &= mask`` writes ``fg``).
    """
    return _cached_disk_mask(
        float(disk.cx), float(disk.cy), float(disk.radius),
        (int(shape[0]), int(shape[1])), float(frac),
    )


# Every path asks at size=2048 and one image is live at a time, so a hit needs only
# the current frame's disk. Within an image the same disk is requested again and
# again -- once per operating point in the sweep, once per (threshold, min_area)
# grid cell in train.py's OOF validation -- and that is where the cache pays.
# Across images the disk changes, so an entry never hits again once its image is
# done; eight entries is therefore ample, and bounded at 8 x ~4 MB of mask.
@lru_cache(maxsize=8)
def _cached_disk_mask(
    cx: float, cy: float, radius: float, shape: tuple[int, int], frac: float
) -> np.ndarray:
    mask = _compute_disk_mask(cx, cy, radius, shape, frac)
    mask.setflags(write=False)
    return mask


def _compute_disk_mask(
    cx: float, cy: float, radius: float, shape: tuple[int, int], frac: float
) -> np.ndarray:
    """The uncached computation, kept separate so the cache can be tested against it."""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.hypot(xx - cx, yy - cy) < frac * radius


def label_instances(
    foreground: np.ndarray, min_area: int = 300, max_area: int = MAX_AREA
) -> list[Instance]:
    """Connected components of a boolean map, as cropped instances.

    8-connectivity, so a filament that thins to a diagonal pixel chain stays one
    component rather than fragmenting.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    out: list[Instance] = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area or area > max_area:
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        out.append(Instance(y, x, np.ascontiguousarray(lab[y : y + h, x : x + w] == i)))
    return out


def mean_prob(inst: Instance, probs: np.ndarray) -> float:
    """Mean probability inside one instance, read from a map at the instance's scale.

    Shared rather than inlined at its two call sites: the sweep's ``top_k`` means
    "the most confident k" and ``guaranteed_instances`` means "the most confident
    one". If those two ever disagreed about what confidence is, the sweep and the
    submission it is tuning would disagree about which filament is best.

    ``probs`` must be in the same coordinate frame as ``inst`` -- instances coming
    out of ``probs_to_instances`` are in ``size`` coordinates, not the raw map's.
    """
    if inst.area == 0:
        return 0.0
    return float(probs[inst.y0 : inst.y1, inst.x0 : inst.x1][inst.patch].mean())


def probs_to_instances(
    probs: np.ndarray,
    threshold: float,
    min_area: int = 300,
    disk: Disk | None = None,
    size: int | None = None,
) -> list[Instance]:
    """Full post-processing chain: upsample -> threshold -> components.

    The probability map is upsampled to ``size`` **before** thresholding.
    Thresholding first and upsampling the binary mask discards sub-pixel
    boundary information, and a median filament tolerates only ~4 px of
    positional error before it drops below the metric's IoU gate.
    """
    if size is not None and probs.shape[0] != size:
        probs = cv2.resize(probs, (size, size), interpolation=cv2.INTER_LINEAR)
    fg = probs > threshold
    if disk is not None:
        fg &= disk_mask(disk, fg.shape)
    return label_instances(fg, min_area=min_area)
