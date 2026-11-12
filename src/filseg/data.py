"""Loading and tabulating the MAGFiLO 1.0 competition data.

The single most important structural fact about this dataset is that
``images`` entries are *annotation passes*, not photographs: the JSON holds
1154 entries covering only 707 distinct H-Alpha exposures, because up to three
annotators labelled the same exposure independently. Every helper here keeps
``file_stem`` (the physical observation) and ``image_id`` (the annotation pass)
separate so that downstream code cannot accidentally conflate them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
from scipy import ndimage

from .paths import IMAGE_SIZE, annotations_path, test_images_dir, train_images_dir

# GONG is a six-station network; the trailing two characters of a filename are
# the station code. 'h' denotes the H-Alpha instrument at that site.
STATIONS = {
    "Bh": "Big Bear (USA)",
    "Ch": "Cerro Tololo (Chile)",
    "Lh": "Learmonth (Australia)",
    "Mh": "Mauna Loa (USA)",
    "Th": "Teide (Canary Is.)",
    "Uh": "Udaipur (India)",
}

CATEGORY_NAMES = {1: "Left", 2: "Right", 3: "Unidentifiable", 4: "Ambiguous"}


@lru_cache(maxsize=1)
def load_coco() -> dict:
    """Read the raw COCO-style annotation JSON (48 MB; cached per process)."""
    with open(annotations_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_stem(stem: str) -> tuple[datetime, str]:
    """Split ``YYYYMMDDHHMMSSII`` into its capture time and station code."""
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S"), stem[14:]


def image_table() -> pd.DataFrame:
    """One row per annotation pass, with the observation identity broken out.

    Columns: image_id, batch, file_stem, file_name, timestamp, station,
    station_name, year, width, height.
    """
    rows = []
    for img in load_coco()["images"]:
        stem = Path(img["file_name"]).stem
        timestamp, station = parse_stem(stem)
        batch, _, _ = img["id"].partition("-")
        rows.append(
            {
                "image_id": img["id"],
                "batch": batch,
                "file_stem": stem,
                "file_name": img["file_name"],
                "timestamp": timestamp,
                "station": station,
                "station_name": STATIONS.get(station, f"Unknown ({station})"),
                "year": timestamp.year,
                "width": img["width"],
                "height": img["height"],
            }
        )
    return pd.DataFrame(rows)


def annotation_table() -> pd.DataFrame:
    """One row per annotated filament, joined to its observation identity.

    Geometry stays out of this table (it is far too large); use
    :func:`polygon_of`, :func:`spine_of` and :func:`ann_to_mask` for that.
    Columns include bbox_x/y/w/h, area, category_id/name, n_poly_points,
    n_spine_points plus every column of :func:`image_table`.
    """
    coco = load_coco()
    rows = []
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        rows.append(
            {
                "ann_id": ann["id"],
                "image_id": ann["image_id"],
                "category_id": ann["category_id"],
                "category_name": CATEGORY_NAMES.get(ann["category_id"], "?"),
                "area": ann["area"],
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "iscrowd": ann["iscrowd"],
                "n_poly_points": len(ann["segmentation"][0]) // 2,
                "n_spine_points": len(ann["spine"]) // 2,
            }
        )
    df = pd.DataFrame(rows)
    return df.merge(image_table(), on="image_id", how="left", validate="many_to_one")


@lru_cache(maxsize=1)
def annotations_by_id() -> dict[str, dict]:
    """Index raw annotation dicts by ``ann_id`` so geometry is fetchable on demand."""
    return {ann["id"]: ann for ann in load_coco()["annotations"]}


def polygon_of(ann: dict | str) -> np.ndarray:
    """Return a filament outline as an ``(n, 2)`` float array of ``(x, y)`` pixels."""
    if isinstance(ann, str):
        ann = annotations_by_id()[ann]
    return np.asarray(ann["segmentation"][0], dtype=np.float64).reshape(-1, 2)


def spine_of(ann: dict | str) -> np.ndarray:
    """Return a filament spine polyline as an ``(n, 2)`` float array of ``(x, y)`` pixels."""
    if isinstance(ann, str):
        ann = annotations_by_id()[ann]
    return np.asarray(ann["spine"], dtype=np.float64).reshape(-1, 2)


def ann_to_mask(ann: dict | str, size: int = IMAGE_SIZE) -> np.ndarray:
    """Rasterise one annotation to a ``(size, size)`` uint8 mask via pycocotools."""
    if isinstance(ann, str):
        ann = annotations_by_id()[ann]
    rles = mask_utils.frPyObjects(ann["segmentation"], size, size)
    return mask_utils.decode(mask_utils.merge(rles))


def union_mask(anns, size: int = IMAGE_SIZE) -> np.ndarray:
    """Rasterise many annotations into a single binary mask (their pixel union)."""
    out = np.zeros((size, size), dtype=np.uint8)
    for ann in anns:
        np.maximum(out, ann_to_mask(ann, size), out=out)
    return out


def load_image(stem: str, split: str = "train") -> np.ndarray:
    """Load one 2048x2048 grayscale H-Alpha observation as a uint8 array.

    Images are 8-bit single channel; PIL mode ``L`` is enforced so a stray
    JPEG colour profile cannot silently produce three channels.
    """
    directory = train_images_dir() if split == "train" else test_images_dir()
    path = directory / f"{stem}.jpeg"
    with Image.open(path) as im:
        return np.array(im.convert("L"))


def list_stems(split: str = "train") -> list[str]:
    """Sorted observation stems present on disk for a split."""
    directory = train_images_dir() if split == "train" else test_images_dir()
    return sorted(p.stem for p in directory.glob("*.jpeg"))


def split_table(split: str = "test") -> pd.DataFrame:
    """Filename-derived metadata for every image on disk in a split.

    Works for ``test`` too, which has no annotations at all -- this is the only
    way to compare train and test acquisition conditions.
    """
    rows = []
    for stem in list_stems(split):
        timestamp, station = parse_stem(stem)
        rows.append(
            {
                "file_stem": stem,
                "timestamp": timestamp,
                "station": station,
                "station_name": STATIONS.get(station, f"Unknown ({station})"),
                "year": timestamp.year,
                "split": split,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Disk:
    """Fitted solar disk in pixel coordinates."""

    cx: float
    cy: float
    radius: float

    def to_dict(self) -> dict[str, float]:
        return {"cx": self.cx, "cy": self.cy, "radius": self.radius}


def detect_disk(img: np.ndarray, threshold: float | None = None) -> Disk:
    """Fit the solar limb as the extent of the largest bright component.

    A naive "count pixels above a low threshold and invert area = pi r^2" fit
    is badly biased on these frames: GONG images carry a scattered-light halo
    and a bright limb ring outside the photospheric edge, plus corner
    vignetting artefacts. Measured against the fit below, that approach
    overestimates the radius by ~12% and can misplace the centre by tens of
    pixels -- which propagates straight into any heliographic coordinate.

    Instead: threshold relative to the frame's own bright level, keep only the
    largest connected component (dropping corner artefacts), fill interior
    holes (so sunspots and dark filaments do not eat into the disk), and take
    the centre and radius from robust percentile extents of that component --
    thin off-limb prominences are trimmed by the percentile rather than
    inflating the radius.
    """
    if threshold is None:
        # Relative to the frame's own brightness: robust to station and to the
        # image-to-image exposure variation across a 12-year archive.
        threshold = max(20.0, 0.40 * float(np.percentile(img, 99.5)))

    bright = img > threshold
    if not bright.any():
        raise ValueError("No on-disk pixels found; image may be blank.")

    labels, n = ndimage.label(bright)
    if n > 1:
        sizes = ndimage.sum(bright, labels, range(1, n + 1))
        bright = labels == (1 + int(np.argmax(sizes)))
    bright = ndimage.binary_fill_holes(bright)

    ys, xs = np.nonzero(bright)
    x0, x1 = np.percentile(xs, [0.02, 99.98])
    y0, y1 = np.percentile(ys, [0.02, 99.98])
    return Disk(
        cx=float((x0 + x1) / 2),
        cy=float((y0 + y1) / 2),
        radius=float(0.25 * ((x1 - x0) + (y1 - y0))),
    )
