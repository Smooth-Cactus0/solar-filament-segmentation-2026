"""Read access to a probability-map cache written by ``scripts/cache_probs.py``.

The cache is what lets post-processing be developed without a GPU: one uint8 PNG
per exposure at the model's working resolution, plus a manifest with the fold and
the fitted solar disk each map was produced with. Everything downstream -- the
fragmentation diagnostic, the instance scorer, hysteresis and extent experiments --
reads maps through :class:`ProbCache` so the quantisation and the disk bookkeeping
live in exactly one place.

Layout::

    <root>/manifest.json          config, size, quant, checkpoints, per-stem records
                                  (``"channel": "spine"`` on a spine-map cache)
    <root>/disks.csv              the same disks, flat, for pandas
    <root>/oof/<stem>.png         fold k's checkpoint over ITS held-out exposure
    <root>/test/<stem>.png        the full ensemble over a test image

A cached map differs from the live one only by quantisation: probabilities survive
to within ``0.5 / quant`` (about 0.002), so thresholding it reproduces the live
result except at pixels that close to the threshold -- the ``NEAR`` class of
``scripts/verify_sweep_equivalence.py``, never a structural difference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import Disk

OOF = "oof"
TEST = "test"
MANIFEST = "manifest.json"


def dequantize(levels: np.ndarray, quant: int = 255) -> np.ndarray:
    """uint8 levels back to float32 probabilities."""
    return levels.astype(np.float32) / quant


def read_map(path: Path, quant: int = 255) -> np.ndarray:
    """Read one cached map as float32 probabilities.

    Raises:
        OSError: if the file cannot be read.
        ValueError: if it is not a single-channel uint8 image.
    """
    levels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if levels is None:
        raise OSError(f"cv2 could not read {path}")
    if levels.ndim != 2 or levels.dtype != np.uint8:
        raise ValueError(f"{path} is not a single-channel uint8 map: "
                         f"shape {levels.shape}, dtype {levels.dtype}")
    return dequantize(levels, quant)


@dataclass(frozen=True)
class ProbCache:
    """One cache directory, opened through its manifest."""

    root: Path
    manifest: dict

    @classmethod
    def open(cls, root: str | Path, allow_spine: bool = False) -> "ProbCache":
        """Open a cache; a spine-map cache only with ``allow_spine`` (its maps are not mask
        probabilities, so every tool that cuts candidates must refuse it).

        Raises:
            ValueError: if the manifest says ``"channel": "spine"`` and ``allow_spine`` is False.
        """
        root = Path(root)
        manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
        if manifest.get("channel") == "spine" and not allow_spine:
            raise ValueError(f"{root} is a spine-map cache, not mask probabilities: pass it as --spine-cache")
        return cls(root=root, manifest=manifest)

    # --- metadata -----------------------------------------------------------

    @property
    def size(self) -> int:
        """Side length of every cached map (the model's working resolution)."""
        return int(self.manifest["size"])

    @property
    def quant(self) -> int:
        return int(self.manifest.get("quant", 255))

    @property
    def config(self) -> str:
        return str(self.manifest["config"])

    def stems(self, split: str) -> list[str]:
        """Sorted stems of one split (``"oof"`` or ``"test"``)."""
        self._check_split(split)
        return sorted(self.manifest[split])

    def fold(self, stem: str) -> int:
        """The fold whose checkpoint produced an out-of-fold map."""
        return int(self.manifest[OOF][stem]["fold"])

    def disk(self, split: str, stem: str) -> Disk:
        """The solar disk the map's input was built with, in 2048 px coordinates."""
        self._check_split(split)
        rec = self.manifest[split][stem]
        return Disk(cx=float(rec["cx"]), cy=float(rec["cy"]), radius=float(rec["radius"]))

    # --- maps ----------------------------------------------------------------

    def path(self, split: str, stem: str) -> Path:
        self._check_split(split)
        return self.root / split / f"{stem}.png"

    def probs(self, split: str, stem: str) -> np.ndarray:
        """Float32 probabilities at ``self.size``; see the module docstring on precision."""
        return read_map(self.path(split, stem), self.quant)

    def verify(self) -> list[str]:
        """Every manifest entry whose PNG is missing on disk; empty when the cache is whole."""
        return [f"{split}/{stem}" for split in (OOF, TEST)
                for stem in self.manifest.get(split, {})
                if not self.path(split, stem).is_file()]

    @staticmethod
    def _check_split(split: str) -> None:
        if split not in (OOF, TEST):
            raise ValueError(f"split must be {OOF!r} or {TEST!r}, got {split!r}")
