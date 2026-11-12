"""Canonical filesystem locations for the Solar Filament Segmentation project.

Resolves the competition data root whether running locally or on a Kaggle kernel,
so the same analysis code works in both places.
"""
from __future__ import annotations

import os
from pathlib import Path

# src/filseg/paths.py -> src/filseg -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCS = PROJECT_ROOT / "docs"
FIGURES = DOCS / "figures"
EDA_STATS = DOCS / "eda_stats"

_LOCAL_DATA = PROJECT_ROOT / "filament-segmentation-2026" / "MAGFiLO_1.0_Kaggle_2026"
_KAGGLE_CANDIDATES = (
    Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    Path("/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
)

IMAGE_SIZE = 2048


def data_root() -> Path:
    """Locate ``MAGFiLO_1.0_Kaggle_2026``, honouring the ``FILSEG_DATA_ROOT`` override."""
    override = os.environ.get("FILSEG_DATA_ROOT")
    if override:
        return Path(override)
    if _LOCAL_DATA.exists():
        return _LOCAL_DATA
    for candidate in _KAGGLE_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate MAGFiLO_1.0_Kaggle_2026. Set FILSEG_DATA_ROOT to its path."
    )


def train_images_dir() -> Path:
    return data_root() / "train" / "train_images"


def test_images_dir() -> Path:
    return data_root() / "test" / "test_images"


def annotations_path() -> Path:
    return data_root() / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"


def output_root() -> Path:
    """Where to write checkpoints, results and submissions.

    On Kaggle the code is mounted read-only under ``/kaggle/input/``, so
    anything derived from ``PROJECT_ROOT`` is unwritable there. Set
    ``FILSEG_OUTPUT_ROOT`` (to ``/kaggle/working``) in that environment;
    locally it defaults to the project itself.
    """
    override = os.environ.get("FILSEG_OUTPUT_ROOT")
    return Path(override) if override else PROJECT_ROOT


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
