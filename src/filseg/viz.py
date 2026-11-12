"""Shared plotting style so every EDA figure reads as one system.

The explainer commits to a **dark** surface: H-Alpha frames are a bright disk on
black sky, so dark chrome lets the science imagery sit on the page without a
glowing white frame around it. Colours below are the dark-mode steps of a
validated categorical palette -- all six checks pass at six slots on surface
``#1a1a19`` (worst adjacent CVD dE 8.4, normal-vision 19.3, all >= 3:1 contrast).

Because the six-slot run sits in the 6-8 CVD floor band, categorical charts here
must carry secondary encoding: direct labels, gaps between marks, or a legend
plus ordering. Scatter/bubble forms compare *all* pairs rather than adjacent
ones, and the palette only clears that at three slots -- use `SERIES[:3]` there
and fold the rest into "Other" or facet.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --- Categorical slots (dark steps, fixed order -- never cycle, never re-order) ---
SERIES = [
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
]
ALL_PAIRS_SAFE = SERIES[:3]

# --- Chrome & ink ---
SURFACE = "#1a1a19"
PAGE = "#0d0d0d"
INK = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRID = "#2c2c2a"
AXIS = "#383835"

# --- Status (reserved -- never used as a series colour) ---
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

# Sequential blue, reversed for a dark surface: near-zero recedes toward the
# surface, high magnitude reads bright.
_BLUE_STEPS = [
    "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6",
    "#3987e5", "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
]
SEQ_BLUE = LinearSegmentedColormap.from_list("filseg_blue", _BLUE_STEPS)
# Second simultaneous sequential context takes the next categorical hue (orange).
SEQ_ORANGE = LinearSegmentedColormap.from_list(
    "filseg_orange",
    ["#2a1206", "#5c2a10", "#8f3f18", "#d95926", "#ee9366", "#f8c9ac"],
)
# Diverging: blue <-> red with a neutral gray midpoint (never a hue at zero).
DIVERGING = LinearSegmentedColormap.from_list(
    "filseg_div", ["#3987e5", "#86b6ef", "#383835", "#e66767", "#d03b3b"]
)
# H-Alpha imagery keeps a plain grayscale ramp -- it is a photograph, not an encoding.
SOLAR_CMAP = "gray"


def use_style() -> None:
    """Apply the project's matplotlib defaults: dark surface, recessive chrome."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.edgecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.edgecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "600",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 9,
            "legend.labelcolor": INK_SECONDARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "image.cmap": SOLAR_CMAP,
        }
    )


def save(fig, path: Path | str, dpi: int = 150) -> Path:
    """Write a figure to ``path``, creating parent directories, and close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def annotate_source(ax, text: str = "MAGFiLO 1.0 / NSO GONG H-Alpha") -> None:
    """Add a recessive provenance caption under an axes."""
    ax.annotate(
        text,
        xy=(0, -0.14),
        xycoords="axes fraction",
        fontsize=8,
        color=INK_MUTED,
        ha="left",
        va="top",
    )
