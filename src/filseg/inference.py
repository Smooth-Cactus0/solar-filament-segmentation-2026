"""Tiled inference: a full frame through a model trained on crops, blended with a cosine window.

A model trained on 1024 px crops of the native 2048 px frame cannot see the whole disk at
once, and a fully-convolutional network's output near a tile edge is poorer than inside it.
Tiles overlap and each pixel's probability is the window-weighted mean over the tiles that
cover it, so seams vanish and every downstream tool receives one full-frame map.
"""
from __future__ import annotations

import numpy as np
import torch


def cosine_window(n: int) -> np.ndarray:
    """A separable raised-cosine weight over an ``n`` x ``n`` tile.

    The 1e-3 floor is applied per axis, before the outer product: flooring the product would
    flatten the weights along a frame edge (where every covering tile has a near-zero weight on
    that axis), so the other axis's blend would be lost exactly where only edge tiles exist.
    """
    i = np.arange(n, dtype=np.float64)
    w = np.maximum(0.5 - 0.5 * np.cos(2 * np.pi * (i + 0.5) / n), 1e-3)
    return np.outer(w, w).astype(np.float32)


def tile_starts(size: int, tile: int, overlap: float) -> list[int]:
    """Tile origins along one axis: a regular grid of step ``tile * (1 - overlap)``, the last
    tile flush with the frame's far edge."""
    if tile >= size:
        return [0]
    step = max(1, int(round(tile * (1.0 - overlap))))
    starts = list(range(0, size - tile, step)) + [size - tile]
    return sorted(set(starts))


def tiled(fn, x: torch.Tensor, tile: int, overlap: float = 0.25) -> torch.Tensor:
    """``fn(view) -> (1, 1, h, w)`` probabilities per tile; returns the blended (1, 1, S, S) map.

    Raises:
        ValueError: if ``tile`` exceeds the frame: the one view would be smaller than the window.
    """
    size = x.shape[-1]
    if tile > size:
        raise ValueError(f"tile {tile} px is larger than the {size} px frame")
    w = torch.from_numpy(cosine_window(tile)).to(x.device)
    acc = torch.zeros((1, 1, size, size), device=x.device, dtype=torch.float32)
    wsum = torch.zeros_like(acc)
    for y0 in tile_starts(size, tile, overlap):
        for x0 in tile_starts(size, tile, overlap):
            p = fn(x[..., y0:y0 + tile, x0:x0 + tile]).float()
            acc[..., y0:y0 + tile, x0:x0 + tile] += p * w
            wsum[..., y0:y0 + tile, x0:x0 + tile] += w
    return acc / wsum
