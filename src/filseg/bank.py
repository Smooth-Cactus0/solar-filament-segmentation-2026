"""Filament bank paste: filaments cut from other training exposures, pasted onto quiet disk.

A much stronger form of the in-place copy-paste screened in wave 1: every annotated
filament of the split's exposures is cut out once (its mask and the flat-field and
raw planes under it, each relative to the filament's local background) and 2-6 of
them are pasted per training image at random quiet-disk positions, rotated, scaled
0.7-1.4 and contrast-scaled 0.7-1.3, with a feathered edge. Donors are chosen at a
similar disk radius (+/-0.1 R) so limb darkening stays consistent, never overlap an
existing filament (4 px margin) and stay fully on the disk. Only the split's own
training exposures are donors, so nothing from validation reaches training.

The paste is additive on the background-subtracted planes: the target keeps its own
texture and gains the donor's darkness, the way a real filament sits on the
chromosphere.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

MARGIN_PX = 4
RADIUS_BAND = 0.10
MAX_R = 0.95
FEATHER_SIGMA = 2.0
RING_PX = 6
TRIES = 20


@dataclass(frozen=True)
class BankItem:
    flat: np.ndarray    # flat-plane values minus the local background
    raw: np.ndarray     # raw-plane values minus the local background
    mask: np.ndarray    # float 0/1
    r: float            # centroid radius / R


@dataclass(frozen=True)
class FilamentBank:
    """The donor items plus, per exposure stem, every annotator's filaments (bit-packed)."""
    items: list
    avoid: dict     # stem -> np.packbits of the flattened (size, size) union mask

    def __len__(self) -> int:
        return len(self.items)

    def avoid_mask(self, stem: str, size: int) -> np.ndarray:
        return unpack_avoid(self.avoid[stem], size)


def unpack_avoid(packed: np.ndarray, size: int) -> np.ndarray:
    """Inverse of the packing in ``build_bank``: a bool (size, size) mask."""
    return np.unpackbits(packed, count=size * size).reshape(size, size).astype(bool)


def _crop(mask: np.ndarray, pad: int):
    ys, xs = np.nonzero(mask)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


def build_bank(rows, by_pass, size: int, disks: dict, load) -> FilamentBank:
    """One item per annotated filament of ``rows``' exposures (``rows``: a folds table).

    Also keeps, per stem, the union of every annotation of every pass (the paste avoid mask).
    """
    from .dataset import build_input, render_mask

    bank, avoid = [], {}
    for stem, g in rows.groupby("file_stem"):
        img = load(stem, "train")
        x = build_input(img, disks[stem], size)
        union = np.zeros((size, size), bool)
        for pid in g.image_id:
            for aid in by_pass.get(pid, []):
                soft = render_mask([aid], size)
                union |= soft > 0          # any coverage: the avoid mask keeps partial edge pixels
                m = soft >= 0.5
                if m.sum() < 4:
                    continue
                sy, sx = _crop(m, RING_PX)
                mc = m[sy, sx]
                ring = cv2.dilate(mc.astype(np.uint8), np.ones((2 * RING_PX + 1,) * 2, np.uint8)) > 0
                ring &= ~mc
                f, r_ = x[0][sy, sx], x[1][sy, sx]
                fb = float(np.median(f[ring])) if ring.any() else 0.0
                rb = float(np.median(r_[ring])) if ring.any() else 0.0
                cy, cx = np.nonzero(mc)
                rr = float(x[2][sy, sx][cy.mean().astype(int), cx.mean().astype(int)] + 0.6)
                bank.append(BankItem((f - fb).astype(np.float32), (r_ - rb).astype(np.float32),
                                     mc.astype(np.float32), rr))
        avoid[stem] = np.packbits(union.ravel())
    return FilamentBank(bank, avoid)


def _warp(item: BankItem, angle: float, scale: float):
    h, w = item.mask.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    warp = lambda a: cv2.warpAffine(a, m, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0.0)  # noqa: E731
    return warp(item.flat), warp(item.raw), warp(item.mask)


def _disk_geometry(r_map: np.ndarray) -> tuple[float, float, float]:
    """(cy, cx, R in px) from the radius plane ``r/R`` (min at the centre; area of r < 1 is pi R^2)."""
    cy, cx = np.unravel_index(int(np.argmin(r_map)), r_map.shape)
    return float(cy), float(cx), float(np.sqrt((r_map < 1.0).sum() / np.pi))


def paste_bank(x: np.ndarray, y: np.ndarray, bank: list[BankItem], rng: np.random.Generator,
               cfg, avoid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Paste ``cfg.bank_n`` bank filaments into (x, y); returns the new arrays and the count.

    ``avoid`` (bool, H x W): extra pixels no paste may touch (4 px margin), e.g. the
    other annotators' filaments of the same exposure.
    """
    if not bank:
        return x, y, 0
    x, y = x.copy(), y.copy()
    r_map = x[2] + 0.6
    cy_d, cx_d, R = _disk_geometry(r_map)
    k = np.ones((2 * MARGIN_PX + 1,) * 2, np.uint8)
    occ0 = y[0] > 0
    if avoid is not None:
        occ0 = occ0 | np.asarray(avoid, bool)
    occupied = cv2.dilate(occ0.astype(np.uint8), k) > 0
    H, W = y.shape[1:]
    n_target = int(rng.integers(cfg.bank_n[0], cfg.bank_n[1] + 1))
    n = 0
    for _ in range(n_target):
        for _try in range(TRIES):
            item = bank[int(rng.integers(len(bank)))]
            fl, rw, mk = _warp(item, float(rng.uniform(0, 360)), float(rng.uniform(*cfg.bank_scale)))
            m = mk > 0.5
            if not m.any():
                continue
            # O(1) placement: a polar draw inside the donor's radius band on the disk.
            lo, hi = max(item.r - RADIUS_BAND, 0.0), min(item.r + RADIUS_BAND, MAX_R)
            if hi <= lo:
                break
            rad, ang = float(np.sqrt(rng.uniform(lo * lo, hi * hi))) * R, float(rng.uniform(0.0, 2 * np.pi))
            py, px = int(round(cy_d + rad * np.sin(ang))), int(round(cx_d + rad * np.cos(ang)))
            h, w = m.shape
            y0, x0 = py - h // 2, px - w // 2
            if y0 < 0 or x0 < 0 or y0 + h > H or x0 + w > W:
                continue
            sy, sx = slice(y0, y0 + h), slice(x0, x0 + w)
            if (occupied[sy, sx] & m).any() or (r_map[sy, sx][m] >= MAX_R).any():
                continue
            alpha = cv2.GaussianBlur(mk, (0, 0), FEATHER_SIGMA)
            alpha = np.clip(alpha / max(float(alpha.max()), 1e-6), 0.0, 1.0)
            con = float(rng.uniform(*cfg.bank_contrast))
            x[0][sy, sx] += alpha * con * fl
            x[1][sy, sx] += alpha * con * rw
            # Pasted filaments get no spine entry: the spine channel is off in r0/r1.
            y[0][sy, sx] = np.maximum(y[0][sy, sx], m.astype(y.dtype))
            occupied[sy, sx] |= cv2.dilate(m.astype(np.uint8), k) > 0
            n += 1
            break
    return x, y, n
