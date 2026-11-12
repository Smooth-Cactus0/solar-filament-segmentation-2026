"""Per-sample augmentation, driven by the `augment:` block of an experiment config.

The defaults are the augmentation every model up to exp8 was trained with, and
`apply` draws from the random generator in exactly the same order as the original
`dataset.augment` did: for the same generator state, this augmentation function
reproduces that old function's outputs exactly (tests/test_augment.py pins this).
Training runs themselves are not seed-reproducible -- torch is not seeded -- so
this is a guarantee about one augmentation call, not about replaying a full
training run byte for byte. The five original gates (flip_rows, flip_cols,
affine_p, photo_p, blur_p) always draw from the generator, even at p = 0, to
keep that legacy draw order; the newer transforms below them draw nothing from
the generator while their probability is 0.

Channel conventions (see dataset.build_input): 0 flat-fielded image, 1 raw
normalised image, 2 normalised radius. Photometric transforms touch channels 0-1
only; channel 2 is geometry and moves only with geometric transforms. In the
normalised channels 0.0 is quiet Sun, which is what cutout fills holes with.

The affine warp fills pixels pulled in from outside the frame with each
channel's own minimum, which is correct for channels 0-1 (that minimum is a
quiet/dark value) but wrong for channel 2: its minimum is the disk-centre
value (built as clip(r/R, 0, 1.2) - 0.6, so -0.6 at centre and +0.6 off-disk),
so warped-in border pixels default to reading as disk centre, which then
counts as on-disk in FilamentDataset's ``valid`` mask. Setting
``radius_border_sky=True`` fills channel 2's border with its own maximum (the
sky value) instead, so those pixels correctly read as off-disk. Default is
False so every model trained through exp8 keeps its exact old behaviour.

Channels 3 and up (``filseg.inputs``' latitude plane) are built so that 0 is their
off-disk value, and the warp fills them with 0.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    flip_rows: float = 0.5
    flip_cols: float = 0.5
    affine_p: float = 0.8
    rotate_deg: float = 15.0
    scale: tuple[float, float] = (0.95, 1.05)
    photo_p: float = 0.7
    contrast: tuple[float, float] = (0.85, 1.15)
    brightness: tuple[float, float] = (-0.1, 0.1)
    per_channel: bool = False
    blur_p: float = 0.25
    blur_kernels: tuple[int, ...] = (3, 5)
    rot90: float = 0.0
    noise_p: float = 0.0
    noise_sigma: float = 0.05
    gradient_p: float = 0.0
    gradient_amp: float = 0.2
    cutout_p: float = 0.0
    cutout_n: int = 4
    cutout_size: tuple[int, int] = (32, 96)
    copy_paste_p: float = 0.0
    copy_paste_max: int = 3
    radius_border_sky: bool = False
    bank_paste_p: float = 0.0
    bank_n: tuple[int, int] = (2, 6)
    bank_scale: tuple[float, float] = (0.7, 1.4)
    bank_contrast: tuple[float, float] = (0.7, 1.3)

    @classmethod
    def from_dict(cls, d: dict | None) -> "AugmentConfig":
        """Build from a config block; unknown keys are an error (typos must not pass silently)."""
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown augment option(s): {', '.join(unknown)}")
        kw = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        return cls(**kw)


def _hit(rng: np.random.Generator, p: float) -> bool:
    """True with probability p; draws from rng only when p > 0."""
    return p > 0 and rng.random() < p


def apply(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
          cfg: AugmentConfig) -> tuple[np.ndarray, np.ndarray]:
    """Augment one sample; see the module docstring for the RNG-order contract."""
    # --- original transforms, original draw order --------------------------------
    if rng.random() < cfg.flip_rows:
        x, y = x[:, ::-1], y[:, ::-1]
    if rng.random() < cfg.flip_cols:
        x, y = x[:, :, ::-1], y[:, :, ::-1]
    x, y = np.ascontiguousarray(x), np.ascontiguousarray(y)

    if rng.random() < cfg.affine_p:
        ang = float(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
        sc = float(rng.uniform(cfg.scale[0], cfg.scale[1]))
        h, w = y.shape[1:]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        x = np.stack([
            cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=float(c.max()) if (cfg.radius_border_sky and i == 2)
                           else 0.0 if i >= 3 else float(c.min()))
            for i, c in enumerate(x)
        ])
        y = np.stack([
            cv2.warpAffine(c, M, (w, h), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
            for c in y
        ])

    if rng.random() < cfg.photo_p:
        if cfg.per_channel:
            for c in (0, 1):
                x[c] = (x[c] * float(rng.uniform(*cfg.contrast))
                        + float(rng.uniform(*cfg.brightness)))
        else:
            x[:2] = (x[:2] * float(rng.uniform(*cfg.contrast))
                     + float(rng.uniform(*cfg.brightness)))

    if rng.random() < cfg.blur_p:
        k = int(rng.choice(list(cfg.blur_kernels)))
        x[:2] = np.stack([cv2.GaussianBlur(c, (k, k), 0) for c in x[:2]])

    # --- Phase-1 transforms: no draws while their p is 0 --------------------------
    if _hit(rng, cfg.rot90):
        k = int(rng.integers(1, 4))
        x = np.ascontiguousarray(np.rot90(x, k, axes=(1, 2)))
        y = np.ascontiguousarray(np.rot90(y, k, axes=(1, 2)))

    if _hit(rng, cfg.noise_p):
        x[:2] = x[:2] + rng.normal(0.0, cfg.noise_sigma, size=x[:2].shape).astype(np.float32)

    if _hit(rng, cfg.gradient_p):
        h, w = x.shape[1:]
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        amp = float(rng.uniform(-cfg.gradient_amp, cfg.gradient_amp))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = (xx / max(w - 1, 1)) * 2 - 1
        v = (yy / max(h - 1, 1)) * 2 - 1
        g = (amp * (np.cos(theta) * u + np.sin(theta) * v)).astype(np.float32)
        x[:2] = x[:2] + g

    if _hit(rng, cfg.cutout_p):
        h, w = x.shape[1:]
        lo, hi = int(cfg.cutout_size[0]), int(cfg.cutout_size[1])
        for _ in range(int(cfg.cutout_n)):
            s = int(rng.integers(lo, hi + 1))
            cy, cx = int(rng.integers(0, h)), int(rng.integers(0, w))
            y0, y1 = max(cy - s // 2, 0), min(cy + s // 2 + 1, h)
            x0, x1 = max(cx - s // 2, 0), min(cx + s // 2 + 1, w)
            x[:2, y0:y1, x0:x1] = 0.0
            y[:, y0:y1, x0:x1] = 0.0

    return np.ascontiguousarray(x), np.ascontiguousarray(y)


def paste_instances(x: np.ndarray, y: np.ndarray, donor_x: np.ndarray,
                    donor_masks: list[np.ndarray], rng: np.random.Generator,
                    max_n: int, margin: int = 2) -> tuple[np.ndarray, np.ndarray, int]:
    """Copy up to ``max_n`` donor filaments into this frame at their own location.

    A donor instance is skipped if its ``margin``-dilated mask touches any target
    pixel already present, so pasted filaments never merge with real ones. Channels
    0-1 are copied inside the dilated mask (the margin carries the filament's edge);
    only the undilated mask is added to the target. Channel 2 (radius) is geometry
    and is never copied.
    """
    x = x.copy()
    y = y.copy()
    order = rng.permutation(len(donor_masks)) if donor_masks else []
    k = np.ones((2 * margin + 1, 2 * margin + 1), dtype=np.uint8)
    n = 0
    for i in order:
        if n >= max_n:
            break
        m = donor_masks[int(i)] > 0.5
        if not m.any():
            continue
        grown = cv2.dilate(m.astype(np.uint8), k) > 0
        if (y[0][grown] > 0).any():
            continue
        x[:2, grown] = donor_x[:2, grown]
        y[0] = np.maximum(y[0], m.astype(y.dtype))
        n += 1
    return x, y, n
