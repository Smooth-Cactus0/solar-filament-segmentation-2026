"""Config-driven training loss: the ``loss:`` block of an experiment config (spec §2c).

Every option defaults to the loss every model through recipe r1 was trained with --
masked BCE with ``pos_weight`` plus 0.25 x soft Dice, per-pass targets, one output
channel -- and at those defaults ``FilamentLoss`` returns exactly what
``model.MaskSpineLoss`` returns (tests/test_losses.py pins it).

Options:

- ``target``: ``pass`` (each annotation pass is a sample with its own mask, drawn by
  the baseline's ``shuffle=True`` loader) or ``consensus`` (every pass of an exposure
  carries the mean mask over all of that exposure's passes). Consensus training
  draws rows with a ``WeightedRandomSampler`` weighted ``1 / n_passes`` of the row's
  exposure (``FilamentDataset.exposure_weights``), so every exposure is seen equally
  often per epoch instead of 3-pass exposures 3x as often as 1-pass ones; the epoch
  length and optimiser step count stay the same as the baseline's.
- ``head``: ``mask`` or ``ordinal`` -- three channels trained on soft cumulative
  agreement levels (``ordinal_targets``: ``level_k = clamp(3 * frac - k, 0, 1)``),
  whose mean is the agreement fraction exactly. Needs ``target: consensus``, whose
  value is that fraction. 411 of 707 exposures have one pass, so for them all three
  levels coincide (0 or 1).
- ``skeleton_weight``: skeleton-recall term (Kirchhoff et al., ECCV 2024), 1 minus
  the predicted probability's recall of the target's tubed skeleton. The skeleton
  is computed from the mask (``filseg.targets``), never from annotated spines.
- ``boundary_weight`` / ``boundary_kernel``: BCE weight ``1 + w`` on a band of
  ``boundary_kernel`` px around every target edge.
- ``small_weight_max`` / ``small_ref_area``: BCE weight on each target instance's
  pixels, ``(ref_area / area) ** 0.5`` clipped to ``[1, small_weight_max]``;
  ``small_ref_area`` is in 2048-px units (1228 px is the median annotation area)
  and the dataset rescales it to the working size. 1 disables it.
- ``contrast_weight_max`` / ``contrast_ref``: BCE weight on each target instance's pixels,
  ``ref / contrast`` clipped to ``[1, contrast_weight_max]``, where contrast is the flat-field
  ring-minus-inside difference the scorer calls ``flat_contrast`` and ``contrast_ref`` its
  median over all annotations (0.116).
  1 disables it. Combines multiplicatively with ``small_weight_max``.
- ``lovasz_weight``: plus the Lovász hinge on the binarised target (Berman et al.,
  CVPR 2018).

Pixel weights move the BCE optimum, so a weighted model's probabilities are no
longer calibrated annotator frequencies. The downstream scorer recalibrates, but a
threshold grid reads differently for such a model.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import head_channels, mask_prob

TARGETS = ("pass", "consensus")
N_ORDINAL_LEVELS = 3


@dataclass(frozen=True)
class LossConfig:
    target: str = "pass"
    head: str = "mask"
    skeleton_weight: float = 0.0
    boundary_weight: float = 0.0
    boundary_kernel: int = 5
    small_weight_max: float = 1.0
    small_ref_area: float = 1228.0
    contrast_weight_max: float = 1.0
    contrast_ref: float = 0.116
    lovasz_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise ValueError(f"loss.target must be one of {TARGETS}, got {self.target!r}")
        head_channels(self.head)  # raises on an unknown head
        if self.head == "ordinal" and self.target != "consensus":
            raise ValueError("loss.head 'ordinal' needs loss.target 'consensus' "
                             "(it predicts agreement levels)")
        if self.boundary_kernel < 3 or self.boundary_kernel % 2 == 0:
            raise ValueError("loss.boundary_kernel must be an odd integer >= 3")
        if self.small_weight_max < 1.0:
            raise ValueError("loss.small_weight_max must be >= 1 (1 disables it)")
        if self.contrast_weight_max < 1.0:
            raise ValueError("loss.contrast_weight_max must be >= 1 (1 disables it)")
        if self.contrast_ref <= 0:
            raise ValueError("loss.contrast_ref must be > 0")
        for k in ("skeleton_weight", "boundary_weight", "lovasz_weight"):
            if getattr(self, k) < 0:
                raise ValueError(f"loss.{k} must be >= 0")

    @classmethod
    def from_dict(cls, d: dict | None) -> "LossConfig":
        """Build from a config block; unknown keys are an error (typos must not pass silently)."""
        if not d:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(d) - known)
        if unknown:
            raise ValueError(f"unknown loss option(s): {', '.join(unknown)}")
        return cls(**d)


def ordinal_targets(frac: torch.Tensor) -> torch.Tensor:
    """(B, 1, H, W) agreement fraction -> (B, 3, H, W) soft cumulative ("thermometer") levels.

    ``level_k = clamp(3 * frac - k, 0, 1)`` for k = 0, 1, 2, so the mean of the three
    levels equals ``frac`` exactly for every value, not only exact thirds. A 2-pass
    exposure's frac = 0.5 becomes [1, 0.5, 0] (mean 0.5) instead of being thresholded
    down to [1, 0, 0] (mean 1/3); a 3-pass exposure's exact-third fracs (0, 1/3, 2/3, 1)
    still give the same hard 0/1 targets as the old threshold-at-1/3-2/3-1 encoding.
    """
    return torch.cat([torch.clamp(3.0 * frac - k, 0.0, 1.0) for k in range(N_ORDINAL_LEVELS)],
                     dim=1)


def boundary_band(target: torch.Tensor, kernel: int) -> torch.Tensor:
    """1 on a ``kernel``-px band around every edge of the binarised target, else 0."""
    t = (target >= 0.5).to(target.dtype)
    pad = kernel // 2
    dil = F.max_pool2d(t, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-t, kernel, stride=1, padding=pad)
    return dil - ero


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    inter = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jac = 1.0 - inter / union
    jac[1:] = jac[1:] - jac[:-1]
    return jac


def lovasz_hinge(logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Mean over images of the Lovász hinge on on-disk pixels (labels binarised at 0.5)."""
    losses = []
    for lg, lb, v in zip(logits, labels, valid):
        m = v.reshape(-1) > 0.5
        z = lg.reshape(-1)[m].float()
        y = (lb.reshape(-1)[m] >= 0.5).float()
        if z.numel() == 0:
            losses.append(logits.sum() * 0.0)
            continue
        errors = 1.0 - z * (2.0 * y - 1.0)
        errors_sorted, perm = torch.sort(errors, descending=True)
        losses.append(torch.dot(F.relu(errors_sorted), _lovasz_grad(y[perm])))
    return torch.stack(losses).mean()


class FilamentLoss(nn.Module):
    """Masked BCE (+ optional pixel weights) + soft Dice + the optional terms above.

    ``target`` channel 0 is the mask target (per-pass mask or consensus fraction);
    channel 1, when present, is the spine map of the legacy spine head, whose output
    channel sits right after the mask head's.
    """

    def __init__(self, cfg: LossConfig | None = None, dice_weight: float = 0.25,
                 pos_weight: float = 3.0, spine_weight: float = 0.0,
                 spine_pos_weight: float = 10.0) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        if self.cfg.head == "ordinal" and spine_weight > 0:
            raise ValueError("the ordinal head and the spine head cannot be combined")
        self.dice_weight = dice_weight
        self.spine_weight = spine_weight
        self.nh = head_channels(self.cfg.head)
        self.register_buffer("pos_weight", torch.tensor(pos_weight))
        self.register_buffer("spine_pos_weight", torch.tensor(spine_pos_weight))

    def forward(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                skel: torch.Tensor | None = None, pixw: torch.Tensor | None = None) -> torch.Tensor:
        c = self.cfg
        t = target[:, :1]
        n = valid.sum().clamp_min(1.0)
        w = valid if pixw is None else valid * pixw
        if c.boundary_weight > 0:
            w = w * (1.0 + c.boundary_weight * boundary_band(t, c.boundary_kernel))
        if c.head == "ordinal":
            bce = F.binary_cross_entropy_with_logits(
                logits[:, :3], ordinal_targets(t), pos_weight=self.pos_weight,
                reduction="none").mean(dim=1, keepdim=True)
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits[:, :1], t, pos_weight=self.pos_weight, reduction="none")
        loss = (bce * w).sum() / n

        p = mask_prob(logits, c.head) if (self.dice_weight > 0 or c.skeleton_weight > 0) else None
        if self.dice_weight > 0:
            pv, tv = p * valid, t * valid
            num = 2.0 * (pv * tv).sum() + 1.0
            den = pv.sum() + tv.sum() + 1.0
            loss = loss + self.dice_weight * (1.0 - num / den)
        if c.skeleton_weight > 0 and skel is not None:
            s = skel * valid
            recall = (p * s).sum() / s.sum().clamp_min(1.0)
            loss = loss + c.skeleton_weight * (1.0 - recall)
        if c.lovasz_weight > 0:
            z = logits[:, :1] if c.head == "mask" else torch.logit(
                mask_prob(logits, c.head).float().clamp(1e-6, 1 - 1e-6))
            loss = loss + c.lovasz_weight * lovasz_hinge(z, t, valid)
        if self.spine_weight > 0 and logits.shape[1] > self.nh and target.shape[1] > 1:
            spine = F.binary_cross_entropy_with_logits(
                logits[:, self.nh:self.nh + 1], target[:, 1:2],
                pos_weight=self.spine_pos_weight, reduction="none")
            loss = loss + self.spine_weight * (spine * valid).sum() / n
        return loss
