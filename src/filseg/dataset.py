"""Dataset and augmentation for filament segmentation.

Two design decisions here come straight from the EDA and are worth stating
plainly, because both look like arbitrary choices and neither is.

**Three input channels, not one.** The raw frame is nearly featureless -- on-disk
brightness spans only 46 of 255 levels and the median filament is just 6.7%
darker than its surroundings, while limb darkening removes ~19% of the
brightness by the edge, roughly three times the signal. So channel 0 is
flat-fielded, channel 1 keeps the raw appearance (absolute brightness and cloud
cues the flat field erases), and channel 2 is the normalised radius, which hands
the model disk geometry for free. Three channels also means the pretrained stem
is used unmodified.

**Rotation is capped at +/-15 degrees.** Free 360-degree rotation is tempting
because the frame is a disk, but filaments cluster near the solar equator (93%
within 40 degrees latitude) and the images are north-up, so arbitrary rotation
would destroy exactly the positional prior we want the model to learn. Flips are
different: filaments split 50.5/49.5 north/south, so a vertical flip is a true
label-preserving symmetry. Elastic and grid distortion are excluded outright --
they deform thin morphology and a median filament tolerates only ~4 px of
positional error before it drops below the metric's IoU gate.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import augment as faug
from . import bank as fbank
from . import data as fdata
from . import inputs as finputs
from . import targets as ftargets
from .losses import LossConfig
from .baseline import illumination
from .cache import read_map
from .metric import Instance
from .paths import PROJECT_ROOT


FLAT_GAIN = 5.0     # channel 0 = clip((flat - 1) * FLAT_GAIN, -6, 6)
CONTRAST_RING = 5   # contrast-weight ring width in 2048-px units


def build_input(img: np.ndarray, disk: fdata.Disk, size: int, latitude: bool = False) -> np.ndarray:
    """Compose the 3-channel model input from one full-resolution frame.

    Returns float32 (3, size, size), or (4, ...) with the latitude plane, each plane
    roughly zero-centred.
    """
    illum = illumination(img).astype(np.float32)
    raw = img.astype(np.float32)

    # ch0: flat-fielded. 1.0 is quiet Sun; filaments sit below it. Centred and
    # scaled so the useful on-disk range spans roughly [-1, 1].
    #
    # Clipping is not cosmetic. Just outside the limb the illumination estimate
    # falls to near zero while the bright limb ring does not, so the quotient
    # explodes (values above 90 were observed). The loss is masked off-disk, but
    # the network still consumes those pixels and they would distort normalisation
    # statistics and near-limb features. Clip to a range that comfortably contains
    # every real filament signal.
    flat = raw / (illum + 1.0)
    flat = np.clip((flat - 1.0) * FLAT_GAIN, -6.0, 6.0)

    # ch1: raw brightness, normalised by the frame's own on-disk level so that
    # station-to-station exposure differences do not shift the input.
    scale = float(np.median(illum[illum > 20])) if (illum > 20).any() else 1.0
    rawn = np.clip(raw / max(scale, 1.0) - 1.0, -1.5, 3.0)

    # ch2: normalised radius. Off-disk (>1) is clipped so sky reads as a
    # constant rather than an unbounded ramp.
    yy, xx = np.mgrid[0 : img.shape[0], 0 : img.shape[1]]
    rr = np.hypot(xx - disk.cx, yy - disk.cy) / disk.radius
    rr = np.clip(rr, 0.0, 1.2).astype(np.float32) - 0.6

    planes = [flat, rawn, rr]
    if latitude:
        planes.append(finputs.latitude_plane(disk, img.shape))
    stack = np.stack(planes).astype(np.float32)
    if size != img.shape[0]:
        stack = np.stack([
            cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA) for c in stack
        ])
    return stack


def model_input(img: np.ndarray, disk: fdata.Disk, size: int, inputs=None) -> np.ndarray:
    """The array the model sees at inference: ``build_input`` then ``inputs.select``."""
    return finputs.select(build_input(img, disk, size, latitude=finputs.wants_latitude(inputs)),
                          inputs)


def render_mask(ann_ids, size: int, full: int = 2048) -> np.ndarray:
    """Rasterise one annotation pass to a (size, size) float mask.

    Polygons are drawn at full resolution and then area-averaged down, so a
    filament thinner than the output pixel grid survives as a soft-edged partial
    value instead of disappearing.
    """
    m = np.zeros((full, full), dtype=np.uint8)
    for aid in ann_ids:
        np.maximum(m, fdata.ann_to_mask(aid), out=m)
    if size != full:
        return cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
    return m.astype(np.float32)


def render_spine(ann_ids, size: int, full: int = 2048, thickness: int = 7) -> np.ndarray:
    """Rasterise the annotated spine polylines to a (size, size) float map.

    Every annotation carries a hand-drawn ``spine`` tracing the filament axis --
    8,199 of them, and the competition metric ignores all of it. Predicting the
    spine as an auxiliary task is therefore free supervision, and on a dataset of
    only 707 exposures the regularisation matters more than the extra capacity
    costs.

    Drawn thick at full resolution and area-averaged down, so the line survives
    downsampling as a soft ridge rather than an aliased dotted trail. Values are
    normalised to [0, 1]: the target is "how centred on a filament axis is this
    pixel", not a hard binary.
    """
    m = np.zeros((full, full), dtype=np.uint8)
    for aid in ann_ids:
        pts = fdata.spine_of(aid)
        if len(pts) < 2:
            continue
        cv2.polylines(m, [np.round(pts).astype(np.int32)], isClosed=False,
                      color=255, thickness=thickness, lineType=cv2.LINE_AA)
    out = m.astype(np.float32) / 255.0
    if size != full:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
    peak = float(out.max())
    return out / peak if peak > 0 else out


def augment(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Physics-constrained augmentation with the default settings -- see filseg.augment.

    ``y`` is (C, H, W): channel 0 is the filament mask and, when the spine head
    is enabled, channel 1 is the spine map; every geometric transform moves all
    target channels together, so the two stay pixel-aligned with ``x``.

    Kept as a function so existing callers and tests keep working; configurable
    augmentation goes through ``filseg.augment.apply``.
    """
    return faug.apply(x, y, rng, faug.AugmentConfig())


class _DiskLookup(dict):
    """``disks[stem]`` for build_bank, detecting and caching through the dataset."""

    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def __missing__(self, stem):
        return self.ds._disk(stem, fdata.load_image(stem, "train"))


def crop_window(disk: fdata.Disk, size: int, crop: int, rng: np.random.Generator | None = None,
                j: int = 0, per: int = 1) -> tuple[int, int]:
    """Top-left of a ``crop`` px window of a ``size`` px frame (the disk is in 2048 px units).

    Training (``rng``): a centre drawn uniformly over the disk's area. Validation (no
    ``rng``): fixed centres, the disk centre for ``per == 1``, else ``per`` points at half
    the radius, so the epoch proxy sees the same crops every time.
    """
    s = size / 2048.0
    if rng is not None:
        r = disk.radius * np.sqrt(rng.random())
        a = rng.random() * 2 * np.pi
    elif per > 1:
        r, a = disk.radius / 2, 2 * np.pi * j / per
    else:
        r, a = 0.0, 0.0
    cy, cx = disk.cy + r * np.sin(a), disk.cx + r * np.cos(a)
    y0 = int(np.clip(round(cy * s - crop / 2), 0, size - crop))
    x0 = int(np.clip(round(cx * s - crop / 2), 0, size - crop))
    return y0, x0


class FilamentDataset(Dataset):
    """One item per annotation pass.

    Every pass is an independent sample: the same exposure appears 2-3 times
    with different labels, and BCE under inconsistent targets converges to the
    per-pixel consensus. That is how the model's sigmoid output becomes
    ``P(an annotator marks this pixel)`` -- the quantity the acceptance rule
    wants to threshold -- with no extra machinery.

    With ``loss.target: consensus`` every pass of an exposure instead carries the
    mean mask over all of that exposure's passes, so the per-pixel agreement is
    the target itself rather than an average the loss has to discover.
    """

    def __init__(
        self,
        folds_df,
        ann_df,
        size: int = 1024,
        train: bool = True,
        seed: int = 0,
        cache_disks: dict | None = None,
        spine: bool = False,
        augment_cfg: dict | None = None,
        loss_cfg: dict | None = None,
        inputs: list | None = None,
        ignore_dir: str | None = None,
        crops: dict | None = None,
    ) -> None:
        self.rows = folds_df.reset_index(drop=True)
        self.size = size
        self.train = train
        # Native-resolution crop mode: a ``crop`` px window of the ``size`` px frame, ``per`` per
        # row. Training draws them at random; validation takes ``per`` fixed crops per row (half
        # the radius out, evenly spaced in angle; the centre when ``per == 1``), so the epoch
        # Dice proxy sees the limb as well as the disk centre, the same crops every epoch.
        self.crop = int(crops["size"]) if crops else 0
        self.per = int(crops.get("per_exposure", 1)) if crops else 1
        if self.crop and not 0 < self.crop <= size:
            raise ValueError(f"crop size {self.crop} must be in (0, {size}]")
        self.seed = seed
        self.disks = cache_disks or {}
        self.spine = spine
        self.aug = faug.AugmentConfig.from_dict(augment_cfg)
        if self.crop and train and (self.aug.copy_paste_p > 0 or self.aug.bank_paste_p > 0):
            raise ValueError("crop mode does not support copy-paste or bank-paste (they work on full frames)")
        self.by_pass = {k: list(v) for k, v in ann_df.groupby("image_id").ann_id}
        self.loss = LossConfig.from_dict(loss_cfg)
        # Every pass of an exposure present in this split, for the consensus target.
        self.passes_of = {k: list(v) for k, v in self.rows.groupby("file_stem").image_id}
        self.inputs = finputs.validate(inputs)
        self.latitude = "latitude" in self.inputs
        derived = [n for n in self.inputs if n in finputs.DERIVED]
        if self.crop and derived:
            # Derived planes are computed on the array they are given: on a crop in training, on
            # the full frame before tiling at inference, so the two would see different planes.
            raise ValueError(f"crop mode does not support derived input planes {derived}")
        # Filament bank for the bank-paste augmentation: the split's own filaments only.
        self.bank = (fbank.build_bank(self.rows, self.by_pass, self.size, _DiskLookup(self),
                                      fdata.load_image)
                     if (train and self.aug.bank_paste_p > 0) else None)
        # Per-exposure ignore masks (training only): pixels dropped from the loss's ``valid``.
        self.ignore_dir = Path(ignore_dir) if ignore_dir else None
        if self.ignore_dir is not None and not self.ignore_dir.is_absolute():
            self.ignore_dir = PROJECT_ROOT / self.ignore_dir
        self.n_ignore = (sum((self.ignore_dir / f"{s}.png").is_file() for s in self.rows.file_stem.unique())
                         if (train and self.ignore_dir is not None) else 0)

    def __len__(self) -> int:
        return len(self.rows) * self.per

    def exposure_weights(self) -> np.ndarray:
        """Sampling weight per row: ``1 / n_passes`` of its exposure (float64).

        Feeds a ``WeightedRandomSampler`` so ``loss.target: consensus`` training
        sees every exposure equally often per epoch, instead of a 3-pass exposure
        3x as often as a 1-pass one.
        """
        w = np.array(
            [1.0 / len(self.passes_of[stem]) for stem in self.rows.file_stem],
            dtype=np.float64,
        )
        return np.repeat(w, self.per)

    def _disk(self, stem: str, img: np.ndarray) -> fdata.Disk:
        if stem not in self.disks:
            self.disks[stem] = fdata.detect_disk(img)
        return self.disks[stem]

    def __getitem__(self, i: int):
        row = self.rows.iloc[i // self.per]
        rng = (np.random.default_rng(self.seed + i * 7919 + int(torch.initial_seed() % 100000))
               if self.train else None)
        img = fdata.load_image(row.file_stem, "train")
        disk = self._disk(row.file_stem, img)

        ann_ids = self.by_pass.get(row.image_id, [])
        x = build_input(img, disk, self.size, latitude=self.latitude)
        if self.loss.target == "consensus":
            pids = self.passes_of.get(row.file_stem, [row.image_id])
            mask = ftargets.consensus([render_mask(self.by_pass.get(p, []), self.size) for p in pids],
                                      (self.size, self.size))
        else:
            mask = render_mask(ann_ids, self.size)
        channels = [mask]
        if self.spine:
            channels.append(render_spine(ann_ids, self.size))
        y = np.stack(channels)

        has_ignore = False
        if self.train and self.ignore_dir is not None:
            p = self.ignore_dir / f"{row.file_stem}.png"
            if p.is_file():
                ign = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if ign is None:
                    raise OSError(f"unreadable ignore mask {p}")
                ign = cv2.resize(ign.astype(np.float32) / 255.0, (self.size, self.size), interpolation=cv2.INTER_AREA)
                y = np.concatenate([y, (ign > 0).astype(np.float32)[None]], axis=0)   # rides along the geometry
                has_ignore = True

        if self.crop:
            y0, x0 = crop_window(disk, self.size, self.crop, rng, j=i % self.per, per=self.per)
            x = x[:, y0:y0 + self.crop, x0:x0 + self.crop]
            y = y[:, y0:y0 + self.crop, x0:x0 + self.crop]

        if self.train:
            if self.aug.copy_paste_p > 0 and rng.random() < self.aug.copy_paste_p:
                j = int(rng.integers(0, len(self.rows)))
                donor = self.rows.iloc[j]
                if donor.file_stem != row.file_stem:
                    dimg = fdata.load_image(donor.file_stem, "train")
                    # paste_instances only ever copies donor_x[:2] (channels 0-1); the
                    # latitude plane is the costliest loader op (~120 ms/item), so it is
                    # never worth building for a donor frame.
                    dx = build_input(dimg, self._disk(donor.file_stem, dimg), self.size,
                                     latitude=False)
                    dmasks = [render_mask([aid], self.size)
                              for aid in self.by_pass.get(donor.image_id, [])]
                    x, y0, _ = faug.paste_instances(x, y[:1], dx, dmasks, rng,
                                                    self.aug.copy_paste_max)
                    y = np.concatenate([y0, y[1:]], axis=0)
            if self.bank is not None and rng.random() < self.aug.bank_paste_p:
                # Every annotator's filaments of this exposure are off limits, not just this pass's
                # (precomputed by build_bank, so no per-item rendering).
                others = self.bank.avoid_mask(row.file_stem, self.size)
                x, y0, _ = fbank.paste_bank(x, y[:1], self.bank.items, rng, self.aug, avoid=others)
                y = np.concatenate([y0, y[1:]], axis=0)
            x, y = faug.apply(x, y, rng, self.aug)

        ignore = None
        if has_ignore:
            ignore, y = y[-1] > 0.5, y[:-1]

        # Loss is masked off-disk: the model should never spend capacity on sky.
        # This stops at 0.98 R (a margin at the limb), tighter than the < 1.0 R full
        # disk the derived input planes use (filseg.inputs.on_disk) -- the 0.98-1.0 R
        # ring is outside the loss either way, so the two boundaries never conflict.
        flat_plane = x[0] / FLAT_GAIN          # the flat-field ratio minus one, the units of flat_contrast
        valid = ((x[2] + 0.6) < 0.98) & (~ignore if ignore is not None else True)
        x = finputs.select(x, self.inputs)
        out = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "valid": torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
            "image_id": row.image_id,
            "file_stem": row.file_stem,
        }
        # Loss-only targets, computed from the *augmented* mask so they stay aligned.
        if self.train and self.loss.skeleton_weight > 0:
            out["skel"] = torch.from_numpy(ftargets.tubed_skeleton(y[0])).unsqueeze(0)
        if self.train and (self.loss.small_weight_max > 1.0 or self.loss.contrast_weight_max > 1.0):
            w = np.ones(y[0].shape, dtype=np.float32)
            if self.loss.small_weight_max > 1.0:
                ref = self.loss.small_ref_area * (self.size / 2048) ** 2
                w *= ftargets.small_instance_weights(y[0], ref, self.loss.small_weight_max)
            if self.loss.contrast_weight_max > 1.0:
                ring = max(2, round(CONTRAST_RING * self.size / 2048))
                w *= ftargets.contrast_instance_weights(y[0], flat_plane, self.loss.contrast_ref,
                                                        self.loss.contrast_weight_max, ring)
            out["pixw"] = torch.from_numpy(w).unsqueeze(0)
        return out


def load_frame(path: str | Path) -> np.ndarray:
    """One grayscale frame from a file path, as uint8 -- decoded the way ``fdata.load_image`` does."""
    with Image.open(path) as im:
        return np.array(im.convert("L"))


class PseudoDataset(Dataset):
    """One item per unlabelled GONG frame; the target is the teacher ensemble's soft map.

    The pretraining set: archive frames from days MAGFiLO never observed, labelled by our own
    fold-averaged ensemble as 8-bit maps (``<targets_dir>/<stem>.png``, ``cache_probs.write_map``).
    The item is built exactly as ``FilamentDataset`` builds one -- ``build_input``, the same
    ``faug.apply`` augmentation, the same 0.98 R loss mask, the same input-plane selection, the
    same keys -- so the trainer's loop, the Dice proxy and ``FilamentLoss`` consume it unchanged.

    The target stays soft. A teacher probability of 0.6 is the BCE target 0.6: the same "how likely
    is an annotator to mark this pixel" quantity per-pass training converges to, so binarising it
    would throw away the one thing the teacher knows that a threshold does not.

    No annotations exist here, so everything built from them is absent: no copy-paste or bank-paste
    (refused), no ignore masks, no spine channel, no consensus target (refused).
    """

    def __init__(
        self,
        frames: list[str],
        images_dir: Path,
        targets_dir: Path,
        size: int,
        train: bool,
        seed: int = 0,
        augment_cfg: dict | None = None,
        loss_cfg: dict | None = None,
        inputs: list | None = None,
    ) -> None:
        self.frames = list(frames)
        self.images_dir = Path(images_dir)
        self.targets_dir = Path(targets_dir)
        self.size = int(size)
        self.train = train
        self.seed = seed
        self.disks: dict[str, fdata.Disk] = {}
        self.aug = faug.AugmentConfig.from_dict(augment_cfg)
        if train and (self.aug.copy_paste_p > 0 or self.aug.bank_paste_p > 0):
            raise ValueError("pseudo-label frames have no annotated instances: copy-paste and "
                             "bank-paste are not supported")
        self.loss = LossConfig.from_dict(loss_cfg)
        if self.loss.target == "consensus":
            raise ValueError("pseudo-label frames carry one soft target, not annotation passes: "
                             "loss.target 'consensus' is not supported")
        self.inputs = finputs.validate(inputs)
        self.latitude = "latitude" in self.inputs

    def __len__(self) -> int:
        return len(self.frames)

    def _disk(self, stem: str, img: np.ndarray) -> fdata.Disk:
        if stem not in self.disks:
            self.disks[stem] = fdata.detect_disk(img)
        return self.disks[stem]

    def target(self, stem: str) -> np.ndarray:
        """The teacher map of ``stem`` as float32 probabilities at the dataset size."""
        t = read_map(self.targets_dir / f"{stem}.png", quant=255)
        if t.shape != (self.size, self.size):
            t = cv2.resize(t, (self.size, self.size), interpolation=cv2.INTER_AREA)
        return t.astype(np.float32, copy=False)

    def __getitem__(self, i: int):
        stem = self.frames[i]
        rng = (np.random.default_rng(self.seed + i * 7919 + int(torch.initial_seed() % 100000))
               if self.train else None)
        img = load_frame(self.images_dir / f"{stem}.jpg")
        disk = self._disk(stem, img)
        x = build_input(img, disk, self.size, latitude=self.latitude)
        y = self.target(stem)[None]
        if self.train:
            x, y = faug.apply(x, y, rng, self.aug)

        # The same loss mask as FilamentDataset: off-disk and the 0.98-1.0 R ring never count,
        # read from the radius plane before the input planes are selected.
        flat_plane = x[0] / FLAT_GAIN
        valid = (x[2] + 0.6) < 0.98
        x = finputs.select(x, self.inputs)
        out = {
            "x": torch.from_numpy(x),
            "y": torch.from_numpy(np.ascontiguousarray(y)),
            "valid": torch.from_numpy(valid.astype(np.float32)).unsqueeze(0),
            "image_id": stem,
            "file_stem": stem,
        }
        # Loss-only targets, as FilamentDataset computes them, from the augmented target (the
        # target helpers binarise at 0.5, so they see the teacher's confident region).
        if self.train and self.loss.skeleton_weight > 0:
            out["skel"] = torch.from_numpy(ftargets.tubed_skeleton(y[0])).unsqueeze(0)
        if self.train and (self.loss.small_weight_max > 1.0 or self.loss.contrast_weight_max > 1.0):
            w = np.ones(y[0].shape, dtype=np.float32)
            if self.loss.small_weight_max > 1.0:
                ref = self.loss.small_ref_area * (self.size / 2048) ** 2
                w *= ftargets.small_instance_weights(y[0], ref, self.loss.small_weight_max)
            if self.loss.contrast_weight_max > 1.0:
                ring = max(2, round(CONTRAST_RING * self.size / 2048))
                w *= ftargets.contrast_instance_weights(y[0], flat_plane, self.loss.contrast_ref,
                                                        self.loss.contrast_weight_max, ring)
            out["pixw"] = torch.from_numpy(w).unsqueeze(0)
        return out


def instances_from_annotations(ann_ids) -> list[Instance]:
    """Ground-truth instances at full 2048 resolution, for scoring."""
    from .metric import from_mask

    out = []
    for aid in ann_ids:
        it = from_mask(fdata.ann_to_mask(aid).astype(bool))
        if it is not None:
            out.append(it)
    return out
