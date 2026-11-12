"""Train one fold of the filament segmentation model, or pretrain it on pseudo-labels.

Usage:
    PYTHONPATH=src python scripts/train.py --config configs/exp1_convnext_1024.yaml --fold 0
    PYTHONPATH=src python scripts/train.py --config configs/... --fold 0 --smoke
    PYTHONPATH=src python scripts/train.py --config configs/pre_r2g_gong_1024.yaml

A config with a ``pseudo`` block is a pretraining run, not a fold: it trains on
unlabelled GONG frames whose target is the teacher ensemble's soft map (``PseudoDataset``),
holds out a share of the frames for the Dice proxy, takes no ``--fold`` and runs no PQ sweep
(there are no annotations to score against). Its checkpoint and results JSON are
``<name>.pt`` / ``<name>.json``, with no fold suffix, so the r2 chain warm-starts from it with
``--init-from checkpoints/<name>.pt --require-init-result``.

Validation deserves a note. Exp 0 measured that scoring the *same* predictions
against different annotators of one exposure moves PQ by 0.015 -- as large as
the gap between our local score and the leaderboard. So validation here scores
each prediction against **every available annotation pass** and averages. Picking
one pass arbitrarily would inject enough noise to fake or hide a real gain.

Model selection during training uses a cheap Dice proxy at working resolution;
the expensive instance-level PQ sweep (threshold x min-area, at full 2048) runs
once at the end on the best checkpoint, which is also where the operating point
gets chosen.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filseg import data as fdata  # noqa: E402
from filseg.dataset import FilamentDataset, PseudoDataset, instances_from_annotations  # noqa: E402
from filseg.inference import tiled  # noqa: E402
from filseg.instances import probs_to_instances  # noqa: E402
from filseg.metric import PQAccumulator  # noqa: E402
from filseg.inputs import validate as validate_inputs  # noqa: E402
from filseg.losses import FilamentLoss, LossConfig  # noqa: E402
from filseg.model import build_from_config, head_of, mask_prob  # noqa: E402
from filseg.paths import PROJECT_ROOT, output_root  # noqa: E402

# Outputs go to output_root(), which FILSEG_OUTPUT_ROOT redirects to
# /kaggle/working when the code itself is mounted read-only.
CKPT_DIR = output_root() / "checkpoints"
RESULT_DIR = output_root() / "results"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_config(path: str) -> dict:
    """Configs may inherit with `base:` -- see filseg.config."""
    from filseg.config import load_config as _load_config

    return _load_config(path)


def already_done(result_dir: Path, name: str) -> bool:
    """True when `result_dir/<name>.json` exists and is non-empty (a finished run)."""
    p = Path(result_dir) / f"{name}.json"
    return p.is_file() and p.stat().st_size > 0


def init_result_missing(result_dir: Path, init_from: str) -> bool:
    """True when the warm-start checkpoint's own results JSON is missing or empty.

    Progressive-resize stage 2 warm-starts from a stage-1 checkpoint file
    (``<name>_fold<k>.pt``); its result is ``<name>_fold<k>.json``. If that JSON
    is absent or empty, stage 1 crashed or never finished, and stage 2 must not
    quietly warm-start from whatever partial weights happen to be on disk.
    """
    return not already_done(result_dir, Path(init_from).stem)


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, tta: bool, scales=None, tile: int | None = None,
                  overlap: float = 0.25) -> torch.Tensor:
    """Sigmoid probabilities, averaged over flip symmetries and input scales.

    Only flips are used: they are true label-preserving symmetries of the solar
    disk (filaments split 50.5/49.5 north/south). Rotation TTA would break the
    latitude prior, so it is deliberately absent.
    """
    if tile:
        if scales:
            raise ValueError("tiled inference and multi-scale TTA are exclusive")
        return tiled(lambda v: predict_probs(model, v, tta), x, tile, overlap)
    # Channel 0 is the filament mask. With the spine auxiliary head enabled,
    # channel 1 carries the spine map -- a training signal only, which must
    # never reach instance extraction.
    #
    # `scales` runs the same image at several input sizes and averages the
    # probability maps at the native size. Resolution is the one lever that has
    # measurably worked on this task (1024 -> 1536 gave +0.022 PQ), and a
    # fully-convolutional encoder can be evaluated at sizes it was not trained
    # at, so this buys some of that benefit without retraining. Averaging happens
    # in probability space, before any threshold, for the same reason the map is
    # upsampled before thresholding: binarising early discards the sub-pixel
    # agreement a 4 px translation budget depends on.
    views = [(x, None)] if not tta else [
        (x, None), (torch.flip(x, [2]), [2]),
        (torch.flip(x, [3]), [3]), (torch.flip(x, [2, 3]), [2, 3]),
    ]
    size = x.shape[-1]
    sizes = [size] if not scales else list(scales)
    total = None
    for s in sizes:
        for v, dims in views:
            vi = v if s == size else F.interpolate(
                v, size=(s, s), mode="bilinear", align_corners=False)
            q = mask_prob(model(vi), head_of(model))
            if s != size:
                q = F.interpolate(q, size=(size, size), mode="bilinear",
                                  align_corners=False)
            if dims is not None:
                q = q.flip(dims)
            total = q if total is None else total + q
    return total / (len(sizes) * len(views))


@torch.no_grad()
def ema_update(ema_model, model, decay: float, step: int) -> float:
    """Move the EMA weights one step toward the live weights; returns the decay used.

    The decay is warmed up as ``min(decay, (1 + step) / (10 + step))`` so the first
    updates copy the live model almost verbatim instead of averaging in the random
    initial weights. Buffers (LayerNorm has none that matter, but keep it general)
    are copied rather than averaged.
    """
    d = min(decay, (1.0 + step) / (10.0 + step))
    live = dict(model.named_parameters())
    for name, p_ema in ema_model.named_parameters():
        p_ema.mul_(d).add_(live[name].detach(), alpha=1.0 - d)
    live_buf = dict(model.named_buffers())
    for name, b_ema in ema_model.named_buffers():
        b_ema.copy_(live_buf[name])
    return d


def make_ema(model):
    """A frozen deep copy of ``model`` to hold the exponential moving average."""
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema.eval()
    return ema


@torch.no_grad()
def swa_update(swa_state: dict | None, model, n_averaged: int) -> dict:
    """Fold ``model``'s weights into an equal-weight running mean of ``n_averaged`` snapshots.

    Stochastic weight averaging over the last epochs of the cosine schedule. Floating
    tensors are averaged in float32; integer buffers take the latest value. The
    snapshot is a copy, never a view of the live weights.
    """
    snap = {k: v.detach().clone().float() if v.is_floating_point() else v.detach().clone()
            for k, v in model.state_dict().items()}
    if swa_state is None:
        return snap
    for k, v in snap.items():
        if v.is_floating_point():
            swa_state[k].mul_(n_averaged / (n_averaged + 1)).add_(v, alpha=1.0 / (n_averaged + 1))
        else:
            swa_state[k] = v
    return swa_state


def update_bn(model, loader, device, amp: bool, amp_dtype, max_batches: int = 150) -> int:
    """Re-estimate BatchNorm running statistics of ``model`` from ``loader``; returns batches used.

    Averaging BatchNorm running statistics across SWA snapshots is not the running
    statistics of the averaged weights (Izmailov et al., 2018) -- the smp UNet
    decoder has BatchNorm2d layers (10 in the r1 model) even though the ConvNeXt-V2
    encoder is LayerNorm-only. A model with no BatchNorm (``_BatchNorm``) modules
    (e.g. a bare encoder) is returned untouched.

    Each BN's running stats are reset and its momentum set to ``None`` (cumulative
    moving average, so every batch counts equally regardless of how many are seen),
    the model runs up to ``max_batches`` forward passes over ``b["x"]`` from
    ``loader`` in train mode under ``torch.no_grad()`` and the same autocast as
    training, then each BN's original momentum is restored and the model is set
    back to eval mode.
    """
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return 0
    momenta = {m: m.momentum for m in bns}
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    model.train()
    n = 0
    with torch.no_grad():
        for b in loader:
            if n >= max_batches:
                break
            x = b["x"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp, dtype=amp_dtype):
                model(x)
            n += 1
    for m, mom in momenta.items():
        m.momentum = mom
    model.eval()
    return n


def worker_init(_: int) -> None:
    """Keep each DataLoader worker single-threaded.

    OpenCV spawns its own thread pool in every worker (cv2.resize, warpAffine,
    GaussianBlur all use it), so eight workers on a 16-thread box oversubscribe it
    several times over. On the RTX 3080 that showed up as 16% system time, a GPU idle
    more than half the time and 7 min epochs against a 4 min compute budget.
    """
    import cv2

    cv2.setNumThreads(0)
    torch.set_num_threads(1)


def dice_proxy(model, loader, device, amp: bool) -> float:
    """Cheap soft-Dice at working resolution, for epoch-to-epoch model selection."""
    model.eval()
    inter = union = 0.0
    with torch.no_grad():
        for b in loader:
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            v = b["valid"].to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                p = mask_prob(model(x), head_of(model))
            p = (p * v).float()
            t = (y[:, :1] * v).float()
            inter += 2.0 * (p * t).sum().item()
            union += p.sum().item() + t.sum().item()
    return inter / max(union, 1e-6)


def make_scheduler(opt, epochs: int, steps_per_epoch: int, lr_min: float, warmup_epochs: int = 0):
    """Per-optimiser-step LR schedule: cosine to ``lr_min``, optionally after a linear warmup.

    ``warmup_epochs == 0`` is exactly the original ``CosineAnnealingLR`` over the whole run.
    Otherwise the LR ramps linearly from 1 % to 100 % over the warmup, then the cosine
    covers the remaining steps.
    """
    total = epochs * steps_per_epoch
    w = int(warmup_epochs) * steps_per_epoch
    if w <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=lr_min)
    return torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=w),
         torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(total - w, 1), eta_min=lr_min)],
        milestones=[w],
    )


def check_init(missing, unexpected, allow_partial: bool) -> None:
    """Refuse a warm start that leaves weights at random init, unless ``allow_partial``."""
    if missing and not allow_partial:
        raise SystemExit(
            f"--init-from left {len(missing)} weight(s) uninitialised, e.g. "
            f"{', '.join(list(missing)[:5])}; pass --allow-partial-init to accept this")
    if missing or unexpected:
        log(f"    missing={len(missing)} unexpected={len(unexpected)}")


@torch.no_grad()
def validate_pq(model, rows, ann_df, cfg, device) -> dict:
    """Full instance-level validation: sweep threshold x min-area on pooled PQ.

    Each exposure is predicted once, then scored against every annotation pass
    it has; the per-pass PQ values are averaged.
    """
    model.eval()
    size = cfg["size"]
    thresholds = cfg["eval"]["thresholds"]
    min_areas = cfg["eval"]["min_areas"]

    # One accumulator per (threshold, min_area) per annotator slot.
    accs = {(t, a): PQAccumulator() for t in thresholds for a in min_areas}
    stems = sorted(rows.file_stem.unique())
    by_stem = {s: sorted(rows[rows.file_stem == s].image_id.unique()) for s in stems}
    by_pass = {k: list(v) for k, v in ann_df.groupby("image_id").ann_id}
    disks: dict[str, fdata.Disk] = {}

    from filseg.dataset import model_input

    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        img = fdata.load_image(stem, "train")
        if stem not in disks:
            disks[stem] = fdata.detect_disk(img)
        disk = disks[stem]
        x = torch.from_numpy(model_input(img, disk, size, cfg.get("inputs"))).unsqueeze(0).to(device)
        with torch.autocast("cuda", enabled=cfg["amp"]):
            p = predict_probs(model, x, cfg["eval"]["tta"], cfg["eval"].get("scales"),
                              tile=cfg["eval"].get("tile"), overlap=cfg["eval"].get("tile_overlap", 0.25))
        probs = p[0, 0].float().cpu().numpy()
        if not np.isfinite(probs).all():
            raise RuntimeError(f"non-finite probabilities for {stem} (NaN/inf in the model output)")

        gts = [instances_from_annotations(by_pass.get(pid, [])) for pid in by_stem[stem]]
        for t in thresholds:
            for a in min_areas:
                preds = probs_to_instances(probs, t, min_area=a, disk=disk, size=2048)
                for g in gts:
                    accs[(t, a)].add(g, preds)
        if i % 25 == 0:
            log(f"    val {i}/{len(stems)}  ({time.time() - t0:.0f}s)")

    grid = []
    for (t, a), acc in accs.items():
        s = acc.summary()
        grid.append({"threshold": t, "min_area": a, **{k: float(v) for k, v in s.items()}})
    best = max(grid, key=lambda r: r["pq"])
    return {"grid": grid, "best": best}


def run_name(cfg_name: str, fold: int, smoke: bool = False) -> str:
    """``<name>_fold<k>``; a ``--smoke`` run is ``<name>_smoke_fold<k>``: outside every
    ``<name>_fold*`` glob (checkpoints, caches) and a distinct experiment in the registry."""
    return f"{cfg_name}{'_smoke' if smoke else ''}_fold{fold}"


PSEUDO_KEYS = ("manifest", "images", "targets", "holdout")


def check_pseudo(cfg: dict) -> None:
    """Refuse a ``pseudo`` block that is incomplete or combined with an option it cannot honour.

    The pretraining frames carry one soft teacher map and no annotation passes, so there is no
    consensus target, no spine polyline and no ignore mask to build, and the frames are used
    whole (no native crops). ``select: swa`` is refused too: it picks the weights by a PQ sweep,
    and a pretraining run has none -- the chain warm-starts from the best-Dice ``<name>.pt``.
    """
    block = cfg.get("pseudo")
    missing = [k for k in PSEUDO_KEYS if not isinstance(block, dict) or k not in block]
    if missing:
        raise SystemExit(f"pseudo block must give {', '.join(PSEUDO_KEYS)}; missing {', '.join(missing)}")
    bad = [what for what, on in (
        ("crops", bool(cfg.get("crops"))),
        ("ignore_dir", bool(cfg.get("ignore_dir"))),
        ("spine_weight > 0", float(cfg.get("spine_weight", 0.0)) > 0),
        ("loss.target: consensus", LossConfig.from_dict(cfg.get("loss")).target == "consensus"),
    ) if on]
    if bad:
        raise SystemExit(f"pseudo-label pretraining cannot be combined with {', '.join(bad)}: "
                         "the GONG frames have one soft teacher map and no annotations")
    if cfg.get("select", "best") == "swa":
        raise SystemExit("pseudo-label pretraining cannot use select: swa (it selects by a PQ sweep, "
                         "and a pretraining run has none); SWA weights are still written with select: best")


def pseudo_split(manifest: pd.DataFrame, holdout: float, seed: int) -> tuple[list[str], list[str]]:
    """The kept stems (``keep == 1``) split into (train, held out), both sorted.

    Stems are ranked by a stable hash, ``crc32(f"{seed}:{stem}")``, and the first
    ``round(holdout * n)`` are held out -- at least one and at most ``n - 1`` once there are two
    frames, so neither side is ever empty. The split depends only on the stems and the seed: the
    same manifest in any row order splits the same way on every machine (Python's ``hash`` is
    salted per process, so it cannot be used).
    """
    if not 0.0 < float(holdout) < 1.0:
        raise ValueError(f"holdout must be in (0, 1), got {holdout}")
    keep = pd.to_numeric(manifest["keep"], errors="coerce").fillna(0).astype(int) == 1
    stems = sorted(set(manifest.loc[keep, "stem"].astype(str)))
    n = len(stems)
    n_val = min(max(int(round(holdout * n)), 1), n - 1) if n >= 2 else 0
    ranked = sorted(stems, key=lambda s: (zlib.crc32(f"{seed}:{s}".encode("utf-8")), s))
    held = set(ranked[:n_val])
    return [s for s in stems if s not in held], [s for s in stems if s in held]


def pseudo_frames(cfg: dict, smoke: bool = False) -> tuple[list[str], list[str], Path, Path]:
    """A ``pseudo`` block's frames: (train stems, held-out stems, images dir, targets dir).

    The manifest's kept rows are split by :func:`pseudo_split` on the config's seed, then each
    side keeps the frames whose image is on disk (the manifest can reach a machine ahead of some
    images); splitting first keeps the held-out set the same on every machine. Paths in the block
    are relative to the project root. Every frame used must have its teacher map: a missing one
    means the labelling run has not finished, and pretraining on whatever part is labelled would
    quietly change the experiment.
    """
    block = cfg["pseudo"]
    images, targets = PROJECT_ROOT / block["images"], PROJECT_ROOT / block["targets"]
    manifest = pd.read_csv(PROJECT_ROOT / block["manifest"], dtype={"stem": str})
    tr, va = pseudo_split(manifest, float(block["holdout"]), int(cfg["seed"]))
    n_kept = len(tr) + len(va)
    tr = [s for s in tr if (images / f"{s}.jpg").is_file()]
    va = [s for s in va if (images / f"{s}.jpg").is_file()]
    log(f"  pseudo: {n_kept} kept frames in {block['manifest']}, {len(tr) + len(va)} with an image "
        f"on disk; train {len(tr)} / held out {len(va)} (holdout {block['holdout']})")
    if smoke:
        tr, va = tr[:8], va[:4]
    if not tr or not va:
        raise SystemExit(f"pseudo-label pretraining needs at least two kept frames with an image in {images}")
    no_map = [s for s in tr + va if not (targets / f"{s}.png").is_file()]
    if no_map:
        raise SystemExit(f"{len(no_map)} of {len(tr) + len(va)} pretraining frames have no teacher map in "
                         f"{targets} (e.g. {', '.join(no_map[:3])}): finish the labelling run first")
    return tr, va, images, targets


def save_swa(cfg: dict, swa_state: dict, swa_n: int, swa_last: int, loader, device, amp: bool,
             amp_dtype, path: Path):
    """The SWA model: averaged weights, BatchNorm re-estimated from ``loader``, saved to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    swa_model = build_from_config(cfg, pretrained=False).to(device)
    swa_model.load_state_dict(swa_state)
    # BatchNorm running stats are averaged along with every other weight by
    # swa_update, but the smp UNet decoder's BatchNorm2d layers need their
    # stats re-estimated from a forward pass of the averaged weights -- the
    # average of the statistics is not the statistics of the average.
    n_bn = update_bn(swa_model, loader, device, amp, amp_dtype)
    log(f"  SWA   BN re-estimated from {n_bn} batches")
    torch.save({"model": swa_model.state_dict(), "cfg": cfg, "epoch": cfg["epochs"],
                "swa_last": swa_last, "n_averaged": swa_n}, path)
    return swa_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", type=int, default=None,
                    help="the held-out fold (required; omit it, or pass -1, for a pseudo-label config)")
    ap.add_argument("--smoke", action="store_true", help="2 epochs on a tiny subset")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights from a checkpoint (progressive resize)")
    ap.add_argument("--allow-partial-init", action="store_true",
                    help="with --init-from, accept a checkpoint that leaves some weights uninitialised")
    ap.add_argument("--skip-if-done", action="store_true",
                    help="exit 0 without training when this run's results JSON already exists")
    ap.add_argument("--require-init-result", action="store_true",
                    help="with --init-from, refuse to warm-start unless the checkpoint's own "
                         "results JSON exists and is non-empty (stage 1 actually finished)")
    args = ap.parse_args()

    if args.require_init_result and args.init_from and init_result_missing(RESULT_DIR, args.init_from):
        log(f"=== refusing to warm-start from {args.init_from}: no results JSON for "
            f"{Path(args.init_from).stem} in {RESULT_DIR} (stage 1 did not finish) ===")
        sys.exit(1)

    cfg = load_config(args.config)
    pseudo = bool(cfg.get("pseudo"))
    if pseudo:
        check_pseudo(cfg)
        if args.fold not in (None, -1):
            raise SystemExit(f"--fold {args.fold}: a pseudo-label pretraining run has no folds; omit --fold")
        args.fold = -1
    elif args.fold is None:
        ap.error("the following arguments are required: --fold")
    name = (f"{cfg['name']}{'_smoke' if args.smoke else ''}" if pseudo
            else run_name(cfg["name"], args.fold, args.smoke))
    if args.smoke:
        cfg["name"] += "_smoke"   # the results JSON records config.name; the registry keys rows by it
    if args.skip_if_done and not args.smoke and already_done(RESULT_DIR, name):
        log(f"=== {name}: results already exist, skipping (--skip-if-done) ===")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = bool(cfg["amp"]) and device == "cuda"
    log(f"=== {name} on {device} (amp={amp}) ===")

    if pseudo:
        fr_tr, fr_va, images, targets = pseudo_frames(cfg, args.smoke)
        if args.smoke:
            cfg["epochs"] = 2
    else:
        folds = pd.read_csv(PROJECT_ROOT / "data" / "folds.csv")
        ann = fdata.annotation_table()[["image_id", "ann_id"]]
        tr = folds[folds.fold != args.fold]
        va = folds[folds.fold == args.fold]

        if args.smoke:
            tr, va = tr.head(8), va.head(4)
            cfg["epochs"] = 2
            cfg["eval"]["thresholds"] = [0.3, 0.5]
            cfg["eval"]["min_areas"] = [300]

    loss_cfg = LossConfig.from_dict(cfg.get("loss"))
    inputs = validate_inputs(cfg.get("inputs"))
    swa_last = max(0, min(int(cfg.get("swa_last", 0)), int(cfg["epochs"])))
    select = cfg.get("select", "best")
    if select not in ("best", "swa") or (select == "swa" and swa_last <= 0):
        raise SystemExit(f"select must be 'best' or 'swa' (swa needs swa_last > 0); "
                         f"got select={select!r}, swa_last={swa_last}")
    disks: dict[str, fdata.Disk] = {}
    use_spine = bool(cfg.get("spine_weight", 0.0) > 0)
    if pseudo:
        # Pretraining: whole GONG frames against the teacher's soft maps; check_pseudo has
        # already refused crops, ignore masks, the spine head and consensus targets.
        log(f"  pseudo-label pretraining: train {len(fr_tr)} frames / val {len(fr_va)} frames "
            f"(targets {targets})")
        ds_tr = PseudoDataset(fr_tr, images, targets, size=cfg["size"], train=True, seed=cfg["seed"],
                              augment_cfg=cfg.get("augment"), loss_cfg=cfg.get("loss"),
                              inputs=cfg.get("inputs"))
        ds_va = PseudoDataset(fr_va, images, targets, size=cfg["size"], train=False,
                              loss_cfg=cfg.get("loss"), inputs=cfg.get("inputs"))
    else:
        log(f"  train {len(tr)} passes / val {len(va)} passes "
            f"({va.file_stem.nunique()} exposures)")
        ds_tr = FilamentDataset(tr, ann, size=cfg["size"], train=True, seed=cfg["seed"],
                                cache_disks=disks, spine=use_spine, augment_cfg=cfg.get("augment"),
                                loss_cfg=cfg.get("loss"), inputs=cfg.get("inputs"),
                                ignore_dir=cfg.get("ignore_dir"), crops=cfg.get("crops"))
        ds_va = FilamentDataset(va, ann, size=cfg["size"], train=False, cache_disks=disks,
                                spine=use_spine, loss_cfg=cfg.get("loss"), inputs=cfg.get("inputs"),
                                crops=cfg.get("crops"))
    if cfg.get("ignore_dir"):
        log(f"  ignore masks: {ds_tr.n_ignore} of {tr.file_stem.nunique()} training exposures ({cfg['ignore_dir']})")
        if ds_tr.n_ignore == 0:
            raise SystemExit("ignore_dir set but no mask found: ship data/ignore/<name>/ to this machine")
    log(f"  augment {cfg.get('augment') or 'defaults'}")
    log(f"  crops {cfg.get('crops') or 'off'}")
    # Per-pass targets keep the baseline's plain shuffle. Consensus targets sample
    # exposures uniformly instead: a WeightedRandomSampler weighted 1/n_passes per
    # row so a 3-pass exposure is not seen 3x as often as a 1-pass one, while the
    # epoch length and optimiser step count stay the same as the baseline's (a
    # sampler and shuffle=True cannot both be given to DataLoader).
    sampler_tr = None
    if loss_cfg.target == "consensus":
        sampler_tr = WeightedRandomSampler(ds_tr.exposure_weights(), num_samples=len(ds_tr),
                                           replacement=True)
    log(f"  loss {cfg.get('loss') or 'defaults'} | inputs {list(inputs)}"
        + (" | sampling exposure-uniform" if sampler_tr is not None else "")
        + (f" | swa last {swa_last} (select {select})" if swa_last else ""))
    dl_tr = DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=(sampler_tr is None),
                       sampler=sampler_tr,
                       num_workers=cfg["workers"], pin_memory=True, drop_last=True,
                       persistent_workers=cfg["workers"] > 0, worker_init_fn=worker_init,
                       prefetch_factor=4 if cfg["workers"] > 0 else None)
    dl_va = DataLoader(ds_va, batch_size=cfg["batch_size"], shuffle=False,
                       num_workers=cfg["workers"], pin_memory=True, worker_init_fn=worker_init,
                       persistent_workers=cfg["workers"] > 0)

    # Autocast precision. fp16 needs the GradScaler and can still overflow in the
    # forward pass: exp7 fold 0 went NaN at epoch 8 and never recovered. bf16 has
    # fp32's exponent range, needs no scaler, and Ampere (RTX 3080) runs it natively.
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.get("amp_dtype", "float16")]
    use_scaler = amp and amp_dtype == torch.float16

    model = build_from_config(cfg).to(device)
    n_par = sum(q.numel() for q in model.parameters()) / 1e6
    log(f"  encoder {cfg['encoder']} | spine head {use_spine} | {n_par:.1f} M params")
    if args.init_from:
        # Progressive resize: a model trained at one input size transfers to a
        # larger one because the encoder is fully convolutional. Only the weights
        # are reused -- optimiser and scheduler restart, since the fine-tune runs
        # a shorter schedule at a lower learning rate.
        sd = torch.load(args.init_from, map_location=device)["model"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        log(f"  warm-started from {args.init_from}")
        check_init(missing, unexpected, args.allow_partial_init)
    crit = FilamentLoss(loss_cfg, cfg["dice_weight"], cfg["pos_weight"],
                        cfg.get("spine_weight", 0.0),
                        cfg.get("spine_pos_weight", 10.0)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    accum = int(cfg.get("grad_accum", 1))
    # The scheduler advances once per OPTIMISER step, not per batch, so its
    # period must count accumulated steps or the cosine finishes early.
    steps_per_epoch = max(math.ceil(len(dl_tr) / accum), 1)
    warmup_epochs = int(cfg.get("warmup_epochs", 0))
    sched = make_scheduler(opt, cfg["epochs"], steps_per_epoch, cfg["lr_min"], warmup_epochs)
    if warmup_epochs > 0:
        log(f"  linear warmup {warmup_epochs} epoch(s) ({warmup_epochs * steps_per_epoch} steps) "
            f"from 1% of lr, then cosine to {cfg['lr_min']}")
    log(f"  batch {cfg['batch_size']} x accum {accum} = effective {cfg['batch_size'] * accum}"
        f" | {steps_per_epoch} optimiser steps/epoch | autocast "
        f"{'off' if not amp else str(amp_dtype).replace('torch.', '')}")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{name}.pt"
    best_dice = -1.0
    history = []

    # EMA of the weights, kept in its own folder so the fold-checkpoint globs the
    # inference kernels use (``<name>_fold*.pt``) never pick up two files per fold.
    # The Dice proxy barely moves between epochs on this task, so which epoch the
    # "best" checkpoint comes from is close to random; the EMA is the model
    # selection that does not depend on that lottery.
    ema_decay = float(cfg.get("ema_decay", 0.0))
    ema = make_ema(model) if ema_decay > 0 else None
    ema_ckpt = CKPT_DIR / "ema" / f"{name}.pt"
    # The EMA's Dice proxy costs a full validation pass (~50 s/epoch on the RTX 3080, a
    # third of the epoch); every k-th epoch and the last is plenty for a curve.
    ema_val_every = int(cfg.get("ema_val_every", 1))
    opt_steps = 0
    skipped_steps = 0
    swa_state, swa_n = None, 0

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        tot = n = 0
        t0 = time.time()
        # Where the epoch's wall time goes: waiting for the loader vs. the GPU step.
        # On the RTX 3080 epoch 1 took 7 min against a 3 min compute budget and the two were
        # indistinguishable from outside; this splits them in the log.
        t_wait = t_step = 0.0
        t_mark = time.time()
        # Gradient accumulation lets a large encoder run at batch 1 while keeping
        # the EFFECTIVE batch equal to the smaller models', so encoder size stays
        # the only variable across the size sweep. The ConvNeXt-V2 encoder is
        # LayerNorm-only, but the smp UNet decoder has BatchNorm2d layers.
        # Accumulation is still safe: BatchNorm's running statistics update once
        # per forward batch (unchanged by how many batches accumulate before an
        # optimiser step), not once per optimiser step.
        opt.zero_grad(set_to_none=True)
        for step, b in enumerate(dl_tr, 1):
            t_wait += time.time() - t_mark
            t_mark = time.time()
            x = b["x"].to(device, non_blocking=True)
            y = b["y"].to(device, non_blocking=True)
            v = b["valid"].to(device, non_blocking=True)
            extra = {k: b[k].to(device, non_blocking=True) for k in ("skel", "pixw") if k in b}
            with torch.autocast("cuda", enabled=amp, dtype=amp_dtype):
                loss = crit(model(x), y, v, **extra)
            loss_val = loss.item()  # .item() syncs, so t_step below is real GPU time
            if not math.isfinite(loss_val):
                # A non-finite loss must not reach the weights (or the EMA). Drop the
                # accumulated gradients and move on; if it keeps happening the
                # epoch-end check below stops the fold instead of burning 20 epochs.
                skipped_steps += 1
                opt.zero_grad(set_to_none=True)
                t_step += time.time() - t_mark
                t_mark = time.time()
                continue
            # Scale down so accumulated gradients average rather than sum.
            scaler.scale(loss / accum).backward()
            if step % accum == 0 or step == len(dl_tr):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema_update(ema, model, ema_decay, opt_steps)
                opt_steps += 1
            tot += loss_val * x.size(0)
            n += x.size(0)
            t_step += time.time() - t_mark
            t_mark = time.time()

        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise RuntimeError(f"non-finite weights after epoch {ep} "
                               f"({skipped_steps} non-finite losses skipped so far); stopping the fold")
        if skipped_steps:
            log(f"    !! {skipped_steps} non-finite loss(es) skipped so far")

        t_train = time.time() - t0
        d = dice_proxy(model, dl_va, device, amp)
        d_ema = None
        if ema is not None and (ep % ema_val_every == 0 or ep == cfg["epochs"]):
            d_ema = dice_proxy(ema, dl_va, device, amp)
        t_val = time.time() - t0 - t_train
        history.append({"epoch": ep, "loss": tot / max(n, 1), "val_dice": d, "val_dice_ema": d_ema,
                        "lr": sched.get_last_lr()[0], "secs": round(time.time() - t0, 1),
                        "secs_wait_data": round(t_wait, 1), "secs_gpu_step": round(t_step, 1),
                        "secs_val": round(t_val, 1), "skipped_steps": skipped_steps})
        flag = ""
        if d > best_dice:
            best_dice, flag = d, "  *"
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "dice": d}, ckpt)
        ema_txt = f"  ema_dice {d_ema:.4f}" if d_ema is not None else ""
        log(f"  ep {ep:3d}  loss {tot / max(n, 1):.4f}  val_dice {d:.4f}{ema_txt}  "
            f"({time.time() - t0:.0f}s: data-wait {t_wait:.0f}s, gpu {t_step:.0f}s, "
            f"val {t_val:.0f}s){flag}")
        if swa_last and ep > cfg["epochs"] - swa_last:
            swa_state = swa_update(swa_state, model, swa_n)
            swa_n += 1

    if ema is not None:
        ema_ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": ema.state_dict(), "cfg": cfg, "epoch": cfg["epochs"],
                    "dice": history[-1]["val_dice_ema"], "ema_decay": ema_decay}, ema_ckpt)
        # The training-time Dice proxy uses the raw autocast dtype; the PQ sweep
        # below runs both models through the same predict_probs path as inference.
        log(f"\n  final EMA weights -> {ema_ckpt}")

    log(f"\n  best proxy dice {best_dice:.4f} -> {ckpt}")
    if pseudo:
        # No annotations, so no PQ sweep: the best-Dice checkpoint is the result, and the
        # results JSON marks the run as pretraining for --require-init-result and the reader.
        if swa_state is not None:
            swa_ckpt = CKPT_DIR / "swa" / f"{name}.pt"
            save_swa(cfg, swa_state, swa_n, swa_last, dl_tr, device, amp, amp_dtype, swa_ckpt)
            log(f"  SWA weights ({swa_n} epochs) -> {swa_ckpt}")
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULT_DIR / f"{name}.json"
        out.write_text(json.dumps(
            {"name": name, "fold": -1, "config": cfg, "best_proxy_dice": best_dice,
             "history": history, "pretrain": True, "n_train": len(ds_tr), "n_val": len(ds_va)},
            indent=1), encoding="utf-8")
        log(f"  wrote {out}")
        return
    log("  running instance-level PQ sweep on the best checkpoint")
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    res = validate_pq(model, va, fdata.annotation_table(), cfg, device)
    res_ema = None
    if ema is not None:
        log("  running instance-level PQ sweep on the final EMA weights")
        res_ema = validate_pq(ema, va, fdata.annotation_table(), cfg, device)
    res_swa = None
    if swa_state is not None:
        swa_ckpt = CKPT_DIR / "swa" / f"{name}.pt"
        swa_model = save_swa(cfg, swa_state, swa_n, swa_last, dl_tr, device, amp, amp_dtype, swa_ckpt)
        log(f"  running instance-level PQ sweep on the SWA weights ({swa_n} epochs) -> {swa_ckpt}")
        res_swa = validate_pq(swa_model, va, fdata.annotation_table(), cfg, device)

    b = res["best"]
    log(f"\n  BEST  PQ {b['pq']:.4f}  (SQ {b['sq']:.3f} RQ {b['rq']:.3f})  "
        f"@ threshold {b['threshold']} min_area {b['min_area']}   [best-dice checkpoint]")
    log(f"        TP {int(b['tp'])}  FP {int(b['fp'])}  FN {int(b['fn'])}")
    if res_ema is not None:
        e = res_ema["best"]
        log(f"  EMA   PQ {e['pq']:.4f}  (SQ {e['sq']:.3f} RQ {e['rq']:.3f})  "
            f"@ threshold {e['threshold']} min_area {e['min_area']}   [final EMA weights]")
        log(f"        TP {int(e['tp'])}  FP {int(e['fp'])}  FN {int(e['fn'])}")
    if res_swa is not None:
        s = res_swa["best"]
        log(f"  SWA   PQ {s['pq']:.4f}  (SQ {s['sq']:.3f} RQ {s['rq']:.3f})  "
            f"@ threshold {s['threshold']} min_area {s['min_area']}   [SWA of last {swa_n} epochs]")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULT_DIR / f"{name}.json"
    primary = res_swa if select == "swa" else res
    out.write_text(json.dumps(
        {"name": name, "fold": args.fold, "config": cfg,
         "best_proxy_dice": best_dice, "history": history, **primary,
         "selected": select, "best_dice_ckpt": res if select == "swa" else None,
         "ema": res_ema, "swa": res_swa}, indent=1
    ), encoding="utf-8")
    log(f"  wrote {out}")


if __name__ == "__main__":
    main()
