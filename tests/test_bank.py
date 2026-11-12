"""Filament bank: build from a split's annotations, paste onto other frames."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import synth_data  # noqa: E402
from filseg import augment as faug  # noqa: E402
from filseg import data as fdata  # noqa: E402
from filseg.bank import build_bank, paste_bank, unpack_avoid  # noqa: E402
from filseg.dataset import FilamentDataset, build_input  # noqa: E402

S = 256


def _bank(monkeypatch):
    folds, ann, disks = synth_data.install(monkeypatch)
    by_pass = {k: list(v) for k, v in ann.groupby("image_id").ann_id}
    return build_bank(folds, by_pass, S, disks, fdata.load_image)


def _items(monkeypatch):
    return _bank(monkeypatch).items


def test_bank_holds_every_annotation_of_the_split(monkeypatch):
    bank = _items(monkeypatch)
    assert len(bank) == 2                                  # a1 and a2 (p3 drew nothing)
    item = bank[0]
    assert item.mask.shape == item.flat.shape == item.raw.shape
    assert item.mask.max() == 1.0 and item.flat[item.mask > 0.5].mean() < -0.1   # darker than quiet Sun
    assert 0.0 < item.r < 0.3                              # the rectangle sits near disk centre


def test_paste_adds_dark_filaments_to_quiet_disk(monkeypatch):
    bank = _items(monkeypatch)
    x = build_input(synth_data.image(), synth_data.DISK, S)
    y = np.zeros((1, S, S), np.float32)
    cfg = faug.AugmentConfig(bank_paste_p=1.0, bank_n=(3, 3))
    x2, y2, n = paste_bank(x.copy(), y.copy(), bank, np.random.default_rng(0), cfg)
    assert n >= 1 and y2[0].sum() > 0
    new = y2[0] > 0.5
    assert x2[0][new].mean() < x[0][new].mean() - 0.05   # pasted pixels got darker
    assert np.array_equal(x2[2], x[2])                    # geometry untouched
    r = x[2] + 0.6
    assert r[new].max() < 0.97                            # pasted fully on the disk


def test_paste_never_overlaps_existing_filaments(monkeypatch):
    bank = _items(monkeypatch)
    x = build_input(synth_data.image(), synth_data.DISK, S)
    y = np.zeros((1, S, S), np.float32)
    y[0, 100:124, 60:200] = 1.0                           # a filament across the top of the centre
    cfg = faug.AugmentConfig(bank_paste_p=1.0, bank_n=(6, 6))
    for seed in range(5):
        _, y2, _ = paste_bank(x.copy(), y.copy(), bank, np.random.default_rng(seed), cfg)
        added = (y2[0] > 0.5) & (y[0] < 0.5)
        grown = np.zeros_like(added)
        grown[96:128, 56:204] = True                      # the block plus its 4 px margin
        assert not (added & grown).any()


def test_dataset_builds_the_bank_only_when_enabled(monkeypatch):
    folds, ann, disks = synth_data.install(monkeypatch)
    off = FilamentDataset(folds, ann, size=S, train=True, cache_disks=disks, augment_cfg=synth_data.NO_AUG)
    assert off.bank is None
    on = FilamentDataset(folds, ann, size=S, train=True, cache_disks=disks,
                         augment_cfg={**synth_data.NO_AUG, "bank_paste_p": 1.0})
    assert len(on.bank) == len(on.bank.items) == 2
    # Pass p3 drew nothing, so any mask is pasted. The other passes' filaments are off limits and
    # the synthetic disk is tiny, so a single draw can find no room: try several seeds.
    got = []
    for seed in range(10):
        torch.manual_seed(seed)
        got.append(float(on[2]["y"][0].sum()))
    assert sum(g > 0 for g in got) >= 3


def test_paste_succeeds_for_most_seeds(monkeypatch):
    bank = _items(monkeypatch)
    x = build_input(synth_data.image(), synth_data.DISK, S)
    y = np.zeros((1, S, S), np.float32)
    cfg = faug.AugmentConfig(bank_paste_p=1.0, bank_n=(3, 3))
    hits = sum(paste_bank(x.copy(), y.copy(), bank, np.random.default_rng(s), cfg)[2] >= 1
               for s in range(5))
    assert hits >= 4


def test_placement_is_cheap(monkeypatch):
    import time
    bank = _items(monkeypatch)
    x = build_input(synth_data.image(), synth_data.DISK, S)
    y = np.zeros((1, S, S), np.float32)
    cfg = faug.AugmentConfig(bank_paste_p=1.0, bank_n=(3, 3))
    times = []
    for s in range(20):
        t = time.perf_counter()
        paste_bank(x.copy(), y.copy(), bank, np.random.default_rng(s), cfg)
        times.append(time.perf_counter() - t)
    assert np.median(times) < 0.05


def test_paste_avoids_the_avoid_mask(monkeypatch):
    bank = _items(monkeypatch)
    x = build_input(synth_data.image(), synth_data.DISK, S)
    y = np.zeros((1, S, S), np.float32)
    avoid = np.zeros((S, S), bool)
    avoid[100:124, 60:200] = True                         # another annotator's filament
    cfg = faug.AugmentConfig(bank_paste_p=1.0, bank_n=(3, 3))
    total = 0
    for seed in range(10):
        _, y2, n = paste_bank(x.copy(), y.copy(), bank, np.random.default_rng(seed), cfg, avoid=avoid)
        total += n
        grown = np.zeros((S, S), bool)
        grown[96:128, 56:204] = True
        assert not ((y2[0] > 0.5) & grown).any()
    assert total >= 1                                     # the property was exercised, not vacuous


def test_bank_keeps_the_union_of_all_passes_as_a_packed_avoid_mask(monkeypatch):
    from filseg.dataset import render_mask
    folds, ann, disks = synth_data.install(monkeypatch)
    by_pass = {k: list(v) for k, v in ann.groupby("image_id").ann_id}
    bank = build_bank(folds, by_pass, S, disks, fdata.load_image)
    assert set(bank.avoid) == set(folds.file_stem)
    for stem, g in folds.groupby("file_stem"):
        packed = bank.avoid[stem]
        assert packed.dtype == np.uint8 and packed.size == (S * S + 7) // 8
        want = np.zeros((S, S), bool)
        for pid in g.image_id:
            want |= render_mask(by_pass.get(pid, []), S) > 0
        assert np.array_equal(unpack_avoid(packed, S), want)
        assert np.array_equal(bank.avoid_mask(stem, S), want)
    assert any(unpack_avoid(v, S).any() for v in bank.avoid.values())



def test_avoid_mask_keeps_partial_coverage_pixels(monkeypatch):
    import cv2
    from filseg.dataset import render_mask
    folds, ann, disks = synth_data.install(monkeypatch)
    thin = np.zeros((synth_data.FULL, synth_data.FULL), np.uint8)
    cv2.line(thin, (700, 900), (1300, 1100), 1, 3)          # 3 px wide, off the 8x8 block grid
    monkeypatch.setattr(fdata, "ann_to_mask", lambda aid, size=synth_data.FULL: thin if aid == "a1"
                        else np.zeros_like(thin))
    by_pass = {k: list(v) for k, v in ann.groupby("image_id").ann_id}
    bank = build_bank(folds, by_pass, S, disks, fdata.load_image)
    soft = render_mask(["a1"], S)
    assert ((soft > 0) & (soft < 0.5)).any()                # the setting really has partial pixels
    assert bank.avoid_mask("S1", S)[soft > 0].all()
