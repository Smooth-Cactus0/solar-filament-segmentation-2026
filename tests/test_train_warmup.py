"""make_scheduler, check_init and the non-finite guard in scripts/train.py (CPU, tiny modules)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import train  # noqa: E402
from train import check_init, make_scheduler  # noqa: E402


def _opt(lr=1e-3):
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=lr)


def _lrs(sched, opt, n):
    out = []
    for _ in range(n):
        out.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    return out


def test_no_warmup_is_todays_cosine():
    epochs, spe, lr_min = 4, 5, 1e-5
    o1, o2 = _opt(), _opt()
    s = make_scheduler(o1, epochs, spe, lr_min, 0)
    assert isinstance(s, torch.optim.lr_scheduler.CosineAnnealingLR)
    ref = torch.optim.lr_scheduler.CosineAnnealingLR(o2, T_max=epochs * spe, eta_min=lr_min)
    assert _lrs(s, o1, epochs * spe) == _lrs(ref, o2, epochs * spe)


def test_one_epoch_warmup_ramps_then_decays_to_lr_min():
    epochs, spe, lr_min, lr = 4, 5, 1e-5, 1e-3
    o = _opt(lr)
    s = make_scheduler(o, epochs, spe, lr_min, 1)
    seq = _lrs(s, o, epochs * spe)
    assert seq[0] == pytest.approx(0.01 * lr)
    assert seq[spe] == pytest.approx(lr)
    assert all(a < b for a, b in zip(seq[:spe], seq[1:spe + 1]))
    assert seq[-1] < 0.05 * lr and seq[-1] >= lr_min


def test_check_init_refuses_missing_weights_unless_allowed():
    check_init([], ["x"], False)
    check_init(["a"], [], True)
    with pytest.raises(SystemExit) as e:
        check_init(["decoder.w", "head.b"], [], False)
    assert "decoder.w" in str(e.value)


def test_validate_pq_rejects_non_finite_probabilities(monkeypatch):
    import pandas as pd
    monkeypatch.setattr(train.fdata, "load_image", lambda *a, **k: np.zeros((8, 8), np.float32))
    monkeypatch.setattr(train.fdata, "detect_disk", lambda img: None)
    import filseg.dataset as ds
    monkeypatch.setattr(ds, "model_input", lambda *a, **k: np.zeros((3, 8, 8), np.float32))
    monkeypatch.setattr(train, "predict_probs",
                        lambda *a, **k: torch.full((1, 1, 8, 8), float("nan")))
    rows = pd.DataFrame({"file_stem": ["s1"], "image_id": [1]})
    ann = pd.DataFrame({"image_id": [1], "ann_id": [1]})
    cfg = {"size": 8, "amp": False, "eval": {"thresholds": [0.5], "min_areas": [1], "tta": False}}
    model = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="s1"):
        train.validate_pq(model, rows, ann, cfg, "cpu")
