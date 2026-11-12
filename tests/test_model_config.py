"""build_from_config sets input/output channels from the config; scripts use it (CPU)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.model import build_from_config, build_model, head_of  # noqa: E402

ENC = "tu-convnextv2_atto"   # smallest ConvNeXt-V2; pretrained=False needs no download


def _cfg(**kw):
    return {"encoder": ENC, "pretrained": False, **kw}


def test_legacy_config_builds_the_same_network_as_build_model():
    a = build_from_config(_cfg())
    b = build_model(ENC, pretrained=False)
    assert list(a.state_dict()) == list(b.state_dict())
    assert head_of(a) == "mask"
    assert a(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_inputs_head_and_spine_set_the_channel_counts():
    m = build_from_config(_cfg(inputs=["flat", "raw", "radius", "latitude", "tophat", "clahe"],
                               loss={"target": "consensus", "head": "ordinal"}))
    assert head_of(m) == "ordinal"
    assert m(torch.zeros(1, 6, 64, 64)).shape == (1, 3, 64, 64)
    assert build_from_config(_cfg(spine_weight=0.2))(torch.zeros(1, 3, 64, 64)).shape == (1, 2, 64, 64)
    assert build_from_config(_cfg(inputs=["flat", "raw"]))(torch.zeros(1, 2, 64, 64)).shape == (1, 1, 64, 64)


SCRIPTS = [s for s in ("train.py", "predict.py", "cache_probs.py", "diagnose_oof.py", "eval_oof.py",
                       "sweep_submissions.py") if (ROOT / "scripts" / s).is_file()]


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_go_through_the_config_aware_helpers(script):
    src = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    assert "build_model(" not in src
    assert "build_input(" not in src
    for pattern in ("torch.sigmoid(model(", "torch.sigmoid(m("):
        assert pattern not in src, pattern
