# tests/test_shallow.py
"""The full- and half-resolution shallow branch and its UNet wrapper."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.shallow import ShallowBranch  # noqa: E402


def test_branch_shapes_even_and_odd():
    b = ShallowBranch(3)
    f1, f2 = b(torch.zeros(2, 3, 64, 96))
    assert f1.shape == (2, 16, 64, 96) and f2.shape == (2, 32, 32, 48)
    f1, f2 = b(torch.zeros(1, 3, 65, 33))
    assert f1.shape == (1, 16, 65, 33) and f2.shape == (1, 32, 33, 17)


def test_branch_channels_follow_the_arguments_and_it_is_tiny():
    b = ShallowBranch(5, c1=8, c2=24)
    f1, f2 = b(torch.zeros(1, 5, 32, 32))
    assert f1.shape[1] == 8 and f2.shape[1] == 24 and (b.c1, b.c2) == (8, 24)
    assert sum(p.numel() for p in ShallowBranch(3).parameters()) < 20_000


def test_branch_uses_groupnorm_not_batchnorm():
    mods = list(ShallowBranch(3).modules())
    assert any(isinstance(m, torch.nn.GroupNorm) for m in mods)
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in mods)


from filseg.model import arch_kwargs, build_from_config, build_model, head_of  # noqa: E402
from filseg.shallow import ShallowSkipUnet  # noqa: E402

ENC = "tu-convnextv2_atto"   # smallest ConvNeXt-V2; pretrained=False needs no download
BLOCK = {"c1": 16, "c2": 32}


def test_switch_off_is_the_legacy_unet():
    a = build_model(ENC, pretrained=False)
    b = build_model(ENC, pretrained=False, shallow_skip=None)
    assert type(a) is type(b) and not isinstance(a, ShallowSkipUnet)
    assert list(a.state_dict()) == list(b.state_dict())


def test_wrapper_keeps_the_legacy_keys_and_adds_branch_and_fuse():
    m = build_model(ENC, pretrained=False, shallow_skip=BLOCK)
    assert isinstance(m, ShallowSkipUnet)
    legacy = set(build_model(ENC, pretrained=False).state_dict())
    keys = set(m.state_dict())
    assert {k for k in keys if k.startswith("encoder.")} == {k for k in legacy if k.startswith("encoder.")}
    assert {k for k in keys if k.startswith("segmentation_head.")} == {k for k in legacy if k.startswith("segmentation_head.")}
    assert {k.split(".")[0] for k in keys - legacy} <= {"branch", "fuse", "decoder"}
    assert {k.split(".")[0] for k in keys} == {"encoder", "decoder", "branch", "fuse", "segmentation_head"}


def test_wrapper_forward_shapes():
    m = build_model(ENC, pretrained=False, shallow_skip=BLOCK)
    assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 1, 64, 64)
    assert m(torch.zeros(1, 3, 96, 64)).shape == (1, 1, 96, 64)
    with pytest.raises(RuntimeError):
        m(torch.zeros(1, 3, 70, 64))   # not divisible by the encoder's output stride


def test_half_skip_widens_the_decoder_and_full_skip_feeds_the_fuse():
    m = build_model(ENC, pretrained=False, shallow_skip=BLOCK)
    legacy = build_model(ENC, pretrained=False)
    # block 3 is the 1/4 -> 1/2 block; its skip slot was empty (0 channels) and is now c2
    assert m.decoder.blocks[3].conv1[0].in_channels == legacy.decoder.blocks[3].conv1[0].in_channels + 32
    assert m.decoder.blocks[4].conv1[0].in_channels == legacy.decoder.blocks[4].conv1[0].in_channels
    assert m.fuse[0].in_channels == 16 + 16 and m.fuse[0].out_channels == 16
    assert m.segmentation_head[0].in_channels == 16


def test_spine_and_inputs_work_through_the_wrapper():
    cfg = {"encoder": ENC, "pretrained": False, "spine_weight": 0.2,
           "inputs": ["flat", "raw", "radius", "latitude"], "arch": {"shallow_skip": {"c1": 8, "c2": 16}}}
    m = build_from_config(cfg)
    assert isinstance(m, ShallowSkipUnet) and head_of(m) == "mask"
    assert m(torch.zeros(1, 4, 64, 64)).shape == (1, 2, 64, 64)


def test_non_unet_decoder_refuses_the_block():
    with pytest.raises(ValueError, match="UNet"):
        build_model(ENC, pretrained=False, decoder="FPN", shallow_skip=BLOCK)


def test_branch_keys_are_validated():
    with pytest.raises(TypeError):
        build_model(ENC, pretrained=False, shallow_skip={"c1": 16, "width": 3})


def test_arch_kwargs_adds_the_key_only_when_set():
    assert arch_kwargs({"arch": {"shallow_skip": BLOCK}})["shallow_skip"] == BLOCK
    assert "shallow_skip" not in arch_kwargs({})
    assert "shallow_skip" not in arch_kwargs({"arch": {"shallow_skip": None}})
    with pytest.raises(ValueError, match="unknown arch option"):
        arch_kwargs({"arch": {"shallow": BLOCK}})


def _decoder_conv2_std(m):
    return [b.conv2[0].weight.std().item() for b in m.decoder.blocks]


def test_rebuilt_decoder_has_smps_decoder_init_scale():
    torch.manual_seed(0)
    a = _decoder_conv2_std(build_model(ENC, pretrained=False, shallow_skip=BLOCK))
    b = _decoder_conv2_std(build_model(ENC, pretrained=False))
    for i, (x, y) in enumerate(zip(a, b)):
        assert 0.8 <= x / y <= 1.25, (i, x / y)


def test_non_empty_half_level_encoder_is_refused():
    with pytest.raises(ValueError):
        build_model("tu-resnet18", pretrained=False, shallow_skip=BLOCK)


def test_scse_bilinear_decoder_kwargs_are_honoured():
    dk = {"decoder_attention_type": "scse", "decoder_interpolation": "bilinear"}
    m = build_model(ENC, pretrained=False, shallow_skip=BLOCK, decoder_kwargs=dk)
    legacy = build_model(ENC, pretrained=False, decoder_kwargs=dk).state_dict()
    for k, v in m.state_dict().items():
        if not k.startswith("decoder."):
            continue
        if k.startswith("decoder.blocks.3.attention1"):
            continue   # the only attention module that sees the widened input
        if k == "decoder.blocks.3.conv1.0.weight":
            assert v.shape[1] == legacy[k].shape[1] + BLOCK["c2"]
            continue
        assert v.shape == legacy[k].shape, k
    assert m.decoder.blocks[0].interpolation_mode == "bilinear"   # no parameters: shapes cannot catch it
    assert m.eval()(torch.zeros(1, 3, 64, 64)).shape == (1, 1, 64, 64)


def test_unsupported_decoder_kwargs_raise():
    with pytest.raises(ValueError, match="decoder_use_batchnorm"):
        build_model(ENC, pretrained=False, shallow_skip=BLOCK, decoder_kwargs={"decoder_use_batchnorm": False})


def test_half_precision_works_on_the_wrapper():
    m = build_model(ENC, pretrained=False, shallow_skip=BLOCK)
    assert m.half() is m
    assert m.branch.full[0][0].weight.dtype == torch.float16


def test_empty_shallow_skip_block_means_defaults():
    m = build_from_config({"encoder": ENC, "pretrained": False, "arch": {"shallow_skip": {}}})
    assert isinstance(m, ShallowSkipUnet) and (m.branch.c1, m.branch.c2) == (16, 32)
