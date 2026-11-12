"""The `arch:` block: decoder, encoder and their arguments, built on CPU without weights."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filseg.model import arch_kwargs, build_from_config, build_model  # noqa: E402

CNV2T = "tu-convnextv2_tiny.fcmae_ft_in22k_in1k"
SWIN = "tu-swin_tiny_patch4_window7_224.ms_in22k_ft_in1k"
EFF = "tu-tf_efficientnetv2_s.in21k_ft_in1k"
MAXVIT = "tu-maxvit_tiny_tf_512.in1k"
DINO = "tu-vit_small_patch16_dinov3.lvd1689m"
COAT1 = "tu-coatnet_rmlp_1_rw2_224.sw_in12k_ft_in1k"
COAT2 = "tu-coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k"

SCREENS = [
    ("UPerNet", CNV2T, {}),
    ("FPN", CNV2T, {}),
    ("DeepLabV3Plus", CNV2T, {}),
    ("Unet", SWIN, {"pass_img_size": True}),
    ("Unet", EFF, {}),
    ("UnetPlusPlus", EFF, {}),
    ("Unet", MAXVIT, {}),
    ("Segformer", "mit_b2", {}),
    ("DPT", DINO, {"encoder_kwargs": {"dynamic_img_size": True}}),
    ("Unet", COAT1, {"pass_img_size": True}),
    ("Unet", COAT2, {"pass_img_size": True}),
]


def _cfg(encoder, size=512, **arch):
    return {"encoder": encoder, "pretrained": False, "size": size, "arch": arch}


def test_no_arch_block_builds_the_legacy_unet():
    legacy = build_model(CNV2T, pretrained=False)
    for cfg in ({"encoder": CNV2T, "pretrained": False},
                {"encoder": CNV2T, "pretrained": False, "arch": {}},
                _cfg(CNV2T, decoder="Unet")):
        m = build_from_config(cfg)
        assert type(m).__name__ == "Unet"
        assert list(m.state_dict()) == list(legacy.state_dict())


def test_arch_kwargs_validates_and_fills_the_image_size():
    assert arch_kwargs({"size": 1024}) == {"decoder": "Unet", "encoder_kwargs": {}, "decoder_kwargs": {}}
    a = arch_kwargs({"size": 1536, "arch": {"decoder": "Unet", "pass_img_size": True,
                                           "encoder_kwargs": {"drop_path_rate": 0.1}}})
    assert a["encoder_kwargs"] == {"drop_path_rate": 0.1, "img_size": 1536}
    with pytest.raises(ValueError, match="unknown arch"):
        arch_kwargs({"size": 512, "arch": {"decoders": "FPN"}})
    with pytest.raises(ValueError, match="decoder"):
        arch_kwargs({"size": 512, "arch": {"decoder": "Mask2Former"}})


@pytest.mark.parametrize("decoder, encoder, arch", SCREENS, ids=[f"{d}-{e.split('.')[0]}" for d, e, _ in SCREENS])
def test_screen_architectures_build_and_run(decoder, encoder, arch):
    m = build_from_config(_cfg(encoder, decoder=decoder, **arch)).eval()
    with torch.no_grad():
        y = m(torch.zeros(1, 3, 512, 512))
    assert y.shape == (1, 1, 512, 512)
    assert type(m).__name__ == decoder


@pytest.mark.parametrize("decoder, encoder, arch", [s for s in SCREENS if s[1] in (SWIN, MAXVIT, DINO, "mit_b2", COAT1, COAT2)],
                         ids=["swin", "maxvit", "segformer", "dpt", "coatnet1", "coatnet2"])
def test_weights_carry_from_the_1024_stage_to_the_1536_stage(decoder, encoder, arch):
    small = build_from_config(_cfg(encoder, 512, decoder=decoder, **arch))
    big = build_from_config(_cfg(encoder, 1024, decoder=decoder, **arch)).eval()
    big.load_state_dict(small.state_dict(), strict=True)
    with torch.no_grad():
        assert big(torch.zeros(1, 3, 1024, 1024)).shape == (1, 1, 1024, 1024)


@pytest.mark.parametrize("decoder, encoder, arch", SCREENS, ids=[f"{d}-{e.split('.')[0]}" for d, e, _ in SCREENS])
def test_training_forward_backward_works_at_batch_one(decoder, encoder, arch):
    # UPerNet / DeepLabV3+ pool to 1x1 and batch-normalise it: batch 1 raised in training mode.
    px = 512 if encoder == MAXVIT else 256  # MaxViT's /32 map must divide its 16-px window
    m = build_from_config(_cfg(encoder, size=px, decoder=decoder, **arch)).train()
    y = m(torch.randn(1, 3, px, px))
    y.mean().backward()
    assert torch.isfinite(y).all()


def test_pooled_branch_norm_swapped_only_for_upernet_and_deeplab():
    import torch.nn as nn

    def n_gn(m):
        return sum(isinstance(x, nn.GroupNorm) for x in m.modules())

    # The legacy UNet gains no GroupNorm.
    assert n_gn(build_model(CNV2T, pretrained=False)) == 0
    assert n_gn(build_from_config({"encoder": CNV2T, "pretrained": False})) == 0
    up = build_model(CNV2T, pretrained=False, decoder="UPerNet")
    assert isinstance(up.decoder.psp.blocks[0][1][1], nn.GroupNorm)
    dl = build_model(CNV2T, pretrained=False, decoder="DeepLabV3Plus")
    assert isinstance(dl.decoder.aspp[0].convs[4][2], nn.GroupNorm)
