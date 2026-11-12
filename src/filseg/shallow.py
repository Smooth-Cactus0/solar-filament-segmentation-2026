"""A full- and half-resolution shallow branch for the UNet.

ConvNeXt's stem is a stride-4 patchify with no stride-2 stage, so the finest skip into the
UNet decoder is at 1/4: a median filament (8.5 px wide at native 2048) is 2 px wide there,
1 px at a 1024 input. The effective receptive field is wide enough (90 % of a pixel's
decision comes from a disc about 800 px across, measured on ``scr_r1_crop1024`` fold 0);
what is coarse is the fine end. This branch runs three 3x3 convolutions on the raw input
planes and hands the decoder a full-resolution feature (``f1``) and a half-resolution one
(``f2``). :class:`ShallowSkipUnet` wires them into segmentation_models_pytorch's UNet:
``f2`` fills the decoder's empty 1/2 skip, ``f1`` is fused into the decoder output before
the segmentation head. GroupNorm, not BatchNorm: training runs at batch 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from segmentation_models_pytorch.base import initialization as smp_init
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder

GROUPS = 8
ALLOWED_DECODER_KWARGS = frozenset(
    {"decoder_channels", "decoder_use_norm", "decoder_attention_type", "decoder_interpolation"})


def _conv_norm_act(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    """3x3 convolution (no bias), GroupNorm, GELU."""
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False),
                         nn.GroupNorm(GROUPS, cout), nn.GELU())


class ShallowBranch(nn.Module):
    """``f1`` at full resolution (``c1`` channels) and ``f2`` at 1/2 (``c2`` channels)."""

    def __init__(self, in_ch: int, c1: int = 16, c2: int = 32) -> None:
        super().__init__()
        self.c1, self.c2 = int(c1), int(c2)
        self.full = nn.Sequential(_conv_norm_act(in_ch, self.c1), _conv_norm_act(self.c1, self.c1))
        # Not ``self.half``: that name is nn.Module.half(), so the submodule would be shadowed.
        self.down = _conv_norm_act(self.c1, self.c2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.full(x)
        return f1, self.down(f1)


class ShallowSkipUnet(nn.Module):
    """An smp ``Unet`` with :class:`ShallowBranch` skips at 1/2 and 1/1.

    The inner UNet's modules are registered here under their own names (``encoder``,
    ``decoder``, ``segmentation_head``), so the state-dict keys and every attribute the
    rest of the code reads stay as for a plain smp UNet, plus ``branch.*`` and ``fuse.*``.
    The decoder is rebuilt once with ``branch.c2`` channels in its 1/2 skip slot (smp fills
    that slot with an empty 0-channel feature for ConvNeXt-style encoders). smp's own
    ``Unet.__init__`` ends with ``initialize_decoder``; the rebuilt decoder gets the same call,
    so its Kaiming-uniform init matches the legacy UNet's (default PyTorch init is about
    2.4x smaller in std, an effective-learning-rate change under AdamW + BatchNorm).
    """

    def __init__(self, unet: nn.Module, branch: ShallowBranch, decoder_kwargs: dict | None = None) -> None:
        super().__init__()
        dk = dict(decoder_kwargs or {})
        extra = sorted(set(dk) - ALLOWED_DECODER_KWARGS)
        if extra:
            raise ValueError(f"shallow_skip cannot re-read decoder_kwargs {extra}; "
                             f"supported: {sorted(ALLOWED_DECODER_KWARGS)}")
        enc_ch = list(unet.encoder.out_channels)
        if len(enc_ch) != 6 or enc_ch[1] != 0:
            raise ValueError("shallow_skip expects an encoder with an empty 1/2 level "
                             f"(ConvNeXt-style strides 4/8/16/32), got out_channels {enc_ch}")
        dec_ch = [b.conv2[0].out_channels for b in unet.decoder.blocks]
        self.encoder = unet.encoder
        self.decoder = UnetDecoder(
            encoder_channels=[enc_ch[0], branch.c2, *enc_ch[2:]],
            decoder_channels=dec_ch,
            n_blocks=len(dec_ch),
            use_norm=dk.get("decoder_use_norm", "batchnorm"),
            attention_type=dk.get("decoder_attention_type"),
            interpolation_mode=dk.get("decoder_interpolation", "nearest"),
        )
        smp_init.initialize_decoder(self.decoder)   # as smp's Unet.initialize() does for its own
        self.branch = branch
        self.fuse = _conv_norm_act(dec_ch[-1] + branch.c1, dec_ch[-1])
        self.segmentation_head = unet.segmentation_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        stride = int(getattr(self.encoder, "output_stride", 32))
        if h % stride or w % stride:
            raise RuntimeError(f"input {h}x{w} is not divisible by the encoder's output stride {stride}")
        feats = self.encoder(x)
        f1, f2 = self.branch(x)
        feats[1] = f2
        d = self.decoder(feats)
        d = self.fuse(torch.cat([d, f1], dim=1))
        return self.segmentation_head(d)
