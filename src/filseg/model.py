"""Model and loss.

The loss is the part worth reading. Because every annotator pass is fed as an
independent sample with its own labels, the network sees the same exposure two
or three times with *disagreeing* targets. Under a proper scoring rule that
converges to the per-pixel conditional mean -- so the sigmoid output becomes
``P(an annotator marks this pixel)``, which is exactly the quantity the
acceptance rule ``p* = 0.5 * PQ / q`` wants to threshold.

Binary cross-entropy is a proper scoring rule. Dice is not: it optimises overlap
directly and will happily push probabilities toward 0/1, destroying the
calibration that makes the threshold meaningful. So BCE dominates, and the small
Dice term exists only to stabilise training against a 0.36% positive rate.
"""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F


DECODERS = ("Unet", "UnetPlusPlus", "UPerNet", "FPN", "DeepLabV3Plus", "Segformer", "DPT")
ARCH_KEYS = ("decoder", "encoder_kwargs", "decoder_kwargs", "pass_img_size", "shallow_skip")


def build_model(
    encoder: str = "tu-convnext_small",
    in_channels: int = 3,
    classes: int = 1,
    pretrained: bool = True,
    decoder: str = "Unet",
    encoder_kwargs: dict | None = None,
    decoder_kwargs: dict | None = None,
    shallow_skip: dict | None = None,
) -> nn.Module:
    """A segmentation_models_pytorch network: ``decoder`` over a (usually timm ``tu-``) encoder.

    The default is the UNet every model through recipe r2 was trained with.
    ConvNeXt's stem is stride 4 with no stride-2 stage, so the finest skip
    connection is at /4. At 1024 input a median filament is ~4 px wide and
    survives there as roughly one pixel, which the decoder can recover. Plain
    patch-16 ViTs (DINOv3) need a multi-scale adapter: smp's ``DPT`` decoder is one.

    Which *weights* load is decided by the timm tag in ``encoder``, not by this
    function: for ``tu-`` encoders smp treats ``encoder_weights`` as a plain on/off
    flag and forwards ``pretrained=True`` to ``timm.create_model``.

    ``encoder_kwargs`` reach the encoder (``timm.create_model`` for ``tu-``
    encoders, e.g. Swin's ``img_size``); ``decoder_kwargs`` are the architecture's
    own arguments (e.g. ``decoder_channels``). ``classes=2`` enables the spine
    auxiliary head (channel 1, loss only): the spine map is on channel 1 and
    never reaches inference.

    The DINOv3 ConvNeXt variants (``tu-convnext_*.dinov3_lvd1689m``) keep strides
    4/8/16/32 and plug into a UNet directly; the DINOv3 ViTs are patch-16 and need
    a multi-scale decoder such as ``DPT``.

    UPerNet and DeepLabV3+ pool the feature map to 1x1 and batch-normalise it
    (``decoder.psp.blocks.0.1.1`` / ``decoder.aspp.0.convs.4.2``): in training
    mode that raises at batch 1 and normalises just two values at batch 2.
    For those two decoders only, every BatchNorm2d that follows an
    AdaptiveAvgPool2d is replaced by ``GroupNorm(1, C)``. Every other
    architecture (including the default UNet) is unchanged.

    ``shallow_skip`` (``{"c1": 16, "c2": 32}``) wraps the UNet in
    :class:`filseg.shallow.ShallowSkipUnet`: a tiny branch on the input planes feeds
    the decoder a half-resolution skip and a full-resolution feature fused before the
    head. UNet-only. ``None`` (the default) builds exactly the network every
    checkpoint so far was trained with.
    """
    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder {decoder!r}; expected one of {DECODERS}")
    if shallow_skip is not None and decoder != "Unet":
        raise ValueError(f"shallow_skip is UNet-only, got decoder {decoder!r}")
    model = getattr(smp, decoder)(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=classes,
        **(decoder_kwargs or {}),
        **(encoder_kwargs or {}),
    )
    if decoder in ("UPerNet", "DeepLabV3Plus"):
        _groupnorm_pooled_branch(model.decoder)
    if shallow_skip is not None:
        from .shallow import ShallowBranch, ShallowSkipUnet

        model = ShallowSkipUnet(model, ShallowBranch(in_channels, **shallow_skip), decoder_kwargs)
    return model


def _groupnorm_pooled_branch(module: nn.Module) -> None:
    """Swap every BatchNorm2d inside a Sequential that holds an AdaptiveAvgPool2d for GroupNorm(1, C)."""
    pooled = [m for m in module.modules()
              if isinstance(m, nn.Sequential) and any(isinstance(c, nn.AdaptiveAvgPool2d) for c in m)]
    for block in pooled:
        for parent in block.modules():
            for name, child in parent.named_children():
                if isinstance(child, nn.BatchNorm2d):
                    setattr(parent, name, nn.GroupNorm(1, child.num_features))


def arch_kwargs(cfg: dict) -> dict:
    """``build_model`` arguments from a config's ``arch:`` block (defaults: the legacy UNet).

    ``pass_img_size: true`` sets the encoder's ``img_size`` to the config's working
    ``size`` (Swin builds its attention windows for a fixed input size, and each
    curriculum stage has its own size).
    """
    a = dict(cfg.get("arch") or {})
    unknown = sorted(set(a) - set(ARCH_KEYS))
    if unknown:
        raise ValueError(f"unknown arch option(s): {', '.join(unknown)}")
    decoder = a.get("decoder", "Unet")
    if decoder not in DECODERS:
        raise ValueError(f"unknown decoder {decoder!r}; expected one of {DECODERS}")
    enc = dict(a.get("encoder_kwargs") or {})
    if a.get("pass_img_size"):
        enc["img_size"] = int(cfg["size"])
    out = {"decoder": decoder, "encoder_kwargs": enc, "decoder_kwargs": dict(a.get("decoder_kwargs") or {})}
    if a.get("shallow_skip") is not None:
        out["shallow_skip"] = dict(a["shallow_skip"])
    return out


HEADS = ("mask", "ordinal")


def head_channels(head: str) -> int:
    """Output channels the mask head occupies: 1 for a plain mask, 3 for the ordinal head."""
    if head not in HEADS:
        raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
    return 3 if head == "ordinal" else 1


def mask_prob(logits: torch.Tensor, head: str = "mask") -> torch.Tensor:
    """(B, 1, H, W) probability that an annotator marks each pixel.

    ``mask``: the sigmoid of channel 0. ``ordinal``: channels 0-2 are trained on
    soft cumulative agreement levels (see ``losses.ordinal_targets``); their mean
    is the expected agreement fraction exactly, the same quantity a plain head
    learns from per-pass targets. Channels after the head (the spine map) never
    reach the result.
    """
    if head == "ordinal":
        return torch.sigmoid(logits[:, :3]).mean(dim=1, keepdim=True)
    if head != "mask":
        raise ValueError(f"unknown head {head!r}; expected one of {HEADS}")
    return torch.sigmoid(logits[:, :1])


def head_of(model: nn.Module) -> str:
    """The head a model was built with (``build_from_config`` records it); ``mask`` by default."""
    return getattr(model, "filseg_head", "mask")


def build_from_config(cfg: dict, pretrained: bool | None = None) -> nn.Module:
    """The network an experiment config describes: its input planes, head and spine channel.

    Inference passes ``pretrained=False`` (the weights come from the checkpoint).
    The head is recorded on the module as ``filseg_head``, so
    ``mask_prob(logits, head_of(model))`` reads the right channels without the
    config being threaded through every inference call. The ``arch:`` block (see
    :func:`arch_kwargs`) picks the decoder and passes encoder and decoder arguments.
    """
    from .inputs import validate
    from .losses import LossConfig

    head = LossConfig.from_dict(cfg.get("loss")).head
    spine = float(cfg.get("spine_weight", 0.0)) > 0
    model = build_model(
        cfg["encoder"],
        in_channels=len(validate(cfg.get("inputs"))),
        classes=head_channels(head) + int(spine),
        pretrained=bool(cfg.get("pretrained", True)) if pretrained is None else pretrained,
        **arch_kwargs(cfg),
    )
    model.filseg_head = head
    return model


class MaskedBCEDice(nn.Module):
    """BCE + weighted Dice, both restricted to the on-disk region.

    Args:
        dice_weight: kept small (0.25) on purpose -- see the module docstring.
        pos_weight: positives are ~0.36% of on-disk pixels; a mild upweight
            speeds early convergence without distorting calibration much.
    """

    def __init__(self, dice_weight: float = 0.25, pos_weight: float = 3.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.register_buffer("pos_weight", torch.tensor(pos_weight))

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight, reduction="none"
        )
        n = valid.sum().clamp_min(1.0)
        bce = (bce * valid).sum() / n

        if self.dice_weight <= 0:
            return bce

        p = torch.sigmoid(logits) * valid
        t = target * valid
        num = 2.0 * (p * t).sum() + 1.0
        den = p.sum() + t.sum() + 1.0
        return bce + self.dice_weight * (1.0 - num / den)


class MaskSpineLoss(nn.Module):
    """Mask loss plus an auxiliary spine term.

    The spine head exists to regularise, not to be used. With 707 exposures and
    a 50-90 M parameter encoder there is ample room to overfit -- and our
    CV-to-LB gap of 0.055-0.067, which `scripts/diagnose_oof.py` showed is not a
    scoring artefact, is consistent with exactly that. Asking the decoder to also
    place each filament's axis forces it to represent the structure's topology
    rather than memorising blobs.

    The spine term is plain masked BCE with no Dice: the target is a soft ridge
    in [0, 1], so overlap-style losses are meaningless on it. Only channel 0
    reaches inference, so a poorly-learned spine head costs nothing at test time.
    """

    def __init__(
        self, dice_weight: float = 0.25, pos_weight: float = 3.0,
        spine_weight: float = 0.2, spine_pos_weight: float = 10.0,
    ) -> None:
        super().__init__()
        self.mask_loss = MaskedBCEDice(dice_weight, pos_weight)
        self.spine_weight = spine_weight
        # The spine occupies far fewer pixels than the mask it sits inside, so
        # it needs a stronger positive weight to produce any gradient at all.
        self.register_buffer("spine_pos_weight", torch.tensor(spine_pos_weight))

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        loss = self.mask_loss(logits[:, :1], target[:, :1], valid)
        if logits.shape[1] < 2 or target.shape[1] < 2 or self.spine_weight <= 0:
            return loss
        spine = F.binary_cross_entropy_with_logits(
            logits[:, 1:2], target[:, 1:2],
            pos_weight=self.spine_pos_weight, reduction="none",
        )
        n = valid.sum().clamp_min(1.0)
        return loss + self.spine_weight * (spine * valid).sum() / n
