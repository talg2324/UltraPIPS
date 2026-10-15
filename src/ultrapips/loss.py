from enum import Enum

import torch
import torch.nn as nn


class BackboneType(str, Enum):
    USFM = "usfm"
    TINYUSFM = "tinyusfm"
    MEDSAM = "medsam"
    TUSA_VIT = "tusa_vit"
    VIT_IMAGENET = "vit_imagenet"
    SWIN_IMAGENET = "swin_imagenet"
    CLIP = "clip"
    BIOMEDCLIP = "biomedclip"
    ULTRASOUND_CLIP = "ultrasound_clip"


class UltraPIPS(nn.Module):
    """
    Perceptual image patch similarity for B-mode ultrasound images.
    """

    def __init__(
        self,
        backbone: str | BackboneType,
        model_path: str | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.reduction = reduction

        if isinstance(backbone, str):
            backbone = BackboneType(backbone.lower())

        if backbone == BackboneType.USFM:
            from .backbones.usfm_vit import USFMBackbone
            self.encoder = USFMBackbone(weights_path=model_path)
        elif backbone == BackboneType.TINYUSFM:
            from .backbones.tinyusfm import TinyUSFMBackbone
            self.encoder = TinyUSFMBackbone(weights_path=model_path)
        elif backbone == BackboneType.MEDSAM:
            from .backbones.medsam import MedSAMBackbone
            self.encoder = MedSAMBackbone(weights_path=model_path)
        elif backbone == BackboneType.TUSA_VIT:
            from .backbones.tusa import TUSA_SwinViTBackbone
            self.encoder = TUSA_SwinViTBackbone(weights_path=model_path)
        elif backbone == BackboneType.VIT_IMAGENET:
            from .backbones.torchvision import ViTImageNetBackbone
            self.encoder = ViTImageNetBackbone(weights_path=model_path)
        elif backbone == BackboneType.SWIN_IMAGENET:
            from .backbones.torchvision import SwinImageNetBackbone
            self.encoder = SwinImageNetBackbone(weights_path=model_path)
        elif backbone in [BackboneType.CLIP, BackboneType.BIOMEDCLIP, BackboneType.ULTRASOUND_CLIP]:
            from .backbones.clip import CLIPBackbone
            self.encoder = CLIPBackbone(model_type=backbone.value, weights_path=model_path)
        else:
            raise ValueError(f"Unknown backbone type: {backbone}")

    def _normalize(self, x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Unit normalize features along channel dimension."""
        norm = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True) + eps)
        return x / norm

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: tensor to compare (B, C, H, W) in [0, 1] range.
            target: reference tensor (B, C, H, W) in [0, 1] range.
        """
        feats_inp = self.encoder(input)
        feats_tgt = self.encoder(target)

        dist = None
        for f_in, f_tg in zip(feats_inp, feats_tgt):
            f_in = self._normalize(f_in)
            f_tg = self._normalize(f_tg)

            diff = (f_in - f_tg) ** 2

            # Spatial mean -> sum channels
            layer_dist = diff.mean(dim=[2, 3]).sum(dim=1)

            if dist is None:
                dist = layer_dist
            else:
                dist = dist + layer_dist

        if self.reduction == "mean":
            return dist.mean()
        elif self.reduction == "sum":
            return dist.sum()
        return dist
