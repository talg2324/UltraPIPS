from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

from .base import UltrasoundImageEncoder
from .utils import ImageNormalizer


class TinyUSFMBackbone(UltrasoundImageEncoder):
    """
    TinyUSFM (ViT-Tiny distilled from USFM): https://github.com/MacDunno/TinyUSFM
    The upstream model is a thin wrapper around timm's VisionTransformer, so the
    checkpoint loads directly into timm after stripping the wrapper prefixes.
    """

    def __init__(self,
                 weights_path: str | None = None,
                 layers: list[int] | None = None):
        super().__init__(weights_path)
        if weights_path is None:
            raise ValueError("TinyUSFMBackbone requires weights_path.")

        self.layers = layers  # block indices to return; None -> all blocks

        self.model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=192,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,
            global_pool='',
        )
        # Upstream preprocessing: RGB, 224x224, ImageNet stats
        self.normalizer = ImageNormalizer(
            img_size=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_output_channels=3
        )

        checkpoint = torch.load(weights_path, map_location='cpu')
        state_dict = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ('module.', 'model.'):
                if k.startswith(prefix):
                    k = k[len(prefix):]
            # Head is unused; the final norm is kept for embed()
            if k.startswith(('head', 'fc_norm')):
                continue
            cleaned[k] = v

        missing, _ = self.model.load_state_dict(cleaned, strict=False)
        missing = [k for k in missing if not k.startswith(('head', 'fc_norm'))]
        if missing:
            raise RuntimeError(f"TinyUSFM checkpoint is missing backbone weights: {missing}")

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        # forward_features applies the final norm; take the CLS token
        return self.model.forward_features(self.normalizer(x))[:, 0]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalizer(x)

        # Per-block outputs as (B, C, H, W), cls token removed
        return self.model.forward_intermediates(
            x,
            indices=self.layers,
            norm=False,
            output_fmt='NCHW',
            intermediates_only=True,
        )
