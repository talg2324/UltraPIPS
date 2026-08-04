import torch
from segment_anything import sam_model_registry

from .base import UltrasoundImageEncoder
from .utils import ImageNormalizer


class MedSAMBackbone(UltrasoundImageEncoder):
    def __init__(self, weights_path: str | None = None):
        super().__init__(weights_path)
        if weights_path is None:
            raise ValueError("MedSAMBackbone requires weights_path (checkpoint).")
        sam = sam_model_registry["vit_b"](checkpoint=weights_path)
        self.model = sam.image_encoder

        self.normalizer = ImageNormalizer(
            img_size=1024,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            num_output_channels=3,
            input_scale=255.0
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalizer(x)

        x = self.model.patch_embed(x)
        if self.model.pos_embed is not None:
            x = x + self.model.pos_embed

        features = []
        for blk in self.model.blocks:
            x = blk(x)
            features.append(x)

        # SAM ViT blocks return (B, H, W, C)
        formatted_features = []
        for feat in features:
            # (B, H, W, C) -> (B, C, H, W)
            formatted_features.append(feat.permute(0, 3, 1, 2))

        return formatted_features
