import torch
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights, Swin_B_Weights

from .base import UltrasoundImageEncoder
from .utils import ImageNormalizer


class ViTImageNetBackbone(UltrasoundImageEncoder):
    """
    ViT-B-16 backbone from torchvision, pretrained on ImageNet-1K.
    Aligned with MedSAM and USFM implementations.
    """

    def __init__(self, weights_path: str | None = None):
        super().__init__(weights_path)
        # weights_path is ignored as we use ImageNet weights by default
        self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # ImageNet normalization stats
        self.normalizer = ImageNormalizer(
            img_size=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_output_channels=3
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalizer(x)

        # ViT forward features logic aligned with torchvision structure
        # Reshape to patches
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand cls token
        cls_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add pos embedding
        x = self.model.encoder.pos_embedding + x
        x = self.model.encoder.dropout(x)

        features = []
        # Pass through each encoder block
        for layer in self.model.encoder.layers:
            x = layer(x)
            # x is (B, L, C) where L = 1 (cls) + 196 (patches)
            # Remove cls token and reshape to (B, C, H, W)
            f = x[:, 1:, :]  # (B, 196, 768)
            B, L, C = f.shape
            H = W = int(L**0.5)
            f = f.permute(0, 2, 1).reshape(B, C, H, W)
            features.append(f)

        return features


class SwinImageNetBackbone(UltrasoundImageEncoder):
    """
    Swin-B backbone from torchvision, pretrained on ImageNet-1K.
    Aligned with TUSA SwinViT implementation.
    """

    def __init__(self, weights_path: str | None = None):
        super().__init__(weights_path)
        self.model = models.swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)

        # ImageNet normalization stats
        self.normalizer = ImageNormalizer(
            img_size=224,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            num_output_channels=3
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalizer(x)

        features = []
        # Swin model.features has 8 blocks:
        # [PatchEmbed, Stage1, PatchMerge, Stage2, PatchMerge, Stage3, PatchMerge, Stage4]
        # We want the output of each Stage (blocks 1, 3, 5, 7)
        for i, block in enumerate(self.model.features):
            x = block(x)
            if i in [1, 3, 5, 7]:
                # Swin outputs (B, H, W, C)
                # Permute to (B, C, H, W) to match TUSA alignment (which uses (B, C, H, W))
                features.append(x.permute(0, 3, 1, 2))

        return features
