import torch
from tusa.texture import tusa_unet

from .utils import ImageNormalizer
from .base import UltrasoundImageEncoder


class TUSA_SwinViTBackbone(UltrasoundImageEncoder):
    def __init__(self, weights_path: str | None = None):
        super().__init__(weights_path)
        try:
            full_model = tusa_unet()
            if weights_path:
                full_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
            self.swinViT = full_model.swinViT
        except ImportError:
            raise ImportError("Please install 'tusa' from the reference repository.")

        self.img_size = 128
        self.normalizer = ImageNormalizer(
            img_size=128,
            mean=[0.5],
            std=[0.5],
            num_output_channels=1
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.normalizer(x)
        return self.swinViT(x, True)
