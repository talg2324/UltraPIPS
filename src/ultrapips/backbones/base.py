from abc import ABC

import torch
import torch.nn as nn


class UltrasoundImageEncoder(nn.Module, ABC):
    """Base class for ultrasound perceptual backbones."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Global (B, D) embedding for FID. Defaults to the mean-pooled last feature map."""
        return self(x)[-1].mean(dim=(2, 3))
