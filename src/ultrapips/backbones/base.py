from abc import ABC

import torch.nn as nn


class UltrasoundImageEncoder(nn.Module, ABC):
    """Base class for ultrasound perceptual backbones."""

    def __init__(self, weights_path: str | None = None):
        super().__init__()
