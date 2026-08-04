import torch
import torch.nn as nn
from torchvision.transforms import v2


class ImageNormalizer(nn.Module):
    def __init__(
        self,
        img_size: int,
        mean: list[float],
        std: list[float],
        num_output_channels: int = 3,
        input_scale: float = 1.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.num_output_channels = num_output_channels
        self.input_scale = input_scale

        self.resize = v2.Resize((img_size, img_size), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)
        self.normalize = v2.Normalize(mean=mean, std=std)

        if num_output_channels == 1:
            self.grayscale = v2.Grayscale(num_output_channels=1)
        else:
            self.grayscale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hard check: Input must be BxCxHxW
        assert x.ndim == 4, f"Input must be BxCxHxW format, got {x.ndim}D"

        # Check dtype and handle conversion/scaling
        if x.dtype == torch.uint8:
            x = x.to(torch.float32) / 255.0
        elif x.dtype == torch.float32:
            assert x.min() >= -1e-6 and x.max() <= 1.0 + 1e-6, "float32 input must be in [0, 1] range"
        else:
            raise ValueError(f"Unsupported dtype: {x.dtype}. Only uint8 and float32 are supported.")

        # Apply input scale (e.g., 255.0 for MedSAM)
        if self.input_scale != 1.0:
            x = x * self.input_scale

        # Handle channel conversion
        C = x.shape[1]
        if self.num_output_channels == 3 and C == 1:
            x = x.repeat(1, 3, 1, 1)
        elif self.num_output_channels == 1 and C == 3:
            x = self.grayscale(x)

        # Ensure correct number of channels after conversion
        assert x.shape[1] == self.num_output_channels, f"Expected {self.num_output_channels} channels, got {x.shape[1]}"

        # Resize and Normalize
        x = self.resize(x)
        x = self.normalize(x)

        return x
