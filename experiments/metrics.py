import torch
import torch.nn as nn
from pytorch_msssim import ssim
from monai.losses import PerceptualLoss
from torchvision.transforms.v2 import Normalize

from ultrapips import UltraPIPS


class MetricWrapper(nn.Module):
    def __init__(self, name, model):
        super().__init__()
        self.name = name
        self.model = model
        self.model.eval()

    def forward(self, x, y):
        return self.model(x, y)


class SSIMMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "SSIM"

    def forward(self, x, y):
        # ssim returns similarity [0, 1], we want loss [0, 1]
        return 1.0 - ssim(x, y, data_range=1.0, win_size=5, size_average=True)


class L2Metric(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "L2"

    def forward(self, x, y):
        return nn.functional.mse_loss(x, y)


class NormalizedPerceptualLoss(nn.Module):
    """Wraps MONAI PerceptualLoss with input remapping [0,1] -> [-1,1].

    LPIPS (alex/vgg) was calibrated on [-1, 1] inputs. MONAI does not
    pass normalize=True to the underlying lpips.LPIPS call, so without
    this wrapper the network operates in an unintended regime.
    RadImageNet handles its own preprocessing internally, so it is
    excluded here.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.normalize = Normalize(mean=[0.5], std=[0.5])  # [0,1] -> [-1,1]
        self.model = model

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.model(self.normalize(x), self.normalize(y))


def get_monai_lpips(network_type, is_fake_3d=False):
    # network_type: "alex", "vgg", "radimagenet_resnet50"
    # Note: monai's PerceptualLoss expects 3D or 2D.
    # For 2D, it handles typical ImageNet backbones if requested.
    model = PerceptualLoss(
        spatial_dims=2,
        network_type=network_type,
        is_fake_3d=is_fake_3d,
        pretrained=True
    )

    if network_type in ('alex', 'vgg'):
        model = NormalizedPerceptualLoss(model)

    return model


def metrics_list(model_name):
    backbones = {
        'usfm': 'assets/weights/USFM.pt',
        'medsam': 'assets/weights/MedSAM.pt',
        'tusa_logits': None,
        'tusa_vit': None,
        'vit_imagenet': None,
        'swin_imagenet': None,
        'clip': None,
        'biomedclip': None,
        'ultrasound_clip': 'assets/weights/ultrasound-clip.pt',
    }

    if model_name == 'classic':
        metrics = [
            L2Metric(),
            SSIMMetric()
        ]
    elif model_name == 'monai':
        metrics = [
            MetricWrapper(f'MONAI_{net}', get_monai_lpips(net)) for net in ['alex', 'vgg', 'radimagenet_resnet50']
        ]
    elif model_name in backbones:
        metrics = [
            MetricWrapper(f'UltraPIPS_{model_name}', UltraPIPS(model_name, backbones[model_name]))
        ]

    return metrics
