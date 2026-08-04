import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pytorch_msssim import ssim
from ultrapips.loss import UltraPIPS, BackboneType

from INR import INR
from metrics import get_monai_lpips
from INR.data import SweepingTransducerDataset


parser = ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
parser.add_argument("--loss-metric", type=str)
parser.add_argument("--log-interval", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--n-epochs", type=int, default=300)


class LossFunction(nn.Module):
    def __init__(self, loss_metric: str):
        super().__init__()

        self.l2 = nn.MSELoss()
        self.additional_loss = None

        if loss_metric == 'l2':
            pass

        elif loss_metric == 'ssim':
            self.additional_loss = ssim_loss

        elif loss_metric == 'alex':
            self.additional_loss = get_monai_lpips('alex')

        elif loss_metric == 'vgg':
            self.additional_loss = get_monai_lpips('vgg')

        elif loss_metric == 'radimagenet':
            self.additional_loss = get_monai_lpips('radimagenet_resnet50')

        elif loss_metric == 'vit_imagenet':
            self.additional_loss = UltraPIPS(BackboneType.VIT_IMAGENET)

        elif loss_metric == 'swin_imagenet':
            self.additional_loss = UltraPIPS(BackboneType.SWIN_IMAGENET)

        elif loss_metric == 'clip':
            self.additional_loss = UltraPIPS(BackboneType.CLIP)

        elif loss_metric == 'biomedclip':
            self.additional_loss = UltraPIPS(BackboneType.BIOMEDCLIP)

        elif loss_metric == 'usfm':
            self.additional_loss = UltraPIPS(BackboneType.USFM, 'assets/weights/USFM.pt')

        elif loss_metric == 'tusa_vit':
            self.additional_loss = UltraPIPS(BackboneType.TUSA_VIT)

        elif loss_metric == 'ultrasound_clip':
            self.additional_loss = UltraPIPS(BackboneType.ULTRASOUND_CLIP, 'assets/weights/ultrasound-clip.pt')

        else:
            raise NotImplementedError(f"Invalid loss metric")

    def forward(self, y, yhat):
        loss = self.l2(y, yhat)

        if self.additional_loss is not None:
            loss += self.additional_loss(
                (y + 1) / 2.,
                (yhat.float() + 1) / 2.,
            )

        return loss


def ssim_loss(im1, im2):
    return 1 - ssim(im1, im2, data_range=1., win_size=5, size_average=True)


def train_loop(model, loss_fxn, dl, optimizer, device):
    total_loss = 0
    for scan_pos, bmode in dl:
        bmode = bmode.to(device).unsqueeze(1)
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            yhat = model(scan_pos.to(device))
            loss = loss_fxn(bmode, yhat.unsqueeze(1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    return dict(loss=total_loss / len(dl))


if __name__ == "__main__":
    args = parser.parse_args()
    device = torch.device('cuda:0')
    dl = SweepingTransducerDataset.dataloader(
        os.path.join('INR', 'data', args.data_dir),
        batch_size=args.batch_size,
        dec_factor=1
    )
    dst = os.path.join('INR', 'data', 'inrs')
    os.makedirs(dst, exist_ok=True)

    torch.manual_seed(7)
    model = INR().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs, eta_min=1e-6)
    loss_fxn = LossFunction(args.loss_metric).to(device)

    for e in range(1, args.n_epochs + 1, args.log_interval):
        with tqdm(range(args.log_interval), desc=f"Epoch #{e}/{args.n_epochs}", bar_format="{l_bar}{r_bar}") as progress_bar:
            for k in progress_bar:
                losses = train_loop(model, loss_fxn, dl, optimizer, device)
                progress_bar.set_postfix(losses)
                scheduler.step()

    # Final evaluation: store reconstructions
    reconstructions = []
    eval_dl = torch.utils.data.DataLoader(dl.dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    with torch.inference_mode():
        for scan_pos, bmode in tqdm(eval_dl, desc="Final Evaluation"):
            yhat = model(scan_pos.to(device))
            reconstructions.append(yhat.cpu().numpy())

    reconstructions = np.concatenate(reconstructions, axis=0)

    # Save results in standardized format
    results_dir = os.path.join("assets", "results", "04", args.data_dir)
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, f"{args.loss_metric}.npz")
    np.savez(
        save_path,
        reconstructions=reconstructions,
    )
