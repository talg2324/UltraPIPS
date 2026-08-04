"""experiments/CAMUS/01_camus.py

Evaluates EchoPrime classification confidence degradation on CAMUS (A2C/A4C)
under pre-computed EchoGains augmentations, alongside UltraPIPS perceptual metrics.

Aug variants (e.g. depth-15, depth-30, ...) are discovered automatically from
the aug directory — no CLI flag needed. Adding a new augmentation type to the
pipeline will be picked up automatically on the next run.

Augmented files produced by echogains follow the naming convention:
  aug/{tgt}/batch_N/{variant}/images/frame{K}_aug0.png   <- augmented
  aug/{tgt}/batch_N/{variant}/images/frame{K}.png        <- original copy
Clean frames are loaded from:
  raw/{tgt}/batch_N/images/frame{K}.png

Output: assets/results/CAMUS/{model}/{tgt}.npz
  variants   : (num_variants+1,) str  — ['clean', 'depth-15', ...]
  y          : (N,)              int  — true EchoPrime class index
  confidence : (N, num_variants+1)   — softmax prob of correct class per step
  yhat       : (N, num_variants+1)   — argmax prediction per step
  {metric}   : (N, num_variants)     — per-frame perceptual loss vs clean (aug steps only)
"""
import os
import glob
import argparse

import cv2
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import functional as TF

from metrics import metrics_list
from EchoPrime.echoprime import EchoPrime

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# EchoPrime 11-class indices for the two CAMUS views
CAMUS_ECHOPRIME_LABEL = {
    'A2C': 0,  # A2C is class 0 in COARSE_VIEWS
    'A4C': 2,  # A4C is class 2 in COARSE_VIEWS
}

RAW_BASE = 'assets/data/CAMUS/raw'
AUG_BASE = 'assets/data/CAMUS/aug'
SAVE_BASE = 'assets/results/CAMUS'
ECHOPRIME_WEIGHTS = 'assets/weights/view_classifier.pt'
ECHOPRIME_SIZE = 224


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def echoprime_norm(x: torch.Tensor) -> torch.Tensor:
    x = x * 255
    return TF.normalize(x, mean=[29.110628, 28.076836, 29.096405],
                        std=[47.989223, 46.456997, 47.20083])


def load_image(path: str) -> torch.Tensor:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
    return TF.resize(t, [ECHOPRIME_SIZE, ECHOPRIME_SIZE], antialias=True)


def discover_variants(tgt: str, augmentation: str) -> list[str]:
    """Return sorted variant names for a given augmentation type (e.g. ['depth-15', 'depth-30', ...]).

    Only returns variants whose directory name starts with '{augmentation}-',
    so different augmentation types stay isolated.
    """
    variants: set[str] = set()
    prefix = f'{augmentation}-'
    for batch_dir in glob.glob(os.path.join(AUG_BASE, tgt, 'batch_*')):
        for name in os.listdir(batch_dir):
            if name.startswith(prefix) and os.path.isdir(os.path.join(batch_dir, name, 'images')):
                variants.add(name)
    return sorted(variants)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CAMUSDataset(Dataset):
    """Returns (clean, aug, label) tuples for a given tgt and aug variant.

    aug_variant=None  →  clean baseline: aug is identical to clean.
    Pairing is done batch-by-batch using the echogains naming convention:
      raw frame  : raw/{tgt}/batch_N/images/frame{K}.png
      aug frame  : aug/{tgt}/batch_N/{variant}/images/frame{K}_aug0.png
    """

    def __init__(self, tgt: str, aug_variant: str | None):
        self.label = CAMUS_ECHOPRIME_LABEL[tgt]
        self.pairs: list[tuple[str, str]] = []

        for raw_batch_dir in sorted(glob.glob(os.path.join(RAW_BASE, tgt, 'batch_*'))):
            batch_name = os.path.basename(raw_batch_dir)
            clean_files = sorted(glob.glob(os.path.join(raw_batch_dir, 'images', 'frame*.png')))

            if aug_variant is None:
                aug_files = clean_files
            else:
                aug_dir = os.path.join(AUG_BASE, tgt, batch_name, aug_variant, 'images')
                aug_files = sorted(glob.glob(os.path.join(aug_dir, '*_aug0.png')))

            if len(clean_files) != len(aug_files):
                raise ValueError(
                    f'[{tgt}/{batch_name}] count mismatch for variant={aug_variant!r}: '
                    f'{len(clean_files)} clean vs {len(aug_files)} aug'
                )

            self.pairs.extend(zip(clean_files, aug_files))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        clean_path, aug_path = self.pairs[idx]
        return load_image(clean_path), load_image(aug_path), self.label


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='EchoPrime confidence degradation on CAMUS under EchoGains augmentations.'
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Perceptual metric (e.g. 'usfm', 'classic', 'monai')")
    parser.add_argument('--augmentation', type=str, required=True,
                        help="EchoGains augmentation type to evaluate (e.g. 'depth')")
    parser.add_argument('--tgt', type=str, default='both', choices=['A2C', 'A4C', 'both'])
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EchoPrime(ECHOPRIME_WEIGHTS).eval().to(device)
    metrics = metrics_list(args.model)

    tgts = ['A2C', 'A4C'] if args.tgt == 'both' else [args.tgt]

    batch_size = 16 if args.model != 'medsam' else 2

    for tgt in tgts:
        variants = discover_variants(tgt, args.augmentation)  # ['depth-15', 'depth-30', ...]
        all_variants = [None] + variants            # None = clean baseline
        label = CAMUS_ECHOPRIME_LABEL[tgt]

        print(f'\n=== {tgt} | discovered variants: {variants} ===')

        n_images = len(CAMUSDataset(tgt, None))
        n_steps = len(all_variants)
        confidence = np.zeros((n_images, n_steps))
        yhat = np.zeros((n_images, n_steps), dtype=np.int32)
        metric_acc = {m.name: np.zeros((n_images, len(variants))) for m in metrics}

        for step_idx, variant in enumerate(all_variants):
            desc = variant or 'clean (baseline)'
            ds = CAMUSDataset(tgt, variant)
            dl = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4
            )

            i = 0
            for clean, aug, _ in tqdm(dl, desc=f'  {desc}'):
                clean, aug = clean.to(device), aug.to(device)
                b = aug.shape[0]

                with torch.no_grad():
                    logits = model(echoprime_norm(aug))
                    probs = F.softmax(logits, dim=1)

                    confidence[i:i+b, step_idx] = probs[:, label].cpu().numpy()
                    yhat[i:i+b, step_idx] = logits.argmax(dim=1).cpu().numpy()

                    if variant is not None:
                        aug_idx = step_idx - 1  # metric arrays exclude the clean baseline
                        for m in metrics:
                            m.to(device)
                            vals = torch.stack([m(aug[k:k+1], clean[k:k+1]) for k in range(b)])
                            metric_acc[m.name][i:i+b, aug_idx] = vals.cpu().numpy()
                            m.cpu()

                i += b

            mean_conf = confidence[:, step_idx].mean()
            acc = np.mean(yhat[:, step_idx] == label)
            print(f'    mean confidence: {mean_conf:.3f}  accuracy: {acc:.3f}')

        save_dir = os.path.join(SAVE_BASE, args.augmentation, args.model)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{tgt}.npz')

        np.savez(
            save_path,
            variants=np.array(['clean'] + variants),
            y=np.full(n_images, label, dtype=np.int32),
            confidence=confidence,
            yhat=yhat,
            **metric_acc,
        )
        print(f'  Saved → {save_path}')
