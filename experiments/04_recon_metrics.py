import os
import glob
from argparse import ArgumentParser

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_laplace
from INR.data import SweepingTransducerDataset
from skimage.metrics import structural_similarity
from skimage.feature import graycoprops, graycomatrix


parser = ArgumentParser()
parser.add_argument("--tgt", choices=['follicles', 'prostate'], default='follicles')


def hfen(img1, img2, sigma):
    """
    High-Frequency Error Norm computed via Laplacian of Gaussian.
    Lower is better. Sigma should reflect expected speckle grain size —
    1.5 is a reasonable default but consider tuning to transducer frequency.
    """
    log1 = gaussian_laplace(img1, sigma=sigma)
    log2 = gaussian_laplace(img2, sigma=sigma)
    return np.linalg.norm(log1 - log2)


def compute_glcm_features(img, levels=64, distances=(1, 3), angles=(0, np.pi/4, np.pi/2, 3*np.pi/4)):
    """Quantize image and extract GLCM texture features, averaged over distances and angles."""
    # Normalize to [0, levels-1] — use per-image range to be robust to intensity drift across volumes
    img_min, img_max = img.min(), img.max()
    quantized = ((img - img_min) / (img_max - img_min + 1e-8) * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(quantized, distances=list(distances), angles=list(angles),
                        levels=levels, symmetric=True, normed=True)

    return {
        prop: graycoprops(glcm, prop).mean()
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    }


def glcm_distance(y, yhat, levels=32):
    """
    L1 distance between GLCM feature vectors of ground truth and reconstruction.
    Features are normalized by their expected range so each contributes equally.
    Lower = textures match better.
    """
    gt_feats = compute_glcm_features(y,    levels=levels)
    pred_feats = compute_glcm_features(yhat, levels=levels)

    diffs = {
        f'GLCM_{k.title()}': abs(gt_feats[k] - pred_feats[k])
        for k in gt_feats
    }
    return diffs


if __name__ == "__main__":
    args = parser.parse_args()
    data_dir = os.path.join(f'assets/results/04/raw/{args.tgt}')
    dst_dir = os.path.join(f'assets/results/04/raw/{args.tgt}/metrics')
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob('**/*.npz', root_dir=data_dir)

    rows = []

    for i, f in enumerate(tqdm(files)):
        data_file, model_name = f.removesuffix('.npz').split('/')
        ds = SweepingTransducerDataset(os.path.join('INR', 'data', args.tgt, data_file), dec_factor=1)
        data = np.load(os.path.join(data_dir, f))

        if model_name == 'ssim':
            continue

        for j, (y, yhat) in enumerate(zip(ds.images.numpy(), data['reconstructions'])):
            s = structural_similarity(y, yhat, win_size=5, data_range=2.)
            h = hfen(y, yhat, sigma=2.5)
            g = glcm_distance(y, yhat)

            rows.append({
                'data_file': data_file,
                'model': model_name,
                'ssim': s,
                'hfen': h,
                **g,
            })

            if args.tgt == 'follicles' and data_file == 'e5e46' and j == 140:
                y = 127.5 * (y + 1)
                yhat = 127.5 * (yhat + 1)
                cv2.imwrite(os.path.join(dst_dir, 'base.png'), y.clip(0, 255).astype(np.uint8))
                cv2.imwrite(os.path.join(dst_dir, f'{model_name}.png'), yhat.clip(0, 255).astype(np.uint8))

    pd.DataFrame(rows).to_csv(os.path.join(dst_dir, 'metrics.csv'), index=False)
