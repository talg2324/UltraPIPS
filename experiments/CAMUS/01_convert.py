import os
import glob
import random

import cv2
import numpy as np
import nibabel as nib
from tqdm import tqdm


src_dir = '/storage/talg/data/camus/database_nifti/'
dst_dir = 'experiments/assets/data/CAMUS/raw'


def load_ims(path, is_mask=False):
    data = nib.load(str(path)).get_fdata()
    if data.ndim == 3:
        data = np.moveaxis(data, -1, 0)
    elif data.ndim == 2:
        data = data[np.newaxis, ...]

    if not is_mask:
        mi, ma = data.min(), data.max()
        data = (data - mi) / (ma - mi + 1e-8) * 255

    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return [cv2.resize(f, (256, 256), interpolation=interp).astype(np.uint8).T for f in data[::20]]


if __name__ == "__main__":
    mask_files = glob.glob(pathname="**/*_gt.nii.gz", root_dir=src_dir, recursive=True)
    mask_files = random.sample(mask_files, k=1000)

    BATCH_SIZE = 64
    # state tracks: { 'tgt': { 'batch_idx': 0, 'frame_count': 0 } }
    state = {
        'A2C': {'batch_idx': 0, 'frame_count': 0},
        'A4C': {'batch_idx': 0, 'frame_count': 0}
    }

    for mask_path in tqdm(mask_files):
        im_path = mask_path.replace('_gt', '')

        ims = load_ims(os.path.join(src_dir, im_path), is_mask=False)
        masks = load_ims(os.path.join(src_dir, mask_path), is_mask=True)

        tgt = 'A2C' if '2CH' in mask_path else 'A4C'

        for im, mask in zip(ims, masks):
            s = state[tgt]

            # If batch is full, move to next one
            if s['frame_count'] >= BATCH_SIZE:
                s['batch_idx'] += 1
                s['frame_count'] = 0

            batch_dir = os.path.join(dst_dir, tgt, f'batch_{s["batch_idx"]}')
            os.makedirs(os.path.join(batch_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(batch_dir, 'labels'), exist_ok=True)

            cv2.imwrite(os.path.join(batch_dir, 'images', f'frame{s["frame_count"]:03d}.png'), im)
            cv2.imwrite(os.path.join(batch_dir, 'labels', f'frame{s["frame_count"]:03d}.png'), mask)

            s['frame_count'] += 1

    print("\nBatching Summary:")
    for tgt in ['A2C', 'A4C']:
        # Total batches is batch_idx + 1
        n_batches = state[tgt]['batch_idx'] + 1
        print(f"  {tgt}: {n_batches} batches")
