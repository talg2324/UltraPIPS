import os
import glob
import pickle

import cv2
import numpy as np
from tqdm import tqdm


src_dir = '/mnt/d/talg/echonet_segtest/test'
dst_dir = 'experiments/assets/data/EchoNet/raw'


if __name__ == "__main__":
    pkl_files = glob.glob(os.path.join(src_dir, "*.pkl"))

    BATCH_SIZE = 64
    batch_idx = 0
    frame_count = 0
    tgt = 'A4C'

    import random

    for pkl_path in tqdm(pkl_files):
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        images = data['images']
        masks = data['masks']

        if len(images) > 0:
            idx = random.randrange(len(images))
            im = images[idx]
            mask = masks[idx]

            # If batch is full, move to next one
            if frame_count >= BATCH_SIZE:
                batch_idx += 1
                frame_count = 0

            batch_dir = os.path.join(dst_dir, tgt, f'batch_{batch_idx}')
            os.makedirs(os.path.join(batch_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(batch_dir, 'labels'), exist_ok=True)

            im_resized = cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

            cv2.imwrite(os.path.join(batch_dir, 'images', f'frame{frame_count:03d}.png'), im_resized)
            cv2.imwrite(os.path.join(batch_dir, 'labels', f'frame{frame_count:03d}.png'), mask_resized)

            frame_count += 1

    print("\nBatching Summary:")
    print(f"  {tgt}: {batch_idx + 1} batches")
