import os
import glob

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class CAMUSAugDataset(Dataset):
    """Pairs augmented images+masks for one (tgt, aug, aug_power).

    Yields: (aug_image:   Tensor[3,H,W] float32 [0,1],
             clean_image: Tensor[3,H,W] float32 [0,1],
             lv_mask:     ndarray[H,W]  uint8   binary,
             la_mask:     ndarray[H,W]  uint8   binary,
             frame_id:    str)           "batch_N/frameK"

    Sources per (tgt, aug, aug_power):
      aug/{tgt}/batch_*/{aug}-{aug_power}/images/frameK_aug0.png   <- augmented
      aug/{tgt}/batch_*/{aug}-{aug_power}/images/frameK.png        <- original
      aug/{tgt}/batch_*/{aug}-{aug_power}/labels/frameK_aug0.png   <- warped mask
                                              0=bg, 1=LV, 2=MYO, 3=LA

    Only frames with both aug image AND aug label are included.
    Order: batch_* ascending, frameK ascending (deterministic).
    """

    def __init__(self, tgt: str, aug: str, aug_power, dataset: str = 'CAMUS'):
        super().__init__()
        self.pairs = []

        aug_base = f'assets/data/{dataset}/aug'
        # Helper to find the correct variant directory since 1 could be 1.0

        def find_variant_dir(batch_dir, aug, power):
            for fmt in [str(power), str(float(power)), str(int(float(power))) if float(power).is_integer() else str(power)]:
                v = f"{aug}-{fmt}"
                if os.path.isdir(os.path.join(batch_dir, v, 'images')):
                    return v
            return f"{aug}-{power}"

        for batch_dir in sorted(glob.glob(os.path.join(aug_base, tgt, 'batch_*'))):
            batch_name = os.path.basename(batch_dir)

            variant = find_variant_dir(batch_dir, aug, aug_power)

            aug_img_dir = os.path.join(batch_dir, variant, 'images')
            aug_lbl_dir = os.path.join(batch_dir, variant, 'labels')

            aug_files = sorted(glob.glob(os.path.join(aug_img_dir, '*_aug0.png')))

            for aug_path in aug_files:
                fname = os.path.basename(aug_path)
                frame_name = fname.replace('_aug0.png', '')
                clean_path = os.path.join(aug_img_dir, f"{frame_name}.png")
                lbl_path = os.path.join(aug_lbl_dir, fname)

                if os.path.exists(lbl_path):
                    self.pairs.append({
                        'aug_path': aug_path,
                        'clean_path': clean_path,
                        'lbl_path': lbl_path,
                        'frame_id': f"{batch_name}/{frame_name}"
                    })

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]

        # Load augmented image
        aug_im = cv2.imread(item['aug_path'], cv2.IMREAD_COLOR)
        aug_im = torch.from_numpy(cv2.cvtColor(aug_im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        # Load clean image
        clean_im = cv2.imread(item['clean_path'], cv2.IMREAD_COLOR)
        clean_im = torch.from_numpy(cv2.cvtColor(clean_im, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0

        # Load labels
        lbl = cv2.imread(item['lbl_path'], cv2.IMREAD_GRAYSCALE)

        lv_mask = (lbl == 1).astype(np.uint8)
        la_mask = (lbl == 3).astype(np.uint8)

        return aug_im, clean_im, lv_mask, la_mask, item['frame_id']
