import os
import sys
import json
import argparse
import tempfile
import subprocess

import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import measure
from pycocotools import mask as maskUtils

from utils import CAMUSAugDataset

# Configuration
ULTRASAM = '/storage/talg/src/MedSAM/work_dir/UltraSam.pth'
MMDET_CONFIG = '/storage/talg/src/UltraSam/configs/UltraSAM/UltraSAM_full/UltraSAM_box_no_refine.py'
ULTRASAM_ROOT = '/storage/talg/src/UltraSam'


def mask_to_rle(binary_mask):
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def run_ultrasam(images, masks):
    if len(images) != len(masks):
        raise ValueError(f"Image and mask count mismatch: {len(images)} vs {len(masks)}")

    parent_dir = os.path.join(ULTRASAM_ROOT, 'UltraSAM_DATA')
    os.makedirs(parent_dir, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=parent_dir) as temp_dir:
        temp_name = os.path.basename(temp_dir)
        img_dir = os.path.join(temp_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        coco_data = {
            "info": {},
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "object", "supercategory": "object"}]
        }

        ann_id = 1
        valid_indices = []

        for idx, (img, mask) in enumerate(zip(images, masks)):
            labeled_mask = measure.label(mask > 0)
            regions = [r for r in measure.regionprops(labeled_mask) if r.area >= 5]

            if not regions:
                continue

            valid_indices.append(idx)
            file_name = f"{idx:05d}.png"

            if img.ndim == 2:
                img_3c = np.stack([img]*3, axis=-1)
            elif img.ndim == 3 and img.shape[0] in [1, 3]:
                img_3c = img.transpose(1, 2, 0)
                if img_3c.shape[-1] == 1:
                    img_3c = np.concatenate([img_3c]*3, axis=-1)
            else:
                img_3c = img

            if img_3c.dtype != np.uint8:
                img_uint8 = (img_3c * 255).clip(0, 255).astype(np.uint8)
            else:
                img_uint8 = img_3c

            Image.fromarray(img_uint8).save(os.path.join(img_dir, file_name))

            coco_data["images"].append({
                "id": len(valid_indices),
                "file_name": file_name,
                "height": int(img_uint8.shape[0]),
                "width": int(img_uint8.shape[1])
            })

            for region in regions:
                region_mask = (labeled_mask == region.label)
                rle = mask_to_rle(region_mask)
                bbox = region.bbox
                coco_bbox = [bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0]]

                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": len(valid_indices),
                    "category_id": 1,
                    "bbox": coco_bbox,
                    "area": float(region.area),
                    "segmentation": rle,
                    "iscrowd": 0
                })
                ann_id += 1

        if not valid_indices:
            return np.zeros_like(masks, dtype=np.uint8)

        ann_file_rel = os.path.join(temp_name, 'annotations.json')
        with open(os.path.join(temp_dir, 'annotations.json'), 'w') as f:
            json.dump(coco_data, f)

        temp_config_path = os.path.join(temp_dir, 'config.py')
        config_content = f"""
_base_ = '{MMDET_CONFIG}'
data_root = 'UltraSAM_DATA'
test_ann_file = '{ann_file_rel}'
train_dataloader = None
train_cfg = None
optim_wrapper = None
param_scheduler = None
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo={{'classes': ('object',)}},
        data_prefix=dict(img='{temp_name}/images/'),
        ann_file=test_ann_file,
        filter_cfg=dict(filter_empty_gt=False),
        test_mode=True,
        backend_args=None,
    ),
)
test_dataloader = val_dataloader
val_evaluator = dict(
    type='CocoMetric',
    metric=['bbox', 'segm'],
    format_only=False,
    classwise=True,
    ann_file='UltraSAM_DATA/{ann_file_rel}',
    outfile_prefix='UltraSAM_DATA/{temp_name}/results',
)
test_evaluator = val_evaluator
"""
        with open(temp_config_path, 'w') as f:
            f.write(config_content)

        cmd = ["mim", "test", "mmdet", temp_config_path, "--checkpoint", ULTRASAM]
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{ULTRASAM_ROOT}:{env.get('PYTHONPATH', '')}"
            subprocess.run(cmd, check=True, env=env, cwd=ULTRASAM_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            if e.stderr:
                print(f"Error running mim test: {e.stderr.decode()}")
            raise e

        results_file = os.path.join(temp_dir, 'results.segm.json')
        if not os.path.exists(results_file):
            for f in os.listdir(temp_dir):
                if f.endswith('.segm.json'):
                    results_file = os.path.join(temp_dir, f)
                    break

        with open(results_file, 'rb') as f:
            results = json.load(f)

        pred_masks = np.zeros_like(masks, dtype=np.uint8)
        for res in results:
            try:
                image_id = res['image_id']
                vol_idx = valid_indices[image_id - 1]
                if 'segmentation' in res:
                    m = maskUtils.decode(res['segmentation'])
                    pred_masks[vol_idx] = np.maximum(pred_masks[vol_idx], m.astype(np.uint8))
            except Exception as e:
                print(f"Error parsing instance result: {e}")

    return pred_masks


def compute_dice(pred, gt):
    if pred.sum() == 0 and gt.sum() == 0:
        return 1.0
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum()
    return 2 * intersection / (union + 1e-6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, required=True)
    parser.add_argument('--aug-power', type=str, required=True)
    parser.add_argument('--tgt', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='CAMUS', choices=['CAMUS', 'EchoNet'])
    args = parser.parse_args()

    ds = CAMUSAugDataset(args.tgt, args.aug, args.aug_power, dataset=args.dataset)
    if len(ds) == 0:
        print("No data found!")
        sys.exit(0)

    images = []
    lv_masks = []
    la_masks = []
    frame_ids = []

    for i in tqdm(range(len(ds)), desc="Loading data"):
        aug_im, clean_im, lv_mask, la_mask, frame_id = ds[i]
        images.append(aug_im.cpu().numpy())
        lv_masks.append(lv_mask)
        la_masks.append(la_mask)
        frame_ids.append(frame_id)

    images = np.array(images)
    lv_masks = np.array(lv_masks)
    la_masks = np.array(la_masks)

    print("Running UltraSam for LV...")
    lv_preds = run_ultrasam(images, lv_masks)
    lv_dice = np.array([compute_dice(lv_preds[i], lv_masks[i]) for i in range(len(lv_preds))])
    print(f"Mean LV Dice: {lv_dice.mean():.4f}")

    if args.dataset == 'CAMUS':
        print("Running UltraSam for LA...")
        la_preds = run_ultrasam(images, la_masks)
        la_dice = np.array([compute_dice(la_preds[i], la_masks[i]) for i in range(len(la_preds))])
        print(f"Mean LA Dice: {la_dice.mean():.4f}")
    else:
        la_dice = np.zeros(len(images))

    save_dir = f"assets/results/{args.dataset}/dsc/{args.aug}/{args.aug_power}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.tgt}.npz")

    np.savez(
        save_path,
        frame_ids=np.array(frame_ids),
        lv_dice=lv_dice,
        la_dice=la_dice
    )
    print(f"Saved to {save_path}")
