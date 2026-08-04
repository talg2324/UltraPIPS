import os
import sys
import json
import argparse
import tempfile
import subprocess
import glob
import cv2

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage import measure
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt

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

        mim_path = os.path.join(os.path.dirname(sys.executable), 'mim')
        cmd = [mim_path, "test", "mmdet", temp_config_path, "--checkpoint", ULTRASAM]
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{ULTRASAM_ROOT}:{env.get('PYTHONPATH', '')}"
            res = subprocess.run(cmd, check=True, env=env, cwd=ULTRASAM_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            if e.stdout:
                print(f"Error running mim test. Output:\n{e.stdout.decode('utf-8', errors='ignore')}")
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


def get_power(d):
    try:
        return float(os.path.basename(d).split('-')[1])
    except:
        return 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run UltraSAM on all depth augmentations for a single image.")
    parser.add_argument('image_path', type=str, help="Path to a single image (e.g. assets/data/CAMUS/aug/A2C/batch_1/depth-15/images/frame053_aug0.png)")
    args = parser.parse_args()

    # Parse path to infer target, batch, and frame
    path_parts = args.image_path.split('/')
    try:
        aug_idx = path_parts.index('aug')
        dataset = path_parts[aug_idx - 1]
        tgt = path_parts[aug_idx + 1]
        batch_name = path_parts[aug_idx + 2]
    except ValueError:
        print("Error: Could not parse dataset, tgt, or batch_name from the path. Expected path like '.../aug/tgt/batch_X/...'")
        sys.exit(1)

    aug_power_dir = path_parts[aug_idx + 3]
    aug_type = aug_power_dir.split('-')[0]

    filename = os.path.basename(args.image_path)
    frame_name = filename.replace('_aug0.png', '').replace('.png', '')

    batch_dir = os.path.join(*path_parts[:aug_idx+3])
    if args.image_path.startswith('/'):
        batch_dir = '/' + batch_dir
        
    aug_dirs = glob.glob(os.path.join(batch_dir, f'{aug_type}-*'))
    aug_dirs.sort(key=get_power)

    if not aug_dirs:
        print(f"No {aug_type} augmentations found in {batch_dir}")
        sys.exit(1)

    images = []
    lv_masks = []
    la_masks = []
    powers = []
    valid_paths = []

    # First, load the base unaugmented image and label
    if aug_dirs:
        d = aug_dirs[0]
        base_img_path = os.path.join(d, 'images', f'{frame_name}.png')
        base_lbl_path = os.path.join(d, 'labels', f'{frame_name}.png')
        if os.path.exists(base_img_path) and os.path.exists(base_lbl_path):
            base_im = cv2.imread(base_img_path, cv2.IMREAD_COLOR)
            if base_im is not None:
                base_im = cv2.cvtColor(base_im, cv2.COLOR_BGR2RGB)
                base_im_tensor = torch.from_numpy(base_im).permute(2, 0, 1).float() / 255.0
                
                base_lbl = cv2.imread(base_lbl_path, cv2.IMREAD_GRAYSCALE)
                base_lv = (base_lbl == 1).astype(np.uint8)
                base_la = (base_lbl == 3).astype(np.uint8)
                
                images.append(base_im_tensor.cpu().numpy())
                lv_masks.append(base_lv)
                la_masks.append(base_la)
                powers.append("base")
                valid_paths.append(base_img_path)

    for d in aug_dirs:
        lbl_path = os.path.join(d, 'labels', f'{frame_name}_aug0.png')
        aug_path = os.path.join(d, 'images', f'{frame_name}_aug0.png')
        
        if not os.path.exists(lbl_path) or not os.path.exists(aug_path):
            continue
            
        aug_im = cv2.imread(aug_path, cv2.IMREAD_COLOR)
        if aug_im is None:
            continue
        # Convert to RGB, shape to (C, H, W) and scale to [0, 1] exactly as in 03_sam_camus.py
        aug_im = cv2.cvtColor(aug_im, cv2.COLOR_BGR2RGB)
        aug_im_tensor = torch.from_numpy(aug_im).permute(2, 0, 1).float() / 255.0
        
        lbl = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        lv_mask = (lbl == 1).astype(np.uint8)
        la_mask = (lbl == 3).astype(np.uint8)
        
        images.append(aug_im_tensor.cpu().numpy())
        lv_masks.append(lv_mask)
        la_masks.append(la_mask)
        powers.append(get_power(d))
        valid_paths.append(aug_path)

    if not images:
        print(f"No valid images/labels found for frame {frame_name} in {aug_type} augmentations.")
        sys.exit(1)

    images = np.array(images)
    lv_masks = np.array(lv_masks)
    la_masks = np.array(la_masks)

    print(f"Found {len(images)} {aug_type} augmentations for {frame_name}. Running UltraSam...")
    print("Running UltraSam for LV...")
    lv_preds = run_ultrasam(images, lv_masks)

    print("Running UltraSam for LA...")
    la_preds = run_ultrasam(images, la_masks)

    out_dir = os.path.join('assets', 'results', '03', 'sam_figure', f'{tgt}_{batch_name}_{frame_name}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Saving images and segmentations to {out_dir} ...")

    for i, p in enumerate(tqdm(powers, desc="Saving results")):
        img_np = (images[i].transpose(1, 2, 0) * 255).astype(np.uint8)
        
        # 1. Save raw image
        Image.fromarray(img_np).save(os.path.join(out_dir, f'{aug_type}_{p}_image.png'))
        
        # 2. Save ground truth masks
        if lv_masks[i].sum() > 0:
            Image.fromarray(lv_masks[i] * 255).save(os.path.join(out_dir, f'{aug_type}_{p}_gt_lv.png'))
        if la_masks[i].sum() > 0:
            Image.fromarray(la_masks[i] * 255).save(os.path.join(out_dir, f'{aug_type}_{p}_gt_la.png'))
            
        # 3. Save predicted masks
        if lv_preds[i].sum() > 0:
            Image.fromarray(lv_preds[i] * 255).save(os.path.join(out_dir, f'{aug_type}_{p}_pred_lv.png'))
        if la_preds[i].sum() > 0:
            Image.fromarray(la_preds[i] * 255).save(os.path.join(out_dir, f'{aug_type}_{p}_pred_la.png'))

    print("Done!")
