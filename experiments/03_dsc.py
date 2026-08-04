import os
import glob
import argparse

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import CAMUSAugDataset
from metrics import metrics_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aug', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--tgt', type=str, default='both')
    parser.add_argument('--dataset', type=str, default='CAMUS', choices=['CAMUS', 'EchoNet'])
    args = parser.parse_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    metrics = metrics_list(args.model)
    tgts = ['A2C', 'A4C'] if args.tgt == 'both' else [args.tgt]
    batch_size = 16 if 'medsam' not in args.model else 2

    for tgt in tgts:
        print(f"Processing {tgt}...")
        base_dir = f"assets/results/{args.dataset}/dsc/{args.aug}"
        power_dirs = sorted(glob.glob(os.path.join(base_dir, '*')))
        powers = []
        for d in power_dirs:
            p = os.path.basename(d)
            if p.isdigit():
                powers.append(int(p))
            elif p.replace('.', '', 1).isdigit():  # For float powers
                powers.append(float(p))

        powers = sorted(powers)
        K = len(powers)
        if K == 0:
            print(f"No power directories found in {base_dir}")
            continue

        first_npz = os.path.join(base_dir, str(powers[0]), f"{tgt}.npz")
        if not os.path.exists(first_npz):
            print(f"Missing {first_npz}")
            continue

        first_data = np.load(first_npz)
        frame_ids = first_data['frame_ids']
        N = len(frame_ids)

        lv_dice = np.zeros((N, K))
        la_dice = np.zeros((N, K))
        metric_acc = {m.name: np.zeros((N, K)) for m in metrics}

        for k, power in enumerate(powers):
            npz_path = os.path.join(base_dir, str(power), f"{tgt}.npz")
            if not os.path.exists(npz_path):
                print(f"Warning: {npz_path} missing!")
                continue

            data = np.load(npz_path)
            # Ensure frame_ids match? The CAMUSAugDataset is deterministic, so order should match.
            lv_dice[:, k] = data['lv_dice']
            la_dice[:, k] = data['la_dice']

            ds = CAMUSAugDataset(tgt, args.aug, power, dataset=args.dataset)
            dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)

            i = 0
            for aug_im, clean_im, _, _, _ in tqdm(dl, desc=f"Power {power}"):
                aug_im, clean_im = aug_im.to(device), clean_im.to(device)
                b = aug_im.shape[0]

                with torch.no_grad():
                    for m in metrics:
                        m.to(device)
                        vals = torch.stack([m(aug_im[j:j+1], clean_im[j:j+1]) for j in range(b)])
                        metric_acc[m.name][i:i+b, k] = vals.cpu().numpy()
                        m.cpu()

                i += b

        save_dir = f"assets/results/{args.dataset}/dsc/{args.aug}/{args.model}"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{tgt}.npz")

        np.savez(
            save_path,
            aug_powers=np.array(powers),
            frame_ids=frame_ids,
            lv_dice=lv_dice,
            la_dice=la_dice,
            **metric_acc
        )
        print(f"Saved to {save_path}")
