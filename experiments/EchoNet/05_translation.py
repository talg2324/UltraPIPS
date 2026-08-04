import os
import shutil
import re
from argparse import ArgumentParser

import echogains

src_dir = 'experiments/assets/data/EchoNet/raw'
dst_dir = 'experiments/assets/data/EchoNet/aug'

parser = ArgumentParser()
parser.add_argument("--aug-power", type=int, required=True)
parser.add_argument("--tgt", choices=['A4C'], required=True)
parser.add_argument("--batch-id", type=int, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    echogains.download_and_set_up_model('CAMUS_diffusion_model')

    # Point to the specific mini-batch
    batch_suffix = f'batch_{args.batch_id}'
    power_suffix = f'translation-{args.aug_power}'

    src_batch_path = os.path.abspath(os.path.join(src_dir, args.tgt, batch_suffix))
    dst_batch_path = os.path.abspath(os.path.join(dst_dir, args.tgt, batch_suffix, power_suffix))

    # Final paths for repaint
    repaint_keep = os.path.join(dst_batch_path, 'repaint', 'keep')
    repaint_masks = os.path.join(dst_batch_path, 'repaint', 'masks')
    buffer_path = os.path.join(dst_batch_path, "buffer")
    aug_output_path = os.path.join(dst_batch_path, "images")

    os.makedirs(dst_batch_path, exist_ok=True)

    # Temporary partitioning root
    temp_root = os.path.join(dst_batch_path, "temp_split")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root, exist_ok=True)

    subsets = ['pos', 'neg']
    configs = {
        'pos': {'val': args.aug_power, 'dir': os.path.join(temp_root, 'pos')},
        'neg': {'val': args.aug_power, 'dir': os.path.join(temp_root, 'neg')}
    }

    for s in subsets:
        d = configs[s]['dir']
        os.makedirs(os.path.join(d, 'src', 'images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'src', 'labels'), exist_ok=True)
        os.makedirs(os.path.join(d, 'dst'), exist_ok=True)

    # Split files by index parity
    src_images_dir = os.path.join(src_batch_path, 'images')
    if not os.path.exists(src_images_dir):
        raise FileNotFoundError(f"Source images directory not found: {src_images_dir}")

    for filename in os.listdir(src_images_dir):
        if not filename.endswith('.png'):
            continue

        # Extract frame index
        match = re.search(r'frame(\d+)', filename)
        if not match:
            continue
        idx = int(match.group(1))

        subset = 'pos' if idx % 2 == 0 else 'neg'
        subset_src = os.path.join(configs[subset]['dir'], 'src')

        # Copy image and label to subset src
        shutil.copy2(os.path.join(src_images_dir, filename), os.path.join(subset_src, 'images', filename))
        label_path = os.path.join(src_batch_path, 'labels', filename)
        if os.path.exists(label_path):
            shutil.copy2(label_path, os.path.join(subset_src, 'labels', filename))

    # Run preparation for each subset
    print(f"Preparing masks for {args.tgt} {batch_suffix} (translation_radius={args.aug_power})")
    for s in subsets:
        val = configs[s]['val']
        subset_dir = configs[s]['dir']
        subset_dst = os.path.join(subset_dir, 'dst')

        # Create repaint subfolders for this subset
        subset_keep = os.path.join(subset_dst, 'repaint', 'keep')
        subset_masks = os.path.join(subset_dst, 'repaint', 'masks')

        # We use displacement_radius_range as per tutorial.
        # Since the user requested special treatment like rotation (alternating parity),
        # we partition the frames. If displacement_radius_range picks a random direction,
        # it is already unbiased. We maintain the partitioning structure for consistency.
        params = [{'type': 'translation', 'prob': 1, 'displacement_radius_range': [val, val + 1]}]

        echogains.prepare_gen_aug_seg(
            os.path.join(subset_dir, 'src'),
            subset_dst,
            subset_keep,
            subset_masks,
            params,
            nb_augmentations=1,
            include_original=True
        )

        # Merge results to final destination
        for folder_rel in ['repaint/keep', 'repaint/masks', 'images', 'labels']:
            src_f = os.path.join(subset_dst, folder_rel)
            dst_f = os.path.join(dst_batch_path, folder_rel)
            if not os.path.exists(src_f):
                continue

            os.makedirs(dst_f, exist_ok=True)
            for f in os.listdir(src_f):
                shutil.move(os.path.join(src_f, f), os.path.join(dst_f, f))

    # Cleanup temp split directories
    shutil.rmtree(temp_root)

    print(f"Running diffusion for {args.tgt} {batch_suffix} with {power_suffix}")
    config = echogains.load_default_config(
        'CAMUS_diffusion_model',
        repaint_keep,
        repaint_masks,
        buffer_path,
        aug_output_path
    )
    config['data']['eval']['inference']['batch_size'] = 16
    config['schedule_jump_params']['jump_n_sample'] = 5
    echogains.run_repaint(config)
