import os
import shutil
import re
from argparse import ArgumentParser

import echogains

src_dir = 'experiments/assets/data/EchoNet/raw'
dst_dir = 'experiments/assets/data/EchoNet/aug'

parser = ArgumentParser()
parser.add_argument("--aug-power", type=float, required=True)
parser.add_argument("--tgt", choices=['A4C'], required=True)
parser.add_argument("--batch-id", type=int, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    echogains.download_and_set_up_model('CAMUS_diffusion_model')

    # Point to the specific mini-batch
    batch_suffix = f'batch_{args.batch_id}'
    # For sector width, we use the float power in the name
    power_suffix = f'sector_width-{args.aug_power}'

    src_batch_path = os.path.abspath(os.path.join(src_dir, args.tgt, batch_suffix))
    dst_batch_path = os.path.abspath(os.path.join(dst_dir, args.tgt, batch_suffix, power_suffix))

    # Final paths for repaint
    repaint_keep = os.path.join(dst_batch_path, 'repaint', 'keep')
    repaint_masks = os.path.join(dst_batch_path, 'repaint', 'masks')
    buffer_path = os.path.join(dst_batch_path, "buffer")
    aug_output_path = os.path.join(dst_batch_path, "images")

    os.makedirs(dst_batch_path, exist_ok=True)

    # We don't need parity for sector width, but we need to follow the same prep/repaint workflow
    # For simplicity and consistency with the other scripts, we'll just run one pass

    # Create repaint subfolders
    os.makedirs(repaint_keep, exist_ok=True)
    os.makedirs(repaint_masks, exist_ok=True)

    # Sector width doesn't need parity partitioning, so we just run prepare_gen_aug_seg once
    # params = [{'type': 'sector_width', 'prob': 1, 'width_factor': [args.aug_power, args.aug_power + 0.01]}]
    # Using +0.01 to ensure np.random.uniform/randint works if it requires low < high
    params = [{'type': 'sector_width', 'prob': 1, 'width_factor': [args.aug_power, args.aug_power + 0.01]}]

    print(f"Preparing masks for {args.tgt} {batch_suffix} (width_factor={args.aug_power})")
    echogains.prepare_gen_aug_seg(
        src_batch_path,
        dst_batch_path,
        repaint_keep,
        repaint_masks,
        params,
        nb_augmentations=1,
        include_original=True
    )

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
