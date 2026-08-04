import os
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

    augmentation_params = [
        {'type': 'depth', 'prob': 1, 'depth_increase_range': [args.aug_power, args.aug_power+1]},  # deterministic control
    ]

    # Point to the specific mini-batch
    batch_suffix = f'batch_{args.batch_id}'
    power_suffix = f'depth-{args.aug_power}'

    src_batch_path = os.path.join(src_dir, args.tgt, batch_suffix)
    dst_batch_path = os.path.join(dst_dir, args.tgt, batch_suffix, power_suffix)

    repaint_keep = os.path.join(dst_batch_path, 'repaint', 'keep')
    repaint_masks = os.path.join(dst_batch_path, 'repaint', 'masks')

    print(f"Processing {args.tgt} {batch_suffix} with {power_suffix}")

    echogains.prepare_gen_aug_seg(
        src_batch_path,
        dst_batch_path,
        repaint_keep,
        repaint_masks,
        augmentation_params,
        nb_augmentations=1,
        include_original=True
    )

    buffer_path = os.path.join(dst_batch_path, "buffer")
    aug_output_path = os.path.join(dst_batch_path, "images")

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
