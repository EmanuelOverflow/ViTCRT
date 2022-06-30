import os
import argparse


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, help='training script name')
    parser.add_argument('--config', type=str, default='baseline', help='yaml configure file name')
    parser.add_argument('--save_dir', type=str, help='root directory to save checkpoints, logs, and tensorboard')
    parser.add_argument('--mode', type=str, choices=["single", "multiple", "multinode"], default="multiple",
                        help="train on single gpu or multiple gpus")
    parser.add_argument('--nproc_per_node', type=int, help="number of GPUs per node")  # specify when mode is multiple
    parser.add_argument('--nnodes', type=int)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    if args.mode == "single":
        train_cmd = f"python lib/ltr/run_training.py --script {args.script} " \
                    f"--config {args.config} --save_dir {args.save_dir}"

    elif args.mode == "multiple":
        train_cmd = f"python -m torch.distributed.launch --nproc_per_node {args.nproc_per_node} " \
                    f"lib/ltr/run_training.py --script {args.script} --config {args.config} --save_dir {args.save_dir}"
    elif args.mode == "multinode":
        train_cmd = f"python -m torch.distributed.launch --nproc_per_node {args.nproc_per_node} " \
                    f"--nnodes {args.nnodes} lib/ltr/run_training.py --script {args.script} " \
                    f"--config {args.config} --save_dir {args.save_dir}"
    else:
        raise ValueError("mode should be 'single' or 'multiple'.")
    os.system(train_cmd)


if __name__ == "__main__":
    main()
