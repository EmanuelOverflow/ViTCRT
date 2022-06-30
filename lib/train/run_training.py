import os
import sys
import argparse
import importlib
import cv2 as cv
import torch.backends.cudnn
import torch.distributed as dist

import random
import numpy as np
torch.backends.cudnn.benchmark = False

import _init_paths
import lib.train.admin.settings as ws_settings


def init_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name, config_name, cudnn_benchmark=True, local_rank=-1, save_dir=None,
                 base_seed=None, save_every_epoch=True):
    """Run the train script.
    args:
        script_name: Name of emperiment in the "experiments/" folder.
        config_name: Name of the yaml file in the "experiments/<script_name>".
        cudnn_benchmark: Use cudnn benchmark or not (default is True).
    """
    if save_dir is None:
        print("save_dir dir is not given. Use the default dir instead.")
    # This is needed to avoid strange crashes related to opencv
    cv.setNumThreads(0)

    torch.backends.cudnn.benchmark = cudnn_benchmark

    print(f'script_name: {script_name}.py  config_name: {config_name}.yaml')

    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    settings = ws_settings.Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    settings.save_every_epoch = save_every_epoch
    settings.project_path = f'ltr/{script_name}/{config_name}'
    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    prj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    settings.cfg_file = os.path.join(prj_dir, f'experiments/{script_name}/{config_name}.yaml')
    expr_module = importlib.import_module(f'lib.train.train_settings.{script_name}.{script_name}')
    expr_func = getattr(expr_module, 'run')

    expr_func(settings)


def main():
    parser = argparse.ArgumentParser(description='Run a train scripts in train_settings.')
    parser.add_argument('--script', type=str, required=True, help='Name of the train script.')
    parser.add_argument('--config', type=str, required=True, help="Name of the config file.")
    parser.add_argument('--cudnn_benchmark', type=bool, default=True, help='Set cudnn benchmark on (1) or off (0) (default is on).')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--save_dir', type=str, help='the directory to save checkpoints and logs')
    parser.add_argument('--seed', type=int, default=42, help='seed for random numbers')
    parser.add_argument('--save_every_epoch', type=bool, default=True, help='Save checkpoints on every epoch')

    args = parser.parse_args()
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl')  # , world_size=2, init_method='env://')
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(0)
    run_training(args.script, args.config, cudnn_benchmark=args.cudnn_benchmark, save_every_epoch=args.save_every_epoch,
                 local_rank=args.local_rank, save_dir=args.save_dir, base_seed=args.seed)


if __name__ == '__main__':
    main()
