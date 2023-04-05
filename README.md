# Tracking Vision Transformer With Class and Regression Tokens

### [Paper](https://doi.org/10.1016/j.ins.2022.11.055)

## Table of Contents

1. [Projects path](#projects-path)
2. [Training](#training)
3. [Tracking](#tracking)
4. [Checkpoint](#checkpoint)
5. [Raw results](#raw-results)
6. [Acknowledgments](#acknowledgments)
7. [Citation](#citation)

## Projects path

Create the file for tracker configuration

```bash
python tracking/create_default_local_file.py --workspace_dir . --data_dir <data_dir> --save_dir .
```

The above script will create `local.py` files for tarining and testing in the following paths:

```bash
lib/train/admin/local.py  # Training
lib/pytracking/evaluation/local.py  # Testing
```

Please edit them with appropriate paths

## Training

For training:

```bash
# Multiple GPU
python tracking/train.py --script vit_crt --config baseline --save_dir . --mode multiple --nproc_per_node 4

# Single GPU
python tracking/train.py --script vit_crt --config baseline --save_dir . --mode single
```

## Tracking

- VOT2022-STB

Trax wrapper is in the following directory:

```bash
<your_path>/ViTCRTracking/pytracking/vot/vit_crt_exp_stb.py
```

- OTB/LaSOT/TrackingNet/GOT10K:

```bash
python tracking/test.py vit_crt baseline --dataset <dataset_name> --threads N --num_gpus N
```

Original PySOT and GOT10K toolkit are provided for comparison

## Checkpoint

To run the experiments it is possible to use the following checkpoint:

- [Google Drive](https://bit.ly/ViTCRT_ckp)
- [OneDrive](https://bit.ly/vitcrt_ckp_onedrive)

Put it in

```bash
<your_path>/ViTCRTracking/checkpoints/train/vit_crt/baseline
```

## Raw results

Raw results for each dataset reported in the article are provided to the following link: 

- [Google Drive](https://bit.ly/vitcrt_raw_results)
- [OneDrive](https://bit.ly/vitcrt_raw_results_onedrive)

Specifically they are:

- GOT10K
- TrackingNet
- LaSOT
- OTB100
- NfS
- TC128
- UAV123

## Acknowledgments

- Thanks to [PyTracking](https://github.com/visionml/pytracking) library that allows easily build a tracker.
- Thanks to [STARK](https://github.com/researchmm/Stark) as starting point for our code

## Citation

```text
@article{ViTCRT,
title = {Tracking Vision Transformer With Class and Regression Tokens},
author = {Emanuel Di Nardo and Angelo Ciaramella},
journal = {Information Sciences},
year = {2022},
doi = {https://doi.org/10.1016/j.ins.2022.11.055}
}
```
