# Tracking Vision Transformer With Class and Regression Tokens


## Projects path

Create the file for tracker configuration

```
python tracking/create_default_local_file.py --workspace_dir . --data_dir <data_dir> --save_dir .
```

The above script will create `local.py` files for tarining and testing in the following paths:

```
lib/train/admin/local.py  # Training
lib/pytracking/evaluation/local.py  # Testing
```

Please edit them with appropriate paths


## Training

For training:

```
# Multiple GPU
python tracking/train.py --script vit_crt --config baseline --save_dir . --mode multiple --nproc_per_node 4

# Single GPU
python tracking/train.py --script vit_crt --config baseline --save_dir . --mode single
```


## Tracking

- VOT2022-STB

Trax wrapper is in the following directory:
```
<your_path>/ViTCRTracking/pytracking/vot/vit_crt_exp_stb.py
```

- OTB/LaSOT/TrackingNet/GOT10K:

```
python tracking/test.py vit_crt baseline --dataset <dataset_name> --threads N --num_gpus N
```

Original PySOT and GOT10K toolkit are provided for comparison


## Checkpoint
To run the experiments it is possible to use the following checkpoint:

- [Google Drive](https://bit.ly/3MPyYaC) 

Put it in 

```
<your_path>/ViTCRTracking/checkpoints/train/vit_crt/baseline
```

## Acknowledgments
* Thanks to [PyTracking](https://github.com/visionml/pytracking) library that allows easily build a tracker.
* Thanks to [STARK](https://github.com/researchmm/Stark) as starting point for our code
