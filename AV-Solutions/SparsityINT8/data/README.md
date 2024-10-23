# Dataset
## A. Raw data download
We are using the ImageNet 2012 dataset (task 1 - image classification), which requires manual downloads due to terms of access agreements.
Please login/sign-up on [the ImageNet website](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and download the "train/validation data".

## B. Conversion to specific format
Uncompress the downloaded `.tar` ImageNet files and move the validation images to labeled subfolders (as instructed [here](https://github.com/NVIDIA/apex/tree/master/examples/imagenet#requirements)):
```
Both train and val datasets should be in the format:

val/
  n01440764/
    ILSVRC2012_val_00000293.JPEG
    ...
  ...
```

Steps:
1. Set `IMAGENET_HOME=/path/to/imagenet/tar/files` in [`imagenet_data_setup.sh`](imagenet_data_setup.sh) (modification of [this](https://github.com/NVIDIA/TensorRT/blob/release/8.6/tools/tensorflow-quantization/examples/data/imagenet_data_setup.sh)).
2. Run `./imagenet_data_setup.sh`.

## Note
- [`valprep.sh`](valprep.sh) was downloaded from [this link](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh) in case the link becomes unreachable in the future.
