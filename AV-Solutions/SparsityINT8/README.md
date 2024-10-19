# About
This repository contains the PyTorch codebase for Sparsity INT8 training and TensorRT inference.  
Please refer to the blogpost titled ["Sparsity in INT8: Training Workflow and Best Practices for NVIDIA TensorRT Acceleration"](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/).

# Requirements
## Hardware
For TensorRT inference: Ampere GPU due to Sparse Tensor Core support.

## Dataset
Download the [ImageNet 2012 dataset](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) and 
 format it according to the instructions in [data/README.md](data/README.md).

## Packages
- PyTorch 1.11.0 (tested, may work with other versions)
- PyTorch Quantization toolkit: [pytorch-quantization](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization)
- PyTorch Sparsity toolkit: [APEX](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity)
- (Manual installation) TensorRT engine deployment: [TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download)

### Docker
See [docker](docker).

### Local
1. Create a Python virtual environment and install dependencies:
```shell
virtualenv -p /usr/bin/python3.8 venv38
source venv38/bin/activate
chmod +x install.sh && ./install.sh
```

2. [Download TensorRT](https://developer.nvidia.com/nvidia-tensorrt-download) and install `tensorrt` python wheel:
```shell
pip install $TRT_PATH/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl 
```

# How to Run
> See each python script for all supported flags. 

## 1. Sparsity fine-tuning
Loads the pre-trained dense weights, sparsifies the model, and fine-tunes it. 
```sh
python step1_sparse_training.py --model_name=resnet34 --data_dir=$DATA_DIR --batch_size=128 --eval_baseline --eval_sparse
```

This saves the sparse checkpoints (best and final), and their respective ONNX files. 
 The best checkpoint will be used for the QAT workflow, and the best ONNX file will be used for the PTQ workflow.  

## 2. Quantization
There are two ways of quantizing the network: Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). 

### 2.1. PTQ calibration
Calibrates an ONNX model via the `entropy` or `minmax` approach: 
```sh
python step2_1_ptq_calibration.py --onnx_path=model_sparse.onnx --onnx_input_name=input.1 --data_dir=$DATA_DIR \
                                  --calibrator_type=entropy --calib_data_size=512
```

To generate the dense-PTQ version, for comparison, use the flag `--is_dense_calibration`.
 This will disable sparse weights when calibrating the dense model.

### 2.2. QAT fine-tuning
Loads the fine-tuned sparsified weights, adds QDQ nodes to relevant layers, calibrates it, and fine-tunes it.
```sh
python step2_2_qat_training.py --model_name=resnet34 --data_dir=$DATA_DIR --batch_size=128 --eval_qat
```

To generate the dense-QAT version, for comparison, use the flag `--is_dense_training`.

# Results for ResNet34
For results, please see refer to our [blogpost](https://developer.nvidia.com/blog/sparsity-in-int8-training-workflow-and-best-practices-for-tensorrt-acceleration/).
