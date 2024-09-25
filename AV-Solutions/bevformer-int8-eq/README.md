# About
This repository contains an end-to-end example of deploying BEVFormer with explicit quantization with [NVIDIA's ModelOpt Toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
 At the end, we show TensorRT deployment results in terms of runtime and accuracy.

# Requirements
- TensorRT 10.x
- ONNX-Runtime 1.18.x
- onnx-graphsurgeon
- onnsim
- [ModelOpt toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer) >= 0.15.0
- [BEVFormer_tensorrt](https://github.com/DerryHub/BEVFormer_tensorrt)

## Prepare dataset
Follow the [Data Preparation steps](https://github.com/DerryHub/BEVFormer_tensorrt#nuscenes-and-can-bus-for-bevformer) for NuScenes and CAN bus.
 This will prepare the full train / validation dataset.

## Prepare docker image
Build docker image:
```bash
$ export TAG=tensorrt_bevformer:24.08
$ docker build -f docker/tensorrt.Dockerfile --no-cache --tag=$TAG .
```

# How to Run

## 1. Export model to ONNX and compile plugins
A. Download model weights from [here](https://github.com/DerryHub/BEVFormer_tensorrt#bevformer-pytorch) 
 and save it in `./models`:
```sh
$ wget -P ./models https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
```

B. Run docker container:
```sh
$ docker run -it --rm --gpus device=0 --network=host --shm-size 20g -v $(pwd):/mnt -v <path to data>:/workspace/BEVFormer_tensorrt/data $TAG
```

C. In docker container, patch the `BEVFormer_tensorrt` folder and compile plugins:
```sh
# 1. Apply patch to BEVFormer_tensorrt with changes necessary for TensorRT 10 support
$ cd /workspace/BEVFormer_tensorrt
$ git apply /mnt/bevformer_trt10.patch

# 2. Compile plugins
$ cd TensorRT/build
$ cmake .. -DCMAKE_TENSORRT_PATH=/usr && make -j$(nproc) && make install
```
> The compiled plugin will be saved in `TensorRT/lib/libtensorrt_ops.so`, which will later be used by both ModelOpt and TensorRT.

D. Export simplified ONNX model from torch:
```sh
$ cd /workspace/BEVFormer_tensorrt
$ python tools/pth2onnx.py configs/bevformer/plugin/bevformer_tiny_trt_p2.py /mnt/models/bevformer_tiny_epoch_24.pth --opset=13 --cuda --flag=cp2_op13
$ cp checkpoints/onnx/bevformer_tiny_epoch_24_cp2_op13.onnx /mnt/models/
```

## 2. Post-process ONNX model
```sh
$ export PLUGIN_PATH=/workspace/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so
$ python /mnt/tools/onnx_postprocess.py --onnx=/mnt/models/bevformer_tiny_epoch_24_cp2_op13.onnx --trt_plugins=$PLUGIN_PATH
```
> This will generate an ONNX file of same name as the input ONNX file with the suffix `*_post_simp.onnx`.
>  May need to use `CUDA_MODULE_LOADING=LAZY` if using CUDA 12.x. No such variable is needed with CUDA 11.8.

This script does the following post-processing actions:
1. Automatically detect custom TRT ops in the ONNX model.
2. Ensure that the custom ops are supported as a TRT plugin in ONNX-Runtime (`trt.plugins` domain).
3. Update all tensor types and shapes in the ONNX graph with `onnx-graphsurgeon`.
4. Simplify model with `onnxsim`.

## 3. Quantize ONNX model
1. Prepare the calibration data:  
```sh
$ cd /workspace/BEVFormer_tensorrt
$ python /mnt/tools/calib_data_prep.py configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
    --onnx_path=/mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp.onnx \
    --trt_plugins=$PLUGIN_PATH
```
> The calibration data will be saved in `data/nuscenes/calib_data.npz`. The script uses 600 calibration samples by default.
>  See [instructions](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/onnx_ptq#quantize-an-onnx-model) in the ModelOpt toolkit for more info on generating the calibration data.

2. Quantize ONNX model with calibration data:  
```bash
$ python /mnt/tools/quantize_model.py --onnx_path=/mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp.onnx \
      --trt_plugins=$PLUGIN_PATH \
      --op_types_to_exclude MatMul \
      --calibration_data_path=/workspace/BEVFormer_tensorrt/data/nuscenes/calib_data.npz
```
> This generates an ONNX model with suffix `.quant.onnx` with Q/DQ nodes around relevant layers.

**Notes**:
- MatMul ops are not being quantized (`--op_types_to_exclude MatMul`). The reasoning for this is that MHA blocks, 
  present in Transformer-based models, are currently recommended to run in FP16. Keep in mind that optimal Q/DQ node
  placement can vary for different models, so there may be cases where quantizing MatMul ops may be more advantageous.
  This is up to the user to decide.
- If you're running out of memory, you may need to add `CUDA_MODULE_LOADING=LAZY` to the beginning of that
 quantization command. This is only valid for CUDA 12.x. No such variable is needed with CUDA 11.8.

## 4. Build TensorRT engine
```sh
$ trtexec --onnx=/mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp.quant.onnx \
	      --saveEngine=/mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp.quant.engine \
	      --staticPlugins=$PLUGIN_PATH \
	      --best
```

**Note**: In order to deploy the quantized ONNX model in another platform or with another TensorRT version, simply
 re-compile the plugin for the required settings and deploy the engine using the same explicitly-quantized ONNX model.

## 5. Evaluate accuracy of TensorRT engine
Run evaluation script:
```sh
$ cd /workspace/BEVFormer_tensorrt
$ python tools/bevformer/evaluate_trt.py \
         configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
         /mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp.quant.engine \
         --trt_plugins=$PLUGIN_PATH
```

# Results
**System**: NVIDIA A40 GPU, TensorRT 10.3.0.26.

BEVFormer tiny with FP16 plugins with `nv_half2` (`bevformer_tiny_epoch_24_cp2_op13_post_simp.onnx`):

| Precision                                       | GPU Compute Time (median, ms) | Accuracy (NDS / mAP)   |
|-------------------------------------------------|-------------------------------|------------------------|
| FP32                                            | 18.82                         | NDS: 0.354, mAP: 0.252 |
| FP16                                            | 9.36                          | NDS: 0.354, mAP: 0.251 |
| BEST (TensorRT PTQ - Implicit Quantization)     | 6.20                          | NDS: 0.353, mAP: 0.250 |
| QDQ_BEST (ModelOpt PTQ - Explicit Quantization) | 6.02                          | NDS: 0.352, mAP: 0.251 |


BEVFormer tiny with FP16 plugins with `nv_half` (`bevformer_tiny_epoch_24_cp_op13_post_simp.onnx`):

| Precision                                       | GPU Compute Time (median, ms) | Accuracy (NDS / mAP)   |
|-------------------------------------------------|-------------------------------|------------------------|
| FP32                                            | 18.80                         | NDS: 0.354, mAP: 0.252 |
| FP16                                            | 9.81                          | NDS: 0.354, mAP: 0.251 |
| BEST (TensorRT PTQ - Implicit Quantization)     | 6.73                          | NDS: 0.353, mAP: 0.250 |
| QDQ_BEST (ModelOpt PTQ - Explicit Quantization) | 6.54                          | NDS: 0.353, mAP: 0.251 |

## Steps to reproduce
To reproduce the results, run:
1. `./deploy_trt.sh` to build/save the TensorRT engine and obtain the runtime;
2. `./evaluate_trt.sh` to evaluate the TensorRT engine's accuracy.
