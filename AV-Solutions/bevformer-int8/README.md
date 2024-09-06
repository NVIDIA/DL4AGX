# About
This repository contains an end-to-end example of how to quantize an ONNX model (BEVFormer) containing a custom TRT plugin with [NVIDIA's ModelOpt Toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
 At the end, we show TensorRT deployment results in terms of runtime and accuracy.

# Requirements
- TensorRT 10.x
- ONNX-Runtime 1.18.x
- onnx-graphsurgeon
- onnsim
- [ModelOpt toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer) >= 0.15.0
- [DerryHub's BEVFormer](https://github.com/DerryHub/BEVFormer_tensorrt)

## Prepare dataset
Follow the [Data Preparation steps](https://github.com/DerryHub/BEVFormer_tensorrt#nuscenes-and-can-bus-for-bevformer) for NuScenes and CAN bus in the DerryHub repo.
 This will prepare the full train / validation dataset.

## Docker
Build docker image:
```bash
$ export TAG=tensorrt_bevformer:24.08
$ docker build -f docker/tensorrt.Dockerfile --no-cache --tag=$TAG .
```

# How to Run

## 1. Export model to ONNX and compile plugins
A. Download model weights from [DerryHub's repository](https://github.com/DerryHub/BEVFormer_tensorrt#bevformer-pytorch), 
 and save it in `./models`:
```sh
$ wget -P ./models https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth
```

B. Run docker container:
```sh
$ docker run -it --rm --gpus device=0 --network=host --shm-size 20g -v $(pwd):/mnt -v <path to data>:/workspace/BEVFormer_tensorrt/data $TAG
```

C. In docker container, patch `BEVFormer_tensorrt` folder, compile plugins, and export ONNX model:
```sh
# 1. Apply patch to BEVFormer_tensorrt with changes necessary for TensorRT 10 support
$ cd /workspace/BEVFormer_tensorrt
$ git apply /mnt/bevformer_trt10.patch

# 2. Compile plugins
$ cd TensorRT/build
$ cmake .. -DCMAKE_TENSORRT_PATH=/usr/include/x86_64-linux-gnu
$ make -j$(nproc) && make install

# 3. Export simplified ONNX model from torch
$ cd /workspace/BEVFormer_tensorrt
$ python tools/pth2onnx.py configs/bevformer/plugin/bevformer_tiny_trt_p2.py /mnt/models/bevformer_tiny_epoch_24.pth --opset=13 --cuda --flag=cp2_op13
$ cp checkpoints/onnx/bevformer_tiny_epoch_24_cp2_op13.onnx /mnt/models/
```

## 2. Post-process ONNX model
```sh
$ export PLUGIN_PATH=/workspace/BEVFormer_tensorrt/TensorRT/lib/libtensorrt_ops.so
$ python /mnt/tools/onnx_postprocess.py --onnx=/mnt/models/bevformer_tiny_epoch_24_cp2_op13.onnx \
  --plugins $PLUGIN_PATH \
  --custom_ops RotateTRT2 MultiScaleDeformableAttnTRT2 \
  --plugins_precision MultiScaleDeformableAttnTRT2:[fp16,int32,fp16,fp16,fp16]:[fp16]
```
> This will generate an ONNX file of same name as the input ONNX file with the suffix `*_post_simp.onnx`.
>  May need to use `CUDA_MODULE_LOADING=LAZY` if using CUDA 12.x. No such variable is needed with CUDA 11.8.

This script does the following post-processing actions:
1. Add `trt.plugins` domain to the ONNX file to enable ORT to detect the custom ops as TRT custom ops. This step generates a new ONNX file with *_ort_support.onnx extension.
2. If the precisions of the plugin inputs are given, ensure them by adding Cast nodes.
3. Infer tensor shapes with `ORT` and modify the ONNX graph accordingly with `onnx-graphsurgeon`.
4. Simplify model with `onnxsim`.

## 3. Quantize ONNX model
1. Prepare the calibration data:  
```sh
$ cd /workspace/BEVFormer_tensorrt
$ python /mnt/tools/calib_data_prep.py configs/bevformer/plugin/bevformer_tiny_trt_p2.py \
    --onnx_path=/mnt/models/bevformer_tiny_epoch_24_cp2_op13_post_simp.onnx \
    --trt_plugins=$PLUGIN_PATH
```
> The calibration data will be saved in `data/nuscenes/calib_data.npz`.
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

## 4. Deploy TensorRT engine
```sh
$ trtexec --onnx=/mnt/models/bevformer_tiny_epoch_24_cp2_post_simp.quant.onnx \
	      --saveEngine=/mnt/models/bevformer_tiny_epoch_24_cp2_post_simp.quant.engine \
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
         /mnt/models/bevformer_tiny_epoch_24_cp2_post_simp.quant.engine \
         --trt_plugins=$PLUGIN_PATH
```

# Results
**Model**: BEVFormer tiny with FP16 plugins with nv_half2 (`bevformer_tiny_epoch_24_cp2_post_simp.onnx`)

**System**: A40, TRT 10.3.0.28 -> **ONGOING**: Requested PBR with TRT 10.3.0.26 to match TRT version on public container!

| Precision                   | GPU Compute Time (median, ms) | Accuracy (NDS / mAP)       |
|-----------------------------|-------------------------------|----------------------------|
| FP32                        | 19.37                         | NDS: 0.354, mAP: 0.252     |
| FP16                        | 9.98                          | NDS: 0.354, mAP: 0.252     |
| BEST (TensorRT PTQ)         | 7.39                          | NDS: 0.353, mAP: 0.250     |
| **QDQ_BEST** (ModelOpt PTQ) | **6.91**                      | **NDS: 0.352, mAP: 0.251** |

> See [results/README.md](results/README.md) to reproduce the results.
