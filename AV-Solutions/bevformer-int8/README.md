# About
This repository contains an end-to-end example of how to quantize an ONNX model (BEVFormer) containing a custom TRT plugin with [NVIDIA's ModelOpt Toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
 At the end, we show TensorRT deployment results in terms of runtime.

# Requirements
- TensorRT 10+
- ONNX-Runtime 1.18+
- onnx-graphsurgeon
- onnsim
- [ModelOpt toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [DerryHub's BEVFormer](https://github.com/DerryHub/BEVFormer_tensorrt)

## Docker
Build docker image:
```bash
$ export TAG=tensorrt_bevformer:24.05
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
$ docker run -it --rm --gpus device=0 --network=host -v $(pwd):/mnt $TAG
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
$ python /mnt/onnx_postprocess.py --onnx=/mnt/models/bevformer_tiny_epoch_24_cp2_op13.onnx \
  --plugins $PLUGIN_PATH \
  --custom_ops RotateTRT2 MultiScaleDeformableAttnTRT2 \
  --plugins_precision MultiScaleDeformableAttnTRT2:[fp16,int32,fp16,fp16,fp16]:[fp16]
```
> This will generate an ONNX file of same name as the input ONNX file with the suffix `*_post.onnx`.
>  May need to use `CUDA_MODULE_LOADING=LAZY` if using CUDA 12.x. No such variable is needed with CUDA 11.8.

This script does the following post-processing actions:
1. Add `trt.plugins` domain to the ONNX file to enable ORT to detect the custom ops as TRT custom ops. This step generates a new ONNX file with *_ort_support.onnx extension.
2. If the precisions of the plugin inputs are given, ensure them by adding Cast nodes.
3. Infer tensor shapes with `ORT` and modify the ONNX graph accordingly with `onnx-graphsurgeon`.
4. Simplify model with `onnxsim`.

## 3. Quantize ONNX model
```bash
$ python -m modelopt.onnx.quantization --onnx_path=/mnt/models/bevformer_tiny_epoch_24_cp2_op13_post.onnx \
      --trt_plugins=$PLUGIN_PATH --op_types_to_exclude MatMul
```

This generates an ONNX model with suffix `.quant.onnx` with Q/DQ nodes.

If you're running out of memory, you may need to add `CUDA_MODULE_LOADING=LAZY` to the beginning of that quantization command. This is only valid for CUDA 12.x. No such variable is needed with CUDA 11.8.

## 4. Deploy TensorRT engine
```sh
$ trtexec --onnx=/mnt/models/bevformer_tiny_epoch_24_cp2_post.quant.onnx \
	      --staticPlugins=$PLUGIN_PATH \
	      --best
```

If you wish to deploy the quantized ONNX model in another platform or with another TensorRT version, all you'd need to 
 do is re-compile the plugin for the required settings. You'd then use the same explicitly-quantized ONNX model
 with the new plugin for deployment.

# Results
To reproduce the results in this section, run `./deploy_trt.sh`.

**System**: RTX 3090, TRT 10.0.1.6 (tensorrt:24.05)

**INTERNAL NUMBERS** (not PBR):

| Precision                                       | GPU Compute Time (median, ms) |
|-------------------------------------------------|-------------------------------|
| FP16                                            | 9.77362                       | 
| BEST                                            | 7.52429                       |
| QDQ_BEST (default quantization)                 | 7.00415                       |
| **QDQ_BEST** (quantization excludes MatMul ops) | **6.23511**                   |
