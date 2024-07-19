# About
This repository contains an end-to-end example of how to quantize an ONNX model containing a custom TRT plugin
 (BEVFormer) with [NVIDIA's ModelOpt Toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
 At the end, we show TensorRT deployment results in terms of runtime and accuracy.

# Requirements
- TensorRT 10+
- ONNX-Runtime 1.18+
- [ModelOpt toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer)
- [DerryHub's BEVFormer](https://github.com/DerryHub/BEVFormer_tensorrt)

## Docker
Build docker image:
```bash
$ export TAG=tensorrt_bevformer:24.05
$ docker build -f docker/tensorrt.Dockerfile --no-cache --tag=$TAG .
```

# How to Run

## 1. Export model to ONNX
A. Download model weights from [DerryHub's repository](https://github.com/DerryHub/BEVFormer_tensorrt#bevformer-pytorch), 
 say `bevformer_tiny_epoch_24.pth`, and save it in `models/`.

B. Run docker container:
```sh
$ docker run -it --rm --gpus device=0 --network=host -v $(pwd):/mnt $TAG
```

C. In docker container, run:
```sh
# 1. Compile plugins
$ cd /workspace/BEVFormer_tensorrt/TensorRT/build
$ perl -pi -e 's/&\(/(int*)&\(/g' /workspace/BEVFormer_tensorrt/TensorRT/plugin/rotate/rotatePlugin.cpp
$ perl -pi -e 's/&\(/(int*)&\(/g' /workspace/BEVFormer_tensorrt/TensorRT/plugin/grid_sampler/gridSamplerPlugin.cpp
$ cmake .. -DCMAKE_TENSORRT_PATH=/usr/include/x86_64-linux-gnu
$ make -j$(nproc) && make install

# 2. Export torch to ONNX file
$ python tools/pth2onnx.py configs/bevformer/plugin/bevformer_tiny_trt_p2.py /mnt/models/bevformer_tiny_epoch_24.pth --opset=13 --cuda --flag=cp2_op13
$ cp checkpoints/onnx/bevformer_tiny_epoch_24_cp2_op13.onnx /mnt/models/
```

D. Post-process ONNX model:
```sh
$ python onnx_postprocess.py --onnx=/mnt/models/bevformer_tiny_epoch_24_cp2_op13.onnx \
  --plugins /workspace/BEVFormer/TensorRT/lib/libtensorrt_ops.so \
  --custom_ops RotateTRT2 MultiScaleDeformableAttn_TRT2
```
> This will generate an ONNX file of same name as the input ONNX file with the suffix `*_post.onnx`.

This script does the follow post-processing actions:
1. Add `trt.plugins` domain to the ONNX file to enable ORT to detect the custom ops as TRT custom ops. This step generates a new ONNX file with *_ort_support.onnx extension.
2. Manually generate tensor shapes with ORT.
3. Manually modify the tensor shapes in the ONNX file with onnx-graphsurgeon.
4. Simplify model with `onnxsim`.

## 2. Quantize ONNX model
```bash
$ python -m modelopt.onnx.quantization --onnx_path=bevformer_tiny_epoch_24_cp2_post.onnx \
  --trt_plugins="${MSDA_PLUGIN};${ROTATE_PLUGIN}" --trt_plugins_precision MultiScaleDeformableAttn_TRT2:fp16
```

This generates an ONNX model with suffix `.quant.onnx` with Q/DQ nodes.

If you're running out of memory, you may need to add `CUDA_MODULE_LOADING=LAZY` to the beginning of that quantization command.
Quantizing the whole BEV model requires CUDA_MODULE_LOADING=LAZY variable with CUDA 12. No such variable is needed with CUDA 11.8.

## 3. Deploy TensorRT engine
```sh
$ trtexec --onnx=model.quant.onnx --best
```

# Results
## Runtime
WIP

## Accuracy
WIP
