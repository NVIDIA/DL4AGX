
## UniAD_tiny ONNX and Engine Build
### Pytorch to ONNX
```
cd /workspace/UniAD
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py ./ckpts/tiny_imgx0.25_e2e_ep20.pth 1
```

Due to legal reasons, we provide an [ONNX](../onnx/uniad_tiny_dummy.onnx) model of UniAD-tiny with random weights. Please follow instructions on training to obtain model with trained weights.

### ONNX to TensorRT

#### TensorRT Plugin Compilation:

To deploy UniAD-tiny with TensorRT, we first need to compile TensorRT plugins for `MultiScaleDeformableAttnTRT`, `InverseTRT` and `RotateTRT` operators that are not supported by Native TensorRT. To do this, change line 31&32 of `/workspace/UniAD/tools/tensorrt_plugin/CMakeList.txt`


from
```
set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu/)
```
to
```
set(TENSORRT_INCLUDE_DIRS <path_to_TensorRT>/include/)
set(TENSORRT_LIBRARY_DIRS <path_to_TensorRT>/lib/)
```


Then complie by

```
cd /workspace/UniAD/tools/tensorrt_plugin
mkdir build
cd build
cmake .. 
make -j$(nproc) && make install
```


#### Engine Build
TensorRT engine build and latency measurement (modify TensorRT version (`TRT_VERSION`) and TensorRT path (`TRT_PATH`) inside `run_trtexec.sh` if needed)
```
cd /workspace/UniAD
./run_trtexec.sh
```

<- Last Page: [Tiny Traning](tiny_training.md)

-> Next Page: [C++ Inference and Visualization](../inference_app/README.md)
