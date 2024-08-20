
## UniAD_tiny ONNX and Engine Build
### Pytorch to ONNX
```
cd /workspace/UniAD
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py ./ckpts/tiny_imgx0.25_e2e_ep20.pth 1
```

We provide an [example ONNX](../onnx/uniad_tiny_dummy.onnx) with same structure but dummy weights for your reference.

### ONNX to TensorRT

#### TensorRT Plugin Compilation:

Change line 16&17 of `/workspace/UniAD/tools/tensorrt_plugin/CMakeList.txt`

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
cmake .. -DCMAKE_TENSORRT_PATH=<path_to_TensorRT>
make -j$(nproc) && make install
```


#### Engine Build
TensorRT FP32 engine build and latency measurement (modify TensorRT version (`TRT_VERSION`) and TensorRT path (`TRT_PATH`) inside `run_trtexec.sh` if needed)
```
cd /workspace/UniAD
./run_trtexec.sh
```

<- Last Page: [Tiny Traning](tiny_training.md)

-> Next Page: [C++ Inference and Visualization](../inference_app/README.md)
