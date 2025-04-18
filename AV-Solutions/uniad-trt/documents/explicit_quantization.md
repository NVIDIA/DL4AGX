## Extension: ONNX Explicit Quantization with modelopt
### About
This is an extended example of deploying [UniAD-tiny](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt) with explicit quantization via [NVIDIA's ModelOpt Toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
### How to Run
#### Prerequisites
- Finish all steps until ONNX exportation from [uniad-trt](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt) repository

#### Steps
The following steps are performed inside the deployment docker container on x86_64 platform.

##### 1. Install Required Dependencies
Install TensorRT-Model-Optimizer(Modelopt) v0.29.0.
```
cd /workspace/
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
cd TensorRT-Model-Optimizer
pip install -e ".[all]" --extra-index-url https://pypi.nvidia.com
```
Install TensorRT-10.7
```
cd /workspace/
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.7.0/tars/TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-11.8.tar.gz
tar -xzvf TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-11.8.tar.gz
mv TensorRT-10.7.0.23 TensorRT-10.7_x86_cu118
rm TensorRT-10.7.0.23.Linux.x86_64-gnu.cuda-11.8.tar.gz
pip install /workspace/TensorRT/TensorRT-10.7_x86_cu118/python/tensorrt-10.7.0.23-cp38-none-linux_x86_64.whl
```
Install ONNX Runtime v1.21.0
```
pip install --upgrade onnxruntime-gpu==1.21.0
```

##### 2. Compile TensorRT Plugins
Compiled plugins will be saved as `/workspace/UniAD/plugins/lib/lib_uniad_plugins_trt10.7_x86_cu118.so`.
```
cd /workspace/UniAD/plugins
TRT_VERSION=10.7 ./compile_plugins_x86.sh
```

##### 3. Prepare Calibration Data
Re-run the UniAD-tiny pytorch evaluation script and cherry-pick the minShape 901 samples based on `prev_track_intances0` input shapes for calibration. Calibration data will be saved as `/workspace/UniAD/calib_data_shape0_901.npz`.
```
cd /workspace/UniAD
PYTHONPATH=$(pwd) python3 ./tools/prepare_calib_data.py
```

##### 4. Quantize ONNX Model
```
cd /workspace/UniAD
MIN=901
SHAPES=prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}

LD_LIBRARY_PATH=<path_to_cudnn>/lib:/workspace/TensorRT-10.7_x86_cu118/lib:$LD_LIBRARY_PATH python -m modelopt.onnx.quantization \
    --onnx_path=<path_to_onnx_model> \
    --trt_plugins=/workspace/UniAD/plugins/lib/lib_uniad_plugins_trt10.7_x86_cu118.so \
    --calibration_eps trt cuda:0 cpu \
    --calibration_shapes=${SHAPES} \
    --simplify \
    --op_types_to_exclude MatMul \
    --calibration_data_path=/workspace/UniAD/calib_data_shape0_901.npz \
    --output_path=<path_to_quantized_onnx_model> \
    --dq_only
```

##### 5. Build TensorRT Engine with Explicit Quantization on Orin
```
MIN=901
OPT=901
MAX=1150
SHAPES=prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}

LD_LIBRARY_PATH=${TRT_PATH}/lib:$LD_LIBRARY_PATH \
${TRT_PATH}/bin/trtexec \
  --onnx=<path_to_quantized_onnx_model> \
  --saveEngine=<path_to_engine> \
  --staticPlugins=/workspace/UniAD/plugins/lib/lib_uniad_plugins_trt10.7_x86_cu118.so \
  --verbose \
  --separateProfileRun \
  --profilingVerbosity=detailed \
  --useCudaGraph \
  --tacticSources=+CUBLAS \
  --minShapes=${SHAPES//${MIN}/${MIN}} \
  --optShapes=${SHAPES//${MIN}/${OPT}} \
  --maxShapes=${SHAPES//${MIN}/${MAX}} \
  --iterations=100 \
  --best
```


### Results
We show the TensorRT-10.7 deployment results on Orin-X in terms of runtime and accuracy. `planning MSE` is the average L2 distance between the TensorRT engine output trajectory and the Pytorch-1.12 model output trajectory. Pytorch DL model latency is measured by Python `time.time()` function. TensorRT engine latency is from `mean` of `GPU Compute Time` measured by `trtexec` with `--iterations=100`.
#### Metrics
| Model | Framework | Precision | DL model latency↓ | avg. L2↓ | avg. Col↓ | planning MSE↓ |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD-tiny | Pytorch-1.12 | FP32 | 843.5172 ms | 0.9986  | 0.27 | 0 |
| UniAD-tiny | TensorRT-10.7 | FP32 | 64.0726 ms | 0.9986 | 0.27 | 9.2417e-07 |
| UniAD-tiny | TensorRT-10.7 | FP16 |  50.1560 ms | 1.0021 | 0.26 | 0.0458 |
| UniAD-tiny | TensorRT-10.7 | INT8(EQ)+FP16 | 39.6104 ms | 1.0029 | 0.27 | 0.0502 |

#### Videos

<table>
  <tr>
    <td align="center">
      <strong>TensorRT-10.7 FP32</strong><br>
      <img src="../assets/uniad_fp32_video.gif" style="max-width:100%; width:700px">
    </td>
    <td align="center">
      <strong>TensorRT-10.7 BEST(EQ)</strong><br>
      <img src="../assets/uniad_best_eq_video.gif" style="max-width:100%; width:700px">
    </td>
  </tr>
</table>