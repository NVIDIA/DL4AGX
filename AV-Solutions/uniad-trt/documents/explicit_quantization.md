## Extension: ONNX Explicit Quantization with modelopt
### About
This is an extended example of deploying [UniAD-tiny](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt) with explicit quantization via [NVIDIA's ModelOpt Toolkit](https://github.com/NVIDIA/TensorRT-Model-Optimizer).
### How to Run
#### Prerequisites
- Finish all steps until ONNX exportation from [uniad-trt](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/uniad-trt) repository

#### Steps
The following steps are performed inside the deployment docker container on x86_64 platform.

##### 1. Install Required Dependencies
```
cd /workspace/
git clone https://github.com/NVIDIA/TensorRT-Model-Optimizer.git
cd TensorRT-Model-Optimizer
pip install -e ".[all]" --extra-index-url https://pypi.nvidia.com
cd /workspace/
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/tars/TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-11.8.tar.gz
tar -xzvf TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-11.8.tar.gz
mv TensorRT-10.9.0.34 TensorRT-10.9.0.34_x86_cu118
rm TensorRT-10.9.0.34.Linux.x86_64-gnu.cuda-11.8.tar.gz
pip install --upgrade onnxruntime-gpu==1.21.0
pip install /workspace/TensorRT/TensorRT-10.9.0.34_x86_cu118/python/tensorrt-10.9.0.34-cp38-none-linux_x86_64.whl
```

##### 2. Compile TensorRT Plugins
```
cd /workspace/UniAD/plugins
TRT_VERSION=10.9.0.34 ./compile_plugins_x86.sh
```

##### 3. Prepare Calibration Data
Re-run the UniAD-tiny pytorch evaluation script and cherry-pick the minShape 901 samples based on `prev_track_intances0` input shapes for calibration. 
```
cd /workspace/UniAD
PYTHONPATH=$(pwd) python3 ./tools/prepare_calib_data.py
```

##### 4. Quantize ONNX Model
```
cd /workspace/UniAD
LD_LIBRARY_PATH=<path_to_cudnn>/lib:/workspace/TensorRT-10.9.0.34_x86_cu118/lib:$LD_LIBRARY_PATH python tools/quantize_uniad.py --onnx_path <path_to_onnx_model> --cali_data_path <path_to_calibration_npz_data> --trt_plugins <path_to_compiled_tensorrt_plugins.so>
```

##### 5. Postprocess Quantized ONNX Model
Note that currently the modelopt toolkit does not support postprocessing quantized onnx model back to dynamic input shapes, so we need to manually apply fixes to the quantized onnx model. In the future, this step will be automatically done by the modelopt toolkit.
```
cd /workspace/UniAD
python3 tools/postprocess_quantized_uniad.py --onnx_path <path_to_quantized_onnx_model>
```

##### 6. Build TensorRT Engine with Explicit Quantization on Orin
```
MIN=901
OPT=901
MAX=1150
SHAPES=prev_track_intances0:${MIN}x512,prev_track_intances1:${MIN}x3,prev_track_intances3:${MIN},prev_track_intances4:${MIN},prev_track_intances5:${MIN},prev_track_intances6:${MIN},prev_track_intances8:${MIN},prev_track_intances9:${MIN}x10,prev_track_intances11:${MIN}x4x256,prev_track_intances12:${MIN}x4,prev_track_intances13:${MIN}

LD_LIBRARY_PATH=${TRT_PATH}/lib:$LD_LIBRARY_PATH \
${TRT_PATH}/bin/trtexec \
  --onnx=<path_to_postprocessed_quantized_onnx_model> \
  --saveEngine=<path_to_engine> \
  --staticPlugins=<path_to_compiled_tensorrt_plugins.so> \
  --verbose \
  --profilingVerbosity=detailed \
  --useCudaGraph \
  --tacticSources=+CUBLAS \
  --minShapes=${SHAPES//${MIN}/${MIN}} \
  --optShapes=${SHAPES//${MIN}/${OPT}} \
  --maxShapes=${SHAPES//${MIN}/${MAX}} \
  --best
```


### Results
We show the TensorRT-10.7.0.23 deployment results on Orin-X in terms of runtime and accuracy. `planning MSE` is the average L2 distance between the TensorRT engine output trajectory and the Pytorch-1.12 model output trajectory. 
#### Metrics
| Model | Framework | Precision | DL model latency↓ | avg. L2↓ | avg. Col↓ | planning MSE↓ |
| :---:| :---: | :---: | :---: | :---: | :---: | :---: |
| UniAD-tiny | Pytorch-1.12 | FP32 | 843.5172 ms | 0.9986  | 0.27 | 0 |
| UniAD-tiny | TensorRT-10.7.0.23 | FP32 | 64.0726 ms | 0.9986 | 0.27 | 9.2417e-07 |
| UniAD-tiny | TensorRT-10.7.0.23 | FP16 |  50.1560 ms | 1.0021 | 0.26 | 0.0458 |
| UniAD-tiny | TensorRT-10.7.0.23 | INT8(EQ) |  54.1927 ms | testing | testing | 0.0124 | 
| UniAD-tiny | TensorRT-10.7.0.23 | BEST(EQ) | 45.8763 ms | testing | testing | 0.0499 |

#### Videos
Videos coming soon.