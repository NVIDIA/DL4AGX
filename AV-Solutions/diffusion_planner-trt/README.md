# Diffusion-Planner ONNX + TensorRT Deployment

[Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) is a diffusion-based trajectory planning model for autonomous vehicles. 
This repository provides guidance to enable ONNX export and deployment on NVIDIA DRIVE platforms with TensorRT.


## Features

The patches extends the original Diffusion-Planner repository with the following features:

- **Model export with mixed precision**: Convert the PyTorch model to ONNX with FP16 conversion using [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer)
- **TensorRT deployment**: Build high-performance inference engines for FP32 and FP16 precision
- **Evaluation**: Run closed-loop nuPlan simulations with TensorRT engines

## Results on nuPlan Benchmark

The table below shows closed-loop nuPlan performance across different settings. Latency and the closed-loop planning results on nuPlan benchmark of TensorRT engine were conducted on NVIDIA **DRIVE Thor-X** and **DRIVE Orin-X** using **TensorRT 10.15**. 
To get access to TensorRT 10.15 used in our experiments on NVIDIA DRIVE platforms, please refer to details on the [NVIDIA DRIVE site](https://developer.nvidia.com/drive/downloads?sortBy=drive_downloads%2Fsort%2Fdate%3Adesc).


<table>
  <thead>
    <tr>
      <th rowspan="2" style="vertical-align: middle;">Framework</th> 
      <th rowspan="2" style="vertical-align: middle;">Precision</th> 
      <th colspan="2" style="text-align: center;">Latency</th>
      <th colspan="2">nuPlan Closed-Loop Score</th>
    </tr>
    <tr>
      <th>Orin-X</th>
      <th>Thor-X</th>
      <th>Non-Reactive</th>
      <th>Reactive</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>PyTorch</td>
      <td>FP32</td>
      <td>217.8 ms</td>
      <td>58.8 ms</td>
      <td>89.9</td>
      <td>82.8</td>
    </tr>
    <tr>
      <td>ONNX (ORT-GPU)</td>
      <td>FP32</td>
      <td>35.5 ms</td>
      <td>24.8 ms</td>
      <td>88.1</td>
      <td>81.4</td>
    </tr>
    <tr>
      <td>TensorRT</td>
      <td>FP16+FP32</td>
      <td>3.9 ms</td>
      <td>2.2 ms</td>
      <td>87.9</td>
      <td>81.2</td>
    </tr>
  </tbody>
</table>


## Getting Started
### Prerequisites
1. Clone and set up the original Diffusion-Planner repository:
   ```bash
   git clone https://github.com/ZhengYinan-AIR/Diffusion-Planner.git
   cd Diffusion-Planner
   ```

2. Follow the [original setup instructions](https://github.com/ZhengYinan-AIR/Diffusion-Planner?tab=readme-ov-file#getting-started) to install dependencies and verify the base model runs successfully.

### Apply the Patch
To enable ONNX export and TensorRT deployment, the original Diffusion-Planner codebase must be modified accordingly.

Apply the deployment patch to enable ONNX export and TensorRT evaluation capabilities:
```bash
git apply path/to/patches/ONNX-export-and-TRT-engine-eval.patch
```

## Generate ONNX Models

### Export Model
Prepare the required files (`args.json`, `checkpoint.ckpt`) from the [evaluation section](https://github.com/ZhengYinan-AIR/Diffusion-Planner?tab=readme-ov-file#closed-loop-evaluation), then export:

```bash
python deployment/onnx_export.py \
    --args_json path/to/args.json \
    --model_ckpt path/to/checkpoint.ckpt \
    --onnx_output path/to/diffusion_planner.onnx
```
The output is a simplified ONNX model generated using the [onnx-simplifier](https://github.com/daquexian/onnx-simplifier) tool.


### FP16 Conversion
Convert the model to FP16 using [NVIDIA Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), leveraging its [AutoCast tool](https://github.com/NVIDIA/Model-Optimizer/blob/main/docs/source/guides/8_autocast.rst) for efficient low-precision deployment with TensorRT :

```bash
python -m modelopt.onnx.autocast \
    --onnx_path path/to/diffusion_planner.onnx \
    --output_path path/to/diffusion_planner_fp16.onnx \
    --low_precision_type fp16
```

It selectively lowers precision while retaining critical operations in FP32 for numerical stability, producing a strongly-typed, mixed-precision ONNX model.


## Build TensorRT Engines

You can generate optimized TensorRT engines for deployment on the target device using the provided build script:
```bash
cd deployment/
# Build an FP32 engine
./build_trt_engine.sh --onnx path/to/diffusion_planner.onnx \
                      --engine path/to/diffusion_planner.engine

# Build an FP16 mixed precision engine (ONNX model must already be cast to FP16)
./build_trt_engine.sh --onnx path/to/diffusion_planner_fp16.onnx \
                      --engine path/to/diffusion_planner_fp16.engine \
                      --fp16
``` 
                                     

## Evaluation on nuPlan Closed-loop Simulation

We evaluate the TensorRT engine built/run on the target against [nuPlan simulation benchmark](https://www.nuscenes.org/nuplan) to validate accuracy and performance across autonomous driving scenarios.

The following evaluation system enables responsive streaming validation between the nuPlan simulation environment running on x86 and the TensorRT engine running on the target. This architecture enables proper evaluation of diffusion-based trajectory planning through continuous interaction between the simulation environment and inference engine.

#### System Architecture

```
+-------------------------------------------------------------+  
|             nuPlan Closed-loop Simulation (x86)             |
|-------------------------------------------------------------|
|  • Runs `planner.py` integrated into nuPlan simulation.     |
|  • Converts scenario data into normalized input tensors:    |
|      - ego_current_state                                    |
|      - neighbor_agents_past                                 |
|      - static_objects                                       |
|      - lanes, speed limits, route lanes                     |
|  • Serializes tensors → sends via TCP socket to Target.     |
|  • <<< Receives decoded trajectories (`decoder_outputs`) >>>|
|  • Converts outputs → future ego trajectory for simulation. |
+-------------------------------------------------------------+
                 ^
                 |  (Persistent TCP Socket Connection)
                 v
+-------------------------------------------------------------+
|            NVIDIA DRIVE Platform (aarch64 target)           |
|-------------------------------------------------------------|
|  • Executes compiled TensorRT app                           |
|     (from trt_infer_app.cpp) on target.                     |
|  • Loads prebuilt TensorRT engine.                          |
|  • Steps:                                                   |
|     1) Receives serialized tensors from nuPlan host.        |
|     2) Copies tensors → GPU memory.                         |
|     3) Runs `context->enqueueV3()` (TensorRT inference).    |
|     4) <<< Sends back `decoder_outputs` over socket >>>     |
+-------------------------------------------------------------+

```

--------------------------------------------------------------------------------------------

### Prerequisites
#### 1. Ensure nuPlan is installed according to the [official instructions](https://github.com/motional/nuplan-devkit).

#### 2. Compile TensorRT Inference App on Target
Copy `trt_infer_app.cpp` source file to your target device.
Compile the C++ inference app that will handle TensorRT engine execution and socket communication:

```bash
g++ trt_infer_app.cpp -o trt_infer_app \
    -I/path/to/TensorRT/include \
    -I/usr/local/cuda/include \
    -L/path/to/TensorRT/lib \
    -L/usr/local/cuda/lib64 \
    -lnvinfer -lcudart
```

The inference app is implemented with [CUDA Graph optimization](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html) by capturing the execution pattern on the first inference and reusing it for subsequent calls, reducing latency through minimized kernel scheduling overhead.

### Launch TensorRT Inference App on Target
Ensure the TensorRT engine file `diffusion_planner.engine` is available on your target device.
Start the inference app to listen for tensor data from the nuPlan simulation (default port: 5555):

```bash
./trt_infer_app /path/to/diffusion_planner.engine
```

The app establishes a persistent socket connection and waits for tensor data from the nuPlan simulation runner host.

### Run nuPlan Simulation on x86 Host

Launch the NuPlan closed-loop simulation on the x86 host, which streams planning requests to the inference application running on the target:

```bash
bash sim_diffusion_planner_runner_trt.sh
```

- **Port Configuration**: Default TCP port is 5555. To change:
  - Update the port in `trt_infer_app.cpp` before compilation
  - Update the corresponding port in `sim_diffusion_planner_runner_trt.sh`
- **Persistent Connection**: The socket connection remains active between simulations, eliminating app restart overhead

## Technical Notes: 

### 1. Installing ONNX Runtime (CUDA-Enabled) on DRIVE Thor and Orin
To enable CUDA support on NVIDIA Thor and Orin, ONNX Runtime needs to be built from source. 
Prebuilt GPU wheels are not officially available for ARM64 with custom Python or CUDA versions, so building locally ensures compatibility with the target’s Python, CUDA/cuDNN stack, and compute architecture.

### **On DRIVE Thor (Python 3.12, CUDA 12.8)**
```bash
git clone https://github.com/microsoft/onnxruntime.git 
cd /path/to/workspace/onnxruntime
git checkout v1.21.0
git submodule update --init --recursive

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pip install -U pip setuptools wheel cmake==3.28 ninja "numpy<2.1" psutil packaging

./build.sh --allow_running_as_root --update --build --build_wheel --parallel 10 \
  --config Release --skip_tests --use_cuda \
  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/lib64 \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES="89;90" \
    FETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/deps/eigen \
    FETCHCONTENT_UPDATES_DISCONNECTED=ON \
    USE_CUBLAS=ON USE_CUDNN=ON

pip install --force-reinstall --no-deps \
  /path/to/workspace/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu-*.whl
```
### **On DRIVE Orin (Python 3.10, CUDA 11.4)**
```bash
git clone https://github.com/microsoft/onnxruntime.git
cd /path/to/workspace/onnxruntime
git checkout v1.16.0
git submodule update --init --recursive

pip install cmake==3.26 psutil

./build.sh --allow_running_as_root --update --build --build_wheel --parallel 10 \
  --config Release --skip_tests --use_cuda \
  --cuda_home /usr/local/cuda --cudnn_home /usr/local/cuda/lib64 \
  --cmake_extra_defines \
    CMAKE_CUDA_ARCHITECTURES=87 \
    FETCHCONTENT_SOURCE_DIR_EIGEN=/path/to/deps/eigen \
    FETCHCONTENT_UPDATES_DISCONNECTED=ON \
    USE_CUBLAS=ON USE_CUDNN=ON

pip install --force-reinstall --no-deps \
  /path/to/workspace/onnxruntime/build/Linux/Release/dist/onnxruntime_gpu-*.whl
```
- **Note**: Eigen was explicitly downloaded and placed in `/path/to/deps/eigen`, and referenced via `FETCHCONTENT_SOURCE_DIR_EIGEN` to avoid online fetching during build.
--------------------------------------------------------------------------------------------

### 2. Cross-Platform Accuracy Discrepancies

#### DPM-Solver++ Behavior
When using DPM-Solver++ in diffusion models, minor output differences between PyTorch and ONNX/TensorRT are expected. These discrepancies are observed as a result of the solver’s multi-step numerical formulation, rather than model export issues.

### Root Cause
DPM-Solver++ approximates second-order ODEs using recursive updates:

```
x_t ≈ (σ_t/σ_t0) × x - α_t × φ_1 × ε_θ(x_t0,t0) - 0.5 × α_t × φ_1 × (1/r_0) × [ε_θ(x_t0,t0) - ε_θ(x_t1,t1)]
```
where:

- `φ_1 = expm1(-h)`  (`torch.expm1(-h)` [in code](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/5659e494250523a603902e1c3dca0651d2e4c6fa/diffusion_planner/model/diffusion_utils/dpm_solver_pytorch.py#L825))  
- `r_0 = h_0 / h`, with `h_0 = λ_t0 - λ_t1` and `h = λ_t - λ_t0`  
- `ε_θ` is the model's noise prediction.

This update uses the difference between consecutive model outputs (`model_prev_0 - model_prev_1` [in code](https://github.com/ZhengYinan-AIR/Diffusion-Planner/blob/5659e494250523a603902e1c3dca0651d2e4c6fa/diffusion_planner/model/diffusion_utils/dpm_solver_pytorch.py#L823)) as an estimate of the first-order derivative. Since each step’s input depends on the outputs of previous steps, even small numerical deviations can lead to cumulative errors. Specifically:

- Even small deviations in earlier outputs (e.g., between PyTorch and ONNX/TRT) lead to different inputs at the next step.
- These differences accumulate and amplify over successive diffusion steps, resulting in divergent final outputs.

In DPM-Solver++, the recursion occurs outside the model, within the numerical solver logic that wraps around it. As a result, any numerical drift in model predictions directly alters the solver's input trajectory at subsequent steps.
Furthermore, both ONNX Runtime and TensorRT operate on the same ONNX model exported from PyTorch. This shared graph restructures aspects of the solver logic (e.g., timestep indexing, control flow), leading to consistent behavior between the two backends—and thus similar divergence trends relative to PyTorch.

