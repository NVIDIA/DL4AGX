# RetinaNet DALI Based Inference Pipeline

## Introduction

This short example modfies the MultiDeviceInferencePipeline application to run a ONNX based RetinaNet Model using DALI and TensorRT. It constructs an inference pipeline that will decode a JPEG Image, and perform inference.

## Usage 

Convert the ONNX Model using TRT Exec

``` sh
CUDA_VISIBLE_DEVICES=1 trtexec --onnx=RetinaNet.onnx --saveEngine=retinanet.trt --int8 --useDLACore=1
```

The Inference Pipeline is configured using a TOML file. An example configuration is provided below. In order to run the application, pass the path to the TOML file with the `--conf` flag:

```sh
.../retinanet_inference --conf=inference_pipeline.toml --trtoplib=<path_to_rootdir>/bazel-bin/plugins/dali/TensorRTInferOp/libtensorrtinferop.so
```

## How to compile
It is recommended to cross compile this program instead of natively compiling it on the Development Kit. In setting up this project you should have gone through the instructions to create a Docker image (if you have not done this refer to `//docker/README.md`) which will have the cross compiling toolchain and necessary libraries setup.

There are two main components to compile, the application (i.e Inference pipeline), and the DALI plugin intergrating TensorRT into DALI. All the applications and plugins can be compiled in one step.

_For aarch64-linux:_
```Shell
dazel build //RetinaNetDALITRT/... //plugins/dali/... --config=D5L-toolchain
```
_For aarch64-qnx_:
```Shell
dazel build //RetinaNetDALITRT/... //plugins/dali/... --config=D5Q-toolchain
```
_For x86_64-linux_:
```Shell
dazel build //RetinaNetDALITRT/... //plugins/dali/...
```

Then after making sure all dependencies are on the target and available for linking, copy the contents of `bazel-out/aarch64-fastbuild` to the target and you should be ready to run.

## Running on a Target 

In order to run on a target you will need the following files to be present, libraries should be in the LD_LIBRARY_PATH:

- **inference** (executable) -> ./bazel-out/aarch64-fastbuild/RetinaNetDALITRT/retinanet_inference

- **libtensorrtinferop.so** (DALI plugin for inference) -> ./bazel-out/aarch64-fastbuild/plugins/dali/TensorRTInferOp/libtensorrtinferop.so 

- **Config.toml** (Pipeline Configuration)

- Dependencies baked into the build container - (copy these files out of the build container): 

  > docker cp <container_id>:[path in docker container to file] [path on host machine]
  >
  > Libraries for aarch64 are located in /usr/aarch64-linux-gnu/lib, for QNX /usr/aarch64-unknown-nto-qnx/aarch64le/lib

  - libdali.so 
  - libprotobuf.so.15
  - libopencv_imgcodecs.so.3.4
  - libopencv_imgproc.so.3.4 
  - libopencv_core.so.3.4
  - libjpeg.so.62

## Example Configuration File (TOML) 

``` toml
input_image = "test1.jpg"
profile = false

# Configurations for the Object Detection Pipeline 
[[inference_pipeline]]
name = "RetinaNet"
device = 1
dla_core = 1
batch_size = 1
async_execution = true
num_threads = 1

[inference_pipeline.engine]
path = "retinanet.trt"

[[inference_pipeline.engine.inputs]]
name = "x"
shape = [3, 960, 1920]

[[inference_pipeline.engine.outputs]]
name = "330"
shape = [720, 120, 240]

[[inference_pipeline.engine.outputs]]
name = "331"
shape = [720, 60, 120]

[[inference_pipeline.engine.outputs]]
name = "332"
shape = [720, 30, 60]

[[inference_pipeline.engine.outputs]]
name = "333"
shape = [720, 15, 30]

[[inference_pipeline.engine.outputs]]
name = "334"
shape = [720, 8, 15]

[[inference_pipeline.engine.outputs]]
name = "293"
shape = [36, 120, 240]

[[inference_pipeline.engine.outputs]]
name = "302"
shape = [36, 60, 120]

[[inference_pipeline.engine.outputs]]
name = "311"
shape = [36, 30, 60]

[[inference_pipeline.engine.outputs]]
name = "320"
shape = [36, 15, 30]

[[inference_pipeline.engine.outputs]]
name = "329"
shape = [36, 8, 15]

[inference_pipeline.preprocessing]
resize = [3, 960, 1920]
mean = [127.5, 127.5, 127.5]
std_dev = [127.5, 127.5, 127.5]

[postprocessing]
detection_threshold = 0.5

```

