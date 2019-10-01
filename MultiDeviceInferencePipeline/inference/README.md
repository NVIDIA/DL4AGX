# Inference Pipeline

## Introduction

The Inference Pipeline is the main component of the app. It uses an object detection TensorRT engine (based on SSD-ResNet18) and a segmentation TensorRT engine (based on DeepLabV3) to construct an inference pipeline that will decode a JPEG Image, conduct preprocessing for both networks, perform inference and post-processing visualization. The inference runs concurrently on iGPU and DLA. The input image to each engine can have a different size which is determined when building the engine.

![Detection and segmentation](../docs/pipeline.png)

## Usage 

The Inference Pipeline is configured using a TOML file. An example configuration is provided below. In order to run the application, pass the path to the TOML file with the `--conf` flag:

```sh
.../inference --conf=inference_pipeline.toml --trtoplib=<path_to_rootdir>/bazel-bin/plugins/dali/TensorRTInferOp/libtensorrtinferop.so
```

## How to compile
It is recommended to cross compile this program instead of natively compiling it on the Development Kit. In setting up this project you should have gone through the instructions to create a Docker image (if you have not done this refer to `//docker/README.md`) which will have the cross compiling toolchain and necessary libraries setup.

There are three main components to compile, the application (i.e Inference pipeline), the DALI plugin intergrating TensorRT into DALI as well as the TensorRT plugin (i.e., Flatten Concatenation) for the object detector. All the applications and plugins can be compiled in one step.

_For aarch64-linux:_
```Shell
dazel build //MultiDeviceInferencePipeline/... //plugins/... --config=D5L-toolchain
```
_For aarch64-qnx_:
```Shell
dazel build //MultiDeviceInferencePipeline/... //plugins/... --config=D5Q-toolchain
```
_For x86_64-linux_:
```Shell
dazel build //MultiDeviceInferencePipeline/... //plugins/...
```

Then after making sure all dependencies are on the target and available for linking, copy the contents of `bazel-out/aarch64-fastbuild` to the target and you should be ready to run.

## Running on a Target 

In order to run on a target you will need the following files to be present, libraries should be in the LD_LIBRARY_PATH:

- **inference** (executable) -> ./bazel-out/aarch64-fastbuild/MultiDeviceInferencePipeline/inference/inference

- **libtensorrtinferop.so** (DALI plugin for inference) -> ./bazel-out/aarch64-fastbuild/plugins/dali/TensorRTInferOp/libtensorrtinferop.so 

- **libflattenconcatplugin.so** (TensorRT Plugin for FlattenConcat support) ->  ./bazel-out/aarch64-fastbuild/plugins/tensorrt/FlattenConcatPlugin/libflattenconcatplugin.so

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
  - libavcodec.so.57
  - libavutil.so.55 


## Example Configuration File (TOML) 

``` toml
input_image = "/path/to/input_image.jpg"
output_image = "/path/to/output_image.jpg"
profile = false

# Configurations for the Segmentation Pipeline
[[inference_pipeline]]
name = "Segmentation"
device = 1
dla_core = 1
batch_size = 1
async_execution = true
num_threads = 1

[inference_pipeline.engine]
path = "experiments/data_mdip/model_res18_small_240x795_fp16_DLA.engine"

[[inference_pipeline.engine.inputs]]
name = "ImageTensor"
shape = [3, 240, 795]

[[inference_pipeline.engine.outputs]]
name = "logits/semantic/BiasAdd"
shape = [2, 15, 50]

[inference_pipeline.preprocessing]
resize = [3, 240, 795]
mean = [0.0, 0.0, 0.0]
std_dev = [1.0, 1.0, 1.0]

# Configurations for the Object Detection Pipeline 
[[inference_pipeline]]
name = "Object Detection"
device = 1
dla_core = -1
batch_size = 1
async_execution = true
num_threads = 1

[inference_pipeline.engine]
path = "experiments/data_mdip/SSD_resnet18_kitti_int8_iGPU.engine"

[[inference_pipeline.engine.inputs]]
name = "Input"
shape = [3, 300, 300]

[[inference_pipeline.engine.outputs]]
name = "NMS"
shape = [1, 100, 7]

[[inference_pipeline.engine.outputs]]
name = "NMS_1"
shape = [1, 1, 1]

[[inference_pipeline.engine.plugins]]
path = "<path_to_rootdir>/bazel-bin/plugins/FlattenConcatPlugin/libflattenconcatplugin.so"

[inference_pipeline.preprocessing]
resize = [3, 300, 300]
mean = [127.5, 127.5, 127.5]
std_dev = [127.5, 127.5, 127.5]

[postprocessing]
detection_threshold = 0.5
```

