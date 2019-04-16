##########################################################################
# Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: DL4AGX/BUILD
# Description: Source Package declarations for the project 
##########################################################################
load("//tools/packaging:src_package.bzl", "src_package")

src_package(
    name = "DRIVE_OS-reference-app-5.1.3.0-aarch64-linux-OPENSOURCE",
    components = [
        "//MultiDeviceInferencePipeline:MultiDeviceInferencePipeline",
        "//plugins/tensorrt/FlattenConcatPlugin:libflattenconcatplugin.so",
        "//plugins/dali/TensorRTInferOp:libtrtinferop.so",
    ],
    documentation = """
## DRIVE OS Reference Application

This app takes you through developing a inference pipeline utilizing the compute capabilities of the NVIDIA DRIVE AGX Developer Kit. The pipeline will do Object Detection and Lane Segementation on an image concurrently using both the integrated GPU of the Xavier Chip and one of the Deep Learning Accelerators onboard.

### Getting Started with the DRIVE OS Reference Application
Follow the instructions above to setup the development enviorment. Then look at the README in the project directory for details on the components of the reference application and how to compile them. 
""",
    min_bazel_version = "0.21.0",
    platform = "DRIVE",
    pdk_platform = "aarch64-linux",
    pdk_version = "5.1.3.0",
)

src_package(
    name = "Jetson-reference-app-4.1-aarch64-linux-OPENSOURCE",
    components = [
        "//MultiDeviceInferencePipeline:MultiDeviceInferencePipeline",
        "//plugins/tensorrt/FlattenConcatPlugin:libflattenconcatplugin.so",
        "//plugins/dali/TensorRTInferOp:libtrtinferop.so",
    ],
    documentation = """
## Jetson Reference Application

This app takes you through developing a inference pipeline utilizing the compute capabilities of the NVIDIA Jetson AGX Developer Kit. The pipeline will do Object Detection and Lane Segementation on an image concurrently using both the integrated GPU of the Xavier Chip and one of the Deep Learning Accelerators onboard.

### Getting Started with the Jetson Reference Application
Follow the instructions above to setup the development enviorment. Then look at the README in the project directory for details on the components of the reference application and how to compile them. 
""",
    min_bazel_version = "0.21.0",
    platform = "Jetson",
    pdk_platform = "aarch64-linux",
    pdk_version = "4.1",
)
