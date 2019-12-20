##############################################################################################
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
# File: DL4AGX/LeNetWithS3Pooling/training/BUILD
##############################################################################################
package(default_visibility = ["//visibility:public"])
load("@pylinter_deps//:requirements.bzl", "requirement")

py_binary(
    name = "lenet_s3pool",
    srcs = [
        "main.py",
        "pooling.py"
    ],
    python_version = "PY3",
    deps = [
        requirement("torch"),
        requirement("torchvision"),
        requirement("torchsummary"),
        requirement("numpy")
    ],
)

filegroup(
    name = "lenet_s3pool_training",
    srcs = ["README.md"],
    data = ["lenet_s3pool"],
)
