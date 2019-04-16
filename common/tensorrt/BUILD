#############################################################################
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
# File: DL4AGX/common/tensorrt/BUILD
############################################################################
config_setting(
    name = "aarch64_linux",
    values = {"crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu"},
)

config_setting(
    name = "aarch64_qnx",
    values = {"crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx"},
)

cc_library(
    name = "tensorrt",
    hdrs = [
        "Logger.h",
        "Profiler.h",
        "utils.h",
    ],
    visibility = ["//visibility:public"],
    deps = select({
        ":aarch64_linux": ["@tensorrt5_aarch64_linux//:nvinfer"],
        ":aarch64_qnx": ["@tensorrt5_aarch64_qnx//:nvinfer"],
        "//conditions:default": ["@tensorrt5_x86_64_linux//:nvinfer"],
    }),
)
