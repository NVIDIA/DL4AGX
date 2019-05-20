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
# File: DL4AGX/plugins/dali/TensorRTInferOp/BUILD 
############################################################################
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    values = {"crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu"},
)

config_setting(
    name = "aarch64_qnx",
    values = {"crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx"},
)

cc_binary(
    name = "libtrtinferop.so",
    srcs = [
        "tensorrtInferOp.cpp",
        "tensorrtInferOp.h",
    ],
    copts = [
        "-pthread",
        "-fPIC",
    ] + select({
        ":aarch64_linux": [
            "-D__AARCH64_GNU__",
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
        ],
        ":aarch64_qnx": [
            "-D__AARCH64_QNX__",
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-DGOOGLE_PROTOBUF_ARCH_X64",
            "-Dgoogle=google_private",
        ],
        "//conditions:default": [
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-DGOOGLE_PROTOBUF_ARCH_X64",
        ],
    }),
    linkopts = select({
        ":aarch64_qnx": ["-lc"],
        "//conditions:default": [
            "-Wl,--no-as-needed",
            "-lpthread",
            "-ldl",
        ],
    }),
    linkshared = True,
    deps = select({
        ":aarch64_linux": [
            "@dali_aarch64_linux//:dali",
            "@tensorrt_aarch64_linux//:nvinfer",
            "@tensorrt_aarch64_linux//:nvinferplugin",
            "@cuda_aarch64_linux//:cudart",
        ],
        ":aarch64_qnx": [
            "@dali_aarch64_qnx//:dali",
            "@tensorrt_aarch64_qnx//:nvinfer",
            "@tensorrt_aarch64_qnx//:nvinferplugin",
            "@cuda_aarch64_qnx//:cudart",
        ],
        "//conditions:default": [
            "@dali_x86_64_linux//:dali",
            "@tensorrt_x86_64_linux//:nvinfer",
            "@tensorrt_x86_64_linux//:nvinferplugin",
            "@cuda_x86_64_linux//:cudart",
        ],
    }),
)
