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
# File: DL4AGX/libs/dali.BUILD
############################################################################
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
)

config_setting(
    name = "aarch64_qnx",
    values = { "crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx" }
)

cc_import(
    name = "dali_lib",
    shared_library = "lib/libdali.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "dali_headers",
    hdrs = glob([
        "include/dali/**/*.h"
    ]),
    includes = ["include/dali"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "dali",
    deps = [
        "dali_headers",
        "dali_lib"
    ] + select({
        ":aarch64_linux": [
            "@protobuf_aarch64_linux//:protobuf",
            "@protobuf_aarch64_linux//:protobuf_lite",
            "@protobuf_aarch64_linux//:protoc",
            "@cuda_aarch64_linux//:cudart",
            "@cuda_aarch64_linux//:cuda_dali_deps",
            "@opencv_aarch64_linux//:opencv_core",
            "@opencv_aarch64_linux//:opencv_imgproc",
            "@opencv_aarch64_linux//:opencv_imgcodecs",
            "@turbojpeg_aarch64_linux//:turbojpeg",
            "@jpeg_aarch64_linux//:jpeg"
        ],
        ":aarch64_qnx": [
            "@protobuf_aarch64_qnx//:protobuf",
            "@protobuf_aarch64_qnx//:protobuf_lite",
            "@protobuf_aarch64_qnx//:protoc",
            "@cuda_aarch64_qnx//:cudart",  
            "@cuda_aarch64_qnx//:cuda_dali_deps",
            "@opencv_aarch64_qnx//:opencv_core",
            "@opencv_aarch64_qnx//:opencv_imgproc",
            "@opencv_aarch64_qnx//:opencv_imgcodecs",
            "@turbojpeg_aarch64_qnx//:turbojpeg",
            "@jpeg_aarch64_qnx//:jpeg"
        ],
        "//conditions:default": [
            "@protobuf_x86_64_linux//:protobuf",
            "@protobuf_x86_64_linux//:protobuf_lite",
            "@protobuf_x86_64_linux//:protoc",
            "@cuda_x86_64_linux//:cudart",
            "@cuda_x86_64_linux//:cuda_dali_deps",
            "@opencv_x86_64_linux//:opencv_core",
            "@opencv_x86_64_linux//:opencv_imgproc",
            "@opencv_x86_64_linux//:opencv_imgcodecs",
            "@turbojpeg_x86_64_linux//:turbojpeg",
            "@jpeg_x86_64_linux//:jpeg",
        ],
    }),
    defines = select({
        ":aarch64_linux":["__AARCH64_GNU__",],
        ":aarch64_qnx":["__AARCH64_QNX__",],
        "//conditions:default":[]
    })
)
