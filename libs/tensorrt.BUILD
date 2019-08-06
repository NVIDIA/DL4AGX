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
# File: DL4AGX/libs/tensorrt.BUILD
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

cc_library(
    name = "nvinfer_headers",
    hdrs = [
        "include/NvInfer.h",
        "include/NvUtils.h"
    ],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_import(
    name = "nvinfer_lib",
    shared_library = "lib/libnvinfer.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinfer",
    deps = [
        "nvinfer_headers",
        "nvinfer_lib"
    ] + select({
        ":aarch64_linux": [
            "@cuda_aarch64_linux//:cudart",
            "@cuda_aarch64_linux//:cublas",
            "@cudnn_aarch64_linux//:cudnn"
        ],
        ":aarch64_qnx": [
            "@cuda_aarch64_qnx//:cudart",
            "@cuda_aarch64_qnx//:cublas",
            "@cudnn_aarch64_qnx//:cudnn"
        ],
        "//conditions:default": [
            "@cuda_x86_64_linux//:cudart",
            "@cuda_x86_64_linux//:cublas",
            "@cudnn_x86_64_linux//:cudnn"
        ]
    }),
    visibility = ["//visibility:public"],
)

####################################################################################

cc_import(
    name = "nvparsers_lib",
    shared_library = "lib/libnvparsers.so",
    visibility = ["//visibility:private"],
)


cc_library(
    name = "nvparsers_headers",
    hdrs = [
        "include/NvCaffeParser.h",
        "include/NvOnnxParser.h",
        "include/NvOnnxParserRuntime.h",
        "include/NvOnnxConfig.h",
        "include/NvUffParser.h"
    ],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvparsers",
    deps = [
        "nvparsers_headers",
        "nvparsers_lib"
    ] + select({
        ":aarch64_linux":["@tensorrt_aarch64_linux//:nvinfer"],
        ":aarch64_qnx":["@tensorrt_aarch64_qnx//:nvinfer"],
        "//conditions:default":["@tensorrt_x86_64_linux//:nvinfer"]
    }),
   visibility = ["//visibility:public"],
)

####################################################################################

cc_import(
    name = "nvonnxparser_lib",
    shared_library = "lib/libnvonnxparser.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_headers",
    hdrs = [
        "include/NvOnnxParser.h",
        "include/NvOnnxParserRuntime.h",
        "include/NvOnnxConfig.h"
    ],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser",
    deps = [
        "nvonnxparser_headers",
        "nvonnxparser_lib"
    ] + select({
        ":aarch64_linux":["@tensorrt_aarch64_linux//:nvinfer"],
        ":aarch64_qnx":["@tensorrt_aarch64_qnx//:nvinfer"],
        "//conditions:default":["@tensorrt_x86_64_linux//:nvinfer"]
    }),
   visibility = ["//visibility:public"],
)

####################################################################################

cc_import(
    name = "nvonnxparser_runtime_lib",
    shared_library = "lib/libnvonnxparser_runtime.so",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "nvonnxparser_runtime_header",
    hdrs = ["include/NvOnnxParserRuntime.h"],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvonnxparser_runtime",
    deps = [
        "nvonnxparser_runtime_header",
        "nvonnxparser_runtime_lib"
    ] + select({
        ":aarch64_linux":[
            "@tensorrt_aarch64_linux//:nvinfer",
            "@tensorrt_aarch64_linux//:nvonnxparser",
        ],
        ":aarch64_qnx":[
            "@tensorrt_aarch64_qnx//:nvinfer",
            "@tensorrt_aarch64_qnx//:nvonnxparser"
        ],
        "//conditions:default":[
            "@tensorrt_x86_64_linux//:nvinfer",
            "@tensorrt_x86_64_linux//:nvonnxparser"
        ]
    }),
   visibility = ["//visibility:public"],
)

####################################################################################

cc_import(
    name = "nvcaffeparser_lib",
    shared_library = "lib/libnvcaffe_parsers.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvcaffeparser_headers",
    hdrs = ["include/NvCaffeParser.h"],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvcaffeparser",
    deps = [
        "nvcaffeparser_headers",
        "nvcaffeparser_lib"
    ] + select({
        ":aarch64_linux":["@tensorrt_aarch64_linux//:nvinfer"],
        ":aarch64_qnx":["@tensorrt_aarch64_qnx//:nvinfer"],
        "//conditions:default":["@tensorrt_x86_64_linux//:nvinfer"]
    }),
    visibility = ["//visibility:public"],
)

####################################################################################

cc_import(
    name = "nvinferplugin_lib",
    shared_library = "lib/libnvinfer_plugin.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinferplugin_headers",
    hdrs = ["include/NvInferPlugin.h"],
    includes = ["include/"],
    visibility = ["//visibility:private"],
)

cc_library(
    name = "nvinferplugin",
    deps = [
        "nvinferplugin_headers",
        "nvinferplugin_lib"
    ] + select({
        ":aarch64_linux":["@tensorrt_aarch64_linux//:nvinfer"],
        ":aarch64_qnx":["@tensorrt_aarch64_qnx//:nvinfer"],
        "//conditions:default":["@tensorrt_x86_64_linux//:nvinfer"]
    }),
    visibility = ["//visibility:public"],
)
