################################################################################
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
# File: DL4AGX/MultiDeviceInferencePipeline/inference/JPEGDecoderPipeline/BUILD
################################################################################
package(default_visibility = ["//visibility:public"])

config_setting(
    name = "aarch64_linux",
    values = {"crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu"},
)

config_setting(
    name = "aarch64_qnx",
    values = {"crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx"},
)

cc_library(
    name = "jpeg_decoder_pipeline",
    srcs = ["JPEGDecoderPipeline.cpp"],
    hdrs = ["JPEGDecoderPipeline.h"],
    linkopts = select({
        ":aarch64_qnx": ["-lc"],
        "//conditions:default": ["-ldl"],
    }),
    deps = select({
        ":aarch64_linux": ["@dali_aarch64_linux//:dali"],
        ":aarch64_qnx": ["@dali_aarch64_qnx//:dali"],
        "//conditions:default": ["@dali_x86_64_linux//:dali"],
    }),
)
