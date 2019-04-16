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
# File: DL4AGX/common/BUILD
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
    name = "common",
    hdrs = ["common.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//common:argparser",
        "//common:macros",
        "//common:timers",
        "//common/tensorrt:tensorrt",
        "//common/image:image",
        "//common/datasets/coco:coco_json",
    ] + select({
        ":aarch64_linux": ["@cuda10_aarch64_linux//:cudart"],
        ":aarch64_qnx": ["@cuda10_aarch64_qnx//:cudart"],
        "//conditions:default": ["@cuda10_x86_64_linux//:cudart"],
    }),
)

cc_library(
    name = "argparser",
    hdrs = ["argparser.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "macros",
    hdrs = ["macros.h"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "timers",
    hdrs = ["timers.h"],
    visibility = ["//visibility:public"],
    deps = ["//common:macros"]
)

# cc_library(
#     name = "cupti_utils",
#     srcs = glob(["cuptiUtils.h"]),
#     hdrs = glob(["cuptiUtils.h"]),
#     visibility = ["//visibility:public"],
#     deps = select({":aarch64_linux":["@cuda10_aarch64_linux//:cuda",
#                                    "@cupti10_aarch64_linux//:cupti"],
#                  ":aarch64_qnx":["@cuda10_aarch64_qnx//:cuda",
#                                  "@cupti10_aarch64_qnx//:cupti"],
#                  "//conditions:default":["@cuda10_x86_64_linux//:cuda",
#                                         "@cupti10_x86_64_linux//:cupti"]})
# )
