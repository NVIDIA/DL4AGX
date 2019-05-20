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
# File: DL4AGX/plugins/tensorrt/FlattenConcatPlugin/BUILD 
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

#If we want to distribute a seperate so file instead of compiling into
#a single executable
cc_binary(
    name = "libflattenconcatplugin.so",
    srcs = [
        "FlattenConcat.cpp",
        "FlattenConcat.h",
    ],
    linkshared = True,
    deps = ["//common:common"] +
           select({
               ":aarch64_linux": [
                   "@tensorrt_aarch64_linux//:nvinferplugin",
                   "@cuda_aarch64_linux//:cublas",
               ],
               ":aarch64_qnx": [
                   "@tensorrt_aarch64_qnx//:nvinferplugin",
                   "@cuda_aarch64_qnx//:cublas",
               ],
               "//conditions:default": [
                   "@tensorrt_x86_64_linux//:nvinferplugin",
                   "@cuda_x86_64_linux//:cublas",
               ],
           }),
)

#Usable dependency for other projects
cc_library(
    name = "flattenconcat",
    srcs = [
        "FlattenConcat.cpp",
        "FlattenConcat.h",
    ],
    hdrs = ["FlattenConcat.h"],
    deps = ["//common:common"] +
           select({
               ":aarch64_linux": [
                   "@tensorrt_aarch64_linux//:nvinferplugin",
                   "@cuda_aarch64_linux//:cublas",
               ],
               ":aarch64_qnx": [
                   "@tensorrt_aarch64_qnx//:nvinferplugin",
                   "@cuda_aarch64_qnx//:cublas",
               ],
               "//conditions:default": [
                   "@tensorrt_x86_64_linux//:nvinferplugin",
                   "@cuda_x86_64_linux//:cublas",
               ],
           }),
)
