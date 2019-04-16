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
# File: DL4AGX/WORKSPACE
# Description: Workspace declarations for the bazel build system 
##########################################################################
workspace(name = "DL4AGX")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
git_repository(
    name = "io_bazel_rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    commit = "cc4cbf2f042695f4d1d4198c22459b3dbe7f8e43",
)

load("@io_bazel_rules_python//python:pip.bzl", "pip_import")
pip_import(
   name = "pylinter_deps",
   requirements = "//tools/linter:requirements.txt",
)

load("@pylinter_deps//:requirements.bzl", "pip_install")
pip_install()

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "0.1.0",  # change this to use a different release
)

####################################################################
# Cross-compilation Toolchains  
####################################################################

new_local_repository(
    name = "aarch64_linux_toolchain",
    path = "/",
    build_file = "toolchains/D5L/BUILD",
)

new_local_repository(
    name = "aarch64_qnx_toolchain",
    path = "/",
    build_file = "toolchains/D5Q/BUILD",
)

####################################################################
# x86 Libraries 
####################################################################

new_local_repository(
    name = "cuda10_x86_64_linux",
    path = "/usr/local/cuda/targets/x86_64-linux/",
    build_file = "libs/cuda.BUILD"
)

new_local_repository(
    name = "tensorrt5_x86_64_linux",
    path = "/usr/local/cuda/dl/targets/x86_64-linux/",
    build_file = "libs/tensorrt.BUILD"
)

new_local_repository(
    name = "cudnn7_x86_64_linux",
    path = "/usr/local/cuda/dl/targets/x86_64-linux/",
    build_file = "libs/cudnn.BUILD"
)

# new_local_repository(
#     name = "cupti10_x86_64_linux",
#     path = "/usr/local/cuda-10.0/targets/x86_64-linux/extras/CUPTI",
#     build_file = "libs/cupti.BUILD"
# )

new_local_repository(
    name = "protobuf_x86_64_linux",
    path = "/usr/local",
    build_file = "libs/protobuf.BUILD",
)

new_local_repository(
    name = "opencv_x86_64_linux",
    path = "/usr/local",
    build_file = "libs/opencv.BUILD",
)

new_local_repository(
    name = "dali_x86_64_linux",
    path = "/usr/local",
    build_file = "libs/dali.BUILD",
)

new_local_repository(
    name = "turbojpeg_x86_64_linux",
    path = "/usr/local/lib",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "turbojpeg",
    srcs = ["libturbojpeg.so"],
)
    """,
)

new_local_repository(
    name = "jpeg_x86_64_linux",
    path = "/usr/local/lib",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jpeg",
    srcs = ["libjpeg.so"],
)
    """,
)

####################################################################
# ARM Libraries 
####################################################################

new_local_repository(
    name = "cuda10_aarch64_linux",
    path = "/usr/local/cuda/targets/aarch64-linux/",
    build_file = "libs/cuda.BUILD"
)

new_local_repository(
    name = "tensorrt5_aarch64_linux",
    path = "/usr/local/cuda/dl/targets/aarch64-linux/",
    build_file = "libs/tensorrt.BUILD"
)

new_local_repository(
    name = "cudnn7_aarch64_linux",
    path = "/usr/local/cuda/dl/targets/aarch64-linux/",
    build_file = "libs/cudnn.BUILD"
)

# new_local_repository(
#     name = "cupti10_aarch64_linux",
#     path = "/usr/local/cuda-10.0/targets/aarch64-linux/extras/CUPTI",
#     build_file = "libs/cupti.BUILD"
# )

new_local_repository(
    name = "protobuf_aarch64_linux",
    path = "/usr/aarch64-linux-gnu",
    build_file = "libs/protobuf.BUILD",
)

new_local_repository(
    name = "opencv_aarch64_linux",
    path = "/usr/aarch64-linux-gnu",
    build_file = "libs/opencv.BUILD",
)

new_local_repository(
    name = "dali_aarch64_linux",
    path = "/usr/aarch64-linux-gnu",
    build_file = "libs/dali.BUILD",
)

new_local_repository(
    name = "turbojpeg_aarch64_linux",
    path = "/usr/aarch64-linux-gnu/lib",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "turbojpeg",
    srcs = ["libturbojpeg.so"],
)
    """,
)

new_local_repository(
    name = "jpeg_aarch64_linux",
    path = "/usr/aarch64-linux-gnu/lib",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jpeg",
    srcs = ["libjpeg.so"],
)
    """,
)

####################################################################
# QNX Libraries 
####################################################################
new_local_repository(
    name = "qnx_toolchain",
    path = "/usr/aarch64-unknown-nto-qnx/aarch64le/",
    build_file = "toolchain/D5Q/BUILD"
)

new_local_repository(
    name = "cuda10_aarch64_qnx",
    path = "/usr/local/cuda/targets/aarch64-qnx/",
    build_file = "libs/cuda.BUILD"
)

new_local_repository(
    name = "tensorrt5_aarch64_qnx",
    path = "/usr/local/cuda/dl/targets/aarch64-qnx/",
    build_file = "libs/tensorrt.BUILD"
)

new_local_repository(
    name = "cudnn7_aarch64_qnx",
    path = "/usr/local/cuda/dl/targets/aarch64-qnx/",
    build_file = "libs/cudnn.BUILD"
)

# new_local_repository(
#     name = "cupti10_aarch64_qnx",
#     path = "/usr/local/cuda-10.0/targets/aarch64-qnx/extras/CUPTI",
#     build_file = "libs/cupti.BUILD"
# )


new_local_repository(
    name = "protobuf_aarch64_qnx",
    path = "/usr/aarch64-unknown-nto-qnx/aarch64le/",
    build_file = "libs/protobuf.BUILD",
)

new_local_repository(
    name = "opencv_aarch64_qnx",
    path = "/usr/aarch64-unknown-nto-qnx/aarch64le/",
    build_file = "libs/opencv.BUILD",
)

new_local_repository(
    name = "dali_aarch64_qnx",
    path = "/usr/aarch64-unknown-nto-qnx/aarch64le/",
    build_file = "libs/dali.BUILD",
)

new_local_repository(
    name = "turbojpeg_aarch64_qnx",
    path = "/usr/aarch64-unknown-nto-qnx/aarch64le/lib",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "turbojpeg",
    srcs = ["libturbojpeg.so"],
)
    """,
)

new_local_repository(
    name = "jpeg_aarch64_qnx",
    path = "/usr/aarch64-unknown-nto-qnx/aarch64le/lib",
    build_file_content = """
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "jpeg",
    srcs = ["libjpeg.so"],
)
    """,
)