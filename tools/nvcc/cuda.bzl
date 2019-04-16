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
# File: DL4AGX/tools/nvcc/cuda.bzl
# Description: Declaration of cuda bazel rules 
##########################################################################
load("//tools/nvcc:private/constants.bzl", _CUDA_ACCEPTABLE_EXTENSIONS = "CUDA_ACCEPTABLE_EXTENSIONS", 
                                           _CUDA_ACCEPTABLE_HDR_EXTENSIONS = "CUDA_ACCEPTABLE_HDR_EXTENSIONS")
load("//tools/nvcc:private/cu_library_impl.bzl", _cu_library_impl = "cu_library_impl") 
load("//tools/nvcc:private/cu_binary_impl.bzl", _cu_binary_impl = "cu_binary_impl")

cu_library = rule(
    implementation = _cu_library_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = _CUDA_ACCEPTABLE_EXTENSIONS,
            doc = "Source files for a CUDA Library"
        ), 
        "hdrs": attr.label_list(
            allow_files = _CUDA_ACCEPTABLE_HDR_EXTENSIONS,
            doc = "Header files for a CUDA Library"
        ),
        "gpu_arch": attr.string(
            doc = "Target GPU Architecture"
        ),
        "gen_code": attr.string_list(
            doc = "Other architectures to compile PTX for"
        ),
        "defines": attr.label_list(),
        "nvcc_copts": attr.string_list(),
        "nvcc_linkopts": attr.string_list(),
        "nvcc_nocopts": attr.string_list(),
        "nvcc_nolinkopts": attr.string_list(),
        "copts": attr.string_list(),
        "linkopts": attr.string_list(),
        "nolinkopts": attr.string_list(),
        "nocopts": attr.string_list(),
        "includes": attr.string_list(), 
        "deps": attr.label_list(),
        "linkshared": attr.bool(
            default = False
        ),
        #TODO: "include_prefix": attr.string(), 
        #TODO: "strip_include_prefix": attr.string(),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),  
    },
    doc = "Builds a CUDA Library",
    fragments = ["cpp"],
    host_fragments = ["cpp"]
)

#TODO: Fix cu_library / cu_binary compatablity 
cu_binary = rule(
    implementation = _cu_binary_impl,
    attrs = {
        "srcs": attr.label_list(
            allow_files = _CUDA_ACCEPTABLE_EXTENSIONS,
            doc = "Source files for a CUDA Library"
        ), 
        "gpu_arch": attr.string(
            doc = "Target GPU Architecture"
        ),
        "gen_code": attr.string_list(
            doc = "Other architectures to compile PTX for"
        ),
        "defines": attr.label_list(),
        "copts": attr.string_list(),
        "linkopts": attr.string_list(),
        "nolinkopts": attr.string_list(),
        "nocopts": attr.string_list(),
        "includes": attr.string_list(), 
        "deps": attr.label_list(),
        # TODO: "linkstatic": attr.boolean(),
        # TODO: "linkshared": attr.boolean(),
        # TODO: "include_prefix": attr.string(), 
        # TODO: "strip_include_prefix": attr.string(),
        "_cc_toolchain": attr.label(default = "@bazel_tools//tools/cpp:current_cc_toolchain"),  
    },
    doc = "Builds a CUDA Binary",
    fragments = ["cpp"],
    host_fragments = ["cpp"],
    executable = True,
)
