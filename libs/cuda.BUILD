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
# File: DL4AGX/libs/cuda.BUILD
############################################################################
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cuda",
    srcs = glob([
        "lib/**/*libcuda.so",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cudart",
    srcs = glob([
        "lib/**/*libcudart.so",
    ]),
    hdrs = glob([
        "include/**/*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)

cc_library(
    name = "cublas",
    srcs = glob([
        "lib/**/*libcublas.so",
    ]),
    hdrs = glob([
        "include/**/*cublas*.h",
        "include/**/*.hpp",
        "include/**/*.inl",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)


# We can either create an entry for every library and its
# header files or we can make groups of cuda dependencies
# on a per project basis (what I am doing right now) - Naren
cc_library(
    name = "cuda_dali_deps",
    srcs = glob([
        "lib/**/*libcudart.so",
        "lib/**/*libcublas.so",
        "lib/**/*libnppc.so",
        "lib/**/*libnpps.so",
    ]),
    hdrs = glob([
        "include/**/*cuda_runtime_api.h",
        "include/**/*cublas_v2.h",
    ]),
    includes = ["include/"],
    linkopts = ["-Wl,-rpath,lib/"],
)


