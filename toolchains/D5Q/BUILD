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
# File: DL4AGX/toolchains/D5Q/BUILD
# Description: Pulling in QNX toolchain files from the system 
##########################################################################
package(default_visibility = ["//visibility:public"])
load("//toolchains/D5Q:D5Q_cc_toolchain_config.bzl", "D5Q_cc_toolchain_config")

D5Q_cc_toolchain_config(
    name = "D5Q_toolchain_config",
    cpu = "aarch64",
)


cc_toolchain_suite(
    name = "aarch64-unknown-nto-qnx",
    toolchains = {
        "aarch64": ":gcc-qnx-aarch64",
    },
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "gcc-qnx-aarch64",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_identifier = "aarch64-unknown-nto-qnx",
    toolchain_config = ":D5Q_toolchain_config"
)

filegroup(
    name = "gcc",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-g++",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-ar",
    ],
)

filegroup(
    name = "c++",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-c++",
    ],
)

filegroup(
    name = "c++filt",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-c++filt",
    ],
)

filegroup(
    name = "cpp",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-cpp",
    ],
)

filegroup(
    name = "dwp",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-dwp",
    ],
)

filegroup(
    name = "g++",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-g++",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-ld.gold",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "usr/bin/aarch64-unknown-nto-qnx7.0.0-as",
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "usr/lib/gcc/aarch64-unknown-nto-qnx7.0.0/5.4.0/**",
        "usr/aarch64-unknown-nto-qnx/aarch64le/**",
        "usr/aarch64-unknown-nto-qnx/aarch64le/lib/**",
        "usr/aarch64-unknown-nto-qnx/aarch64le/lib/gcc/5.4.0/**",
    ]),
)

filegroup(
    name = "compiler_components",
    srcs = [
        ":ar",
        ":as",
        ":g++",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)
