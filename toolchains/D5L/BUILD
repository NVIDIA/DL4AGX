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
# File: DL4AGX/toolchains/D5L/BUILD
# Description: Pulling in aarch64 toolchain files from the system 
############################################################################
package(default_visibility = ["//visibility:public"])
load("//toolchains/D5L:D5L_cc_toolchain_config.bzl", "D5L_cc_toolchain_config")

config_setting(
    name = "gcc_7.4",
    values = {"define": "platforms=drive_pdk_5.1.6.0+linux"}
)

config_setting(
    name = "gcc_5.4",
    values = {"define": "platforms=drive_pdk_5.1.3.0+linux"}
)

D5L_cc_toolchain_config(
    name = "D5L_toolchain_config",
    cpu = "aarch64",
    gcc_toolchain_version = select({
        ":gcc_5.4": "gcc5.4",
        "//conditions:default": "gcc7.4",
    })
)

cc_toolchain_suite(
    name = "aarch64-unknown-linux-gnu",
    toolchains = {
        "aarch64": ":gcc-linux-aarch64",
    },
)

filegroup(
    name = "empty",
    srcs = [],
)

cc_toolchain(
    name = "gcc-linux-aarch64",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = "empty",
    strip_files = "empty",
    supports_param_files = 1,
    toolchain_identifier = "aarch64-unknown-linux-gnu",
    toolchain_config = ":D5L_toolchain_config"
)

filegroup(
    name = "gcc",
    srcs = [
        "usr/bin/aarch64-linux-gnu-gcc",
    ],
)

filegroup(
    name = "ar",
    srcs = [
        "usr/bin/aarch64-linux-gnu-ar",
    ],
)

filegroup(
    name = "ld",
    srcs = [
        "usr/bin/aarch64-linux-gnu-ld.gold",
    ],
)

filegroup(
    name = "nm",
    srcs = [
        "usr/bin/aarch64-linux-gnu-nm",
    ],
)

filegroup(
    name = "objcopy",
    srcs = [
        "usr/bin/aarch64-linux-gnu-objcopy",
    ],
)

filegroup(
    name = "objdump",
    srcs = [
        "usr/bin/aarch64-linux-gnu-objdump",
    ],
)

filegroup(
    name = "strip",
    srcs = [
        "usr/bin/aarch64-linux-gnu-strip",
    ],
)

filegroup(
    name = "as",
    srcs = [
        "usr/bin/aarch64-linux-gnu-as",
    ],
)

filegroup(
    name = "compiler_pieces",
    srcs = glob([
        "usr/lib/gcc-cross/aarch64-linux-gnu/5/**",
        "usr/aarch64-linux-gnu/**",
    ]),
)

filegroup(
    name = "compiler_components",
    srcs = [
        ":ar",
        ":as",
        ":gcc",
        ":ld",
        ":nm",
        ":objcopy",
        ":objdump",
        ":strip",
    ],
)
