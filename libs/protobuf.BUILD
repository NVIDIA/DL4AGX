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
# File: DL4AGX/libs/protobuf.BUILD
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
    name = "protobuf_headers",
    hdrs = glob([
        "include/google/protobuf/**/*.h",
        "include/google/protobuf/**/*.inc",
    ]),
    includes = ["include/google/protobuf"],
    visibility = ["//visibility:private"],
)

################################ PROTOBUF LITE ########################################
cc_import(
    name = "protobuf_lite_lib",
    shared_library = "lib/libprotobuf-lite.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "protobuf_lite",
    deps = ["protobuf_headers",
            "protobuf_lite_lib"],
    copts = select({
        ":aarch64_linux": ["-DGOOGLE_PROTOBUF_ARCH_64_BIT"],
        ":aarch64_qnx": [
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-Dgoogle = google_private"
        ],
        "//conditions:default": [
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-DGOOGLE_PROTOBUF_ARCH_X64"
        ]
    })
)
################################ PROTOBUF ########################################
cc_import(
    name = "protobuf_lib",
    shared_library = "lib/libprotobuf.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "protobuf",
    deps = ["protobuf_headers",
            "protobuf_lib"],
    copts = select({
        ":aarch64_linux": ["-DGOOGLE_PROTOBUF_ARCH_64_BIT"],
        ":aarch64_qnx": [
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-Dgoogle = google_private"
        ],
        "//conditions:default": [
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-DGOOGLE_PROTOBUF_ARCH_X64"
        ]
    })
)

################################ PROTOC ########################################
cc_import(
    name = "protoc_lib",
    shared_library = "lib/libprotoc.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "protoc",
    deps = ["protobuf_headers",
            "protoc_lib"],
    copts = select({
        ":aarch64_linux": ["-DGOOGLE_PROTOBUF_ARCH_64_BIT"],
        ":aarch64_qnx": [
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-Dgoogle = google_private"
        ],
        "//conditions:default": [
            "-DGOOGLE_PROTOBUF_ARCH_64_BIT",
            "-DGOOGLE_PROTOBUF_ARCH_X64"
        ]
    })
)
