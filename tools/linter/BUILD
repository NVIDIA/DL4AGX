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
# File: DL4AGX/tools/linter/BUILD
##########################################################################
load("@pylinter_deps//:requirements.bzl", "requirement")

py_binary(
    name = "cpplint",
    srcs = ["cpplint.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

py_binary(
    name = "cpplint_diff",
    srcs = ["cpplint_diff.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [":utils"],
)

py_binary(
    name = "pylint",
    srcs = ["pylint.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        requirement("yapf"),
    ],
)

py_binary(
    name = "pylint_diff",
    srcs = ["pylint_diff.py"],
    default_python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":utils",
        requirement("yapf"),
    ],
)

py_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
)
