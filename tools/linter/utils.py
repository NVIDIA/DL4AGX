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
# File: DL4AGX/tools/linter/utils.py
# Description: Utils for navigating bazel in python 
##########################################################################
import os
import sys
import glob
import subprocess

BLACKLISTED_BAZEL_TARGETS = [
    "//external", "//tools", "//libs", "//docker", "//toolchains", "//third_party", "//bazel-bin", "//bazel-genfiles",
    "//bazel-out", "//bazel-DL4AGX", "//bazel-testlogs"
]


def CHECK_PROJECTS(projs):
    for p in projs:
        if p[:2] != "//":
            sys.exit(p + " is not a valid bazel target")
    return projs


def find_bazel_root():
    """
    Finds the root directory of the bazel space
    """
    curdir = os.path.dirname(os.path.realpath(__file__))
    while 1:
        if os.path.exists(curdir + "/WORKSPACE"):
            return curdir
        if curdir == "/":
            sys.exit("Error: was unable to find a bazel workspace")
        curdir = os.path.dirname(curdir)


def glob_files(project, file_types):
    files = []
    for t in file_types:
        files += glob.glob(project + "/**/*" + t, recursive=True)
    return files
