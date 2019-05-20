#!/usr/bin/python3
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
# File: DL4AGX/tools/linter/cpplint_diff.py
# Description: Lint files with clang format and print the differences 
##########################################################################
import os
import sys
import glob
import subprocess
import utils

VALID_CPP_FILE_TYPES = [".cpp", ".cc", ".c", ".cu", ".hpp", ".h", ".cuh"]


def lint(target_files):
    failure = False
    for f in target_files:
        with open("/tmp/changes.txt", "w") as changes:
            subprocess.run(['clang-format', f], stdout=changes)
        output = subprocess.run(["git", "diff", "-u", "--color", f, "/tmp/changes.txt"])
        if output.returncode != 0:
            failure = True
    return failure


if __name__ == "__main__":
    BAZEL_ROOT = utils.find_bazel_root()
    projects = utils.CHECK_PROJECTS(sys.argv[1:])
    if "//..." in projects:
        projects = [p.replace(BAZEL_ROOT, "/")[:-1] for p in glob.glob(BAZEL_ROOT + '/*/')]
        projects = [p for p in projects if p not in utils.BLACKLISTED_BAZEL_TARGETS]

    failure = False
    for p in projects:
        if p.endswith("/..."):
            p = p[:-4]
        path = BAZEL_ROOT + '/' + p[2:]
        files = utils.glob_files(path, VALID_CPP_FILE_TYPES)
        if files != []:
            failure = lint(files)
    if failure:
        print("\033[91mERROR:\033[0m Some files do not conform to style guidelines")
        sys.exit(1)
