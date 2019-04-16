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
# File: DL4AGX/tools/linter/pylint.py
# Description: Lint files using yapf and overwrite with suggested formating 
##########################################################################
import os
import sys
import glob
import utils
import subprocess
import yapf

VALID_PY_FILE_TYPES = [".py"]


def lint(user, target_files, conf, change_file=True):
    return yapf.FormatFiles(
        filenames=target_files,
        lines=None,
        style_config=conf,
        no_local_style=None,
        in_place=change_file,
        print_diff=False,
        verify=True,
        parallel=True,
        verbose=True)


if __name__ == "__main__":
    BAZEL_ROOT = utils.find_bazel_root()
    STYLE_CONF_PATH = BAZEL_ROOT + "/.style.yapf"
    USER = BAZEL_ROOT.split('/')[2]
    subprocess.run(["useradd", USER])
    projects = utils.CHECK_PROJECTS(sys.argv[1:])
    if "//..." in projects:
        projects = [p.replace(BAZEL_ROOT, "/")[:-1] for p in glob.glob(BAZEL_ROOT + '/*/')]
        projects = [p for p in projects if p not in utils.BLACKLISTED_BAZEL_TARGETS]

    for p in projects:
        if p.endswith("/..."):
            p = p[:-4]
        path = BAZEL_ROOT + '/' + p[2:]
        files = utils.glob_files(path, VALID_PY_FILE_TYPES)
        if files != []:
            changed = lint(USER, files, STYLE_CONF_PATH)
            if changed:
                print(
                    "\033[93mWARNING:\033[0m This command modified your files with the recommended linting, you should review the changes before committing"
                )
                sys.exit(1)
