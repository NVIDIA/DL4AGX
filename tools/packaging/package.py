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
# File: DL4AGX/tools/packaging/package.py
# Description: Script to package a subset of the repo into a tarball 
##########################################################################
import os
import sys
import glob
import shutil
from distutils.dir_util import copy_tree
from distutils.errors import DistutilsFileError
import errno
import argparse
import tarfile
from string import Template

# The files that will be necessary in every tarball
REQUIRED_FILES = set(["//WORKSPACE", "//.bazelrc", "//toolchains", "//tools", "//libs"])

# Bazel targets that should not be copied
BLACKLISTED_BAZEL_TARGETS = ["//external"]

PARSER = argparse.ArgumentParser(description="Package a subset of the repo")
PARSER.add_argument("components", nargs="+", help="Targets to include")
PARSER.add_argument("--platform", nargs="+", help="Platform")
PARSER.add_argument(
    "-p",
    "--pdk",
    required=True,
    help="Specify the pdk version and platform to target <platform>-<pdk version>\n\
    Supported platforms: qnx, aarch64-linux, both")
PARSER.add_argument("-b", "--bazel_version", required=True, help="Specify the bazel version")
PARSER.add_argument("-d", "--docs", help="additional docs to include in the readme", nargs="?", default="")
PARSER.add_argument("-o", "--out", help="custom output file name for the tarball")
ARGS = PARSER.parse_args()


def copy(src, dest):
    """
    Copy directories and files
    """
    try:
        copy_tree(src, dest)
    except DistutilsFileError:
        shutil.copy(src, dest)
    except Exception as e:
        print('Directory not copied. Error: %s' % e)


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


def determine_component_deps(bazel_bin, name, component_name):
    """
    Takes a bazel target and determines all local dependencies 
    """
    component_deps = set()
    with open(bazel_bin + '/' + component_name.split(":")[1] + "_deps_" + name, 'r') as dep_file:
        content = dep_file.readlines()

    content = [x.strip() for x in content]
    for dep_str in content:
        if dep_str[0] is not '@':  #Do not want external dependencies
            dep = dep_str.strip().split(":")[0]
            if dep not in BLACKLISTED_BAZEL_TARGETS:
                component_deps.add(dep)

    return component_deps


def toolchain_command(platform):
    """
    Generate the correct toolchain commands to insert into the README
    """
    if platform == "qnx":
        return "dazel build //exampleApp --config=\"D5Q-toolchain\" #Compile with QNX Toolchain"
    elif platform == "aarch64-linux":
        return "dazel build //exampleApp --config=\"D5L-toolchain\" #Compile with aarch64 Toolchain"
    return "dazel build //exampleApp --config=\"D5Q-toolchain\" #Compile with QNX Toolchain\ndazel build //exampleApp --config=\"D5L-toolchain\" #Compile with aarch64 Toolchain"


def format_toc(toc):
    toc_str = "## Included Components\n"
    for s in toc:
        toc_str += "- {}\n".format(s)
    return toc_str


def fill_out_readme(readme_tpl_file, title, toc, platform, other=""):
    """
    Fill out the README template
    """
    fields = {
        'title': title,
        'toc': format_toc(toc),
        'platform': platform,
        'toolchain_command': toolchain_command(platform),
        'component_specific_instructions': other
    }

    readme_template = Template(readme_tpl_file.read())
    return readme_template.substitute(fields)


def fill_out_dockerfile_dazel(dockerfile_tpl_file, version):
    """
    Fill out the README template
    """
    fields = {'BAZEL_VERSION': version}

    dockerfile_template = Template(dockerfile_tpl_file.read())
    return dockerfile_template.substitute(fields)


def select_dockerfile(platform, os, version):
    '''
    Set up the docker base containers
    '''
    dockerfile_name = "Dockerfile.{}.{}".format(os, version)
    dockerfile_path = BAZEL_ROOT + "/docker/" + platform + "/" + dockerfile_name

    if not os.path.exists(STAGING_AREA + "/docker"):
        os.mkdir(STAGING_AREA + "/docker")

    copy(dockerfile_path, STAGING_AREA + "/docker")


if __name__ == "__main__":
    #Find the bazel root
    BAZEL_ROOT = find_bazel_root()
    os.chdir(BAZEL_ROOT)
    BAZEL_GENFILES = BAZEL_ROOT + "/bazel-genfiles"
    BAZEL_BIN = BAZEL_ROOT + "/bazel-bin"

    #Name of the package
    OUTFILE_NAME = ARGS.out if ARGS.out else "DL4AGX"

    #Location to assemble the tarball (create if does not exist)
    STAGING_AREA = BAZEL_GENFILES + '/' + OUTFILE_NAME
    if not os.path.exists(STAGING_AREA):
        os.mkdir(STAGING_AREA)

    #Figure out all of the directories needed
    required_targets = set()
    table_of_contents = []
    for component in ARGS.components:
        deps = determine_component_deps(BAZEL_BIN, OUTFILE_NAME, component)
        required_targets = required_targets.union(deps)
        table_of_contents.append(component.split(":")[0][2:])
    required_targets = REQUIRED_FILES.union(required_targets)
    required_dirs = [d[1:] for d in required_targets]
    for d in required_dirs:
        copy(BAZEL_ROOT + d, STAGING_AREA + d)

    PLATFORM = ARGS.platform
    #Figure out the specific PDK version and platform
    PDK_PLATFORM = ARGS.pdk.split(':')[0]
    PDK_VERSION = ARGS.pdk.split(':')[1]

    #Use PDK version and platform to select dockerfile
    if PDK_PLATFORM == 'both':
        select_dockerfile(PLATFORM, "qnx", PDK_VERSION)
        select_dockerfile(PLATFORM, "aarch64-linux", PDK_VERSION)
    else:
        select_dockerfile(PLATFORM, PDK_PLATFORM, PDK_VERSION)

    copy(BAZEL_ROOT + "/docker/README.md", STAGING_AREA + "/docker")
    copy(BAZEL_ROOT + "/tools/packaging/.dazelrc", STAGING_AREA)

    #Fill out the readme template
    readme_tpl = open(BAZEL_ROOT + "/tools/packaging/README.md.tpl")
    readme = fill_out_readme(readme_tpl, OUTFILE_NAME, table_of_contents, PDK_PLATFORM, ARGS.docs if ARGS.docs else "")

    with open(STAGING_AREA + "/README.md", "w") as f:
        f.write(readme)

    #Fill out the dockerfile template
    dockerfile_dazel_tpl = open(BAZEL_ROOT + "/tools/packaging/Dockerfile.dazel.tpl")
    dockerfile_dazel = fill_out_dockerfile_dazel(dockerfile_dazel_tpl, ARGS.bazel_version)

    with open(STAGING_AREA + "/Dockerfile.dazel", "w") as f:
        f.write(dockerfile_dazel)

    # Remove code owner files
    for f in glob.glob(STAGING_AREA + "/**/OWNERS"):
        os.remove(f)

    # Remove dazel_run files
    for f in glob.glob(STAGING_AREA + "/**/.dazel_run"):
        os.remove(f)

    with tarfile.open(BAZEL_BIN + '/' + OUTFILE_NAME + ".tar.gz", "w:gz") as tar:
        tar.add(STAGING_AREA, arcname=os.path.basename(STAGING_AREA))

    shutil.rmtree(STAGING_AREA)
    for f in glob.glob(BAZEL_BIN + "/*deps*"):
        os.remove(f)
