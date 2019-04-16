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
# File: DL4AGX/tools/packaging/src_package.bzl
# Description: Bazel rule to package a subset of projects in the repo into
#              a tarball 
##########################################################################
def src_package(name, components, platform, pdk_version, pdk_platform, documentation="", min_bazel_version="0.21.0", visibility=None):
    component_str = ""
    dependency_lists = []
    for c in components:
        get_deps(c.split(':')[1] + "_deps_" + name , c)
        dependency_lists.append(c.split(':')[1] + "_deps_" + name)
        component_str += c + " "
    pdk = pdk_platform + ":" + pdk_version
    
    native.genrule(
        name=name,
        message="Generating Source Package for %s" % (name),
        outs=[name + ".tar.gz"],
        output_to_bindir=1,
        srcs=dependency_lists,
        cmd="$(location //tools/packaging:package) %s --platform %s --pdk %s --bazel_version %s --out %s --docs \"%s\"" % (component_str, platform, pdk, min_bazel_version, name, documentation), 
        tools=["//tools/packaging:package"],
        visibility=visibility
    )

def get_deps(name, target):
    native.genquery(
        name=name,
        expression="deps(%s)" % (target),
        scope=["%s" % (target)]
    )


    
