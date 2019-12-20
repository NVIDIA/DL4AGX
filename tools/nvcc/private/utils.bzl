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
# File: DL4AGX/tools/nvcc/private/utils.bzl
# Description: Utilities to help implement cuda rules
##########################################################################
load("@bazel_skylib//lib:shell.bzl", "shell")
load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain")
load(
    "@bazel_tools//tools/build_defs/cc:action_names.bzl",
    "CPP_COMPILE_ACTION_NAME",
    "CPP_LINK_DYNAMIC_LIBRARY_ACTION_NAME",
    "C_COMPILE_ACTION_NAME",
    "CPP_LINK_EXECUTABLE_ACTION_NAME",
    "CPP_LINK_STATIC_LIBRARY_ACTION_NAME"
)

load("//tools/nvcc:private/constants.bzl", CUDA_ACCEPTABLE_SRC_EXTENSIONS = "CUDA_ACCEPTABLE_SRC_EXTENSIONS", 
                                           CUDA_ACCEPTABLE_BIN_EXTENSIONS = "CUDA_ACCEPTABLE_BIN_EXTENSIONS", 
                                           CUDA_ACCEPTABLE_HDR_EXTENSIONS = "CUDA_ACCEPTABLE_HDR_EXTENSIONS")


'''
Providers for CUDA rules 
Cuda rules can consume CC rules and their providers but preference is given to CUDA provdiers 
TODO: Move to using CC providers entirely 
'''
CuLinkingContext = provider(fields = ["objects", "link_flags"])
CuInfo = provider(fields = ["compilation_context", "cu_linking_context"])


def get_toolchain_info(ctx):
    '''
    Get information about the C++ toolchain to back NVCC, used to support cross compiling 
    '''
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )
    compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = ctx.fragments.cpp.copts,
    )
    compiler_options = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = C_COMPILE_ACTION_NAME,
        variables = compile_variables,
    )
    compiler = str(
        cc_common.get_tool_for_action(
            feature_configuration = feature_configuration,
            action_name = C_COMPILE_ACTION_NAME,
        ),
    )
    exec_linker = str(
        cc_common.get_tool_for_action(
            feature_configuration = feature_configuration,
            action_name = CPP_LINK_EXECUTABLE_ACTION_NAME,
        ),
    )
    lib_linker = str(
        cc_common.get_tool_for_action(
            feature_configuration = feature_configuration,
            action_name = CPP_LINK_STATIC_LIBRARY_ACTION_NAME,
        ),
    )
    return cc_toolchain, compiler, exec_linker, lib_linker


def get_depset_info(ctx, flag_prefix):
    linkopts = []
    objects = []
    for d in ctx.attr.deps:
        # This implementation prefer CuInfo providers but fall back on CcInfo ones if there is no CuInfo providers
        if CuInfo in d: 
            # Add object files and linkopts 
            objects += d[CuInfo].cu_linking_context.objects.to_list()
            # TODO: Should these be filtered by ctx.attr.nolinkopts
            linkopts += list(d[CuInfo].cu_linking_context.link_flags)
        # TODO: Potentially move to full CcInfo
        elif CcInfo in d:
            for lib in d[CcInfo].linking_context.libraries_to_link.to_list():
                # TODO: Verify this ordering of preference 
                # Prefers pic static libraries, static libraries then dynamically libraries
                # Appends the appropriate linker options to the list of linkopts and the file 
                # to objects so it will get moved into the sandbox when the linker is invoked   
                if lib.pic_static_library != None:
                    linkopts.append(flag_prefix + lib.pic_static_library.dirname 
                                            + " -L" + lib.pic_static_library.dirname 
                                            + " -l" + lib.pic_static_library.basename.split("lib")[1].split(".")[0])
                    objects.append(lib.pic_static_library)
                elif lib.static_library != None:
                    linkopts.append(flag_prefix + lib.static_library.dirname 
                                                + " -L" + lib.static_library.dirname 
                                                + " -l" + lib.static_library.basename.split("lib")[1].split(".")[0])
                    objects.append(lib.static_library)
                elif lib.dynamic_library != None:
                    linkopts.append(flag_prefix + lib.dynamic_library.dirname 
                                                + " -L" + lib.dynamic_library.dirname 
                                                + " -l" + lib.dynamic_library.basename.split("lib")[1].split(".")[0])
                    objects.append(lib.dynamic_library)
    return linkopts, objects

###################################################################
# The following code was taken from the rules_go repo the modified (Apache License)
###################################################################

# Copyright 2014 The Bazel Authors. All rights reserved.
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

def as_iterable(v):
    if type(v) == "list":
        return v
    if type(v) == "tuple":
        return v
    if type(v) == "depset":
        return v.to_list()
    fail("Cannot convert {} to an iterable type".format(v))

def split_srcs(srcs):
    sources = struct(
        cu = [],
        headers = [],
        bin = []
    )
    ext_pairs = (
        (sources.cu, CUDA_ACCEPTABLE_SRC_EXTENSIONS),
        (sources.headers, CUDA_ACCEPTABLE_HDR_EXTENSIONS),
        (sources.bin, CUDA_ACCEPTABLE_BIN_EXTENSIONS),
    )
    extmap = {}
    for outs, exts in ext_pairs:
        for ext in exts:
            ext = ext[1:]  # strip the dot
            if ext in extmap:
                break
            extmap[ext] = outs
    for src in as_iterable(srcs):
        extouts = extmap.get(src.extension)
        if extouts == None:
            fail("Unknown source type {0}".format(src.basename))
        extouts.append(src)
    return sources

def join_srcs(source):
    return source.cu + source.headers + source.bin
