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
# File: DL4AGX/tools/nvcc/private/device_link.bzl
# Description: Subroutines to manage linking of device code for cuda rules 
##########################################################################
load("//tools/nvcc:private/utils.bzl", "CuInfo", "CuLinkingContext", "get_depset_info")

def cu_lib_device_link(ctx, name, objs, out, linkopts):
    '''
    Links up all of the obj files into a single obj file 
    '''
    cmd = "nvcc {linkopts} --compiler-options -fPIC -dlink -rdc=true {objs} -o {out}".format(linkopts=" ".join([l for l in linkopts]), 
                                                                                             objs=" ".join([obj.path for obj in objs]), 
                                                                                             out=out.path)
    ctx.actions.run_shell(
        command=cmd,
        inputs=objs, 
        outputs=[out],
        env=None,
        arguments=[],
        mnemonic="NVLINK",
        progress_message="Linking CUDA Device Code for {}".format(name),
        use_default_shell_env = True
    )

def generate_nvcc_linkopts(ctx, host_compiler):
    '''
    Get the relevant linker options and files to pass into NVCC
    '''
    # Get the specified linkopts 
    linkopts = []
    # Add in all the specified linkopts 
    for l in ctx.attr.linkopts:
        # Filter linkopts through the black list 
        if l not in ctx.attr.nolinkopts:
            linkopts.append("--linker-options " + l)
     # Add in all the specified linkopts 
    for l in ctx.attr.nvcc_linkopts:
        # Filter linkopts through the black list 
        if l not in ctx.attr.nvcc_nolinkopts:
            linkopts.append(l)
    # Set the toolchain host compilation (for cross compilation)
    linkopts.append("-ccbin " + host_compiler)
    # Get all the linkopts from the dependencies 
    depset_linkopts, objects = get_depset_info(ctx, "--linker-options -rpath,")
    return linkopts + depset_linkopts, objects


def cu_link_device_code(ctx, host_compiler, objects):
    '''
    Link CUDA object files together into a single device library
    '''
    # Get all the linkopts and files necessary for the linker
    device_linkopts, depset_objects = generate_nvcc_linkopts(ctx, host_compiler)
    objects += depset_objects

    # Object files will be placed in /bazel-out/.../[project]/_objs
    device_lib = ctx.actions.declare_file("_objs/" + ctx.label.name + "/" + ctx.label.name + "_device.pic.o")
    
    # Link with NVCC
    cu_lib_device_link(
        ctx,
        name = ctx.label.name, 
        objs = objects, 
        out = device_lib,
        linkopts = device_linkopts,
    )

    return device_lib
