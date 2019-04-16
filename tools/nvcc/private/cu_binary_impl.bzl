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
# File: DL4AGX/tools/nvcc/private/cu_binary_impl.bzl
# Description: Implementation of the cu_binary rule 
##########################################################################

load("//tools/nvcc:private/constants.bzl", "CUDA_ACCEPTABLE_BIN_EXTENSIONS", "CUDA_ACCEPTABLE_SRC_EXTENSIONS", "CUDA_ACCEPTABLE_HDR_EXTENSIONS")
load("//tools/nvcc:private/utils.bzl", "get_toolchain_info", "split_srcs", "CuInfo", "CuLinkingContext")
load("//tools/nvcc:private/compile.bzl", "cu_compile_sources")
load("//tools/nvcc:private/device_link.bzl", "cu_link_device_code")

def cu_bin_link(ctx, name, objs, out, linkopts):
    '''
    Links up all of the obj files into a executable binary 
    '''
    cmd = "nvcc {objs} {linkopts} -o {out}".format(linkopts=" ".join([l for l in linkopts]), 
                                                   objs=" ".join([obj.path for obj in objs]), 
                                                   out=out.path)

    ctx.actions.run_shell(
        command=cmd,
        inputs=objs, 
        outputs=[out],
        env=None,
        arguments=[],
        mnemonic="NVLINK",
        progress_message="Linking CUDA executable {}".format(name),
        use_default_shell_env = True,
    )

def generate_bin_linkopts(ctx):
    '''
    Generate the linkopts for the linking phase
    '''
    linkopts = []
    objects = []
    # Get all the passed in linkopts
    for l in ctx.attr.linkopts:
        # Filter through the black list 
        if l not in ctx.attr.nolinkopts:
            linkopts.append("--linker-options " + l)
    # Wrap depset arguments in groups 
    linkopts.append("--linker-options --start-group")
    for d in ctx.attr.deps:
        # Prefers CuInfo Providers
        if CuInfo in d: 
            objects += list(d[CuInfo].cu_linking_context.objects)
            linkopts += list(d[CuInfo].cu_linking_context.link_flags)
        elif CcInfo in d:
             for lib in d[CcInfo].linking_context.libraries_to_link:
                # TODO: Verify this ordering of preference 
                # Prefers pic static libraries, static libraries then dynamically libraries
                # Appends the appropriate linker options to the list of linkopts and the file 
                # to objects so it will get moved into the sandbox when the linker is invoked   
                if lib.pic_static_library != None:
                    linkopts.append("--linker-options -rpath," + lib.pic_static_library.dirname 
                                            + " -L" + lib.pic_static_library.dirname 
                                            + " -l" + lib.pic_static_library.basename.split("lib")[1].split(".")[0])
                    objects.append(lib.pic_static_library)
                elif lib.static_library != None:
                    linkopts.append("--linker-options -rpath," + lib.static_library.dirname 
                                                + " -L" + lib.static_library.dirname 
                                                + " -l" + lib.static_library.basename.split("lib")[1].split(".")[0])
                    objects.append(lib.static_library)
                elif lib.dynamic_library != None:
                    linkopts.append("--linker-options -rpath," + lib.dynamic_library.dirname 
                                                + " -L" + lib.dynamic_library.dirname 
                                                + " -l" + lib.dynamic_library.basename.split("lib")[1].split(".")[0])
                    objects.append(lib.dynamic_library)
    linkopts.append("--linker-options --end-group")
    return linkopts, objects


def cu_binary_impl(ctx):
    '''
    Compiles a CUDA Binary 
    '''
    # Get the info about the toolchain - Set by the CROSSTOOL
    cc_toolchain, compiler, exec_linker, lib_linker = get_toolchain_info(ctx)
    # Compile the source code into objects
    objects, _, _, _ = cu_compile_sources(ctx, compiler)
    # Add in any passed in binary files to the list of objects to link 
    objects += [o for o in ctx.files.srcs if o.extension in CUDA_ACCEPTABLE_BIN_EXTENSIONS]
    # Link the CUDA code into a single device library 
    device_lib = cu_link_device_code(ctx, compiler, objects)
    # Add the new device library to the list of objects
    objects.append(device_lib)

    # Generate all the linkopts necessary for linking 
    linkopts, dep_objects = generate_bin_linkopts(ctx)
    linkopts.append("-ccbin " + compiler)
    objects += dep_objects

    # Declare the executable 
    executable = ctx.actions.declare_file(ctx.label.name)
    cu_bin_link(
        ctx,
        name = ctx.label.name, 
        objs = objects, 
        out = executable,
        linkopts = linkopts,
    )

    return [DefaultInfo(
        files = depset([executable]),
        executable = executable,
    )]
