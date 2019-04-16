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
# File: DL4AGX/tools/nvcc/private/cu_library_impl.bzl
# Description: Implementation of the cu_library rule 
##########################################################################

load("//tools/nvcc:private/constants.bzl", "CUDA_ACCEPTABLE_BIN_EXTENSIONS", "CUDA_ACCEPTABLE_SRC_EXTENSIONS", "CUDA_ACCEPTABLE_HDR_EXTENSIONS")
load("//tools/nvcc:private/utils.bzl", "get_toolchain_info", "get_depset_info", "split_srcs", "CuInfo", "CuLinkingContext")
load("//tools/nvcc:private/compile.bzl", "cu_compile_sources")
load("//tools/nvcc:private/device_link.bzl", "cu_link_device_code")

def cu_lib_link_shared(ctx, name, linker, objs, out, linkopts):
    '''
    Links up all of the obj files into a .so file 
    '''
    cmd = "{linker} -shared -o {out} {linkopts} -fPIC {objs} ".format(linker=linker,
                                                                      linkopts=" ".join([l for l in linkopts]), 
                                                                      objs=" ".join([obj.path for obj in objs]), 
                                                                      out=out.path)
    ctx.actions.run_shell(
        command=cmd,
        inputs=objs, 
        outputs=[out],
        env=None,
        arguments=[],
        mnemonic="CUDAHostLD",
        progress_message="Linking CUDA code into shared library {}".format(name),
        use_default_shell_env = True
    )

def cu_lib_link_static(ctx, name, linker, objs, out, linkopts):
    '''
    Links up all of the obj files into a .a file 
    '''
    cmd = "{linker} crf {out} {linkopts} {objs} ".format(linker=linker,  
                                                         linkopts=" ",#.join([l for l in linkopts]), 
                                                         objs=" ".join([obj.path for obj in objs]), 
                                                         out=out.path)
    ctx.actions.run_shell(
        command=cmd,
        inputs=objs, 
        outputs=[out],
        env=None,
        arguments=[],
        mnemonic="CUDAHostAR",
        progress_message="Linking CUDA code into static library {}".format(name),
        use_default_shell_env = True
    )

def generate_host_compiler_linkopts(ctx):
    '''
    Generate the linkopts for the linking phase
    '''
    linkopts = ["-Wl,-rpath,lib/", "-Wl,-S","-Wl,-no-as-needed","-Wl,-z,relro,-z,now","-pass-exit-codes"]
    # Append the same linker options to commands for the host linker 
    # TODO: Is this correct?
    for l in ctx.attr.linkopts:
        if l not in ctx.attr.nolinkopts:
            linkopts.append(l)
    linkopts += get_depset_info(ctx, "-Wl,-rpath,")[0] #just the linkopts
    return linkopts

def cu_library_impl (ctx):
    '''
    Compiles a CUDA library to a static and shared library 
    Will propogate a set of lib files, header files, includes and defines  
    '''
    # Get the info about the toolchain - Set by the CROSSTOOL
    cc_toolchain, compiler, exec_linker, lib_linker = get_toolchain_info(ctx)
    # Compile the source code into objects
    objects, headers, includes, defines = cu_compile_sources(ctx, compiler)
    # Add in any passed in binary files to the list of objects to link 
    objects += [o for o in ctx.files.srcs if o.extension in CUDA_ACCEPTABLE_BIN_EXTENSIONS]
    # Link the CUDA code into a single device library 
    device_lib = cu_link_device_code(ctx, compiler, objects)
    # Add the new device library to the list of objects
    objects.append(device_lib)

    # Generate the linker options for the host linker
    host_linkopts = generate_host_compiler_linkopts(ctx)
    # Declare a static and a shared library file 
    lib_name = ctx.label.name
    if lib_name[:3] != "lib":
        lib_name = "lib" + lib_name 
    static_lib_name = lib_name + ".a"
    shared_lib_name = lib_name + ".so"
    static_lib = ctx.actions.declare_file(static_lib_name)
    shared_lib = ctx.actions.declare_file(shared_lib_name)

    # Link all objects into a shared object file
    cu_lib_link_shared(
        ctx,
        name = ctx.label.name,
        linker = exec_linker, 
        objs = objects, 
        out = shared_lib,
        linkopts = host_linkopts,
    )

    # Link all objects into a static library as well
    cu_lib_link_static(
        ctx,
        name = ctx.label.name,
        linker = lib_linker,
        objs = objects, 
        out = static_lib,
        linkopts = host_linkopts,
    )

    # Collect the feature configuration to populate the CcInfo provider
    feature_configuration = cc_common.configure_features(
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    # Create both a CuInfo Provider and a CcInfo Provider to support 
    # cu_binary and the cc targets
    return [CuInfo(
                compilation_context = cc_common.create_compilation_context(
                    defines = depset(defines),
                    headers = depset(headers),
                    includes = depset(includes),
                    quote_includes = depset([]),
                    system_includes = depset([]),
                ),
                cu_linking_context = CuLinkingContext(
                    objects = depset(objects),
                    link_flags = [], 
                )
            ), CcInfo(
                compilation_context = cc_common.create_compilation_context(
                    defines = depset(defines),
                    headers = depset(headers),
                    includes = depset(includes),
                    quote_includes = depset([]),
                    system_includes = depset([]),
                ),
                linking_context= cc_common.create_linking_context(
                    libraries_to_link = [cc_common.create_library_to_link(
                        actions=ctx.actions,
                        feature_configuration=feature_configuration,
                        cc_toolchain=cc_toolchain, 
                        alwayslink = False,
                        dynamic_library = shared_lib, 
                        pic_static_library = static_lib, 
                    )],
                    user_link_flags = host_linkopts
                )
            ), DefaultInfo(
                files = depset([shared_lib, static_lib] + ctx.files.hdrs)
            )]