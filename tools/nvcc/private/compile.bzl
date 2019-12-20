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
# File: DL4AGX/tools/nvcc/private/compile.bzl
# Description: Common code to compile cuda projects before linking  
##########################################################################
load("//tools/nvcc:private/utils.bzl", "split_srcs")

def cu_compile(ctx, name, srcs_outs, includes, defines, hdrs, copts):
    '''
    Compiles each source file to an obj file
    '''
    for s,o in srcs_outs:
        cmd = "nvcc -x cu --compiler-options -fPIC {copts} -I. {includes} {defines} -rdc=true -c -dc {src} -o {out}".format(copts=" ".join([c for c in copts]), 
		                                                                                                            includes=" ".join(["-I" + i + " " for i in includes]),
                   		                                                                                            defines=" ".join(["-D" + d +  " " for d in defines]),
                                      		                                                                            src=s.path,
                                                                                                                            out=o.path)
        ctx.actions.run_shell(
            command=cmd,
            inputs=[s] + hdrs, 
            outputs=[o],
            env=None,
            arguments=[],
            mnemonic="NVCC",
            progress_message="Compiling CUDA Source code for  {}".format(name),
            use_default_shell_env = True
        )

def generate_copts(ctx, compiler):
    '''
    Generate the copts for the compiling phase
    '''
    # Standard flags
    copts = ['--compiler-options -fdiagnostics-color=always',
             '--compiler-options -fno-canonical-system-headers',
             '--compiler-options -Wno-builtin-macro-redefined']
    # Get the target gpu architecture
    arch = "--gpu-architecture=" + ctx.attr.gpu_arch
    copts.append(arch)
    # Other supported architectures 
    for a in ctx.attr.gen_code:
        copts.append("-gencode=" + a)
    # Append any other passed copts (if they are not black listed)
    for c in ctx.attr.copts:
        if c not in ctx.attr.nocopts:
            copts.append("--compiler-options " + c)
    # Append any other passed copts (if they are not black listed)
    for c in ctx.attr.nvcc_copts:
        if c not in ctx.attr.nvcc_nocopts:
            copts.append(c)
    # Set the host compiler (i.e. cross-compiler)
    # compiler is determined by local cc_toolchain
    copts.append("-ccbin " + compiler)
    return copts


def cu_compile_sources(ctx, compiler):
    '''
    Compiles a set of files to object files
    Takes a bazel context and returns object files, headers, includes and defines 
    '''
    # Generate all of the applicable copts and select the compiler (for cross compiling)
    copts = generate_copts(ctx, compiler)
    # Split up source files into classes (src, headers, bin)
    src_files = split_srcs(ctx.files.srcs)
    # Parse dependency list
    deps = []
    defines = ['__DATE__="redacted"', '__TIMESTAMP__="redacted"', '__TIME__="redacted"'] + ctx.attr.defines
    includes = [] + ctx.attr.includes
    for d in ctx.attr.deps:
        if CcInfo in d:
            deps += d[CcInfo].compilation_context.headers.to_list()
            defines += d[CcInfo].compilation_context.defines.to_list()
            includes += d[CcInfo].compilation_context.includes.to_list()
    # Split up the deps in the same way 
    dep_files = split_srcs(deps)

    # Object files will be placed in /bazel-out/.../[project]/_objs
    prefix = "_objs/" + ctx.label.name + "/"
    source_objects_pairs = [(s, ctx.actions.declare_file(prefix + s.basename.split('.')[0] + ".pic.o")) for s in src_files.cu]

    # Combine the list of headers from the target with headers from dependencies 
    headers = src_files.headers + dep_files.headers
    if hasattr(ctx.files, "hdrs"): 
        headers += ctx.files.hdrs

    cu_compile(
        ctx,
        name = ctx.label.name, 
        srcs_outs = source_objects_pairs,
        includes = includes,
        defines = defines, 
        hdrs = headers,  
        copts = copts,
    )

    # Pull out the object files from the compilation stage 
    objects = [so[1] for so in source_objects_pairs]
    # Add the binary objects from the dependencies 
    objects += dep_files.bin[::-1]
    return objects, headers, includes, defines
