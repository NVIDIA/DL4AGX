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
# File: DL4AGX/toolchains/D5L/D5L_cc_toolchain_config.bzl
# Description: Create a aarch64 toolchain spec mapped to --config=[L4T/D5L]-toolchain 
############################################################################
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "action_config",
    "artifact_name_pattern",
    "env_entry",
    "env_set",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
    "make_variable",
    "tool",
    "tool_path",
    "variable_with_value",
    "with_feature_set",
)
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

def _impl(ctx):
    toolchain_identifier = "aarch64-unknown-linux-gnu"

    host_system_name = "x86_64-unknown-linux-gnu"

    target_system_name = "aarch64"

    target_cpu = "aarch64"

    target_libc = "aarch64"

    compiler = "compiler"

    abi_version = "aarch64"

    abi_libc_version = "aarch64"

    cc_target_os = None

    builtin_sysroot = None

    all_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.clif_match,
        ACTION_NAMES.lto_backend,
    ]

    all_cpp_compile_actions = [
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.clif_match,
    ]

    preprocessor_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_header_parsing,
        ACTION_NAMES.cpp_module_compile,
        ACTION_NAMES.clif_match,
    ]

    codegen_compile_actions = [
        ACTION_NAMES.c_compile,
        ACTION_NAMES.cpp_compile,
        ACTION_NAMES.linkstamp_compile,
        ACTION_NAMES.assemble,
        ACTION_NAMES.preprocess_assemble,
        ACTION_NAMES.cpp_module_codegen,
        ACTION_NAMES.lto_backend,
    ]

    all_link_actions = [
        ACTION_NAMES.cpp_link_executable,
        ACTION_NAMES.cpp_link_dynamic_library,
        ACTION_NAMES.cpp_link_nodeps_dynamic_library,
    ]

    action_configs = []

    coverage_feature = feature(
        name = "coverage",
        flag_sets = [
            flag_set(
                actions = [
                    ACTION_NAMES.preprocess_assemble,
                    ACTION_NAMES.c_compile,
                    ACTION_NAMES.cpp_compile,
                    ACTION_NAMES.cpp_header_parsing,
                    "c++-header-preprocessing",
                    ACTION_NAMES.cpp_module_compile,
                ],
                flag_groups = [flag_group(flags = ["-fprofile-arcs", "-ftest-coverage"])],
            ),
            flag_set(
                actions = [
                    "c++-link-interface-dynamic-library",
                    ACTION_NAMES.cpp_link_dynamic_library,
                    ACTION_NAMES.cpp_link_executable,
                ],
                flag_groups = [flag_group(flags = ["-lgcov"])],
            ),
        ],
        provides = ["profile"],
    )

    debug_feature = feature(
        name = "debug",
        flag_sets = [
            flag_set (
                actions = [
                    "ACTION_NAMES.c_compile",
                    "ACTION_NAMES.cpp_compile"
                ],
                flag_groups = [
                    flag_group(
                        flags = ["-g"],
                    ),
                ],
            )
        ],
    )

    features = [coverage_feature, debug_feature]

    gcc_include_directories_by_version = {
        "gcc5.4": [
            "/usr/aarch64-linux-gnu/include/c++/5/",
            "/usr/aarch64-linux-gnu/include/c++/5/backward",
            "/usr/aarch64-linux-gnu/include/",
            "/usr/lib/gcc-cross/aarch64-linux-gnu/5/include",
            "/usr/lib/gcc-cross/aarch64-linux-gnu/5/include-fixed",
        ],
        "gcc7.4": [
            "/usr/aarch64-linux-gnu/include/c++/7/",
            "/usr/aarch64-linux-gnu/include/c++/7/backward",
            "/usr/aarch64-linux-gnu/include/",
            "/usr/lib/gcc-cross/aarch64-linux-gnu/7/include",
            "/usr/lib/gcc-cross/aarch64-linux-gnu/7/include-fixed",
        ]
    }
    
    cxx_builtin_include_directories = gcc_include_directories_by_version[ctx.attr.gcc_toolchain_version]
    artifact_name_patterns = []

    make_variables = []

    tool_paths = [
        tool_path(
            name = "ld",
            path = "/usr/bin/aarch64-linux-gnu-ld.gold",
        ),
        tool_path(
            name = "cpp",
            path = "/usr/bin/aarch64-linux-gnu-cpp",
        ),
        tool_path(
            name = "dwp",
            path = "/usr/bin/aarch64-linux-gnu-dwp",
        ),
        tool_path(
            name = "gcov",
            path = "/usr/bin/aarch64-linux-gnu-gcov",
        ),
        tool_path(
            name = "nm",
            path = "/usr/bin/aarch64-linux-gnu-nm",
        ),
        tool_path(
            name = "objcopy",
            path = "/usr/bin/aarch64-linux-gnu-objcopy",
        ),
        tool_path(
            name = "objdump",
            path = "/usr/bin/aarch64-linux-gnu-objdump",
        ),
        tool_path(
            name = "strip",
            path = "/usr/bin/aarch64-linux-gnu-strip",
        ),
        tool_path(
            name = "gcc",
            path = "/usr/bin/aarch64-linux-gnu-g++",
        ),
        tool_path(
            name = "ar",
            path = "/usr/bin/aarch64-linux-gnu-ar",
        ),
    ]


    out = ctx.actions.declare_file(ctx.label.name)
    ctx.actions.write(out, "Fake executable")
    return [
        cc_common.create_cc_toolchain_config_info(
            ctx = ctx,
            features = features,
            action_configs = action_configs,
            artifact_name_patterns = artifact_name_patterns,
            cxx_builtin_include_directories = cxx_builtin_include_directories,
            toolchain_identifier = toolchain_identifier,
            host_system_name = host_system_name,
            target_system_name = target_system_name,
            target_cpu = target_cpu,
            target_libc = target_libc,
            compiler = compiler,
            abi_version = abi_version,
            abi_libc_version = abi_libc_version,
            tool_paths = tool_paths,
            make_variables = make_variables,
            builtin_sysroot = builtin_sysroot,
            cc_target_os = cc_target_os
        ),
        DefaultInfo(
            executable = out,
        ),
    ]
D5L_cc_toolchain_config =  rule(
    implementation = _impl,
    attrs = {
        "cpu": attr.string(mandatory=True, values=["aarch64"]),
        "gcc_toolchain_version": attr.string(mandatory=True, values=["gcc5.4", "gcc7.4"]),
    },
    provides = [CcToolchainConfigInfo],
    executable = True,
)
