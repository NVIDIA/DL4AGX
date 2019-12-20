package(default_visibility = ["//visibility:public"])
load("//tools/nvcc:cuda.bzl", "cu_binary", "cu_library")

config_setting(
    name = "aarch64_linux",
    values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
)

config_setting(
    name = "aarch64_qnx",
    values = { "crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx" }
)


#If we want to distribute a seperate so file instead of compiling into
#a single executable
cc_binary(
    name="libs3poolplugin.so",
    linkshared=True,
    srcs=[
        "S3Pool.h",
        "S3Pool.cpp",
        "plugin.h"
    ],
    deps=[
        "avgPool"
    ]
    + select({
        ":aarch64_linux": [
            "@tensorrt_aarch64_linux//:nvinferplugin",
            "@cuda_aarch64_linux//:cublas"
        ],
        ":aarch64_qnx": [
            "@tensorrt_aarch64_qnx//:nvinferplugin",
            "@cuda_aarch64_qnx//:cublas"
        ],
        "//conditions:default":[
            "@tensorrt_x86_64_linux//:nvinferplugin",
            "@cuda_x86_64_linux//:cuda",
        ]}),
)

# Cuda kernel
cu_library(
    name="avgPool",
    srcs=[
        "averagePool.cu"
    ],
    hdrs=[
        "averagePool.h"
        ],
    gpu_arch="sm_70",
    gen_code=[
        "arch=compute_75,code=sm_75", #Turing
        "arch=compute_70,code=sm_70"
    ],
    deps=select({
        ":aarch64_linux": [
            "@cuda_aarch64_linux//:cudart",
        ],
        ":aarch64_qnx": [
            "@cuda_aarch64_qnx//:cudart",
        ],
        "//conditions:default": [
            "@cuda_x86_64_linux//:cudart",
        ],
    })
)
