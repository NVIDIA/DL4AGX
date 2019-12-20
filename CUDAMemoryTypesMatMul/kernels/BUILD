package(default_visibility = ["//visibility:public"])
load("//tools/nvcc:cuda.bzl", "cu_library")

config_setting(
    name = "aarch64_linux",
    values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
)

config_setting(
    name = "aarch64_qnx",
    values = { "crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx" }
)

# Cuda kernel
cu_library(
    name="matrix_mul_mem",
    srcs=[
        "matrixMulMem.cu",
    ],
    hdrs=[
        "matrixMulMem.h",
    ],
    gpu_arch="sm_70",
    gen_code=[
        "arch=compute_75,code=sm_75", #Turing
        "arch=compute_70,code=sm_70"
    ],
    deps=[
        "//CUDAMemoryTypesMatMul/helpers:helpers"
    ] + select({
        ":aarch64_linux": [
            "@cuda_aarch64_linux//:cudart",
        ],
        ":aarch64_qnx": [
            "@cuda_aarch64_qnx//:cudart",
        ],
        "//conditions:default": [
            "@cuda_x86_64_linux//:cudart",
        ],
    }),
)
