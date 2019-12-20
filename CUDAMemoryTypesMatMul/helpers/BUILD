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

cc_library(
    name = "helpers",
    hdrs = [
        "cuda.h",
        "string.h",
    ],
    deps = select({
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
