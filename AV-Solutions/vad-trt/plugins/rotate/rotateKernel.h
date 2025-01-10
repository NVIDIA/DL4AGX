//
// Created by Derry Lin on 2022/11/21.
//

#ifndef TENSORRT_OPS_ROTATEKERNEL_H
#define TENSORRT_OPS_ROTATEKERNEL_H

// #include "cuda_int8.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>

enum class RotateInterpolation { Bilinear, Nearest };

// template <typename T>
// void rotate(T *output, T *input, T *angle, T *center, int *input_dims,
//             RotateInterpolation interp, cudaStream_t stream);

#endif // TENSORRT_OPS_ROTATEKERNEL_H
