/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __LAUNCH_CUH__
#define __LAUNCH_CUH__

#include "check.hpp"

namespace nv {

#define LINEAR_LAUNCH_THREADS 512
#define cuda_linear_index (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_x (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_y (blockDim.y * blockIdx.y + threadIdx.y)
#define divup(a, b) ((static_cast<int>(a) + static_cast<int>(b) - 1) / static_cast<int>(b))

#ifdef CUDA_DEBUG
#define cuda_linear_launch(kernel, stream, num, ...)                                   \
  do {                                                                                 \
    size_t __num__ = (size_t)(num);                                                    \
    size_t __blocks__ = (__num__ + LINEAR_LAUNCH_THREADS - 1) / LINEAR_LAUNCH_THREADS; \
    kernel<<<__blocks__, LINEAR_LAUNCH_THREADS, 0, stream>>>(__num__, __VA_ARGS__);    \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);             \
    nv::check_runtime(cudaStreamSynchronize(stream), #kernel, __LINE__, __FILE__);     \
  } while (false)

#define cuda_2d_launch(kernel, stream, nx, ny, ...)                                \
  do {                                                                             \
    dim3 __threads__(32, 32);                                                      \
    dim3 __blocks__(divup(nx, 32), divup(ny, 32));                                 \
    kernel<<<__blocks__, __threads__, 0, stream>>>(nx, ny, __VA_ARGS__);           \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);         \
    nv::check_runtime(cudaStreamSynchronize(stream), #kernel, __LINE__, __FILE__); \
  } while (false)
#else  // CUDA_DEBUG
#define cuda_linear_launch(kernel, stream, num, ...)                                \
  do {                                                                              \
    size_t __num__ = (size_t)(num);                                                 \
    size_t __blocks__ = divup(__num__, LINEAR_LAUNCH_THREADS);                      \
    kernel<<<__blocks__, LINEAR_LAUNCH_THREADS, 0, stream>>>(__num__, __VA_ARGS__); \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);          \
  } while (false)

#define cuda_2d_launch(kernel, stream, nx, ny, nz, ...)                      \
  do {                                                                       \
    dim3 __threads__(32, 32);                                                \
    dim3 __blocks__(divup(nx, 32), divup(ny, 32), nz);                       \
    kernel<<<__blocks__, __threads__, 0, stream>>>(nx, ny, nz, __VA_ARGS__); \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);   \
  } while (false)
#endif  // CUDA_DEBUG
};      // namespace nv

#endif  // __LAUNCH_CUH__