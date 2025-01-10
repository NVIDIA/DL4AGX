/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-20202524 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 *
 */

#pragma once

#include <iostream>
#include <algorithm>
#include <cublas_v2.h>
#include <cuda.h>
#include <NvInfer.h>

#if DEBUG
#define print_log(...) {\
    char str[100];\
    sprintf(str, __VA_ARGS__);\
    std::cout << "CUSTOM PLUGIN TRACE----> call " << "[" \
              << __FILE__ << ":" << __LINE__ \
              << ", " << __FUNCTION__ << " " << str << std::endl;\
    }
#else
#define print_log(...)
#endif

#if __CUDACC__
#define __hostdev__ __host__ __device__
#else
#define __hostdev__
#endif // __CUDACC__

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = DIVUP(N, THREADS_PER_BLOCK);
  int max_block_num = 4096;
  return std::min(optimal_block_num, max_block_num);
}

#define PLUGIN_VALIDATE(condition)                                             \
  {                                                                            \
    if (!(condition)) {                                                        \
    }                                                                          \
  }

template <typename scalar_t>
cublasStatus_t cublasGemmWrap(cublasHandle_t handle, cublasOperation_t transa,
                              cublasOperation_t transb, int m, int n, int k, const scalar_t* alpha,
                              const scalar_t* A, int lda, const scalar_t* B, int ldb,
                              const scalar_t* beta, scalar_t* C, int ldc);

template <class scalar_t>
void memcpyPermute(scalar_t* dst, const scalar_t* src, int* src_size, int64_t* permute, int src_dim,
                   cudaStream_t stream = 0);

#define cudaCheckError()                                                               \
  {                                                                                    \
    cudaError_t e = cudaGetLastError();                                                \
    if (e != cudaSuccess) {                                                            \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      exit(0);                                                                         \
    }                                                                                  \
  }

#define PLUGIN_ASSERT(assertion)                                                                                       \
  {                                                                                                                  \
    if (!(assertion))                                                                                              \
    {                                                                                                              \
    }                                                                                                              \
  }

inline void caughtError(std::exception const& e) {
  std::cerr << e.what() << std::endl;
}

#if NV_TENSORRT_MAJOR > 7
#define TRT_NOEXCEPT noexcept
#else
#define TRT_NOEXCEPT
#endif

#if NV_TENSORRT_MAJOR >= 10
  using dim_t = int64_t;
#else
  using dim_t = int32_t;
#endif
