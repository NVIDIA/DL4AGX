/*
MIT License

Copyright (c) 2022 OpenGVLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/*!
**************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************
* Modified from DCN (https://github.com/msracver/Deformable-ConvNets)
* Copyright (c) 2018 Microsoft
**************************************************************************
*/

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

// Modified from https://github.com/OpenGVLab/DCNv4/blob/main/DCNv4_op/src/cuda/common.h
// with some tpye adjustments

#ifndef FMSDACOMMON
#define FMSDACOMMON
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

constexpr int kWarpSize = 32;
#define opmath_t scalar_t
#define oplogic_t float

#if __CUDA_ARCH__ < 750
#else
#endif

inline int GET_BLOCKS(const int N, const int num_threads) {
  return (N + num_threads - 1) / num_threads;
}

#define CUDA_KERNEL_LOOP(i, n)                                                 \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);                 \
       i += blockDim.x * gridDim.x)

template <typename scalar_t, typename transfer_t, int c_per_thread>
__device__ void ms_deform_attn_im2col_bilinear(
    opmath_t out_reg_array[], const scalar_t *&p_value, const int &height,
    const int &width, const oplogic_t &h_px, const oplogic_t &w_px,
    const scalar_t &attn, const int &w_stride, const int &base_ptr) {

  const int h_low = floor(h_px);
  const int w_low = floor(w_px);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;
  const oplogic_t lh = h_px - h_low;
  const oplogic_t lw = w_px - w_low;
  const oplogic_t hh = 1 - lh;
  const oplogic_t hw = 1 - lw;

  const scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  const int h_stride = width * w_stride;
  const int h_low_ptr_offset = h_low * h_stride;
  const int h_high_ptr_offset = h_low_ptr_offset + h_stride;
  const int w_low_ptr_offset = w_low * w_stride;
  const int w_high_ptr_offset = w_low_ptr_offset + w_stride;

  int idx1 = h_low_ptr_offset + w_low_ptr_offset + base_ptr;
  int idx2 = h_low_ptr_offset + w_high_ptr_offset + base_ptr;
  int idx3 = h_high_ptr_offset + w_low_ptr_offset + base_ptr;
  int idx4 = h_high_ptr_offset + w_high_ptr_offset + base_ptr;

  scalar_t v1_array[c_per_thread] = {0.};
  scalar_t v2_array[c_per_thread] = {0.};
  scalar_t v3_array[c_per_thread] = {0.};
  scalar_t v4_array[c_per_thread] = {0.};

  if (h_low >= 0 && w_low >= 0) {
    auto p1 = p_value + idx1;
    *(transfer_t *)(v1_array) = *(transfer_t *)(p1);
  }
  if (h_low >= 0 && w_high < width) {
    auto p2 = p_value + idx2;
    *(transfer_t *)(v2_array) = *(transfer_t *)(p2);
  }
  if (h_high < height && w_low >= 0) {
    auto p3 = p_value + idx3;
    *(transfer_t *)(v3_array) = *(transfer_t *)(p3);
  }
  if (h_high < height && w_high < width) {
    auto p4 = p_value + idx4;
    *(transfer_t *)(v4_array) = *(transfer_t *)(p4);
  }
#pragma unroll
  for (int i = 0; i < c_per_thread; i++) {
    out_reg_array[i] +=
        (opmath_t)attn *
        (w1 * v1_array[i] + 
         w2 * v2_array[i] +
         w3 * v3_array[i] + 
         w4 * v4_array[i]);
  }
}

#endif
