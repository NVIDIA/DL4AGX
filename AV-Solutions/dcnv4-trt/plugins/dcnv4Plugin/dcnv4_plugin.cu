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

// Modified from https://github.com/OpenGVLab/DCNv4/blob/main/DCNv4_op/src/cuda/dcnv4_im2col_cuda.cuh
// Removed torch related code, changed data types and slight re-organize the kernel function.

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>

#include "./common.cuh"
#include "./dcnv4_plugin.h"

template <typename scalar_t, int d_stride, typename transfer_t, int L, int K,
          bool softmax>
__global__ void forward_kernel_dcn_reg(
    const scalar_t *p_value, const scalar_t *p_offset, scalar_t *p_output,
    const int G, const int D, const int Q, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, const int dilation_h, const int dilation_w,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const float offset_scale, const int remove_center,
    const int block_multiplier, const int padded_offset_dim) {

  const int &qi = (blockIdx.x * block_multiplier % Q) + threadIdx.z;
  const int &bi = blockIdx.x * block_multiplier / Q;

  const int &di_s = threadIdx.x * d_stride;
  const int &gi = threadIdx.y;
  constexpr int li = 0;

  opmath_t p_mask_shm[K] = {0.};
  opmath_t p_out_shm[d_stride] = {0.};

  const scalar_t *p_offset_ptr = p_offset + (bi*Q + qi)*padded_offset_dim + gi*K*3;
  const int mask_length = K;
  const int num_thread = (D / d_stride);
  const int num_iter = mask_length / num_thread;
  const int remainder = mask_length - num_iter * num_thread;

  for (int i=0; i < K; i++){
    p_mask_shm[i] = *(p_offset_ptr + K*2 + i);
  }
  
  int offset_idx = 0;
  int mask_idx = 0;

  const int w_stride = G * D;
  const int base_ptr = gi * D + di_s;
  const scalar_t *p_value_ptr =
      p_value + (bi * (height_in * width_in)) * (G * D);

  const int p0_w = ((dilation_w * (kernel_w - 1)) >> 1) - pad_w +
                   (qi % width_out) * stride_w;
  const int p0_h = ((dilation_h * (kernel_h - 1)) >> 1) - pad_h +
                   (qi / width_out) * stride_h;
  const oplogic_t p0_w_ =
      p0_w - ((dilation_w * (kernel_w - 1)) >> 1) * offset_scale;
  const oplogic_t p0_h_ =
      p0_h - ((dilation_h * (kernel_h - 1)) >> 1) * offset_scale;
  const int center_h = kernel_h / 2;
  const int center_w = kernel_w / 2;

  int out_idx = ((bi * Q + qi) * G + gi) * D + di_s;

  for (int i = 0; i < kernel_w; ++i) {
    for (int j = 0; j < kernel_h; ++j) {
      if (i != center_w || j != center_h || !remove_center) {
        const oplogic_t w_im =
            p0_w_ + ((i * dilation_w) + (oplogic_t)p_offset_ptr[offset_idx]) * offset_scale;
        const oplogic_t h_im =
            p0_h_ + ((j * dilation_h) + (oplogic_t)p_offset_ptr[offset_idx + 1]) * offset_scale;
        const oplogic_t attn = p_mask_shm[mask_idx];

        if (h_im > -1 && w_im > -1 && h_im < height_in && w_im < width_in) {
          ms_deform_attn_im2col_bilinear<scalar_t, transfer_t, d_stride>(
              p_out_shm, p_value_ptr, height_in, width_in, 
              (opmath_t)h_im, (opmath_t)w_im, (opmath_t)attn,
              w_stride, base_ptr);
        }
        offset_idx += 2;
        mask_idx += 1;
      }
    }
  }
  *(transfer_t *)(p_output + out_idx) = *(transfer_t *)(p_out_shm);
}

template <typename scalar_t, typename stride_type, int d_stride>
void _dcnv4_im2col_cuda(cudaStream_t stream,
                              const scalar_t *value,    // B, H * W, (G * D)
                              const scalar_t *p_offset, // B, H * W, G * K * 3)
                              scalar_t *output,         // B, H_out*W_out, G * D
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w,
                              const int pad_h, const int pad_w,
                              const int dilation_h, const int dilation_w,
                              const int G, const int D, const int B,
                              const int height_in, const int width_in,
                              const int height_out, const int width_out,
                              const float offset_scale,
                              const int remove_center, const int block_thread,
                              const int softmax,
                              const int padded_offset_dim) {

  constexpr int L = 1;

  auto kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, true>;

  int N = height_in * width_in;
  int Q = height_out * width_out;
  int K = kernel_h * kernel_w;

  if (remove_center) {
    K -= 1;
  }
  if (softmax) {
    switch (K) {
    case 9:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, true>;
      break;
    case 8:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 8, true>;
      break;
    default:
      printf("K=%d\n", K);
      throw std::invalid_argument("invalid kernel shape");
    }
  } else {
    switch (K) {
    case 9:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 9, false>;
      break;
    case 8:
      kernel = forward_kernel_dcn_reg<scalar_t, d_stride, stride_type, 1, 8, false>;
    break;
    default:
      printf("K=%d\n", K);
      throw std::invalid_argument("invalid kernel shape");
    }
  }

  const int block_multiplier = block_thread / (D / d_stride) / G;
  bool flag = (B*Q) % block_multiplier == 0;
  if( !flag ) {
    printf("B=%d, Q=%d, block_multiplier=%d\n", B, Q, block_multiplier);
  }

  dim3 num_blocks(B*Q / block_multiplier);
  dim3 num_threads(D / d_stride, G, block_multiplier);

  int shm_size = 0;

  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       shm_size);

  kernel<<<num_blocks, num_threads, shm_size, stream>>>(
      value, p_offset, output, 
      G, D, Q, 
      kernel_h, kernel_w, stride_h, stride_w,
      pad_h, pad_w, dilation_h, dilation_w, height_in, width_in, height_out,
      width_out, 
      offset_scale, remove_center, block_multiplier, padded_offset_dim);

  // cudaStreamSynchronize(stream);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in dcnv4_im2col_cuda: %s\n", cudaGetErrorString(err));
    printf("launch arguments: gridDim=(%d, %d, %d), blockDim=(%d, %d, %d), "
           "shm_size=%d\n\n",
           num_blocks.x, num_blocks.y, num_blocks.z, num_threads.x,
           num_threads.y, num_threads.z, shm_size);
  }
}

template <typename scalar_t>
void dcnv4_im2col_cuda(
    cudaStream_t stream,
    const scalar_t *value,    // B, H * W, (G * D)
    const scalar_t *p_offset, // B, H * W, G * K * 3)
    scalar_t *output,         // B, H_out*W_out, G * D
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w, const int dilation_h,
    const int dilation_w, const int G, const int D, const int B,
    const int height_in, const int width_in, const int height_out,
    const int width_out, const float offset_scale, const int remove_center,
    const int d_stride, const int block_thread, const bool softmax,
    const int padded_offset_dim
) {
  int N = height_in * width_in;
  int Q = height_out * width_out;
  int K = kernel_h * kernel_w;
  assert(D % d_stride == 0);  

  if (sizeof(scalar_t) == 2) {
    switch (d_stride) {
    case 1:
      _dcnv4_im2col_cuda<scalar_t, scalar_t, 1>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_im2col_cuda<scalar_t, uint, 2>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_im2col_cuda<scalar_t, uint2, 4>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_im2col_cuda<scalar_t, uint4, 8>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 16:
      _dcnv4_im2col_cuda<scalar_t, ulonglong4, 16>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    }
  } else {
    assert(sizeof(scalar_t) == 4);
    switch (d_stride) {
    case 1:
      _dcnv4_im2col_cuda<scalar_t, uint, 1>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 2:
      _dcnv4_im2col_cuda<scalar_t, uint2, 2>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 4:
      _dcnv4_im2col_cuda<scalar_t, uint4, 4>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    case 8:
      _dcnv4_im2col_cuda<scalar_t, ulonglong4, 8>(
          stream, value, p_offset, output, kernel_h, kernel_w, stride_h,
          stride_w, pad_h, pad_w, dilation_h, dilation_w, G, D, B, height_in,
          width_in, height_out, width_out, offset_scale, remove_center,
          block_thread, softmax, padded_offset_dim);
      break;
    default:
      printf("not supported for d_stride > 8 for fp32");
      throw std::invalid_argument("invalid d_stride");
    }
  }
}

using namespace nvinfer1;
using nvinfer1::plugin::DCNv4_Plugin;

#ifdef USE_PTX
DCNv4Elf* nvinfer1::plugin::getElf() {
  static DCNv4Elf* g_elf = nullptr;
  if( g_elf == nullptr ) {
      g_elf = new DCNv4Elf();
  }
  return g_elf;
}
#endif

std::pair<int, int> findspec(int B, int H, int W, int G, int C) {
  int d_stride = 8;
  int q = B * H * W;
  int m = 1;

  for( int i=1; i<q+1; i++ ) {
    if( q % i == 0 ) {
      if( i<=64 && (i * G * C / d_stride) <= 512) {
        m = i;
      }
    }
  }
  int n_thread = m * G * C / d_stride;
  return std::make_pair(d_stride, n_thread);
}

int32_t DCNv4_Plugin::enqueue(
    int32_t batchSize, 
    const void* const* inputs, void* const* outputs, void* workspace,
    cudaStream_t stream
) noexcept {
  print_log("enqueue");

  const int kernel_h = this->kh; const int kernel_w = this->kw;
  const int stride_h = this->sh; const int stride_w = this->sw; 
  const int pad_h = this->ph; const int pad_w = this->pw; 
  const int dilation_h = this->dh; const int dilation_w = this->dw; 
  const int group = this->group; const int group_channels = this->group_channels;
  const float offset_scale = this->offscale; 
  const int im2col_step = 256; const int remove_center = this->remove_center;

  const bool softmax = false;

  const int batch = batchSize;           // value.size(0);
  const int height_in = mInputDims.d[0]; // value.size(1);
  const int width_in = mInputDims.d[1];  // value.size(2);
  const int channels = mInputDims.d[2];  // value.size(3);

#ifdef USE_PTX
  if( this->mDataType == nvinfer1::DataType::kHALF ) {
    const __half* value_hptr = reinterpret_cast<const __half*>(inputs[0]);
    const __half* p_offset_hptr = reinterpret_cast<const __half*>(inputs[1]);
    __half* columns_hptr = reinterpret_cast<__half*>(outputs[0]);
    if( batchSize == 128 && height_in == 56 && width_in == 56 && channels == 64
    ) {
      // trigger ptx kernel for specific dsize      
      elf->stream = stream;
      elf->launch(value_hptr, p_offset_hptr, columns_hptr);
      return 0;
    } else {
      int stage = -1;
      if( batchSize == 1 && height_in == 56 && width_in == 56 && channels ==  64) { stage = 0; }
      if( batchSize == 1 && height_in == 28 && width_in == 28 && channels == 128) { stage = 1; }
      if( batchSize == 1 && height_in == 14 && width_in == 14 && channels == 256) { stage = 2; }
      if( batchSize == 1 && height_in ==  7 && width_in ==  7 && channels == 512) { stage = 3; }
      if( stage != -1 ) {
        print_log("%d", stage);
        elf_v2.Launch(stage, value_hptr, p_offset_hptr, columns_hptr, stream);
        return 0;
      } else {
        print_log("didn't trigger ptx");
      }
    }
  } // end of this->mDataType == nvinfer1::DataType::kHALF
#endif

  const int height_out =
      (height_in + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int width_out =
      (width_in + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

  auto spec = findspec(batch, height_out, width_out, group, group_channels);
  int d_stride = spec.first;
  int block_thread = spec.second; 
  const int im2col_step_ = std::min(batch, im2col_step);

  const int batch_n = im2col_step_;
  auto per_value_size = height_in * width_in * channels;  
  auto per_offset_size = height_out * width_out * this->padded_offset_dim;  

  if( this->mDataType == nvinfer1::DataType::kFLOAT ) {
    print_log("float");
    const float* value_ptr = reinterpret_cast<const float*>(inputs[0]);
    const float* p_offset_ptr = reinterpret_cast<const float*>(inputs[1]);
    float* columns_ptr = reinterpret_cast<float*>(outputs[0]);

    int n = 0;
    dcnv4_im2col_cuda<float>(
      stream,
      value_ptr + n * im2col_step_ * per_value_size,
      p_offset_ptr + n * im2col_step_ * per_offset_size,
      columns_ptr, 
      kernel_h, kernel_w, 
      stride_h, stride_w, 
      pad_h, pad_w, 
      dilation_h, dilation_w, 
      group, group_channels, 
      batch_n, 
      height_in, width_in, 
      height_out, width_out, 
      offset_scale, remove_center, d_stride, block_thread, softmax, this->padded_offset_dim
    );
  } else if( this->mDataType == nvinfer1::DataType::kHALF ) {
    print_log("half");
    const __half* value_hptr = reinterpret_cast<const __half*>(inputs[0]);
    const __half* p_offset_hptr = reinterpret_cast<const __half*>(inputs[1]);
    __half* columns_hptr = reinterpret_cast<__half*>(outputs[0]);
    int n = 0;
   
    dcnv4_im2col_cuda<__half>(
      stream,
      value_hptr + n * im2col_step_ * per_value_size,
      p_offset_hptr + n * im2col_step_ * per_offset_size,
      columns_hptr, 
      kernel_h, kernel_w, 
      stride_h, stride_w, 
      pad_h, pad_w, 
      dilation_h, dilation_w, 
      group, group_channels, 
      batch_n, 
      height_in, width_in, 
      height_out, width_out, 
      offset_scale, remove_center, d_stride, block_thread, softmax, padded_offset_dim
    );
  } else {
    throw std::runtime_error("!");
  }
  return 0;
}
