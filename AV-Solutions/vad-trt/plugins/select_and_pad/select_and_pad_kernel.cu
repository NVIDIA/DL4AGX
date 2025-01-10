/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/cub.cuh>
#include "select_and_pad.h"

using namespace nvinfer1;
using nvinfer1::plugin::SelectAndPadPlugin;

// indices: B, P
__global__ void init_select_and_pad(
  int* indices, int* buf,
  const int P
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i < P ) {
    indices[i] = i;
    buf[i] = -1;
  }
}

// feat:    B, Q, C
// indices: B, P
// invalid: C
// out:     B, P, C
// x = indices[b, p]
// out[b, p, :] = feat[b, x, :] if x 
template<typename T>
__global__ void select_or_pad(
  T* feat,
  int* indices,
  T* invalid,
  const int B, const int Q, const int C, const int P,
  T* out
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i >= B * P ) return;

  int b = i / Q;
  int index = indices[i];
  T* curr_f = invalid;
  if( index != -1 ) {
    // printf("%d, %d\n", i, index);
    curr_f = feat + b * Q * C + index * C;
  }
  T* curr_out = out + b * P * C + i * C;
  for( int c = 0; c < C; c++ ) {
    curr_out[c] = curr_f[c];
  }
}

size_t SelectAndPadPlugin::decideTemp() {
  void *d_temp_storage = NULL;  
  size_t _temp_storage_bytes = 0;
  void* _no_use_ptr; cudaMalloc(&_no_use_ptr, sizeof(int) * (600 + P + 1));
  int* _no_use = (int*)_no_use_ptr;
#if CUDA_VERSION >=11040
  cudaError_t err = cub::DeviceSelect::Flagged(
    d_temp_storage, _temp_storage_bytes,
    _no_use, _no_use, _no_use, _no_use, P);
#else
  // TODO: find correct legacy api here
#endif
  if( err != cudaSuccess ) {
    print_log("%s", cudaGetErrorString(err));
  }
  print_log("_temp_storage_bytes=%d", _temp_storage_bytes);
  if( _temp_storage_bytes == 0 ) {
    _temp_storage_bytes = 767; // TODO: solve this issue? not sure why
  }
  cudaFree(_no_use);
  return _temp_storage_bytes;
}

template<typename T>
void select_and_pad_launch(
  T* feat, 
  int* flags,
  T* invalid,
  int B, int Q, int C, int P,
  T* out, 
  void* tmp, size_t tmp_bytes, cudaStream_t stream
) {
  // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceSelect.html?highlight=deviceselect#_CPPv4I000EN3cub12DeviceSelect7FlaggedE11cudaError_tPvR6size_t9IteratorT12FlagIterator20NumSelectedIteratorTi12cudaStream_t
  int* buf = reinterpret_cast<int*>(tmp);
  int* indices = reinterpret_cast<int*>(buf + sizeof(int) * Q);
  int* ftmp = reinterpret_cast<int*>(indices + sizeof(int) * Q);
  int* d_num = reinterpret_cast<int*>(ftmp + tmp_bytes);

  init_select_and_pad<<<((Q + 255) / 256), 256, 0, stream>>>(indices, buf, Q);

  // int* h_flags = new int[Q];
  // cudaMemcpy(h_flags, flags, sizeof(int) * Q, cudaMemcpyDeviceToHost);
  // for( int i=0; i<Q; i++ ) {
  //   printf("%d ", h_flags[i]);
  // }
  // printf("\n");
  // print_log("tmp_bytes=%d", tmp_bytes);

  cudaError_t err = cub::DeviceSelect::Flagged(
    ftmp, tmp_bytes,
    indices, flags, buf, d_num, Q, stream);

  // cudaStreamSynchronize(stream);
  // if( err != cudaSuccess ) {
  //   printf("err = %s\n", cudaGetErrorString(err));
  // }
  
  // int h_num = 0;
  // cudaMemcpyAsync(&h_num, d_num, sizeof(int), cudaMemcpyDeviceToHost, stream);
  // cudaStreamSynchronize(stream);
  // printf("h_num = %d\n", h_num);

  // int* h_buf = new int[P];
  // cudaMemcpy(h_buf, buf, sizeof(int) * P, cudaMemcpyDeviceToHost);
  // for( int i=0; i<P; i++ ) {
  //   printf("%d ", h_buf[i]);
  // }
  // printf("\n");

  // printf("B=%d, Q=%d, C=%d, P=%d\n", B, Q, C, P);
  select_or_pad<<<((P + 255) / 256), 256, 0, stream>>>(
    feat, buf, invalid,
    B, Q, C, P,
    out
  );

  // delete[] h_flags;
  // delete[] h_buf;
}

int32_t SelectAndPadPlugin::enqueue(
  const nvinfer1::PluginTensorDesc *inputDesc,
  const nvinfer1::PluginTensorDesc *outputDesc,
  const void *const *inputs, 
  void *const *outputs,
  void *workSpace,
  cudaStream_t stream
) noexcept {
  nvinfer1::Dims feat_dims = inputDesc[0].dims;     // bqc
  nvinfer1::Dims out_dims = outputDesc[0].dims;     // bpc
  // print_log("feat %d, %d", feat_dims.d[1], feat_dims.d[2]);
  // print_log("input[2], %d", inputDesc[2].dims.d[0]);
  int B = feat_dims.d[0];
  int Q = feat_dims.d[1];
  int C = feat_dims.d[2];
  auto data_type = inputDesc[0].type;  
  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      // cudaMemsetAsync((float *)outputs[0], 0, sizeof(float) * n_elem, stream);
      select_and_pad_launch(
        (float *)inputs[0], 
        (int *)inputs[1], 
        (float *)inputs[2],
        B, Q, C, P,
        (float *)outputs[0],
        workSpace, tmp_bytes, stream);
      break;
    // case nvinfer1::DataType::kHALF:
    //   bev_pool_v2(feat_dims.d[3], interval_dims.d[0], num_points,
    //               (__half *)inputs[0], (__half *)inputs[1], (int *)inputs[2],
    //               (int *)inputs[3], (int *)inputs[4], (int *)inputs[5],
    //               (int *)inputs[6], (__half *)outputs[0], stream);
    //   break;
    default:
      return 1;
  };
  return 0;
}
