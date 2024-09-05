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

#pragma once

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

__constant__ float mean[3] = {123.675, 116.28, 103.53};
__constant__ float stddev[3] = {58.395, 57.12, 57.375};

__global__ void normalize_kernel(unsigned char* images, float* buffer, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            buffer[k * size + idx] = ((float)images[idx * 3 + k] - mean[k]) / stddev[k];
        }
    }
}

void normalize(uint8_t* in, float* out, int HW, cudaStream_t stream) {
    int block = (HW + 511) / 512;
    int thread = 512;
    normalize_kernel<<<block, thread, 0, stream>>>(in, out, HW);
}
