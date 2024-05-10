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

#include <stdio.h>
#include "memory.cuh"
 
__global__ void ApplyDeltaFromMem(double delta, double* mem, float* buf, int n_elem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n_elem ) {
        double v = mem[idx] + delta;
        buf[idx] = (float)v;
    }
}

__global__ void ApplyDeltaToMem(double delta, double* mem, float* buf, int n_elem) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < n_elem ) {
        double v = (double)buf[idx];
        mem[idx] = v - delta;
    }
}

void Memory::StepReset() {
    // reset pre_buf to zero
    cudaMemsetAsync(pre_buf, 0, sizeof(float) * 512, mem_stream);
}

void Memory::StepPre(double ts) {
    // update timestamp in pre_update_memory
    // NOTE: 512 is harded coded number
    ApplyDeltaFromMem<<<512, 1, 0, mem_stream>>>(ts, reinterpret_cast<double*>(mem_buf), pre_buf, 512);
}

void Memory::StepPost(double ts) {
    // update timestamp in post_update_memory
    // NOTE: 640 is harded coded number
    ApplyDeltaToMem<<<640, 1, 0, mem_stream>>>(ts, reinterpret_cast<double*>(mem_buf), post_buf, 640);
}

void Memory::DebugPrint() {
    double temp_buf[16];
    cudaMemcpy(reinterpret_cast<void*>(temp_buf), mem_buf, sizeof(double) * 16, cudaMemcpyDeviceToHost);
    for( int i=0; i<16; i++ ) {
        printf("%f ", temp_buf[i]);
    }
}