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

#include "kernel.cuh"

namespace optimize {

__device__ static int transform_float4_to_int8x4(float4 val) {
    int ix, iy, iz, iw;
    // be cautious for rni vs rzi
    asm volatile("cvt.rzi.s8.f32 %0, %1;" : "=r"(ix) : "f"(val.x));
    asm volatile("cvt.rzi.s8.f32 %0, %1;" : "=r"(iy) : "f"(val.y));
    asm volatile("cvt.rzi.s8.f32 %0, %1;" : "=r"(iz) : "f"(val.z));
    asm volatile("cvt.rzi.s8.f32 %0, %1;" : "=r"(iw) : "f"(val.w));

    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(ix) : "r"(iy));
    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(iz) : "r"(iw));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(ix) : "r"(iz));
    return ix;
}

__device__ __forceinline__ static float4 transform_int8x4_to_float4(int val) {
    int ix, iy, iz, iw = val;

    // Extract the 4 bytes
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4440;" : "=r"(ix) : "r"(iw));
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4441;" : "=r"(iy) : "r"(iw));
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4442;" : "=r"(iz) : "r"(iw));
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4443;" : "=r"(iw) : "r"(iw));
    // the floats
    float fx, fy, fz, fw;

    // convert to floats (make sure we generate I2F.F32.S8)
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fx) : "r"(ix));
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fy) : "r"(iy));
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fz) : "r"(iz));
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fw) : "r"(iw));

    return ::make_float4(fx, fy, fz, fw);
}

inline __device__ float4 operator/(float4 a, float b) {
    return make_float4(a.x / b, a.y / b, a.z / b,  a.w / b);
}

__device__ __forceinline__ float4 ldg128(const float4* addr){
    float a, b, c, d;
    asm volatile(
        "ld.global.nc.v4.f32 {%0, %1, %2, %3}, [%4];\n"
        : "=f"(a),
          "=f"(b),
          "=f"(c),
          "=f"(d)
        : "l"(addr)
    );
    return ::make_float4(a, b, c, d);
}

__global__ void convert_float_to_int8_kernel(const float* a, int8_t* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const float4* va = reinterpret_cast<const float4*>(a);
    int* vb = reinterpret_cast<int*>(b);
    if (idx < size) {
        float4 v = ldg128(va + idx);
        v = v / scale;
        v.x = fmaxf(-128.0, fminf(v.x, 127.0));
        v.y = fmaxf(-128.0, fminf(v.y, 127.0));
        v.z = fmaxf(-128.0, fminf(v.z, 127.0));
        v.w = fmaxf(-128.0, fminf(v.w, 127.0));
        vb[idx] = transform_float4_to_int8x4(v);
    }
}

int convert_float_to_int8(void* d_a, int8_t* d_b, int size, float scale, cudaStream_t stream) {
    int32_t const blocksize = 512;
    int32_t const sizev = size / 4; // TODO: assert size mod 4 == 0
    // Launch kernel
    convert_float_to_int8_kernel<<<(sizev + blocksize) / blocksize, blocksize, 0, stream>>>(
        reinterpret_cast<float*>(d_a), 
        d_b, 
        sizev, scale
    );
    return 0;
}

__global__ void convert_half_to_int8_kernel(const __half* a, int8_t* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t* va = reinterpret_cast<const int64_t*>(a);
    int* vb = reinterpret_cast<int*>(b);
    if (idx < size) {
        int64_t iv = va[idx];  // 64bits, 4 half
        __half* hv; hv = reinterpret_cast<__half*>(&iv);
        float4 v;
        v.x = fmaxf(-128.0, fminf((float)(hv[0]) / scale, 127.0));
        v.y = fmaxf(-128.0, fminf((float)(hv[1]) / scale, 127.0));
        v.z = fmaxf(-128.0, fminf((float)(hv[2]) / scale, 127.0));
        v.w = fmaxf(-128.0, fminf((float)(hv[3]) / scale, 127.0));
        vb[idx] = transform_float4_to_int8x4(v);
    }
}

int convert_half_to_int8(void* d_a, int8_t* d_b, int size, float scale, cudaStream_t stream) {
    int32_t const blocksize = 512;
    int32_t const sizev = size / 4; // TODO: assert size mod 4 == 0
    // Launch kernel
    convert_half_to_int8_kernel<<<(sizev + blocksize) / blocksize, blocksize, 0, stream>>>(
        reinterpret_cast<__half*>(d_a), 
        d_b, 
        sizev, scale
    );
    return 0;
}

__global__ void convert_int8_to_float_kernel(int8_t* a, float* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = static_cast<float>(a[idx]) * scale;
    }
}

int convert_int8_to_float(int8_t* d_a, float* d_b, int size, float scale) {
    const int threads_per_block = 512;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_int8_to_float_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, size, scale);
    return 0;
}

} // namespace optimize

__constant__ float mean[3] = {123.675, 116.28, 103.53};
__constant__ float stddev[3] = {58.395, 57.12, 57.375};

__global__ void preprocess_image_kernel(unsigned char* images, float* buffer, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            buffer[k * size + idx] = ((float)images[idx * 4 + k] - mean[k]) / stddev[k];
        }
    }
}

int preprocess_image(unsigned char* d_images, float* d_buffer, const int size, cudaStream_t stream) {
    int image_size = 4 * size;
    int buffer_size = 3 * size;

    // Launch kernel
    const int threads_per_block = 512;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    preprocess_image_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_images, d_buffer, size);

    return 0;
}
