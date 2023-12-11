/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
 
#include "kernels.h"


// ----------------------- Float to int8 -----------------------
__global__ void convert_float_to_int8_kernel(float* a, int8_t* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float v = (a[idx] / scale);
        if(v < -128) v = -128;
        if(v > 127) v = 127;
        b[idx] = (int8_t)v;
    }
}
int convert_float_to_int8(float* h_a, int8_t* h_b, int size, float scale, cudaStream_t stream){

    float* d_a;
    int8_t* d_b;

    // Allocate device memory
    cudaMalloc((void**)&d_a, size * sizeof(float));
    cudaMalloc((void**)&d_b, size * sizeof(int8_t));

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, size * sizeof(float), cudaMemcpyHostToDevice);

    int32_t const blocksize = 512;
    // Launch kernel
    convert_float_to_int8_kernel<<<(size + blocksize) / blocksize, blocksize, 0, stream>>>(d_a, d_b, size, scale);

    cudaStreamSynchronize(stream);
    // Copy output data from device to host
    cudaMemcpy(h_b, d_b, size * sizeof(int8_t), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}


// ----------------------- int8 to float -----------------------
__global__ void convert_int8_to_float_kernel(int8_t* a, float* b, int size, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        b[idx] = static_cast<float>(a[idx]) * scale;
    }
}

int convert_int8_to_float(int8_t* h_a, float* h_b, int size, float scale, cudaStream_t stream) {

    int8_t* d_a;
    float* d_b;

    // Allocate device memory
    cudaMalloc((void**)&d_a, size * sizeof(int8_t));
    cudaMalloc((void**)&d_b, size * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_a, h_a, size * sizeof(int8_t), cudaMemcpyHostToDevice);

    const int threads_per_block = 512;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    // Launch kernel
    convert_int8_to_float_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, size, scale);

    cudaStreamSynchronize(stream);
    // Copy output data from device to host
    cudaMemcpy(h_b, d_b, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}

// ----------------------- preprocess images -----------------------

__constant__ float mean[3] = {123.675, 116.28, 103.53};
__constant__ float stddev[3] = {58.395, 57.12, 57.375};


__global__ void preprocess_image_kernel(unsigned char* images, float* buffer, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int k = 0; k < 3; ++k) {
            buffer[k * size + idx] = ((float)images[idx * 4 + k] - mean[k]) / stddev[k];
        }
    }
}

__global__ void test_kernel(float* buffer, const int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        for (int k = 0; k < 3; ++k) {
            buffer[k * size + idx] = 1.0;
        }
    }
}

int preprocess_image(unsigned char* h_images, float* h_buffer, const int size, cudaStream_t stream){

    unsigned char* d_images;
    float* d_buffer;

    int image_size = 4 * size;
    int buffer_size = 3 * size;
    // Allocate device memory
    cudaMalloc((void**)&d_images, image_size * sizeof(unsigned char));
    cudaMalloc((void**)&d_buffer, buffer_size * sizeof(float));

    // Copy input data from host to device
    cudaMemcpy(d_images, h_images, image_size * sizeof(unsigned char), cudaMemcpyHostToDevice);
    // Launch kernel
    const int threads_per_block = 512;
    const int num_blocks = (size + threads_per_block - 1) / threads_per_block;
    preprocess_image_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_images, d_buffer, size);
    // test_kernel<<<num_blocks, threads_per_block, 0, stream>>>(d_buffer, size);
    cudaStreamSynchronize(stream);
    // Copy output data from device to host
    cudaMemcpy(h_buffer, d_buffer, buffer_size * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_images);
    cudaFree(d_buffer);

    return 0;
}