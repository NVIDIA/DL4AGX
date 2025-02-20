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

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "pre_process.hpp"

__global__ void permute_img_channel_kernel(float * images, const int width, const int height, const int channels) {
    const int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx < width * height) {
        auto tmp = images[th_idx * channels];
        images[th_idx * channels] = images[th_idx * channels + 2];
        images[th_idx * channels + 2] = tmp;
    }
}

__global__ void transpose_img_kernel(float * images, float * processed_images, const int width, const int height, const int channels) {
    const int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx < width * height * channels) {
        // originally height*width*channels
        // after transpose channels*height*width
        // idx = h*(W*C) + w*(C) + c to (c, h, w)
        int idx = th_idx;
        int c = idx % channels;
        idx /= channels;
        int w = idx % width;
        idx /= width;
        int h = idx % height;
        processed_images[c * (width*height) + h*(width) + w] = images[th_idx];
    }
}

__constant__ float mean[3] = {123.675, 116.28, 103.53};
__constant__ float stddev[3] = {58.395, 57.12, 57.375};

__global__ void norm_img_kernel(float* images, float* processed_images, const int width, const int height, const int channels) {
    const int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx < width * height * channels) {
        // the images must be in width*height*channels format (stb format)
        int k = th_idx % channels;
        processed_images[th_idx] = (images[th_idx] - mean[k]) / stddev[k];
    }
}

__global__ void padding_img_kernel(float* images, float* processed_images, const int width, const int height, const int channels, const int pad_w, const int pad_h) {
    // padding to the bottom and right
    const int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx < pad_w * pad_h * channels) {
        // the images must be in height*width*channels format (stb format)
        // idx = h * (W*C) + w * (C) + c
        int idx = th_idx;
        int c = idx % channels;
        idx /= channels;
        int w = idx % pad_w;
        idx /= pad_w;
        int h = idx % pad_h;
        if (w < width && h < height) {
            processed_images[th_idx] = images[h*(width*channels)+w*(channels)+c];
        } else {
            processed_images[th_idx] = 0.0;
        }
    }
}

__global__ void resizing_img_kernel(float* images, float* processed_images, const int width, const int height, const int channels, const int resized_w, const int resized_h) {
    const int th_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (th_idx < resized_w * resized_h * channels) {
        // the images must be in height*width*channels format (stb format)
        // idx = h * (W*C) + w * (C) + c
        int idx = th_idx;
        int c = idx % channels;
        idx /= channels;
        int w = idx % resized_w;
        idx /= resized_w;
        int h = idx % resized_h;

        float ratio_w = width / (1.0f*resized_w);
        float ratio_h = height / (1.0f*resized_h);

        int w_l = std::floor(ratio_w * w);
        int h_l = std::floor(ratio_h * h);
        int w_h = std::ceil(ratio_w * w);
        int h_h = std::ceil(ratio_h * h);

        float w_weight = (ratio_w * w) - w_l;
        float h_weight = (ratio_h * h) - h_l;

        float img_a = images[h_l * (width*channels) + w_l * (channels) + c];
        float img_b = images[h_l * (width*channels) + w_h * (channels) + c];
        float img_c = images[h_h * (width*channels) + w_l * (channels) + c];
        float img_d = images[h_h * (width*channels) + w_h * (channels) + c];

        processed_images[th_idx] = img_a * (1-w_weight) * (1-h_weight) + img_b * w_weight * (1-h_weight) + img_c * h_weight * (1-w_weight) + img_d * w_weight * h_weight;
    }
}

ImgPreProcess::ImgPreProcess(const int width, const int height, const int channels, const float resize_scale, const int padding_divider) {
    width_ = width; height_ = height; channels_ = channels;
    width_resize_ = (int)(width_ * resize_scale); height_resize_ = (int)(height_ * resize_scale);
    height_pad_ = (int)(std::ceil(height_resize_/(1.0f*padding_divider))) * padding_divider;
    width_pad_ = (int)(std::ceil(width_resize_/(1.0f*padding_divider))) * padding_divider;
    checkRuntime(cudaMallocHost(&img_h, width_*height_*channels_*sizeof(float)));
    checkRuntime(cudaMalloc(&img_d, width_*height_*channels_*sizeof(float)));
    checkRuntime(cudaMalloc(&img_normed_d, width_*height_*channels_*sizeof(float)));
    checkRuntime(cudaMalloc(&img_resized_d, width_resize_*height_resize_*channels_*sizeof(float)));
    checkRuntime(cudaMalloc(&img_padded_d, width_pad_*height_pad_*channels_*sizeof(float)));
    checkRuntime(cudaMalloc(&img_transposed_d, width_pad_*height_pad_*channels_*sizeof(float)));
}

ImgPreProcess::~ImgPreProcess() {
    if (img_h) checkRuntime(cudaFreeHost(img_h));
    if (img_d) checkRuntime(cudaFree(img_d));
    if (img_normed_d) checkRuntime(cudaFree(img_normed_d));
    if (img_resized_d) checkRuntime(cudaFree(img_resized_d));
    if (img_padded_d) checkRuntime(cudaFree(img_padded_d));
    if (img_transposed_d) checkRuntime(cudaFree(img_transposed_d));
}

std::vector<float> ImgPreProcess::img_pre_processing(const std::vector<unsigned char*>& img, void *stream) {
    // image ptr: h, w, c
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    size_t num_imgs = img.size();
    std::vector<float> processed_img(1*num_imgs*channels_*height_pad_*width_pad_, 0);
    for (size_t imgid=0; imgid<img.size(); ++imgid) {
        const int threads_per_block = 512;
        int num_blocks;
        // 0. memHostToDevice
        for (int i=0; i<width_*height_*channels_; ++i) img_h[i] = (float)(img[imgid][i]);
        checkRuntime(cudaMemcpyAsync(img_d, img_h, width_*height_*channels_*sizeof(float), cudaMemcpyHostToDevice, _stream));
        // 1. norm
        num_blocks = (width_*height_*channels_ + threads_per_block - 1) / threads_per_block;
        norm_img_kernel<<<num_blocks, threads_per_block, 0, _stream>>>(img_d, img_normed_d, width_, height_, channels_);
        // 2. resize
        num_blocks = (width_resize_*height_resize_*channels_ + threads_per_block - 1) / threads_per_block;
        resizing_img_kernel<<<num_blocks, threads_per_block, 0, _stream>>>(img_normed_d, img_resized_d, width_, height_, channels_, width_resize_, height_resize_);
        // 3. padding
        num_blocks = (width_pad_*height_pad_*channels_ + threads_per_block - 1) / threads_per_block;
        padding_img_kernel<<<num_blocks, threads_per_block, 0, _stream>>>(img_resized_d, img_padded_d, width_resize_, height_resize_, channels_, width_pad_, height_pad_);
        // 4. whc to chw
        num_blocks = (width_pad_*height_pad_*channels_ + threads_per_block - 1) / threads_per_block;
        transpose_img_kernel<<<num_blocks, threads_per_block, 0, _stream>>>(img_padded_d, img_transposed_d, width_pad_, height_pad_, channels_);
        // 5. concatenate
        checkRuntime(cudaMemcpy(processed_img.data()+imgid*channels_*height_pad_*width_pad_, img_transposed_d, width_pad_*height_pad_*channels_*sizeof(float), cudaMemcpyDeviceToHost));
    }
    return processed_img;
}
