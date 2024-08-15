/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef __PRE_PROCESS_CUH__
#define __PRE_PROCESS_CUH__

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>
#include "dtype.hpp"
#include <numeric>
#include <math.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <string.h>
#include <fstream>
#include <sys/stat.h>
#include "uniad.hpp"
#include "check.hpp"
#include "timer.hpp"
#include "tensorrt.hpp"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

static std::vector<unsigned char*> load_images(const std::vector<std::vector<std::string>>& infos, int idx) {
    // orders:
    // "0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg", "3-BACK.jpg",  "4-BACK_LEFT.jpg",   "5-BACK_RIGHT.jpg"};
    std::vector<unsigned char*> images;
    if (infos[idx].size() != 6) {
        printf("[ERROR] Incorrect info file.\n");
        return images;
    }
    for (size_t i=0; i<infos[idx].size(); ++i) {
        int width, height, channels;
        images.push_back(stbi_load(infos[idx][i].c_str(), &width, &height, &channels, 0));
    }
    return images;
}

static std::vector<unsigned char*> load_images(const std::vector<std::vector<std::string>>& infos, int idx, int& width, int& height, int &channels) {
    // orders:
    // "0-FRONT.jpg", "1-FRONT_RIGHT.jpg", "2-FRONT_LEFT.jpg", "3-BACK.jpg", "4-BACK_LEFT.jpg", "5-BACK_RIGHT.jpg"
    std::vector<unsigned char*> images;
    if (infos[idx].size() != 6) {
        printf("[ERROR] Incorrect info file.\n");
        return images;
    }
    for (size_t i=0; i<infos[idx].size(); ++i) {
        images.push_back(stbi_load(infos[idx][i].c_str(), &width, &height, &channels, 0));
    }
    return images;
}

static void free_images(std::vector<unsigned char*>& images) {
    for (size_t i = 0; i < images.size(); ++i) stbi_image_free(images[i]);
    images.clear();
}

class ImgPreProcess {
public:
    ImgPreProcess(const int width, const int height, const int channels, const float resize_scale, const int padding_divider);
    ~ImgPreProcess();
    std::vector<float> img_pre_processing(const std::vector<unsigned char*>& img, void *stream);
private:
    int width_, height_, channels_;
    int width_resize_, height_resize_;
    int width_pad_, height_pad_;
    float * img_h = nullptr;
    float * img_d = nullptr;
    float * img_normed_d = nullptr;
    float * img_resized_d = nullptr;
    float * img_padded_d = nullptr;
    float * img_transposed_d = nullptr;
};

#endif  // __PRE_PROCESS_CUH__
