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
