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

#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace optimize {
    int convert_float_to_int8(void* d_a, int8_t* d_b, int size, float scale, cudaStream_t stream);
    int convert_half_to_int8(void* d_a, int8_t* d_b, int size, float scale, cudaStream_t stream);
    int convert_int8_to_float(int8_t* d_a, float* d_b, int size, float scale);
} // namespace optimize

int preprocess_image(unsigned char* h_images, float* h_buffer, const int size, cudaStream_t stream);
