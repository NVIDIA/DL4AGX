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
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>
 
struct Memory {
    int mem_len = 640;
    int pre_len = 512;
    void* mem_buf; // 640
    float* pre_buf;  // 1, 512, 1
    float* post_buf; // 1, 640, 1

    cudaStream_t mem_stream;

    Memory() {
        cudaMalloc(&mem_buf, sizeof(double) * 640);
    }

    void StepReset();
    void StepPre(double ts);
    void StepPost(double ts);

    void DebugPrint();
}; // struct Memory
