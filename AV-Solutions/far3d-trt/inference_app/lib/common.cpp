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
 
#include "tensor.hpp"

namespace nv
{
    size_t numel(const nvinfer1::Dims& dims)
    {
        size_t output = 1;
        for(int32_t i = 0; i < dims.nbDims; ++i)
        {
            output *= dims.d[i];
        }
        return output;
    }

    void cudaFreeWrapper(void* ptr)
    {
        cudaError_t err = cudaFree(ptr);
        // TODO error handling
    }

    void cudaFreeHostWrapper(void* ptr)
    {
        cudaError_t err = cudaFreeHost(ptr);
        // TODO error handling
    }
}
