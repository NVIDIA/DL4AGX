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

#ifdef USE_ORIN

#include <sys/stat.h>
#include <iostream>
#include <vector>

#include <cudla.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInferRuntime.h>

class CudlaContext
{
public:
    CudlaContext(const char *loadableFilePath, const int dla_core);
    ~CudlaContext();

    uint64_t getInputTensorSizeWithIndex(uint8_t index);
    uint64_t getOutputTensorSizeWithIndex(uint8_t index);

    bool initialize(const int dla_core);
    bool bufferPrep(std::vector<void*> in_bufs, std::vector<void*> out_bufs, cudaStream_t m_stream);
    bool submitDLATask(cudaStream_t m_stream);
    void cleanUp();
    
private:
    bool readDLALoadable(const char *loadableFilePath);
    bool getTensorAttr();

    cudlaDevHandle m_DevHandle;
    cudlaModule m_ModuleHandle;
    unsigned char* m_LoadableData = NULL;
    uint32_t m_NumInputTensors;
    uint32_t m_NumOutputTensors;
    cudlaModuleTensorDescriptor* m_InputTensorDesc;
    cudlaModuleTensorDescriptor* m_OutputTensorDesc;
    void** m_InputBufferGPU;
    void** m_OutputBufferGPU;
    uint64_t** m_InputBufferRegisteredPtr;
    uint64_t** m_OutputBufferRegisteredPtr;

    size_t m_File_size;
    bool m_Initialized;
};

#endif // ORIN
