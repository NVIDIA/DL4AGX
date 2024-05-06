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

#include "cudla_context.h"

#ifdef USE_ORIN

CudlaContext::CudlaContext(const char *loadableFilePath, const int dla_core)
{
    readDLALoadable(loadableFilePath);
    if (!m_Initialized) {
        initialize(dla_core);
    } else {
	    std::cerr << "The CudlaContext was initialized." << std::endl; 
    }  
}

bool CudlaContext::readDLALoadable(const char *loadableFilePath) {
    FILE* fp = NULL;
    struct stat st;
    size_t actually_read = 0;

    // Read loadable into buffer.
    fp = fopen(loadableFilePath, "rb");
    if (fp == NULL) {
        std::cerr << "Cannot open file " << loadableFilePath << std::endl;
        return false;
    }

    if (stat(loadableFilePath, &st) != 0) {
        std::cerr << "Cannot stat file" << std::endl;
        return false;
    }

    m_File_size = st.st_size;

    m_LoadableData = (unsigned char*)malloc(m_File_size);
    if (m_LoadableData == NULL) {
        std::cerr << "Cannot Allocate memory for loadable" << std::endl;
        return false;
    }

    actually_read = fread(m_LoadableData, 1, m_File_size, fp);
    if (actually_read != m_File_size) {
        free(m_LoadableData);
        std::cerr << "Read wrong size" << std::endl;
        return false;
    }
    fclose(fp);
    return true;
}

bool
CudlaContext::initialize(const int dla_core) {
    cudlaStatus err;
    // err = cudlaCreateDevice(dla_core, &m_DevHandle, CUDLA_CUDA_DLA);
    err = cudlaCreateDevice(dla_core, &m_DevHandle, CUDLA_CUDA_DLA);
    if (err != cudlaSuccess)
    {
        std::cerr << "Error in cuDLA create device = " << err << std::endl;
        cleanUp();
        return false;
    }
    std::cout << "m_File_size = " << m_File_size << std::endl;

    err = cudlaModuleLoadFromMemory(
        m_DevHandle, m_LoadableData, m_File_size,
        &m_ModuleHandle, 0);

    if (err != cudlaSuccess) {
        std::cerr << "Error in cudlaModuleLoadFromMemory = " << err << std::endl;
        cleanUp();
        return false;
    }

    m_Initialized = true;
    if (!getTensorAttr()) {
        cleanUp();
        return false;
    }
    std::cout << "DLA CTX INIT !!!" << std::endl;

    for( int i=0; i<m_NumInputTensors; i++ ) {
        std::cout << m_InputTensorDesc[i].name << std::endl;
    }

    for( int i=0; i<m_NumOutputTensors; i++ ) {
        std::cout << m_OutputTensorDesc[i].name << std::endl;
    }
    return true;
}

uint64_t CudlaContext::getInputTensorSizeWithIndex(uint8_t index){
    return m_InputTensorDesc[index].size;
}

uint64_t CudlaContext::getOutputTensorSizeWithIndex(uint8_t index){
    return m_OutputTensorDesc[index].size;
}

bool CudlaContext::getTensorAttr() {
    if (!m_Initialized) {
	    return false;
    }

    // Get tensor attributes.
    cudlaStatus err;
    cudlaModuleAttribute attribute;

    err = cudlaModuleGetAttributes(
        m_ModuleHandle, 
        CUDLA_NUM_INPUT_TENSORS,
        &attribute);

    if (err != cudlaSuccess) {
        std::cerr << "Error in getting numInputTensors = " << err << std::endl;
        cleanUp();
        return false;
    }
    m_NumInputTensors = attribute.numInputTensors;

    err = cudlaModuleGetAttributes(
        m_ModuleHandle, 
        CUDLA_NUM_OUTPUT_TENSORS,
        &attribute);

    if (err != cudlaSuccess) {
        std::cerr << "Error in getting numOutputTensors = " << err << std::endl;
        cleanUp();
        return false;
    }
    m_NumOutputTensors = attribute.numOutputTensors;

    m_InputTensorDesc =
        (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor) *
                                           m_NumInputTensors);
    m_OutputTensorDesc =
        (cudlaModuleTensorDescriptor*)malloc(sizeof(cudlaModuleTensorDescriptor) *
                                           m_NumOutputTensors);

    if ((m_InputTensorDesc == NULL) || (m_OutputTensorDesc == NULL)) {
        std::cerr << "Error in allocating memory for TensorDesc" << std::endl;
        cleanUp();
        return false;
    }

    attribute.inputTensorDesc = m_InputTensorDesc;
    err = cudlaModuleGetAttributes(
        m_ModuleHandle, 
        CUDLA_INPUT_TENSOR_DESCRIPTORS,
        &attribute);

    if (err != cudlaSuccess) {
        std::cerr << "Error in getting input tensor descriptor = " << err << std::endl;
        cleanUp();
        return false;
    }
  
    attribute.outputTensorDesc = m_OutputTensorDesc;
    err = cudlaModuleGetAttributes(
        m_ModuleHandle, 
        CUDLA_OUTPUT_TENSOR_DESCRIPTORS,
        &attribute);

    if (err != cudlaSuccess) {
	    std::cerr << "Error in getting output tensor descriptor = " << err << std::endl;
        cleanUp();
        return false;
    }

    return true;
}

bool CudlaContext::bufferPrep(std::vector<void*> in_bufs, std::vector<void*> out_bufs, cudaStream_t m_stream) {
    if (!m_Initialized) {
        return false;
    }
    cudlaStatus err;

    m_InputBufferGPU = (void **)malloc(sizeof(void *)*m_NumInputTensors);
    if (m_InputBufferGPU == NULL) {
        std::cerr << "Error in allocating memory for input buffer GPU array" << std::endl;
	    cleanUp();
        return false;
    }

    for (int i = 0; i < in_bufs.size(); ++i) {
        m_InputBufferGPU[i] = in_bufs[i];
    }
    

    m_OutputBufferGPU = (void **)malloc(sizeof(void *)*m_NumOutputTensors);
    if (m_OutputBufferGPU == NULL)
    {
        std::cerr << "Error in allocating memory for output buffer GPU array" << std::endl;
        cleanUp();
        return false;
    }

    for (int i = 0; i < out_bufs.size(); ++i) {
        m_OutputBufferGPU[i] = out_bufs[i];
    }

    // Register the CUDA-allocated buffers.
    m_InputBufferRegisteredPtr = (uint64_t **)malloc(sizeof(uint64_t*)*m_NumInputTensors);
    m_OutputBufferRegisteredPtr = (uint64_t **)malloc(sizeof(uint64_t*)*m_NumOutputTensors);

    if ((m_InputBufferRegisteredPtr == NULL) || (m_OutputBufferRegisteredPtr == NULL))
    {
        std::cerr << "Error in allocating memory for BufferRegisteredPtr" << std::endl;
        cleanUp();
        return false;
    }

    for (uint32_t ii = 0; ii < m_NumInputTensors; ii++)
    {
        err = cudlaMemRegister( m_DevHandle,
                                (uint64_t* )(m_InputBufferGPU[ii]),
                                m_InputTensorDesc[ii].size,
                                &(m_InputBufferRegisteredPtr[ii]),
                                0);
        if (err != cudlaSuccess)
        {
            std::cerr << "Error in registering input tensor memory " << ii << ": "<< err << std::endl;
            cleanUp();
            return false;
        }
    }

    for (uint32_t ii = 0; ii < m_NumOutputTensors; ii++)
    {
        err = cudlaMemRegister(m_DevHandle,
                               (uint64_t* )(m_OutputBufferGPU[ii]),
                               m_OutputTensorDesc[ii].size,
                               &(m_OutputBufferRegisteredPtr[ii]),
                               0);
        if (err != cudlaSuccess)
        {
            std::cerr << "Error in registering output tensor memory " << ii << ": "<< err << std::endl;
            cleanUp();
            return false;
        }
    }
    
    std::cout << "ALL MEMORY REGISTERED SUCCESSFULLY" << std::endl;

    // Memset output buffer on GPU to 0.
    for (uint32_t ii = 0; ii < m_NumOutputTensors; ii++)
    {
        cudaError_t result = cudaMemsetAsync(m_OutputBufferGPU[ii], 0, m_OutputTensorDesc[ii].size, m_stream);
        if (result != cudaSuccess)
        {
            std::cerr << "Error in enqueueing memset for output tensor " << ii << std::endl;
	        cleanUp();
            return false;
        }
    }

    return true;
}

bool CudlaContext::submitDLATask(cudaStream_t m_stream) {
    if (m_InputBufferGPU == NULL || m_InputBufferGPU[0] == NULL)
	    return false;

    // std::cout << "SUBMIT CUDLA TASK" << std::endl;
    // std::cout << "    Input Tensor Num: " << m_NumInputTensors << std::endl;
    // std::cout << "    Output Tensor Num: " << m_NumOutputTensors << std::endl;

    cudlaTask task;
    task.moduleHandle = m_ModuleHandle;
    task.inputTensor = m_InputBufferRegisteredPtr;
    task.outputTensor = m_OutputBufferRegisteredPtr;
    task.numOutputTensors = m_NumOutputTensors;
    task.numInputTensors = m_NumInputTensors;
    task.waitEvents = NULL;
    task.signalEvents = NULL;

    // Enqueue a cuDLA task.
    cudlaStatus err = cudlaSubmitTask(m_DevHandle, &task, 1, m_stream, 0);
    if (err != cudlaSuccess) {
        std::cerr << "Error in submitting task : " << err << std::endl;
	    cleanUp();
        return false;
    }

    // std::cout << "SUBMIT IS DONE !!!" << std::endl;
    return true;
}

void CudlaContext::cleanUp() {
    if (m_InputTensorDesc != NULL) {
        free(m_InputTensorDesc);
        m_InputTensorDesc = NULL;
    }
    if (m_OutputTensorDesc != NULL) {
        free(m_OutputTensorDesc);
        m_OutputTensorDesc = NULL;
    }

    if (m_LoadableData != NULL) {
        free(m_LoadableData);
        m_LoadableData = NULL;
    }

    if (m_ModuleHandle != NULL) {
        cudlaModuleUnload(m_ModuleHandle, 0);
        m_ModuleHandle = NULL;
    }

    if (m_DevHandle != NULL) {
        cudlaDestroyDevice(m_DevHandle);
        m_DevHandle = NULL;
    }

    if (m_InputBufferGPU != NULL) {
        for (uint32_t ii = 0; ii < m_NumInputTensors; ii++) {
            if ((m_InputBufferGPU)[ii] != NULL) {
                cudaFree(m_InputBufferGPU[ii]);
                m_InputBufferGPU[ii] = NULL;
            }
        }
        free(m_InputBufferGPU);
        m_InputBufferGPU = NULL;
    }

    if (m_OutputBufferGPU != NULL) {
	    for (uint32_t ii = 0; ii < m_NumOutputTensors; ii++) {
            if ((m_OutputBufferGPU)[ii] != NULL) {
                cudaFree(m_OutputBufferGPU[ii]);
                m_OutputBufferGPU[ii] = NULL;
            }
        }
        free(m_OutputBufferGPU);
        m_OutputBufferGPU = NULL;
    }

    if (m_InputBufferRegisteredPtr != NULL) {
        free(m_InputBufferRegisteredPtr);
        m_InputBufferRegisteredPtr = NULL;
    }

    if (m_OutputBufferRegisteredPtr != NULL) {
        free(m_OutputBufferRegisteredPtr);
        m_OutputBufferRegisteredPtr = NULL;
    }

    m_NumInputTensors = 0;
    m_NumOutputTensors = 0;
    std::cout << "DLA CTX CLEAN UP !!!" << std::endl;
}

CudlaContext::~CudlaContext() {
    cudlaStatus err;
    for (uint32_t ii = 0; ii < m_NumInputTensors; ii++) {
        err = cudlaMemUnregister(m_DevHandle, m_InputBufferRegisteredPtr[ii]);
        if (err != cudlaSuccess) {
            std::cerr << "Error in unregistering input tensor memory " 
                      << ii << ": " << err << std::endl;
        }
    }

    for (uint32_t ii = 0; ii < m_NumOutputTensors; ii++) {
        err = cudlaMemUnregister(m_DevHandle, m_OutputBufferRegisteredPtr[ii]);
        if (err != cudlaSuccess) {
	        std::cerr << "Error in unregistering output tensor memory " 
                      << ii << ": " << err << std::endl;
        }
    }
}

#endif // USE_ORIN
