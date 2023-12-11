/**
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 *
 */

#ifndef CUDLA_CONTEXT_H
#define CUDLA_CONTEXT_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <vector>
#include <unordered_map>

#include <cudla.h>
#include <cuda_runtime.h>

#define checkCudaErrors(status)                                   \
{                                                                 \
  if (status != 0)                                                \
  {                                                               \
    std::cout << "Cuda failure: " << cudaGetErrorString(status)   \
              << " at line " << __LINE__                          \
              << " in file " << __FILE__                          \
              << " error status: " << status                      \
              << std::endl;                                       \
              abort();                                            \
    }                                                             \
}

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

#endif
