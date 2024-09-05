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

#pragma once
#ifndef _DCNV4_PTX_H_
#define _DCNV4_PTX_H_
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>

#define CUDA_SAFE_CALL(x)                                               \
    do {                                                                \
        CUresult result = x;                                            \
        if (result != CUDA_SUCCESS) {                                   \
            const char *msg;                                            \
            cuGetErrorName(result, &msg);                               \
            printf("error: %s failed with error %s\n", #x, msg);        \
            exit(1);                                                    \
        }                                                               \
    } while(0)

class DCNv4Elf {
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;    
public:
    cudaStream_t stream;
    char* elf;
    size_t elf_size;

    DCNv4Elf();
    DCNv4Elf(const char* elf, size_t elf_size);

    ~DCNv4Elf();
    void launch(
        const half* value, const half* offset, half* out
    );
}; // class DCNv4Elf

struct DCNv4Code {
    int mGridX, mGridY, mGridZ;
    int mBlockX, mBlockY, mBlockZ;
    const char* code; const char* entry; 
};
extern DCNv4Code kernel_codes[];

struct DCNv4Kernel {
    CUmodule module;
    CUfunction kernel;
    DCNv4Code code;

    const char* mElf;
    size_t mElfSize;

    DCNv4Kernel();
    DCNv4Kernel(const char* elf, size_t elf_size, const char* entry);

    ~DCNv4Kernel();
    void Launch(const half* value, const half* offset, half* out, cudaStream_t stream);
};

struct DCNv4Elf_v2 {
    CUdevice cuDevice;
    CUcontext context;
    CUmodule module;
    DCNv4Kernel kernels[4];
    DCNv4Elf_v2();
    ~DCNv4Elf_v2() {}
    void Compile(int stage);
    void Setup(int stage, const char* elf, size_t elf_size);
    void Launch(int stage, const half* value, const half* offset, half* out, cudaStream_t stream);
}; // class DCNv4Elf_v2

#endif // _DCNV4_PTX_H_
