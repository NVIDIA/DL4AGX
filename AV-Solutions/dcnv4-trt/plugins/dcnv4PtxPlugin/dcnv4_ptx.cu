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

#include <stdio.h>
#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvPTXCompiler.h>

#include "dcnv4_ptx.h"
#define STRING(s) #s
#define TO_LITERAL(s) STRING(s)
#define ARCH_NAME "--gpu-name=" TO_LITERAL(PTX_ARCH)
    
constexpr char *dcnv4_stage0_b128_code =
#include "./dcnv4_stage0_b128.ptx.h"
;
constexpr char *dcnv4_stage0_b128_kernel_entry = "dcn_kernel";

struct PtxCode { const char* code; const char* entry; };
constexpr char* stage0_ = 
#include "./dcnv4_stage0_b1.ptx.h"
;

constexpr char* stage1_ = 
#include "./dcnv4_stage1_b1.ptx.h"
;

constexpr char* stage2_ = 
#include "./dcnv4_stage2_b1.ptx.h"
;

constexpr char* stage3_ = 
#include "./dcnv4_stage3_b1.ptx.h"
;

#define NVPTXCOMPILER_SAFE_CALL(x)                                       \
    do {                                                                 \
        nvPTXCompileResult result = x;                                   \
        if (result != NVPTXCOMPILE_SUCCESS) {                            \
            printf("error: %s failed with error code %d\n", #x, result); \
            exit(1);                                                     \
        }                                                                \
    } while(0)

DCNv4Code kernel_codes[] = {
    {56, 7, 1, 128, 1, 1, stage0_, "dcn_kernel"},
    {14, 7, 1, 128, 1, 1, stage1_, "dcn_kernel"},
    { 7, 7, 1, 128, 1, 1, stage2_, "dcn_kernel"},
    { 7, 7, 1, 128, 1, 1, stage3_, "dcn_kernel"},
};

DCNv4Elf::DCNv4Elf() {
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult status;

    size_t infoSize, errorSize;
    char *infoLog, *errorLog;
    unsigned int minorVer, majorVer;

    const char* compile_options[] = {
        ARCH_NAME,
        "--verbose"};
    printf("arch=%s\n", ARCH_NAME);

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
    printf("Current PTX Compiler API Version : %d.%d\n", majorVer, minorVer);

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler,
                                                (size_t)strlen(dcnv4_stage0_b128_code),  /* ptxCodeLen */
                                                dcnv4_stage0_b128_kernel_entry)          /* ptxCode */
                            );

    status = nvPTXCompilerCompile(compiler,
                                  2,                 /* numCompileOptions */
                                  compile_options);  /* compileOptions */

    if (status != NVPTXCOMPILE_SUCCESS) {
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));

        if (errorSize != 0) {
            errorLog = (char*)malloc(errorSize+1);
            NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(compiler, errorLog));
            printf("Error log: %s\n", errorLog);
            free(errorLog);
        }
        exit(1);
    }

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler, &elf_size));

    elf = (char*) malloc(elf_size);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler, (void*)elf));
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));

    if (infoSize != 0) {
        infoLog = (char*)malloc(infoSize+1);
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));
    
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, dcnv4_stage0_b128_kernel_entry));
}

DCNv4Elf::DCNv4Elf(const char* elf, size_t elf_size) {
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, dcnv4_stage0_b128_kernel_entry));
}


DCNv4Elf::~DCNv4Elf() {
    CUDA_SAFE_CALL(cuModuleUnload(module));
    free(elf);
}

void DCNv4Elf::launch(
    const half* value, const half* offset, half* out
) {
    void* args[3];
    CUdeviceptr _val = (CUdeviceptr)value;
    CUdeviceptr _off = (CUdeviceptr)offset;
    CUdeviceptr _out = (CUdeviceptr)out;

    args[0] = &_val;
    args[1] = &_off;
    args[2] = &_out;

    // hard-coded grid/block size for given ptx kernel
    // currently assume: 
    //    value_proj [128 x 56 x 56 x 64], offset_proj [128 x 56 x 56 x 112]
    //    output [128 x 56 x 56 x 64]
    // and the grid size and block size are determined according to the assumed shapes
    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
                                   3584, 7, 1,      // grid dim
                                   128, 1, 1,       // block dim
                                   0, this->stream, // shared mem and stream configurations when we call the kernel
                                   args, 0));       // arguments
}

DCNv4Kernel::DCNv4Kernel() { /* wait for external init */ }
DCNv4Kernel::DCNv4Kernel(const char* elf, size_t elf_size, const char* entry) {
    mElf = new char[elf_size];
    std::memcpy((void*)mElf, elf, sizeof(char) * elf_size);
    mElfSize = elf_size;
    CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, elf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, entry));
}

DCNv4Kernel::~DCNv4Kernel() {
    // delete[] mElf;    
}

void DCNv4Kernel::Launch(
    const half* value, const half* offset, half* out, cudaStream_t stream
) {
    void* args[3];
    CUdeviceptr _val = (CUdeviceptr)value;
    CUdeviceptr _off = (CUdeviceptr)offset;
    CUdeviceptr _out = (CUdeviceptr)out;

    args[0] = &_val;
    args[1] = &_off;
    args[2] = &_out;

    CUDA_SAFE_CALL( cuLaunchKernel(kernel,
        code.mGridX, code.mGridY, code.mGridZ,
        code.mBlockX, code.mBlockY, code.mBlockZ,
        0, stream, 
        args, // kernel params
        0));  // extra params
}

DCNv4Elf_v2::DCNv4Elf_v2() {}

void DCNv4Elf_v2::Compile(int stage) {
    nvPTXCompilerHandle compiler = NULL;
    nvPTXCompileResult status;

    size_t infoSize, errorSize;
    char *infoLog, *errorLog;
    unsigned int minorVer, majorVer;

    const char* compile_options[] = {
        ARCH_NAME,
        "--verbose"};
    printf("arch=%s\n", ARCH_NAME);

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetVersion(&majorVer, &minorVer));
    printf("Current PTX Compiler API Version : %d.%d\n", majorVer, minorVer);

    DCNv4Code& code = kernel_codes[stage];

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerCreate(&compiler, (size_t)strlen(code.code), code.code));
    status = nvPTXCompilerCompile(compiler, 2, compile_options);

    if (status != NVPTXCOMPILE_SUCCESS) {
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLogSize(compiler, &errorSize));

        if (errorSize != 0) {
            errorLog = (char*)malloc(errorSize + 1);
            NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetErrorLog(compiler, errorLog));
            printf("Error log: %s\n", errorLog);
            free(errorLog);
        }
        exit(1);
    }

    DCNv4Kernel& k = kernels[stage];
    k.code = kernel_codes[stage];

    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgramSize(compiler, &k.mElfSize));

    k.mElf = (char*) malloc(k.mElfSize);
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetCompiledProgram(compiler, (void*)k.mElf));
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLogSize(compiler, &infoSize));

    if (infoSize != 0) {
        infoLog = (char*)malloc(infoSize + 1);
        NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerGetInfoLog(compiler, infoLog));
        printf("Info log: %s\n", infoLog);
        free(infoLog);
    }
    NVPTXCOMPILER_SAFE_CALL(nvPTXCompilerDestroy(&compiler));

    CUDA_SAFE_CALL(cuModuleLoadDataEx(&k.module, k.mElf, 0, 0, 0));
    CUDA_SAFE_CALL(cuModuleGetFunction(&k.kernel, k.module, code.entry));
}

void DCNv4Elf_v2::Setup(int stage, const char* elf, size_t elf_size) {
    kernels[stage] = DCNv4Kernel(elf, elf_size, kernel_codes[stage].entry);
    kernels[stage].code = kernel_codes[stage];
}

void DCNv4Elf_v2::Launch(int stage, const half* value, const half* offset, half* out, cudaStream_t stream) {
    kernels[stage].Launch(value, offset, out, stream);
}
