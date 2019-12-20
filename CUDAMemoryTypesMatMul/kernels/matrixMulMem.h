#ifndef MATRIX_MUL_MEM_H_
#define MATRIX_MUL_MEM_H_
#pragma once
#include <cuda_runtime.h>

int matMul(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB, int enableSharedMemory);
int matMul_pinned(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB,int enableSharedMemory);

#endif //MATRIX_MUL_MEM_H_
