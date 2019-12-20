/***************************************************************************************************
 * Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Project: CUDAMemoryTypesMatMul 
 * 
 * File: DL4AGX/CUDAMemoryTypesMatMul/main.cpp
 * 
 * Description: An application to show the difference between different CUDA Memory Types
 ***************************************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#include "CUDAMemoryTypesMatMul/helpers/cuda.h"
#include "CUDAMemoryTypesMatMul/helpers/string.h"

#include "CUDAMemoryTypesMatMul/kernels/matrixMulMem.h"

/**
 * Run a simple test of matrix multiplication using unified memory
 */

/**
 * Program main
 */
int main(int argc, char **argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	if (checkCmdLineFlag(argc, (const char **)argv, "help") ||
			checkCmdLineFlag(argc, (const char **)argv, "?"))
	{
		printf("Usage -device=n (n >= 0 for deviceID)\n");
		printf("      -wA=WidthA -hA=HeightA (Width x Height of Matrix A)\n");
		printf("      -wB=WidthB -hB=HeightB (Width x Height of Matrix B)\n");
		printf("	  -type=(0=Global memory, 1=Pinned memory)\n");
		printf("  Note: Outer matrix dimensions of A & B matrices must be equal.\n");

		exit(EXIT_SUCCESS);
	}




	int memType = 0;    
	// By default, we use global memory, otherwise we choose the memory type based on what is provided at the command line
	if (checkCmdLineFlag(argc, (const char **)argv, "type"))
	{
		memType = getCmdLineArgumentInt(argc, (const char **)argv, "type");
		if(memType != 0 &&  memType != 1)
		{
			printf("Enter --type value as 0 or 1\n");
			exit(EXIT_FAILURE);
		}


	}
	// By default, we disable shared memory optimization
	int enableSharedMemory=0;
	if (checkCmdLineFlag(argc, (const char **)argv, "enableSharedMemory"))
	{
		enableSharedMemory = getCmdLineArgumentInt(argc, (const char **)argv, "enableSharedMemory");
		if(enableSharedMemory != 0 &&  enableSharedMemory != 1)
		{
			printf("Enter --enableSharedMemory value as 0 or 1\n");
			exit(EXIT_FAILURE);
		}


	}

	// By default, we use device 0, otherwise we override the device ID based on what is provided at the command line
	int devID = 0;

	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
	{
		devID = getCmdLineArgumentInt(argc, (const char **)argv, "device");
		cudaSetDevice(devID);
	}

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Use a larger block size for Fermi and above
	int block_size = (deviceProp.major < 2) ? 16 : 32;

	dim3 dimsA(5*2*block_size, 5*2*block_size, 1);
	dim3 dimsB(5*4*block_size, 5*2*block_size, 1);

	// width of Matrix A
	if (checkCmdLineFlag(argc, (const char **)argv, "wA"))
	{
		dimsA.x = getCmdLineArgumentInt(argc, (const char **)argv, "wA");
	}

	// height of Matrix A
	if (checkCmdLineFlag(argc, (const char **)argv, "hA"))
	{
		dimsA.y = getCmdLineArgumentInt(argc, (const char **)argv, "hA");
	}

	// width of Matrix B
	if (checkCmdLineFlag(argc, (const char **)argv, "wB"))
	{
		dimsB.x = getCmdLineArgumentInt(argc, (const char **)argv, "wB");
	}

	// height of Matrix B
	if (checkCmdLineFlag(argc, (const char **)argv, "hB"))
	{
		dimsB.y = getCmdLineArgumentInt(argc, (const char **)argv, "hB");
	}

	if (dimsA.x != dimsB.y)
	{
		printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
				dimsA.x, dimsB.y);
		exit(EXIT_FAILURE);
	}

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n\n", dimsA.x, dimsA.y, dimsB.x, dimsB.y);
	int matrix_result = 0;
	if(memType == 0)
		matrix_result = matMul(argc, argv, block_size, dimsA, dimsB,enableSharedMemory);
	else if(memType == 1)	
		matrix_result = matMul_pinned(argc, argv, block_size, dimsA, dimsB,enableSharedMemory);
	else
		printf("Enter --type value as 0 or 1\n");
	exit(matrix_result);
}
