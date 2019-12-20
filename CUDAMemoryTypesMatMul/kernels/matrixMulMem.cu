/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling approach.
 * It has been written for clarity of exposition to illustrate various CUDA programming
 * principles, not with the goal of providing the most performant generic kernel for matrix multiplication.
 * See also:
 * V. Volkov and J. Demmel, "Benchmarking GPUs to tune dense linear algebra,"
 * in Proc. 2008 ACM/IEEE Conf. on Supercomputing (SC '08),
 * Piscataway, NJ: IEEE Press, 2008, pp. Art. 31:1-11.
 */



// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <sys/time.h>

// Helper functions and utilities to work with CUDA
#include "CUDAMemoryTypesMatMul/helpers/cuda.h"

#define nIter (1)


/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
	template <int BLOCK_SIZE> __global__ void
matrixMulCUDA(float *C, float *A, float *B, int wA, int wB, int enableSharedMemory)
{
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;

	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	// Index of the first sub-matrix of A processed by the block
	int aBegin = wA * BLOCK_SIZE * by;

	// Index of the last sub-matrix of A processed by the block
	int aEnd   = aBegin + wA - 1;

	// Step size used to iterate through the sub-matrices of A
	int aStep  = BLOCK_SIZE;

	// Index of the first sub-matrix of B processed by the block
	int bBegin = BLOCK_SIZE * bx;

	// Step size used to iterate through the sub-matrices of B
	int bStep  = BLOCK_SIZE * wB;

	// Csub is used to store the element of the block sub-matrix
	// that is computed by the thread
	float Csub = 0;

	// Loop over all the sub-matrices of A and B
	// required to compute the block sub-matrix
	for (int a = aBegin, b = bBegin;
			a <= aEnd;
			a += aStep, b += bStep)
	{
		if(enableSharedMemory == 1)
		{
			// Declaration of the shared memory array As used to
			// store the sub-matrix of A
			__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

			// Declaration of the shared memory array Bs used to
			// store the sub-matrix of B
			__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

			// Load the matrices from device memory
			// to shared memory; each thread loads
			// one element of each matrix
			As[ty][tx] = A[a + wA * ty + tx];
			Bs[ty][tx] = B[b + wB * ty + tx];

			// Synchronize to make sure the matrices are loaded
			__syncthreads();
#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; ++k)
			{
				Csub += As[ty][k] * Bs[k][tx];
			}

		}
		else
		{	
#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; ++k)
			{

				Csub += A[a + wA * ty + k] * B[b + wB * k + tx];
			}
		}		
		// Multiply the two matrices together;
		// each thread computes one element
		// of the block sub-matrix


		// Synchronize to make sure that the preceding
		// computation is done before loading two new
		// sub-matrices of A and B in the next iteration
		__syncthreads();
	}

	// Write the block sub-matrix to device memory;
	// each thread writes one element
	int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
	C[c + wB * ty + tx] = Csub;
}

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

void checkResult( float *h_C, dim3 &dimsC, dim3 &dimsA, float valB)
{
#if 0
	bool correct = true;

	printf("Checking computed result for correctness: ");

	// test relative error by the formula
	//     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
	double eps = 1.e-6 ; // machine zero

	for (int i = 0; i < (int)(dimsC.x * dimsC.y); i++)
	{
		double abs_err = fabs(h_C[i] - (dimsA.x * valB));
		double dot_length = dimsA.x;
		double abs_val = fabs(h_C[i]);
		double rel_err = abs_err/abs_val/dot_length ;

		if (rel_err > eps)
		{
			printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i, h_C[i], dimsA.x*valB, eps);
			correct = false;
		}
	}

	printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
#endif
}

/**
 * Run a simple test of matrix multiplication using CUDA global memory
 */
int matMul(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB, int enableSharedMemory)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *h_B = (float *)malloc(mem_size_B);
	printf("Test#1: Global/Conventional Memory Test:\n");
	// Initialize host memory
	const float valB = 0.01f;
	constantInit(h_A, size_A, 1.0f);
	constantInit(h_B, size_B, valB);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	// Allocate host matrix C
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *h_C = (float *) malloc(mem_size_C);
	if (h_C == NULL) {
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaMalloc((void **) &d_A, mem_size_A));
	checkCudaErrors(cudaMalloc((void **) &d_B, mem_size_B));
	checkCudaErrors(cudaMalloc((void **) &d_C, mem_size_C));

	// copy host memory to device
	struct timeval old_tv, cur_tv;

	gettimeofday(&old_tv, NULL);
	checkCudaErrors(cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice));
	gettimeofday(&cur_tv, NULL);
	printf("cudaMemcpy (%.2f KB): %.2f msec\n", ((float)(mem_size_A + mem_size_B)/1024),
			((float)(cur_tv.tv_sec - old_tv.tv_sec) * 1000000 + (cur_tv.tv_usec - old_tv.tv_usec))/1000);

	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	// Create and start timer
	//printf("Performs warmup operation...");

	// Performs warmup operation using matrixMul CUDA kernel
	if (block_size == 16) {
		matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
	} else {
		matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
	}

	//printf("done\n");

	cudaDeviceSynchronize();

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));

	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));

	// Execute the kernel

	for (int j = 0; j < nIter; j++) {
		if (block_size == 16) {
			matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
		} else {
			matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
		}
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
			"Performance= %.2f GFlop/s, Kernel Execution Time= %.3f msec\n",
			gigaFlops,
			msecPerMatrixMul);

	// Copy result from device to host
	checkCudaErrors(cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost));

	checkResult(h_C, dimsC, dimsA, valB);

	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));


	return EXIT_SUCCESS;
}

/**
 * Run a simple test of matrix multiplication using pinned memory
 */
int matMul_pinned(int argc, char **argv, int block_size, dim3 &dimsA, dim3 &dimsB,int enableSharedMemory)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = dimsA.x * dimsA.y;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float *A;
	unsigned int size_B = dimsB.x * dimsB.y;
	unsigned int mem_size_B = sizeof(float) * size_B;
	float *B;
	dim3 dimsC(dimsB.x, dimsA.y, 1);
	unsigned int mem_size_C = dimsC.x * dimsC.y * sizeof(float);
	float *C;
	printf("Test#2: Pinned memory Test:\n");
	checkCudaErrors(cudaMallocHost(&A, mem_size_A, cudaHostAllocMapped));
	checkCudaErrors(cudaMallocHost(&B, mem_size_B, cudaHostAllocMapped));
	checkCudaErrors(cudaMallocHost(&C, mem_size_C, cudaHostAllocMapped));

	// Initialize host memory
	const float valB = 0.01f;
	constantInit(A, size_A, 1.0f);
	constantInit(B, size_B, valB);

	// device pointers
	float *d_A, *d_B, *d_C;
	struct timeval old_tv, cur_tv;

	gettimeofday(&old_tv, NULL);
	checkCudaErrors(cudaHostGetDevicePointer((void **)&d_A, (void *)A, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void **)&d_B, (void *)B, 0));
	checkCudaErrors(cudaHostGetDevicePointer((void **)&d_C, (void *)C, 0));
	gettimeofday(&cur_tv, NULL);
	printf("cudaHostGetDevicePointer (%.2f KB): %.f msec\n", (float)(mem_size_A + mem_size_B)/1024,
			((float)(cur_tv.tv_sec - old_tv.tv_sec) * 1000000 + (cur_tv.tv_usec - old_tv.tv_usec))/1000);


	// Setup execution parameters
	dim3 threads(block_size, block_size);
	dim3 grid(dimsB.x / threads.x, dimsA.y / threads.y);

	// Create and start timer
	//printf("Performs warmup operation...");

	// Performs warmup operation using matrixMul CUDA kernel
	if (block_size == 16) {
		matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
	} else {
		matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
	}

	//printf("done\n");

	cudaDeviceSynchronize();

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	checkCudaErrors(cudaEventCreate(&start));

	cudaEvent_t stop;
	checkCudaErrors(cudaEventCreate(&stop));

	// Record the start event
	checkCudaErrors(cudaEventRecord(start, NULL));

	// Execute the kernel

	for (int j = 0; j < nIter; j++) {
		if (block_size == 16) {
			matrixMulCUDA<16><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
		} else {
			matrixMulCUDA<32><<< grid, threads >>>(d_C, d_A, d_B, dimsA.x, dimsB.x,enableSharedMemory);
		}
	}

	// Record the stop event
	checkCudaErrors(cudaEventRecord(stop, NULL));

	// Wait for the stop event to complete
	checkCudaErrors(cudaEventSynchronize(stop));

	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));

	// Compute and print the performance
	float msecPerMatrixMul = msecTotal / nIter;
	double flopsPerMatrixMul = 2.0 * (double)dimsA.x * (double)dimsA.y * (double)dimsB.x;
	double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
	printf(
			"Performance= %.2f GFlop/s, Kernel Execution Time= %.3f msec\n",
			gigaFlops,
			msecPerMatrixMul);

	checkResult(C, dimsC, dimsA, valB);

	// Clean up memory
	checkCudaErrors(cudaFreeHost(A));
	checkCudaErrors(cudaFreeHost(B));
	checkCudaErrors(cudaFreeHost(C));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	return EXIT_SUCCESS;
}

