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

 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

__global__ void averagePoolNCHW(const float *A, float *C, int batchSize, int channels, int width, int height, int stride, int kernelSize)
{

	int tid = blockDim.x * blockIdx.x + threadIdx.x;// Get global thread ID
	int tpr = (width+stride-1)/stride;// Threads to work in a row
	int tpc = (height+stride-1)/stride;// Threads to work in a col
	int tgrpSize = tpr * tpc;// Number of threads to work on a image
	int tgrpNum = tid/tgrpSize;// Image number
	int tgrpLocaltid = (tid % tgrpSize);// local id in working image

	int startRow = tgrpNum * height;// starting row value of Image
	int localRow =  (tgrpLocaltid/tpr)*stride;// local row number in the working image
	int col = (tid%tpr) * stride;
	//check row boundaries
	if(startRow + localRow > (channels * batchSize * height -1))
		return;

	float outTemp = 0.0f;
	//Access elements from pooling window to compute Average pooling
	for(int i = 0 ; i < kernelSize && localRow + i < height ; i++)
		for( int j =0; j < kernelSize && col+ j < width; j++)
		{	  
			outTemp = outTemp + A[(startRow + localRow + i)*width+ col+ j];
		}

	C[tid] = outTemp/(kernelSize * kernelSize);// Store output
}
__global__ void averagePoolShared(const float *A, float *C, int batchSize, int channels, int width, int height, int stride, int kernelSize)
{
	extern __shared__ float sArr[];// Shared memory to store image data
	int size = height * width;// Image size
	int tpr = (width+stride-1)/stride;// Threads to work in a row
	int tpc = (height+stride-1)/stride;// Threads to work in a col
	int startOutIndex = blockIdx.x * tpr * tpc;// Starting output index of image

	//load image data into shared memory
	for(int i = threadIdx.x; i < size ; i += blockDim.x)
	{
		sArr[i] = A[blockIdx.x * size + i];
	}
	__syncthreads();

	if(threadIdx.x > tpr*tpc)
		return;
	// loop over image output indices
	for(int k = threadIdx.x ; k < tpr * tpc ; k += blockDim.x)
	{
		float outTemp = 0.0f;
		int row = (k/tpr) * stride;
		int col = (k%tpr) * stride;
		// Compute average pooling
		for(int i = 0 ; i < kernelSize && row + i < height ; i++)
			for( int j =0; j < kernelSize && col+ j < width; j++)
			{
				outTemp = outTemp+ sArr[(row + i)*width+ col+ j];
			}
		C[startOutIndex+k] = outTemp/(kernelSize * kernelSize); // Store output

	}
}

int avgPool(int N,
		const float* inputs,
		float* outputs,
		int C,
		int H,
		int W,
		int kernelSize,
		int begPad,
		int endPad,
		int stride,
		cudaStream_t stream)
{
	//size_t sharedMemorySize = sizeof(float)*((H * W) + (((W + stride -1)/stride)*((H + stride - 1)/stride)));
	size_t sharedMemorySize = sizeof(float)*((H * W));
	float* inArr = (float*)malloc(sizeof(float)*H*W);
	float* outArr = (float*)malloc(sizeof(float)*((W+stride-1)/stride)*((H+stride-1)/stride));
	cudaMemcpy(inArr,inputs,sizeof(float)*H*W,cudaMemcpyDeviceToHost);
	if (stride <  kernelSize)
	{	
		unsigned int blocksPerGrid = N * C;
		unsigned int threadsPerBlock = (W * H < 1024)?(W * H):1024;
		averagePoolShared<<<blocksPerGrid, threadsPerBlock, sharedMemorySize,stream>>>(inputs,
				outputs,
				N,
				C,
				W,
				H,
				stride,
				kernelSize);
	}
	else
	{			
		unsigned int threadsPerBlock = 1024;
		unsigned int blocksPerGrid =((N*C*H*W) + threadsPerBlock - 1) / threadsPerBlock;
		averagePoolNCHW<<<blocksPerGrid, threadsPerBlock,0,stream>>>(inputs,
				outputs,
				N,
				C,
				W,
				H,
				1,
				kernelSize);
	}
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n",
				__FILE__, __LINE__, cudaGetErrorString( err ) );

	}
	return 0;
}