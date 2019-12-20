#pragma once
#include <cuda_runtime.h>

int avgPool(int batchSize,
            const float* inputs,
			float* outputs,
			int mC,
			int mH,
			int mW,
			int nKernelShape,
			int nBegPad,
			int nEndPad,
			int nStrides,
			cudaStream_t stream);

