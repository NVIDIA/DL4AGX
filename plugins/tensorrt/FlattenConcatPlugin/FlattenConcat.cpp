/**************************************************************************
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
 * File: DL4AGX/plugins/tensorrt/FlattenConcatPlugin/FlattenConcat.cpp
 * Description: Implementation of FlattenConcat Plugin
 *************************************************************************/

#include "FlattenConcat.h"
#include "NvInferPlugin.h"
#include "cublas_v2.h"

using namespace nvinfer1;

FlattenConcat::FlattenConcat(const int* flattenedInputSize, int numInputs, int flattenedOutputSize)
    : mFlattenedOutputSize(flattenedOutputSize)
{
    for (int i = 0; i < numInputs; ++i)
    {
        mFlattenedInputSize.push_back(flattenedInputSize[i]);
    }
}

FlattenConcat::FlattenConcat(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;

    size_t numInputs = read<size_t>(d);
    for (size_t i = 0; i < numInputs; ++i)
    {
        mFlattenedInputSize.push_back(read<int>(d));
    }
    mFlattenedOutputSize = read<int>(d);

    assert(d == a + length);
}

int FlattenConcat::getNbOutputs() const
{
    // We always return one output
    return 1;
}

Dims FlattenConcat::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // At least one input
    assert(nbInputDims >= 1);
    // We only have one output, so it doesn't
    // make sense to check index != 0
    assert(index == 0);

    size_t flattenedOutputSize = 0;
    int inputVolume = 0;

    for (int i = 0; i < nbInputDims; ++i)
    {
        // We only support NCHW. And inputs Dims are without batch num.
        assert(inputs[i].nbDims == 3);

        inputVolume = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        flattenedOutputSize += inputVolume;
    }

    return DimsCHW(flattenedOutputSize, 1, 1);
}

int FlattenConcat::initialize()
{
    // Called on engine initialization, we initialize cuBLAS library here,
    // since we'll be using it for inference
    CHECK(cublasCreate(&mCublas));
    return 0;
}

void FlattenConcat::terminate()
{
    // Called on engine destruction, we destroy cuBLAS data structures,
    // which were created in initialize()
    CHECK(cublasDestroy(mCublas));
}

size_t FlattenConcat::getWorkspaceSize(int maxBatchSize) const
{
    // The operation is done in place, it doesn't use GPU memory
    return 0;
}

int FlattenConcat::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    // Does the actual concat of inputs, which is just
    // copying all inputs bytes to output byte array
    size_t inputOffset = 0;
    float* output = reinterpret_cast<float*>(outputs[0]);
    cublasSetStream(mCublas, stream);

    for (size_t i = 0; i < mFlattenedInputSize.size(); ++i)
    {
        const float* input = reinterpret_cast<const float*>(inputs[i]);
        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx)
        {
            CHECK(cublasScopy(mCublas, mFlattenedInputSize[i],
                              input + batchIdx * mFlattenedInputSize[i], 1,
                              output + (batchIdx * mFlattenedOutputSize + inputOffset), 1));
        }
        inputOffset += mFlattenedInputSize[i];
    }

    return 0;
}

size_t FlattenConcat::getSerializationSize() const
{
    // Returns FlattenConcat plugin serialization size
    size_t size = sizeof(mFlattenedInputSize[0]) * mFlattenedInputSize.size()
        + sizeof(mFlattenedOutputSize)
        + sizeof(size_t); // For serializing mFlattenedInputSize vector size
    return size;
}

void FlattenConcat::serialize(void* buffer) const
{
    // Serializes FlattenConcat plugin into byte array

    // Cast buffer to char* and save its beginning to a,
    // (since value of d will be changed during write)
    char* d = reinterpret_cast<char*>(buffer);
    char* a = d;

    size_t numInputs = mFlattenedInputSize.size();

    // Write FlattenConcat fields into buffer
    write(d, numInputs);
    for (size_t i = 0; i < numInputs; ++i)
    {
        write(d, mFlattenedInputSize[i]);
    }
    write(d, mFlattenedOutputSize);

    // Sanity check - checks if d is offset
    // from a by exactly the size of serialized plugin
    assert(d == a + getSerializationSize());
}

void FlattenConcat::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputDims, int nbOutputs, nvinfer1::DataType type, nvinfer1::PluginFormat format, int maxBatchSize)
{
    // We only support one output
    assert(nbOutputs == 1);

    // Reset plugin private data structures
    mFlattenedInputSize.clear();
    mFlattenedOutputSize = 0;

    // For each input we save its size, we also validate it
    for (int i = 0; i < nbInputs; ++i)
    {
        int inputVolume = 0;

        // We only support NCHW. And inputs Dims are without batch num.
        assert(inputs[i].nbDims == 3);

        // All inputs dimensions along non concat axis should be same
        for (size_t dim = 1; dim < 3; dim++)
        {
            assert(inputs[i].d[dim] == inputs[0].d[dim]);
        }

        // Size of flattened input
        inputVolume = inputs[i].d[0] * inputs[i].d[1] * inputs[i].d[2];
        mFlattenedInputSize.push_back(inputVolume);
        mFlattenedOutputSize += mFlattenedInputSize[i];
    }
}

bool FlattenConcat::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* FlattenConcat::getPluginType() const { return FLATTENCONCAT_PLUGIN_NAME; }

const char* FlattenConcat::getPluginVersion() const { return FLATTENCONCAT_PLUGIN_VERSION; }

IPluginV2* FlattenConcat::clone() const
{
    return new FlattenConcat(mFlattenedInputSize.data(), mFlattenedInputSize.size(), mFlattenedOutputSize);
}

void FlattenConcat::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* FlattenConcat::getPluginNamespace() const
{
    return mPluginNamespace.c_str();
}
