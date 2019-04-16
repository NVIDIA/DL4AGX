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
 * File: DL4AGX/plugins/tensorrt/FlattenConcatPlugin/FlattenConcat.h
 * Description: Interface of FlattenConcat Plugin
 *************************************************************************/
#include "common/common.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>

#include <cublas_v2.h>

#include "NvInferPlugin.h"

using namespace nvinfer1;

namespace
{
const char* FLATTENCONCAT_PLUGIN_VERSION{"1"};
const char* FLATTENCONCAT_PLUGIN_NAME{"FlattenConcat_TRT"};
}

// Flattens all input tensors and concats their flattened version together
// along the major non-batch dimension, i.e axis = 1
class FlattenConcat : public IPluginV2
{
public:
    // Ordinary ctor, plugin not yet configured for particular inputs/output
    FlattenConcat(){};
    // Ctor for clone()
    FlattenConcat(const int* flattenedInputSize, int numInputs, int flattenedOutputSize);
    // Ctor for loading from serialized byte array
    FlattenConcat(const void* data, size_t length);
    int getNbOutputs() const override;
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override;
    int initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    int enqueue(int batchSize,
                const void* const* inputs,
                void** outputs,
                void*,
                cudaStream_t stream) override;
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    void configureWithFormat(const Dims* inputs,
                             int nbInputs,
                             const Dims* outputDims,
                             int nbOutputs,
                             nvinfer1::DataType type,
                             nvinfer1::PluginFormat format,
                             int maxBatchSize) override;
    bool supportsFormat(DataType type, PluginFormat format) const override;
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    void destroy() override {}
    IPluginV2* clone() const override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;

private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    // Number of elements in each plugin input, flattened
    std::vector<int> mFlattenedInputSize;
    // Number of elements in output, flattened
    int mFlattenedOutputSize{0};
    // cuBLAS library handle
    cublasHandle_t mCublas;
    // We're not using TensorRT namespaces in
    // this sample, so it's just an empty string
    std::string mPluginNamespace = "";
};

// PluginCreator boilerplate code for FlattenConcat plugin
class FlattenConcatPluginCreator : public IPluginCreator
{
public:
    FlattenConcatPluginCreator()
    {
        mFC.nbFields = 0;
        mFC.fields = 0;
    }

    ~FlattenConcatPluginCreator() {}

    const char* getPluginName() const override { return FLATTENCONCAT_PLUGIN_NAME; }

    const char* getPluginVersion() const override { return FLATTENCONCAT_PLUGIN_VERSION; }

    const PluginFieldCollection* getFieldNames() override { return &mFC; }

    IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) override
    {
        return new FlattenConcat();
    }

    IPluginV2* deserializePlugin(const char* name,
                                 const void* serialData,
                                 size_t serialLength) override
    {

        return new FlattenConcat(serialData, serialLength);
    }

    void setPluginNamespace(const char* pluginNamespace) override
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const override
    {
        return mPluginNamespace.c_str();
    }

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace = "";
};

PluginFieldCollection FlattenConcatPluginCreator::mFC{};
std::vector<PluginField> FlattenConcatPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(FlattenConcatPluginCreator);
