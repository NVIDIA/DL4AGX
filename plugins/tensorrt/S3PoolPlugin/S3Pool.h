/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_S3POOL_PLUGIN_H
#define TRT_S3POOL_PLUGIN_H
#include "plugin.h"
#include <string>
#include <vector>

// #include "NvInferPluginUtils.h"

typedef struct
{
    std::vector<int> kernel_shape, beg_padding, end_padding, strides;    
}S3PoolingParams;

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace nvinfer1
{
namespace plugin
{

class S3Pool : public IPluginV2Ext
{
public:
    S3Pool(S3PoolingParams params, int nKernelShape, int nBegPad, int nEndPad, int nStrides, int N, int C, int H, int W);

    S3Pool(const void* buffer, size_t length);

    ~S3Pool() override = default;

    int getNbOutputs() const override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    int initialize() override;

    void terminate() override;

    size_t getWorkspaceSize(int maxBatchSize) const override;

    int enqueue(
        int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override;

    void serialize(void* buffer) const override;

    bool supportsFormat(nvinfer1::DataType type, PluginFormat format) const override;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    void destroy() override;

    IPluginV2Ext* clone() const override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;

    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;

    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const override;

    void attachToContext(
        cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator) override;

    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
        const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes, const bool* inputIsBroadcast,
        const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize) override;

    void detachFromContext() override;


private:
    template <typename T>
    void write(char*& buffer, const T& val) const
    {
        std::memcpy(buffer, &val, sizeof(T));
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer)
    {
        T val;
        std::memcpy(&val, buffer, sizeof(T));
        buffer += sizeof(T);
        return val;
    }

    S3PoolingParams mParams;
    int nKernelShape, nBegPad, nEndPad, nStrides;
    int mN, mC, mH, mW;

    const char* mPluginNamespace = "";
};

class S3PoolPluginCreator : public BaseCreator
{
public:
    S3PoolPluginCreator();

    ~S3PoolPluginCreator() override = default;

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const PluginFieldCollection* getFieldNames() override;

    IPluginV2Ext* createPlugin(const char* name, const PluginFieldCollection* fc) override;

    IPluginV2Ext* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

private:
    static PluginFieldCollection mFC;
    S3PoolingParams params;
    int nkernelShape, nStrides, nBegPad, nEndPad;
    static std::vector<PluginField> mPluginAttributes;
    std::string mPluginNamespace = "";

};

} // namespace plugin
} // namespace nvinfer1

PluginFieldCollection S3PoolPluginCreator::mFC{};
std::vector<PluginField> S3PoolPluginCreator::mPluginAttributes;
REGISTER_TENSORRT_PLUGIN(S3PoolPluginCreator);




#endif // TRT_S3Pool_PLUGIN_H
