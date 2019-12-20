#include "S3Pool.h"
#include "NvInferPlugin.h"
#include "averagePool.h"
#include <cassert>
#include <string.h>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

#define ASSERT assert

namespace 
{
    constexpr const char* S3POOL_PLUGIN_NAME{"S3Pool_TRT"};
    constexpr const char* S3POOL_PLUGIN_VERSION{"001"};
}

S3Pool::S3Pool(S3PoolingParams params, int nKernelShape, int nBegPad, int nEndPad, int nStrides, int N = 0, int C = 0, int H = 0, int W = 0)
    : mParams(params)
    , nKernelShape(nKernelShape)
    , nBegPad(nBegPad)
    , nEndPad(nEndPad)
    , nStrides(nStrides)
    , mN(N)
    , mC(C)
    , mH(H)
    , mW(W)
{

}

S3Pool::S3Pool(const void* data, size_t length)
{
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;

    mN = read<int>(d);
    mC = read<int>(d);
    mH = read<int>(d);
    mW = read<int>(d);
    nKernelShape = read<int>(d);
    for (int i=0; i<nKernelShape; i++)
    {
        mParams.kernel_shape.push_back(read<int>(d));
    }

    nBegPad = read<int>(d);
    for (int i=0; i<nBegPad; i++)
    {
        mParams.beg_padding.push_back(read<int>(d));
    }

    nEndPad = read<int>(d);
    for (int i=0; i<nEndPad; i++)
    {
        mParams.end_padding.push_back(read<int>(d));
    }

    nStrides = read<int>(d);
    for (int i=0; i<nStrides; i++)
    {
        mParams.strides.push_back(read<int>(d));
    }
    assert(d == a + length);
}

int S3Pool::getNbOutputs() const
{
    // We always return one output
    return 1;
}

nvinfer1::Dims S3Pool::getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims)
{
    return *inputs;
}

int S3Pool::initialize()
{
    return 0;
}

void S3Pool::terminate()
{
}

size_t S3Pool::getWorkspaceSize(int maxBatchSize) const
{
    // The operation is done in place, it doesn't use GPU memory
    return 0;
}

int S3Pool::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{

    return avgPool(batchSize,
                   static_cast<const float *>(inputs[0]),
                   static_cast<float *>(outputs[0]),
                   mC,
                   mH,
                   mW,
                   nKernelShape,
                   nBegPad,
                   nEndPad,
                   nStrides,
                   stream);
}

void S3Pool::serialize(void* buffer) const
{
    char *d = reinterpret_cast<char *>(buffer), *a = d;
    write(d, mN);
    write(d, mC);
    write(d, mH);
    write(d, mW);
    write(d, nKernelShape);
    for (int i=0; i<nKernelShape; i++)
    {
        write(d, mParams.kernel_shape[i]);
    }
    write(d, nBegPad);
    for (int i=0; i<nBegPad; i++)
    {
        write(d, mParams.beg_padding[i]);
    }

    write(d, nEndPad);
    for (int i=0; i<nEndPad; i++)
    {
        write(d, mParams.beg_padding[i]);
    }

    write(d, nStrides);
    for (int i=0; i<nStrides; i++)
    {
        write(d, mParams.strides[i]);
    }
    ASSERT(d == a + getSerializationSize());
}

size_t S3Pool::getSerializationSize() const
{
    // nKernelShape, nBegPad, nEndPad, nStrides, N, C, H, W => 8
    return  (8 +
                    mParams.kernel_shape.size() + 
                    mParams.beg_padding.size() +
                    mParams.end_padding.size() +
                    mParams.strides.size()) * sizeof(int);
                     
}
// Set plugin namespace
void S3Pool::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* S3Pool::getPluginNamespace() const
{
    return mPluginNamespace;
}

void S3Pool::configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs,
    const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes, const bool* inputIsBroadcast,
    const bool* outputIsBroadcast, PluginFormat floatFormat, int maxBatchSize)
{

    ASSERT(inputDims->nbDims == 3);
    // Configured with batch size = 1
    mN = 1;
    mC = inputDims->d[0];
    mH = inputDims->d[1];
    mW = inputDims->d[2];
}


// Return the nvinfer1::DataType of the plugin output at the requested index
DataType S3Pool::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    ASSERT(index == 0);
    return nvinfer1::DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool S3Pool::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool S3Pool::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

bool S3Pool::supportsFormat(DataType type, PluginFormat format) const
{
    return (type == nvinfer1::DataType::kFLOAT && format == PluginFormat::kNCHW);
}

const char* S3Pool::getPluginType() const
{
    return S3POOL_PLUGIN_NAME;
}

void S3Pool::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

void S3Pool::detachFromContext()
{
}

void S3Pool::destroy()
{
    delete this;
}

const char* S3Pool::getPluginVersion() const
{
    return S3POOL_PLUGIN_VERSION;
}

IPluginV2Ext* S3Pool::clone() const
{
    return new S3Pool(mParams, nKernelShape, nBegPad, nEndPad, nStrides, mN, mC, mH, mW);
}

S3PoolPluginCreator::S3PoolPluginCreator()
{
    mPluginAttributes.emplace_back(PluginField("n_kernel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("kernel_size", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("n_beg_padding", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("beg_padding", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("n_end_padding", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("end_padding", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("n_strides", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("strides", nullptr, PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();

}

const char* S3PoolPluginCreator::getPluginName() const
{
    return S3POOL_PLUGIN_NAME;
}

const char* S3PoolPluginCreator::getPluginVersion() const
{
    return S3POOL_PLUGIN_VERSION;
}

const PluginFieldCollection* S3PoolPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2Ext* S3PoolPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;
    int nKernelShape = 0, nStrides = 0, nBegPad = 0, nEndPad = 0;
    for (int i=0; i < fc->nbFields; i++)
    {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "n_kernel_shape"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            nKernelShape = *(static_cast<const  int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "kernel_shape"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.kernel_shape.resize(fields[i].length);
            
            const auto *kernelVal = static_cast<const int*>(fields[i].data);
            for(size_t j=0; j < params.kernel_shape.size(); j++)
            {
                params.kernel_shape[j] = *kernelVal;
                kernelVal++;
            }
        }
        else if (!strcmp(attrName, "n_beg_padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            nBegPad = *(static_cast<const  int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "beg_padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.beg_padding.resize(fields[i].length);
            const auto *begPadVal = static_cast<const int*>(fields[i].data);
            for (size_t j=0; j < params.beg_padding.size(); j++)
            {
                params.beg_padding[j] = *begPadVal;
                begPadVal++;
            }
        }
        else if (!strcmp(attrName, "n_end_padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            nEndPad = *(static_cast<const  int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "end_padding"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.end_padding.resize(fields[i].length);
            const auto *endPadVal = static_cast<const int*>(fields[i].data);
            for (size_t j=0; j < params.end_padding.size(); j++)
            {
                params.end_padding[j] = *endPadVal;
                endPadVal++;
            }
        }
        else if (!strcmp(attrName, "n_strides"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            nStrides = *(static_cast<const  int*>(fields[i].data));
        }
        else if (!strcmp(attrName, "strides"))
        {
            ASSERT(fields[i].type == PluginFieldType::kINT32);
            params.strides.resize(fields[i].length);
            const auto *strideVal = static_cast<const int*>(fields[i].data);
            for (size_t j=0; j < params.strides.size(); j++)
            {
                params.strides[j] = *strideVal;
                strideVal++;
            }
        }
    }

    S3Pool *obj = new S3Pool(params, params.kernel_shape[0], nBegPad, nEndPad, params.strides[0]);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2Ext* S3PoolPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    S3Pool *obj = new S3Pool(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}