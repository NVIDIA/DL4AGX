
#ifndef DEBUG
#define DEBUG 0 // set debug mode, if you want to see the api call, set it to 1
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "rotate_plugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::RotatePlugin;
using nvinfer1::plugin::RotatePluginCreator;


namespace
{
static const char* RotatePlugin_VERSION{"1"};
static const char* RotatePlugin_NAME{"RotatePlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection RotatePluginCreator::mFC{};
std::vector<PluginField> RotatePluginCreator::mPluginAttributes{};

// Implementation of plugin class
RotatePlugin::RotatePlugin(PluginFieldCollection const& fc) noexcept {    
    (void) fc;
    const PluginField* fields = fc.fields;
    for (int i=0; i<fc.nbFields; i++) {
        auto curr_field = fields[i];
        if( strcmp(curr_field.name, "interpolation") == 0 ) {
            mMode = (RotateInterpolation)reinterpret_cast<const int*>(curr_field.data)[0];
        }
    }
}

RotatePlugin::RotatePlugin(const std::string name, const void* data, size_t length)
    :mName(name)
{
    print_log("Constructor from serial data");
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    mMode = read<RotateInterpolation>(d);
}

int RotatePlugin::getNbOutputs() const noexcept {
    print_log("Get number of outputs");
    return 1;
}

DimsExprs RotatePlugin::getOutputDimensions(
    int index, DimsExprs const* inputs, int nbInputDims, IExprBuilder& exprBuilder
) noexcept {
    print_log("Get output dimensions");
    DimsExprs outputDim;
    outputDim.nbDims = 3;
    outputDim.d[0] = inputs[0].d[0];
    outputDim.d[1] = inputs[0].d[1];
    outputDim.d[2] = inputs[0].d[2];
    return outputDim;
}

int RotatePlugin::initialize() noexcept {
    size_t stackSizeLimit = 0;    
    cudaDeviceGetLimit(&stackSizeLimit, cudaLimitStackSize);
    return 0;
}

void RotatePlugin::terminate() noexcept {}

size_t RotatePlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, 
    PluginTensorDesc const* outputs,
    int32_t nbOutputs
) const noexcept {
    return 0;
}

size_t RotatePlugin::getSerializationSize() const noexcept {
    // Calculate the serialization size required for your plugin's data
    size_t serializationSize = 0;
    serializationSize += sizeof(mMode);
    return serializationSize;
}

void RotatePlugin::serialize(void* buffer) const noexcept {
    print_log("Serialize RotatePlugin");
    char* d = reinterpret_cast<char*>(buffer);
    const char* a = d;
    write(d, (int)mMode);
}

void RotatePlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInput, 
    DynamicPluginTensorDesc const* out, int32_t nbOutput
) noexcept {
    print_log("RotatePlugin configure plugin");
}

bool RotatePlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, 
    int32_t nbInputs, int32_t nbOutputs
) noexcept {
    if (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) {
        if ((pos == 1) || (pos == 2)) {
            return (inOut[pos].type == nvinfer1::DataType::kFLOAT) || 
                   (inOut[pos].type == nvinfer1::DataType::kHALF);
        } else {
            return ((inOut[pos].type == inOut[0].type) &&
                    ((inOut[pos].type == nvinfer1::DataType::kFLOAT) ||
                    (inOut[pos].type == nvinfer1::DataType::kHALF)));
        }
    } else {
    return false;
  }
}

DataType RotatePlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs
) const noexcept {
    return inputTypes[0];
}

const char* RotatePlugin::getPluginType() const noexcept {
    return RotatePlugin_NAME;
}

const char* RotatePlugin::getPluginVersion() const noexcept {
    return RotatePlugin_VERSION;
}

void RotatePlugin::destroy() noexcept {
    delete this;
}

IPluginV2DynamicExt* RotatePlugin::clone() const noexcept {
    print_log("clone");
    auto* plugin = new RotatePlugin(*this);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void RotatePlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* RotatePlugin::getPluginNamespace() const noexcept {
    return mPluginNamespace;
}

// Implementation of plugin checker
RotatePluginCreator::RotatePluginCreator() {
    setupPluginAttributes(mPluginAttributes);
    mFC.nbFields = (size_t)(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

void RotatePluginCreator::setupPluginAttributes(std::vector<PluginField>& attributes) {
    attributes.clear();
}

const char* RotatePluginCreator::getPluginName(
) const noexcept {
    return RotatePlugin_NAME;
}

const char* RotatePluginCreator::getPluginVersion(
) const noexcept {
    return RotatePlugin_VERSION;
}

void RotatePluginCreator::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    mNamespace = pluginNamespace;
}

const char* RotatePluginCreator::getPluginNamespace(
) const noexcept {
    return mNamespace.c_str();
}

const PluginFieldCollection* RotatePluginCreator::getFieldNames(
) noexcept {
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    return &mFC;
}

IPluginV2* RotatePluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc
) noexcept {
    auto plugin = new RotatePlugin(*fc);
    plugin->setPluginNamespace(mNamespace.c_str());
    mFC = *fc;
    return plugin;
}

IPluginV2* RotatePluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength
) noexcept {
    return new RotatePlugin(name, serialData, serialLength);
}
