/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#ifndef DEBUG
#define DEBUG 0 // set debug mode, if you want to see the api call, set it to 1
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "ms_deform_attn.h"

using namespace nvinfer1;
using nvinfer1::plugin::MultiScaleDeformableAttentionPlugin;
using nvinfer1::plugin::MultiScaleDeformableAttentionPluginCreator;


namespace
{
static const char* MultiScaleDeformableAttentionPlugin_VERSION{"1"};
static const char* MultiScaleDeformableAttentionPlugin_NAME{"MultiScaleDeformableAttentionPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection MultiScaleDeformableAttentionPluginCreator::mFC{};
std::vector<PluginField> MultiScaleDeformableAttentionPluginCreator::mPluginAttributes{};

// Implementation of plugin class
MultiScaleDeformableAttentionPlugin::MultiScaleDeformableAttentionPlugin(PluginFieldCollection const& fc) noexcept {    
    (void) fc;
    const PluginField* fields = fc.fields;
    for (int i=0; i<fc.nbFields; i++) {
        auto curr_field = fields[i];
        // if( strcmp(curr_field.name, "kh") == 0 ) {
        //     kh = reinterpret_cast<const int*>(curr_field.data)[0];
        // }
    }
}

MultiScaleDeformableAttentionPlugin::MultiScaleDeformableAttentionPlugin(const std::string name, const void* data, size_t length)
    :mName(name)
{
    print_log("Constructor from serial data");
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
}

int MultiScaleDeformableAttentionPlugin::getNbOutputs() const noexcept {
    print_log("Get number of outputs");
    return 1;
}

DimsExprs MultiScaleDeformableAttentionPlugin::getOutputDimensions(
    int index, DimsExprs const* inputs, int nbInputDims, IExprBuilder& exprBuilder
) noexcept {
    print_log("Get output dimensions");
    nvinfer1::DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = inputs[3].d[1];

    ret.d[2] = exprBuilder.operation(DimensionOperation::kPROD, *inputs[0].d[2], *inputs[0].d[3]);

    return ret;
}

int MultiScaleDeformableAttentionPlugin::initialize() noexcept {
    size_t stackSizeLimit = 0;    
    cudaDeviceGetLimit(&stackSizeLimit, cudaLimitStackSize);
    return 0;
}

void MultiScaleDeformableAttentionPlugin::terminate() noexcept {}

size_t MultiScaleDeformableAttentionPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, 
    PluginTensorDesc const* outputs,
    int32_t nbOutputs
) const noexcept {
    return 0;
}

size_t MultiScaleDeformableAttentionPlugin::getSerializationSize() const noexcept {
    // Calculate the serialization size required for your plugin's data
    size_t serializationSize = 0;
    return serializationSize;
}

void MultiScaleDeformableAttentionPlugin::serialize(void* buffer) const noexcept {
    print_log("Serialize MultiScaleDeformableAttentionPlugin");
    char* d = reinterpret_cast<char*>(buffer);
    const char* a = d;
}

void MultiScaleDeformableAttentionPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInput, 
    DynamicPluginTensorDesc const* out, int32_t nbOutput
) noexcept {
    print_log("MultiScaleDeformableAttentionPlugin configure plugin");
}

bool MultiScaleDeformableAttentionPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, 
    int32_t nbInputs, int32_t nbOutputs
) noexcept {
    if (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) {
        if ((pos == 1) || (pos == 2)) {
            return (inOut[pos].type == nvinfer1::DataType::kINT32);
        } else {
            bool f1 = (inOut[pos].type == nvinfer1::DataType::kFLOAT) || (inOut[pos].type == nvinfer1::DataType::kHALF);
            // bool f1 = (inOut[pos].type == nvinfer1::DataType::kFLOAT);
            return ((inOut[pos].type == inOut[0].type) &&
                    (f1));
        }
    } else {
    return false;
  }
}

DataType MultiScaleDeformableAttentionPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs
) const noexcept {
    return inputTypes[0];
}

const char* MultiScaleDeformableAttentionPlugin::getPluginType() const noexcept {
    return MultiScaleDeformableAttentionPlugin_NAME;
}

const char* MultiScaleDeformableAttentionPlugin::getPluginVersion() const noexcept {
    return MultiScaleDeformableAttentionPlugin_VERSION;
}

void MultiScaleDeformableAttentionPlugin::destroy() noexcept {
    delete this;
}

IPluginV2DynamicExt* MultiScaleDeformableAttentionPlugin::clone() const noexcept {
    print_log("clone");
    auto* plugin = new MultiScaleDeformableAttentionPlugin(*this);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void MultiScaleDeformableAttentionPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* MultiScaleDeformableAttentionPlugin::getPluginNamespace() const noexcept {
    return mPluginNamespace;
}

// Implementation of plugin checker
MultiScaleDeformableAttentionPluginCreator::MultiScaleDeformableAttentionPluginCreator() {
    setupPluginAttributes(mPluginAttributes);
    mFC.nbFields = (size_t)(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

void MultiScaleDeformableAttentionPluginCreator::setupPluginAttributes(std::vector<PluginField>& attributes) {
    attributes.clear();
}

const char* MultiScaleDeformableAttentionPluginCreator::getPluginName(
) const noexcept {
    return MultiScaleDeformableAttentionPlugin_NAME;
}

const char* MultiScaleDeformableAttentionPluginCreator::getPluginVersion(
) const noexcept {
    return MultiScaleDeformableAttentionPlugin_VERSION;
}

void MultiScaleDeformableAttentionPluginCreator::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    mNamespace = pluginNamespace;
}

const char* MultiScaleDeformableAttentionPluginCreator::getPluginNamespace(
) const noexcept {
    return mNamespace.c_str();
}

const PluginFieldCollection* MultiScaleDeformableAttentionPluginCreator::getFieldNames(
) noexcept {
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    return &mFC;
}

IPluginV2* MultiScaleDeformableAttentionPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc
) noexcept {
    auto plugin = new MultiScaleDeformableAttentionPlugin(*fc);
    plugin->setPluginNamespace(mNamespace.c_str());
    mFC = *fc;
    return plugin;
}

IPluginV2* MultiScaleDeformableAttentionPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength
) noexcept {
    return new MultiScaleDeformableAttentionPlugin(name, serialData, serialLength);
}
