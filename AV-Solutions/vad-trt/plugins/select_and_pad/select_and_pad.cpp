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
#include "select_and_pad.h"

using namespace nvinfer1;
using nvinfer1::plugin::SelectAndPadPlugin;
using nvinfer1::plugin::SelectAndPadPluginCreator;


namespace
{
static const char* SelectAndPadPlugin_VERSION{"1"};
static const char* SelectAndPadPlugin_NAME{"SelectAndPadPlugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection SelectAndPadPluginCreator::mFC{};
std::vector<PluginField> SelectAndPadPluginCreator::mPluginAttributes{};

// Implementation of plugin class
SelectAndPadPlugin::SelectAndPadPlugin(PluginFieldCollection const& fc) noexcept {    
    (void) fc;
    const PluginField* fields = fc.fields;
    for (int i=0; i<fc.nbFields; i++) {
        auto curr_field = fields[i];
        if( strcmp(curr_field.name, "P") == 0 ) {
            P = reinterpret_cast<const int*>(curr_field.data)[0];
        }
    }
}

SelectAndPadPlugin::SelectAndPadPlugin(const std::string name, const void* data, size_t length)
    :mName(name)
{
    print_log("Constructor from serial data");
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    P = read<int>(d);
}

int SelectAndPadPlugin::getNbOutputs() const noexcept {
    print_log("Get number of outputs");
    return 1;
}

DimsExprs SelectAndPadPlugin::getOutputDimensions(
    int index, DimsExprs const* inputs, int nbInputDims, IExprBuilder& exprBuilder
) noexcept {
    Q = inputs[0].d[1]->getConstantValue();

    print_log("Get output dimensions, P=%d, Q=%d", P, Q);
    nvinfer1::DimsExprs ret;
    ret.nbDims = 3;
    ret.d[0] = inputs[0].d[0];
    ret.d[1] = exprBuilder.constant(P);
    ret.d[2] = inputs[0].d[2];
    return ret;
}

int SelectAndPadPlugin::initialize() noexcept {
    size_t stackSizeLimit = 0;    
    cudaDeviceGetLimit(&stackSizeLimit, cudaLimitStackSize);
    return 0;
}

void SelectAndPadPlugin::terminate() noexcept {}

size_t SelectAndPadPlugin::getWorkspaceSize(
    PluginTensorDesc const* inputs, int32_t nbInputs, 
    PluginTensorDesc const* outputs,
    int32_t nbOutputs
) const noexcept {
    print_log("Q=%d", Q);
    // indices, buf, tmp, n_elem
    return sizeof(int) * Q * 2 + tmp_bytes + sizeof(int);
}

size_t SelectAndPadPlugin::getSerializationSize() const noexcept {
    // Calculate the serialization size required for your plugin's data
    size_t serializationSize = 0;
    serializationSize += sizeof(int); // P
    return serializationSize;
}

void SelectAndPadPlugin::serialize(void* buffer) const noexcept {
    print_log("Serialize SelectAndPadPlugin");
    char* d = reinterpret_cast<char*>(buffer);
    const char* a = d;
    write(d, P);
}

void SelectAndPadPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInput, 
    DynamicPluginTensorDesc const* out, int32_t nbOutput
) noexcept {
    print_log("SelectAndPadPlugin configure plugin");
    tmp_bytes = decideTemp();
}

bool SelectAndPadPlugin::supportsFormatCombination(
    int32_t pos, PluginTensorDesc const* inOut, 
    int32_t nbInputs, int32_t nbOutputs
) noexcept {
    print_log("pos=%d, fmt=%d, type=%d", pos, int(inOut[pos].format), int(inOut[pos].type));
    if (inOut[pos].format == nvinfer1::TensorFormat::kLINEAR) {
        if (pos == 1) {
            return (inOut[pos].type == nvinfer1::DataType::kINT32);
        } else {
            return ((inOut[pos].type == inOut[0].type) &&
                    (inOut[pos].type == nvinfer1::DataType::kFLOAT));
        }
    } else {
    return false;
  }
}

DataType SelectAndPadPlugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs
) const noexcept {
    return inputTypes[0];
}

const char* SelectAndPadPlugin::getPluginType() const noexcept {
    return SelectAndPadPlugin_NAME;
}

const char* SelectAndPadPlugin::getPluginVersion() const noexcept {
    return SelectAndPadPlugin_VERSION;
}

void SelectAndPadPlugin::destroy() noexcept {
    delete this;
}

IPluginV2DynamicExt* SelectAndPadPlugin::clone() const noexcept {
    print_log("clone");
    auto* plugin = new SelectAndPadPlugin(*this);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void SelectAndPadPlugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* SelectAndPadPlugin::getPluginNamespace() const noexcept {
    return mPluginNamespace;
}

// Implementation of plugin checker
SelectAndPadPluginCreator::SelectAndPadPluginCreator() {
    setupPluginAttributes(mPluginAttributes);
    mFC.nbFields = (size_t)(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

void SelectAndPadPluginCreator::setupPluginAttributes(std::vector<PluginField>& attributes) {
    attributes.clear();
    attributes.emplace_back(PluginField("P", nullptr, PluginFieldType::kINT32, 1));
}

const char* SelectAndPadPluginCreator::getPluginName(
) const noexcept {
    return SelectAndPadPlugin_NAME;
}

const char* SelectAndPadPluginCreator::getPluginVersion(
) const noexcept {
    return SelectAndPadPlugin_VERSION;
}

void SelectAndPadPluginCreator::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    mNamespace = pluginNamespace;
}

const char* SelectAndPadPluginCreator::getPluginNamespace(
) const noexcept {
    return mNamespace.c_str();
}

const PluginFieldCollection* SelectAndPadPluginCreator::getFieldNames(
) noexcept {
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    return &mFC;
}

IPluginV2* SelectAndPadPluginCreator::createPlugin(
    const char* name, const PluginFieldCollection* fc
) noexcept {
    auto plugin = new SelectAndPadPlugin(*fc);
    plugin->setPluginNamespace(mNamespace.c_str());
    mFC = *fc;
    return plugin;
}

IPluginV2* SelectAndPadPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength
) noexcept {
    return new SelectAndPadPlugin(name, serialData, serialLength);
}
