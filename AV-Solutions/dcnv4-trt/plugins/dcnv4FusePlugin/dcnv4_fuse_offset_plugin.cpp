/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
 */

#ifndef DEBUG
#define DEBUG 0 // set debug mode, if you want to see the api call, set it to 1
#endif

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include "dcnv4_fuse_offset_plugin.h"

using namespace nvinfer1;
using nvinfer1::plugin::DCNv4FuseOffset_Plugin;
using nvinfer1::plugin::DCNv4FuseOffset_PluginChecker;

namespace {
static const char* DCNV4FUSEOFFSET_PLUGIN_VERSION{"1"};
static const char* DCNV4FUSEOFFSET_PLUGIN_NAME{"DCNv4FuseOffset_Plugin"};
} // namespace

// Static class fields initialization
PluginFieldCollection DCNv4FuseOffset_PluginChecker::mFC{};
std::vector<PluginField> DCNv4FuseOffset_PluginChecker::mPluginAttributes{};

// Implementation of plugin class
DCNv4FuseOffset_Plugin::DCNv4FuseOffset_Plugin(PluginFieldCollection const& fc) noexcept {
    (void) fc;
    const PluginField* fields = fc.fields;
    for (int i=0; i<fc.nbFields; i++) {
        auto curr_field = fields[i];
        if( strcmp(curr_field.name, "kh") == 0 ) {
            kh = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "kw") == 0 ) {
            kw = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "sh") == 0 ) {
            sh = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "sw") == 0 ) {
            sw = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "ph") == 0 ) {
            ph = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "pw") == 0 ) {
            pw = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "dh") == 0 ) {
            dh = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "dw") == 0 ) {
            dw = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "group") == 0 ) {
            group = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "group_channels") == 0 ) {
            group_channels = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "offscale") == 0 ) {
            offscale = reinterpret_cast<const float*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "step") == 0 ) {
            step = reinterpret_cast<const int*>(curr_field.data)[0];
        } else if( strcmp(curr_field.name, "remove_center") == 0 ) {
            remove_center = reinterpret_cast<const int*>(curr_field.data)[0];
        } else {
            throw std::runtime_error("bad field");
        }
    }
    print_log("ctor, %d %d", kh, kw);
    print_log("group=%d group_channels=%d", group, group_channels);
}

DCNv4FuseOffset_Plugin::DCNv4FuseOffset_Plugin(const std::string name, const void* data, size_t length)
    :mName(name)
{
    print_log("DCNv4FuseOffset_Plugin::DCNv4FuseOffset_Plugin, %s", name.c_str());
    const char* d = reinterpret_cast<const char*>(data);
    const char* a = d;
    kh = read<int>(d); kw = read<int>(d);
    sh = read<int>(d); sw = read<int>(d);
    ph = read<int>(d); pw = read<int>(d);
    dh = read<int>(d); dw = read<int>(d);
    group = read<int>(d); group_channels = read<int>(d);
    offscale = read<float>(d);
    step = read<int>(d); remove_center = read<int>(d);
    mDataType = read<nvinfer1::DataType>(d);
    mInputDims = read<nvinfer1::Dims>(d);
    mOutputDims = read<nvinfer1::Dims>(d);
    padded_offset_dim = read<int>(d);
}

int DCNv4FuseOffset_Plugin::getNbOutputs() const noexcept {
    print_log("Get number of outputs");
    return 1;
}

Dims DCNv4FuseOffset_Plugin::getOutputDimensions(
    int index, const Dims* inputs, int nbInputDims
) noexcept {
    print_log("getOutputDimensions");
    print_log("%d %d", this->sh, this->sw);

    assert(index == 0 && nbInputDims == 1);

    const int batch = 1; // inputs[0].d[0];
    const int height_in = inputs[0].d[0];
    const int width_in = inputs[0].d[1];
    const int channel_in = inputs[0].d[2];

    const int height_out = (height_in + 2 * ph - (dh * (kh - 1) + 1)) / sh + 1;
    const int width_out = (width_in + 2 * pw - (dw * (kw - 1) + 1)) / sw + 1;
    print_log("%d %d", height_out, width_out);

    Dims ret;
    ret.nbDims = 3;
    ret.d[0] = height_out;
    ret.d[1] = width_out;
    ret.d[2] = group * group_channels;
    print_log("nbInputDims=%d,input=[%d,%d,%d]", nbInputDims, height_in, width_in, channel_in);
    print_log("index=%d,out=[%d,%d,%d]", index, height_out, width_out, group * group_channels);
    return ret;
}

int DCNv4FuseOffset_Plugin::initialize() noexcept {
    size_t stackSizeLimit = 0;    
    cudaDeviceGetLimit(&stackSizeLimit, cudaLimitStackSize);
    return 0;
}

void DCNv4FuseOffset_Plugin::terminate() noexcept {}

size_t DCNv4FuseOffset_Plugin::getWorkspaceSize(int maxBatchSize) const noexcept {
    return 0;
}

size_t DCNv4FuseOffset_Plugin::getSerializationSize() const noexcept {
    // Calculate the serialization size required for your plugin's data
    size_t serializationSize = 0;
    serializationSize += sizeof(kh); serializationSize += sizeof(kw);
    serializationSize += sizeof(sh); serializationSize += sizeof(sw);
    serializationSize += sizeof(ph); serializationSize += sizeof(pw);
    serializationSize += sizeof(dh); serializationSize += sizeof(dw);
    serializationSize += sizeof(group); serializationSize += sizeof(group_channels);
    serializationSize += sizeof(offscale); 
    serializationSize += sizeof(step); serializationSize += sizeof(remove_center);
    serializationSize += sizeof(static_cast<int>(mDataType));
    serializationSize += sizeof(nvinfer1::Dims);
    serializationSize += sizeof(nvinfer1::Dims);
    serializationSize += sizeof(padded_offset_dim);
    return serializationSize;
}

void DCNv4FuseOffset_Plugin::serialize(void* buffer) const noexcept {
    print_log("Serialize DCNv4FuseOffset_Plugin");
    char* d = reinterpret_cast<char*>(buffer);
    const char* a = d;
    write(d, kh); write(d, kw);
    write(d, sh); write(d, sw);
    write(d, ph); write(d, pw);
    write(d, dh); write(d, dw);
    write(d, group); write(d, group_channels);
    write(d, offscale);
    write(d, step); write(d, remove_center);
    write(d, mDataType);
    write(d, mInputDims);
    write(d, mOutputDims);
    write(d, padded_offset_dim);
}

void DCNv4FuseOffset_Plugin::configurePlugin(
    PluginTensorDesc const* in, int32_t nbInput, 
    PluginTensorDesc const* out, int32_t nbOutput
) noexcept {
    print_log("DCNv4FuseOffset_Plugin configure plugin");
    mDataType = in[0].type;
    mInputDims = in[0].dims;
    mOutputDims = out[0].dims;
    padded_offset_dim = in[0].dims.d[2] - out[0].dims.d[2];
}

bool DCNv4FuseOffset_Plugin::supportsFormatCombination(
    int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs
) const noexcept {
    bool f1 = inOut[pos].format == TensorFormat::kLINEAR;
    bool f2 = inOut[pos].type == DataType::kHALF || inOut[pos].type == DataType::kFLOAT;
    bool f3 = inOut[pos].type == inOut[0].type;
    return f1 && f2 && f3;
}

DataType DCNv4FuseOffset_Plugin::getOutputDataType(
    int index, const DataType* inputTypes, int nbInputs
) const noexcept {
    return inputTypes[0];
}

const char* DCNv4FuseOffset_Plugin::getPluginType() const noexcept {
    return DCNV4FUSEOFFSET_PLUGIN_NAME;
}

const char* DCNv4FuseOffset_Plugin::getPluginVersion() const noexcept {
    return DCNV4FUSEOFFSET_PLUGIN_VERSION;
}

void DCNv4FuseOffset_Plugin::destroy() noexcept {
    delete this;
}

IPluginV2IOExt* DCNv4FuseOffset_Plugin::clone() const noexcept {
    print_log("clone");
    auto* plugin = new DCNv4FuseOffset_Plugin(*this);
    plugin->setPluginNamespace(mPluginNamespace);
    return plugin;
}

void DCNv4FuseOffset_Plugin::setPluginNamespace(const char* pluginNamespace) noexcept {
    mPluginNamespace = pluginNamespace;
}

const char* DCNv4FuseOffset_Plugin::getPluginNamespace() const noexcept {
    return mPluginNamespace;
}

bool DCNv4FuseOffset_Plugin::isOutputBroadcastAcrossBatch(
    int outputIndex, const bool* inputIsBroadcasted, int nbInputs
) const noexcept {
    return false;
}

bool DCNv4FuseOffset_Plugin::canBroadcastInputAcrossBatch(int inputIndex) const noexcept {
    return false;
}

// Implementation of plugin checker
DCNv4FuseOffset_PluginChecker::DCNv4FuseOffset_PluginChecker() {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("kh", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("kw", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sh", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("sw", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("ph", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("pw", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dh", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("dw", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("group_channels", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("offscale", nullptr, PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(PluginField("step", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("remove_center", nullptr, PluginFieldType::kINT32, 1));

    mFC.nbFields = (size_t)(mPluginAttributes.size());
    mFC.fields = mPluginAttributes.data();
}

bool DCNv4FuseOffset_PluginChecker::validate(
    char const *name, void const *serialData, size_t serialLength, 
    nvinfer1::PluginTensorDesc const *in, size_t nbInputs, 
    nvinfer1::PluginTensorDesc const *out, size_t nbOutputs, 
    int64_t workspaceSize
) const noexcept {
    print_log("validate");
    // Custom logic can be written here to validate the UnaryPlugin.
    bool valid = true;
    bool const validNbInputsAndOutputs = (nbOutputs == 1) && (nbInputs == 1);
    valid &= validNbInputsAndOutputs;
    if (!valid) {
        return false;
    }
    bool const validDataType1 = (in[0].type == DataType::kHALF) && (out->type == DataType::kHALF);
    bool const validDataType2 = (in[0].type == DataType::kFLOAT) && (out->type == DataType::kFLOAT);
    // bool const validDataType3 = (in[1].type == DataType::kINT8) && (out->type == DataType::kINT8);
    bool const validDataType = validDataType1 || validDataType2;
    valid &= validDataType;
    return valid;
}

const char* DCNv4FuseOffset_PluginChecker::getPluginName(
) const noexcept {
    return DCNV4FUSEOFFSET_PLUGIN_NAME;
}

const char* DCNv4FuseOffset_PluginChecker::getPluginVersion(
) const noexcept {
    return DCNV4FUSEOFFSET_PLUGIN_VERSION;
}

void DCNv4FuseOffset_PluginChecker::setPluginNamespace(
    const char* pluginNamespace
) noexcept {
    mNamespace = pluginNamespace;
}

const char* DCNv4FuseOffset_PluginChecker::getPluginNamespace(
) const noexcept {
    return mNamespace.c_str();
}

const PluginFieldCollection* DCNv4FuseOffset_PluginChecker::getFieldNames(
) noexcept {
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
    return &mFC;
}

IPluginV2IOExt* DCNv4FuseOffset_PluginChecker::createPlugin(
    const char* name, const PluginFieldCollection* fc
) noexcept {
    auto plugin = new DCNv4FuseOffset_Plugin(*fc);
    plugin->setPluginNamespace(mNamespace.c_str());
    mFC = *fc;
    return plugin;
}

IPluginV2IOExt* DCNv4FuseOffset_PluginChecker::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength
) noexcept {
    return new DCNv4FuseOffset_Plugin(name, serialData, serialLength);
}
