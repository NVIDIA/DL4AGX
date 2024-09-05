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

#pragma once

#include <cassert>
#include <vector>
#include <string>
#include <iostream>
#include <cstring>

#include "NvInfer.h"
#include "NvInferConsistency.h"

#include <cuda_fp16.h>
#include "common.h"

namespace nvinfer1 {
namespace plugin {

class DCNv4FuseOffset_Plugin: public IPluginV2IOExt {
public:

    DCNv4FuseOffset_Plugin() noexcept {};
    DCNv4FuseOffset_Plugin(PluginFieldCollection const& fc) noexcept;
    DCNv4FuseOffset_Plugin(const std::string name, const void* serialData, size_t serialLength);

    ~DCNv4FuseOffset_Plugin() override = default;

    int getNbOutputs() const noexcept override;
    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) noexcept override;
    int initialize() noexcept override;
    void terminate() noexcept override;
    size_t getWorkspaceSize(int maxBatchSize) const noexcept override;

    int32_t enqueue(
        int32_t batchSize, 
        const void* const* inputs, void* const* outputs, void* workspace,
        cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;

    bool supportsFormatCombination(
        int pos, 
        const PluginTensorDesc* inOut, int nbInputs, 
        int nbOutputs) const noexcept override;

    const char* getPluginType() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    void destroy() noexcept override;
    IPluginV2IOExt* clone() const noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    DataType getOutputDataType(
        int index, 
        const DataType* inputTypes, 
        int nbInputs) const noexcept override;

    void configurePlugin(
        const PluginTensorDesc* in, 
        int nbInput, 
        const PluginTensorDesc* out, 
        int nbOutput) noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int outputIndex, 
        const bool* inputIsBroadcasted, 
        int nbInputs) const noexcept override;

    bool canBroadcastInputAcrossBatch(int inputIndex) const noexcept override;

private:
    template <typename T>
    void write(char*& buffer, const T& val) const noexcept {
        *reinterpret_cast<T*>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    T read(const char*& buffer) const noexcept {
        T val = *reinterpret_cast<const T*>(buffer);
        buffer += sizeof(T);
        return val;
    }

    const char* mPluginNamespace = "custom_op";
    const std::string mName = "DCNv4FuseOffset_Plugin"; 
    PluginTensorDesc mInputTensor;
    PluginTensorDesc mOutputTensor;

    nvinfer1::Dims mInputDims;
    int padded_offset_dim;
    nvinfer1::Dims mOutputDims;
    nvinfer1::DataType mDataType;

    // params
    int kh, kw;
    int sh, sw;
    int ph, pw;
    int dh, dw;
    int group, group_channels;
    float offscale;
    int step, remove_center;
    
    int ih, iw, oh, ow;
    int channels;
    int batch_size;
}; // class DCNv4FuseOffset_Plugin

class DCNv4FuseOffset_PluginChecker : public nvinfer1::consistency::IPluginChecker {
public:
    DCNv4FuseOffset_PluginChecker();
    ~DCNv4FuseOffset_PluginChecker() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const PluginFieldCollection* getFieldNames() noexcept override;
    IPluginV2IOExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override;
    IPluginV2IOExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* libNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

    bool validate(
        char const *name, void const *serialData, size_t serialLength, 
        nvinfer1::PluginTensorDesc const *in, size_t nbInputs, 
        nvinfer1::PluginTensorDesc const *out, size_t nbOutputs, 
        int64_t workspaceSize) const noexcept override;

private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;        
}; // class DCNv4FuseOffset_PluginChecker

REGISTER_TENSORRT_PLUGIN(DCNv4FuseOffset_PluginChecker);

}
} // namespace nvinfer1
