/***************************************************************************************************
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
 * Project: MultiDeviceInferencePipeline > enginecreator
 * 
 * File: DL4AGX/MultiDeviceInferencePipeline/Int8Calibrator/Int8Calibrator.h
 * 
 * Description: Implementation of a Int8Calibrator for use in creating TRT engines 
 ***************************************************************************************************/
#pragma once
// TRT dependencies
#include "NvInfer.h"
#include "NvInferPlugin.h"

// DALI dependencies
#include "MultiDeviceInferencePipeline/enginecreator/DALIStream/DALIStream.h"

namespace multideviceinferencepipeline
{
namespace enginecreator
{
class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator
{
public:
    Int8EntropyCalibrator(DALIStream& stream, std::string cacheFile, std::map<std::string, nvinfer1::Dims3>& inputDimensions, bool readCache = false);
    ~Int8EntropyCalibrator();
    int getBatchSize() const override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override;
    const void* readCalibrationCache(size_t& length) override;
    void writeCalibrationCache(const void* cache, size_t length) override;

private:
    DALIStream mStream;
    std::string mCacheFile;
    bool mReadCache;
    std::map<std::string, void*> mInputDeviceBuffers;
    std::vector<char> mCalibrationCache;
    std::map<std::string, nvinfer1::Dims3> mInputDimensions;
};
} // namepsace enginecreator
} // namespace multideviceinferencepipeline
