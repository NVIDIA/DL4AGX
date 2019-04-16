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
 * File: DL4AGX/MultiDeviceInferencePipeline/Int8Calibrator/Int8Calibrator.cpp
 * 
 * Description: Implementation of a Int8Calibrator for use in creating TRT engines 
 ***************************************************************************************************/
#include "MultiDeviceInferencePipeline/enginecreator/Int8Calibrator/Int8Calibrator.h"

#include "common/macros.h"
#include "common/tensorrt/utils.h"

using namespace multideviceinferencepipeline::enginecreator;

//Constructor
Int8EntropyCalibrator::Int8EntropyCalibrator(DALIStream& stream, std::string cacheFile, std::map<std::string, nvinfer1::Dims3>& inputDimensions, bool readCache)
    : mStream(stream)
    , mCacheFile(cacheFile)
    , mReadCache(readCache)
    , mInputDimensions(inputDimensions)
{
    for (auto& elem : mInputDimensions)
    {
        int elemCount = common::tensorrt::volume(elem.second);
        void* data;
        CHECK(cudaMalloc(&data, mStream.getBatchSize() * elemCount * sizeof(float)));
        mInputDeviceBuffers.insert(std::make_pair(elem.first, data));
    }
    mStream.reset();
}

//destructor
Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    for (auto& elem : mInputDeviceBuffers)
        CHECK(cudaFree(elem.second));
}

int Int8EntropyCalibrator::getBatchSize() const
{
    return mStream.getBatchSize();
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)
{
    if (!mStream.next())
        return false;

    for (int i = 0; i < nbBindings; ++i)
    {
        assert(i == 0);
        int inputCount = common::tensorrt::volume(mStream.getDims());
        CHECK(cudaMemcpy(mInputDeviceBuffers[names[i]], mStream.getBatch(),
                         inputCount * sizeof(float), cudaMemcpyHostToDevice));
        bindings[i] = mInputDeviceBuffers[names[i]];
    }
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length)
{
    mCalibrationCache.clear();
    std::ifstream input(mCacheFile, std::ios::binary);
    input >> std::noskipws;
    if (input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
                  std::back_inserter(mCalibrationCache));
        std::cout << "Calibrating using the table: " << mCacheFile << std::endl;
    }
    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length)
{
    std::ofstream output(mCacheFile, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    std::cout << "Calibration table written to " << mCacheFile << std::endl;
}