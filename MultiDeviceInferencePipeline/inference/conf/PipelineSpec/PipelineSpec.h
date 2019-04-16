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
 * Project: MultiDeviceInferencePipeline > Inference 
 * 
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/conf/PipelineSpec/PipelineSpec.h
 * 
 * Description: A struct to contain settings for a DALITRTPipeline 
 ***************************************************************************************************/
#pragma once
#include "MultiDeviceInferencePipeline/inference/conf/EngineSettings.h"
#include "MultiDeviceInferencePipeline/inference/preprocessing/PreprocessingSettings.h"
#include "third_party/cpptoml/cpptoml.h"
#include <string>
#include <vector>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace conf
{
class PipelineSpec
{
public:
    PipelineSpec(const std::shared_ptr<cpptoml::table>& conf);
    PipelineSpec(const PipelineSpec&) = default;
    PipelineSpec(){};
    ~PipelineSpec() = default;
    void printSpec();

    std::string name;
    std::string enginePath;
    std::vector<std::string> enginePluginPaths;
    EngineSettings engineSettings;
    preprocessing::PreprocessingSettings preprocessingSettings;
    int deviceId = 0;
    int DLACore = -1;
    int numThreads = 1;
    int batchSize = 1;
    bool pipelineExecution = true;
    int prefetchQueueDepth = 2;
    bool asyncExecution = true;
};
} //namespace conf
} //namespace inference
} //namespace multideviceinferencepipeline
