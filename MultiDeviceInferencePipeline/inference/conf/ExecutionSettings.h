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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/conf/ExecutionSettings.h
 * 
 * Description: Struct to hold execution settings for the application 
 ***************************************************************************************************/
#pragma once
#include <map>
#include <string>
#include <vector>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace conf
{
struct ExecutionSettings
{
    std::vector<std::string> inFiles;
    std::vector<std::string> outFiles;
    uint32_t batchSize;
    bool profile;
    uint32_t iters = 1;
    uint32_t timed_iters = 0;
    float detectionThreshold = 0.5;
    std::map<std::string, EngineSettings> pipelineBindings;
};
} //namespace conf
} //namespace inference
} //namespace multideviceinferencepipeline
