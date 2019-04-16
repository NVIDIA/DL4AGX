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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/conf/EngineSettings.h
 * 
 * Description: Struct to hold binding information for a TensorRT engine
 ***************************************************************************************************/
#pragma once
#include <map>
#include <tuple>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace conf
{
struct EngineSettings
{
    std::map<std::string, std::tuple<int, int, int>> inputBindings;
    std::map<std::string, std::tuple<int, int, int>> outputBindings;
};

inline int bindingSize(const std::tuple<int, int, int> binding)
{
    return std::get<0>(binding) * std::get<1>(binding) * std::get<2>(binding);
}
} //namespace conf
} //namespace inference
} //namespace multideviceinferencepipeline
