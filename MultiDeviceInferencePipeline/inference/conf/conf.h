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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/conf/conf.h
 * 
 * Description: Parse a toml file containing configurations for the application's execution
 ***************************************************************************************************/
#pragma once
#include "MultiDeviceInferencePipeline/inference/conf/EngineSettings.h"
#include "MultiDeviceInferencePipeline/inference/conf/ExecutionSettings.h"
#include "MultiDeviceInferencePipeline/inference/conf/PipelineSpec/PipelineSpec.h"
#include "MultiDeviceInferencePipeline/inference/preprocessing/preprocessing.h"
#include "third_party/cpptoml/cpptoml.h"
#include <iomanip>
#include <map>
#include <tuple>
#include <utility>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace conf
{

const uint32_t kNUM_HOST_PROFILE_ITERS = 105;
const uint32_t kNUM_TIMED_HOST_ITERS = 100;

inline void parseInferencePipelineConfFile(std::string specFile,
                                           ExecutionSettings& settings,
                                           std::map<std::string, PipelineSpec>& specs)
{
    auto config = cpptoml::parse_file(specFile);
    auto inferencePipelines = config->get_table_array("inference_pipeline");
    settings.inFiles.push_back(config->get_as<std::string>("input_image").value_or("UNKNOWN PATH TO INPUT FILE"));
    std::cout << "Input Image: " << settings.inFiles[0] << std::endl;
    settings.outFiles.push_back(config->get_as<std::string>("output_image").value_or("./output.jpg"));
    std::cout << "Output Image: " << settings.outFiles[0] << std::endl;
    settings.batchSize = 1; //THIS IS HARD CODED RIGHT BECAUSE THE PIPELINE IS NOT CURRENTLY BATCH FREINDLY
    settings.profile = config->get_as<bool>("profile").value_or(false);
    if (settings.profile)
    {
        settings.iters = kNUM_HOST_PROFILE_ITERS;
        settings.timed_iters = kNUM_TIMED_HOST_ITERS;
        std::cout << "Profiling Execution" << std::endl;
    }

    for (const auto& pipeline : *inferencePipelines)
    {
        auto spec = PipelineSpec(pipeline);
        specs.emplace(spec.name, spec);
        settings.pipelineBindings[spec.name] = spec.engineSettings;
        specs[spec.name].printSpec();
    }

    auto postprocessingConf = config->get_table("postprocessing");
    settings.detectionThreshold = (float) postprocessingConf->get_as<double>("detection_threshold").value_or(0.5);
    std::cout << "Postprocessing: \n    Detection Threshold: " << settings.detectionThreshold << std::endl;
}
} // namespace conf
} // namespace inference
} // namespace multideviceinferencepipeline