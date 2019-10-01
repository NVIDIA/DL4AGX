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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/preprocessing/preprocessing.h
 * 
 * Description: Preprocessing Pipeline Template 
 ***************************************************************************************************/
#pragma once
#pragma once
#include "MultiDeviceInferencePipeline/inference/preprocessing/PreprocessingSettings.h"
#include <string>
#include <vector>

// DALI dependencies
#include "dali/core/common.h"
#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"
#include "dali/util/user_stream.h"

namespace multideviceinferencepipeline
{
namespace inference
{
namespace preprocessing
{

inline void AddOpsToPipeline(dali::Pipeline* pipe,
                             const std::string prefix,
                             const std::pair<std::string, std::string> externalInput,
                             const std::vector<std::pair<std::string, std::string>> pipelineOutputs,
                             const preprocessing::PreprocessingSettings& settings,
                             bool gpuMode)
{
    int nChannel = settings.imgDims[0]; //Channels
    int nHeight = settings.imgDims[1];  //Height
    int nWidth = settings.imgDims[2];   //Width
    std::string executionPlatform = gpuMode ? "gpu" : "cpu";

    pipe->AddOperator(
        dali::OpSpec("Resize")
            .AddArg("device", executionPlatform)
            .AddArg("interp_type", dali::DALI_INTERP_CUBIC)
            .AddArg("resize_x", (float) nWidth)
            .AddArg("resize_y", (float) nHeight)
            .AddArg("image_type", dali::DALI_RGB)
            .AddInput("decoded_jpegs", executionPlatform)
            .AddOutput("resized_images", executionPlatform),
        prefix + "_Resize");
    pipe->AddOperator(
        dali::OpSpec("NormalizePermute")
            .AddArg("device", executionPlatform)
            .AddArg("output_type", dali::DALI_FLOAT)
            .AddArg("mean", settings.imgMean)
            .AddArg("std", settings.imgStd)
            .AddArg("height", nHeight)
            .AddArg("width", nWidth)
            .AddArg("channels", nChannel)
            .AddInput("resized_images", executionPlatform)
            .AddOutput(pipelineOutputs[0].first, pipelineOutputs[0].second),
        prefix + "_NormalizePermute");
}
} // namespace preprocessing
} // namespace inference
} // namespace multideviceinferencepipelines
