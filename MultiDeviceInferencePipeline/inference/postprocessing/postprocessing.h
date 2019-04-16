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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/postprocessing/postprocessing.h
 * 
 * Description: Post-process inference results, annotate input image with results
 ***************************************************************************************************/
#pragma once
#include "MultiDeviceInferencePipeline/inference/conf/EngineSettings.h"
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace postprocessing
{
void processInferenceResults(cv::Mat& inputImage,
                             conf::EngineSettings segSettings,
                             std::vector<float>* segResults,
                             conf::EngineSettings detectionSettings,
                             std::pair<std::vector<float>*, int> detectionResults,
                             float detectionThreshold,
                             cv::Mat& outputImage);
} // namespace postprocessing
} // namespace inference
} // namespace multideviceinferencepipeline
