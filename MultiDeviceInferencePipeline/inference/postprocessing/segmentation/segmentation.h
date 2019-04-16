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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/postprocessing/segmentation/segmentation.h
 * 
 * Description: Process segmentation results, create a segementation mask
 ***************************************************************************************************/
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <unordered_map>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace postprocessing
{
namespace segmentation
{
void visualizeSegmentation(cv::Mat& img, int mapHeight, int mapWidth, int inputHeight, int inputWidth, float* segOut, std::unordered_map<int, cv::Vec3b>& colorMap, cv::Mat& output);
} //namespace segmentation
} //namepsace postprocessing
} //namespace inference
} //namespace multideviceinferencepipeline
