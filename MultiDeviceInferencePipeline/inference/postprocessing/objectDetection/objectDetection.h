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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/postprocessing/objectDetection/objectDetection.h
 * 
 * Description: Post process object detection results and annotate input images with bounding boxes
 ***************************************************************************************************/
#pragma once
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <unordered_map>
#include <vector>

namespace multideviceinferencepipeline
{
namespace inference
{
namespace postprocessing
{
namespace objectdetection
{
void visualizeObjectDetection(cv::Mat& img, std::vector<float>* detectionOut, std::vector<int>* keepCount, float detectionThreshold, cv::Mat& output);
} //namespace objectdetection
} //namespace postprocessing
} //namespace inference
} //namespace multideviceinferencepipeline
