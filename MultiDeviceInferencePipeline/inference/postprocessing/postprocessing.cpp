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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/postprocessing/postprocessing.cpp
 * 
 * Description: Post-process inference results, annotate input image with results
 ***************************************************************************************************/
#include "MultiDeviceInferencePipeline/inference/postprocessing/postprocessing.h"
#include "MultiDeviceInferencePipeline/inference/postprocessing/objectDetection/objectDetection.h"
#include "MultiDeviceInferencePipeline/inference/postprocessing/segmentation/segmentation.h"

#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace multideviceinferencepipeline::inference;

void postprocessing::processInferenceResults(cv::Mat& inputImage,
                                             conf::EngineSettings segSettings,
                                             std::vector<float>* segResults,
                                             conf::EngineSettings detectionSettings,
                                             std::pair<std::vector<float>*, int> detectionResults,
                                             float detectionThreshold,
                                             cv::Mat& outputImage)
{
    std::unordered_map<int, cv::Vec3b> colorMap;
    colorMap.insert({0, cv::Vec3b(0, 0, 0)});   // background is black
    colorMap.insert({1, cv::Vec3b(0, 0, 255)}); // road is red

    int mapHeight = std::get<1>(segSettings.outputBindings["logits/semantic/BiasAdd"]);
    int mapWidth = std::get<2>(segSettings.outputBindings["logits/semantic/BiasAdd"]);
    int inputHeight = std::get<1>(segSettings.inputBindings["ImageTensor"]);
    int inputWidth = std::get<2>(segSettings.inputBindings["ImageTensor"]);

    cv::Mat segOutput;
    std::cout << "Generating segmentation mask" << std::endl;
    segmentation::visualizeSegmentation(inputImage, mapHeight, mapWidth, inputHeight, inputWidth, segResults->data(), colorMap, segOutput);
    //TODO: Make this batch friendly
    std::vector<int> keepCount = {detectionResults.second};
    std::cout << "Generating bounding boxes" << std::endl;
    objectdetection::visualizeObjectDetection(segOutput, detectionResults.first, &keepCount, detectionThreshold, outputImage);
    return;
}
