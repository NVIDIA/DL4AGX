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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/postprocessing/segmentation/segmentation.cpp
 * 
 * Description: Process inference results and generate a segmentation mask 
 ***************************************************************************************************/
#include "MultiDeviceInferencePipeline/inference/postprocessing/segmentation/segmentation.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace multideviceinferencepipeline::inference;

void postprocessing::segmentation::visualizeSegmentation(cv::Mat& img, int mapHeight, int mapWidth, int inputHeight, int inputWidth, float* segOut, std::unordered_map<int, cv::Vec3b>& colorMap, cv::Mat& output)
{
    int imgHeight = img.rows;
    int imgWidth = img.cols;

    // first resize to 300x300 from 19x19
    cv::Mat segC1(cv::Size(mapWidth, mapHeight), CV_32FC1, cv::Scalar(0));
    cv::Mat segC2(cv::Size(mapWidth, mapHeight), CV_32FC1, cv::Scalar(0));
    int count = 0;
    for (int i = 0; i < mapHeight; ++i)
    {
        for (int j = 0; j < mapWidth; ++j)
        {
            int idx_0 = count;
            int idx_1 = count + 1;
            segC1.at<float>(i, j) = segOut[idx_0];
            segC2.at<float>(i, j) = segOut[idx_1];
            count += 2;
        }
    }

    cv::resize(segC1, segC1, cv::Size(inputWidth, inputHeight), 0, 0, cv::INTER_LINEAR);
    cv::resize(segC2, segC2, cv::Size(inputWidth, inputHeight), 0, 0, cv::INTER_LINEAR);

    cv::Mat segMap(cv::Size(inputWidth, inputHeight), CV_8UC3, cv::Scalar(0, 0, 0));
    int cnt = 0;
    for (int i = 0; i < inputHeight; ++i)
    {
        for (int j = 0; j < inputWidth; ++j)
        {
            if (segC1.at<float>(i, j) > segC2.at<float>(i, j))
            { // background class
                segMap.at<cv::Vec3b>(i, j) = colorMap[0];
            }
            else
            { // road class
                segMap.at<cv::Vec3b>(i, j) = colorMap[1];

                cnt++;
            }
        }
    }

    cv::resize(segMap, segMap, cv::Size(imgWidth, imgHeight), 0, 0, CV_INTER_CUBIC);
    // overlay images
    float alpha = 0.5;
    cv::addWeighted(img, alpha, segMap, 1 - alpha, 0.0, output);
}
