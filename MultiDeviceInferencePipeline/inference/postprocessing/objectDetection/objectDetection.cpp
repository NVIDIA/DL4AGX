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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/postprocessing/objectDetection/objectDetection.cpp
 * 
 * Description: Post process object detection results and annotate input images with bounding boxes
 ***************************************************************************************************/
#include "MultiDeviceInferencePipeline/inference/postprocessing/objectDetection/objectDetection.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace multideviceinferencepipeline::inference;

void postprocessing::objectdetection::visualizeObjectDetection(cv::Mat& img, std::vector<float>* detectionOut, std::vector<int>* keepCount, float detectionThreshold, cv::Mat& output)
{
    int imgHeight = img.rows;
    int imgWidth = img.cols;
    int keepTopK = 100;
    int detFields = 7;
    int nBatch = keepCount->size();
    output = img;
    for (int p = 0; p < nBatch; ++p)
    {
        for (int i = 0; i < (int) ((*keepCount)[p]); ++i)
        {
            float* det = detectionOut->data() + (p * keepTopK + i) * detFields;
            int category_id = (int) det[1];
            float score = det[2];
            if (score > detectionThreshold)
            {
                int x = (int) (det[3] * imgWidth);
                int y = (int) (det[4] * imgHeight);
                int bboxWidth = (int) (det[5] * imgWidth - det[3] * imgWidth);
                int bboxHeight = (int) (det[6] * imgHeight - det[4] * imgHeight);
                cv::Rect box(x, y, bboxWidth, bboxHeight);
                switch (category_id)
                {
                case 1:
                    cv::rectangle(output, box, cv::Scalar(255, 0, 0), 2);
                    break;
                case 2:
                    cv::rectangle(output, box, cv::Scalar(0, 255, 0), 2);
                    break;
                case 3:
                    cv::rectangle(output, box, cv::Scalar(200, 200, 0), 2);
                    break;
                default:
                    cv::rectangle(output, box, cv::Scalar(255, 255, 255), 2);
                    break;
                }
            }
        }
    }
}
