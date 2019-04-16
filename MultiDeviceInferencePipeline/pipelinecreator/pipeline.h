/**************************************************************************
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
 * File: DL4AGX/MultiDeviceInferencePipeline/pipelinecreator/pipeline.h
 * Description: Create a serialized preprocessing pipeline 
 *************************************************************************/
#pragma once
#include <string>
#include <vector>

// DALI dependencies
#include "dali/common.h"
#include "dali/pipeline/data/allocator.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"
#include "dali/util/user_stream.h"

inline std::string serializePipe(const std::vector<int>& imageDims, const std::vector<float>& imgMean, const std::vector<float>& imgStd, bool cpuMode)
{
    // default dali pipeline
    int BATCH_SIZE = 50;
    int N_THREAD = 1;
    int DEVICE_ID = 0;
    int SEED = -1;
    bool PIPELINED_EXECUTION = true;
    int PREFETCH_QUEUE_DEPTH = 2;
    bool ASYNC_EXECUTION = true;

    int nChannel = imageDims[0]; //Channels
    int nHeight = imageDims[1];  //Height
    int nWidth = imageDims[2];   //Width
    // start DALI pipeline
    dali::Pipeline pipe(BATCH_SIZE, N_THREAD, DEVICE_ID, SEED, PIPELINED_EXECUTION, PREFETCH_QUEUE_DEPTH, ASYNC_EXECUTION);
    pipe.AddExternalInput("raw_jpegs");
    if (cpuMode)
    { // cpu mode
        std::cout << "Creating CPU pipeline" << std::endl;
        pipe.AddOperator(
            dali::OpSpec("HostDecoder")
                .AddArg("device", "cpu")
                .AddArg("output_type", dali::DALI_RGB)
                .AddInput("raw_jpegs", "cpu")
                .AddOutput("decoded_jpegs", "cpu"));
        pipe.AddOperator(
            dali::OpSpec("Resize")
                .AddArg("interp_type", dali::DALI_INTERP_CUBIC)
                .AddArg("resize_x", (float) nWidth)
                .AddArg("resize_y", (float) nHeight)
                .AddArg("image_type", dali::DALI_RGB)
                .AddInput("decoded_jpegs", "cpu")
                .AddOutput("resized_images", "cpu"));
        pipe.AddOperator(
            dali::OpSpec("NormalizePermute")
                .AddArg("device", "cpu")
                .AddArg("output_type", dali::DALI_FLOAT)
                .AddArg("mean", imgMean)
                .AddArg("std", imgStd)
                .AddArg("height", nHeight)
                .AddArg("width", nWidth)
                .AddArg("channels", nChannel)
                .AddInput("resized_images", "cpu")
                .AddOutput("preprocessed_images", "cpu"));
    }
    else
    { // gpu mode
        std::cout << "Creating GPU pipeline" << std::endl;
        pipe.AddOperator(
            dali::OpSpec("HostDecoder")
                .AddArg("device", "cpu")
                .AddArg("output_type", dali::DALI_RGB)
                .AddInput("raw_jpegs", "cpu")
                .AddOutput("decoded_jpegs", "cpu"));
        pipe.AddOperator(
            dali::OpSpec("Resize")
                .AddArg("device", "gpu")
                .AddArg("interp_type", dali::DALI_INTERP_CUBIC)
                .AddArg("resize_x", (float) nWidth)
                .AddArg("resize_y", (float) nHeight)
                .AddArg("image_type", dali::DALI_RGB)
                .AddInput("decoded_jpegs", "gpu")
                .AddOutput("resized_images", "gpu"));
        pipe.AddOperator(
            dali::OpSpec("NormalizePermute")
                .AddArg("device", "gpu")
                .AddArg("output_type", dali::DALI_FLOAT)
                .AddArg("mean", imgMean)
                .AddArg("std", imgStd)
                .AddArg("height", nHeight)
                .AddArg("width", nWidth)
                .AddArg("channels", nChannel)
                .AddInput("resized_images", "gpu")
                .AddOutput("preprocessed_images", "gpu"));
    }
    // serialize pipeline
    std::string serialized = pipe.SerializeToProtobuf();
    return serialized;
}