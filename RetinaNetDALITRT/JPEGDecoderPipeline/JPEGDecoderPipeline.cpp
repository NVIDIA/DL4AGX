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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/JPEGDecoderPipeline/JPEGDecoderPipeline.cpp
 * 
 * Description: DALI pipeline to decode JPEG images
 ***************************************************************************************************/
#include "RetinaNetDALITRT/JPEGDecoderPipeline/JPEGDecoderPipeline.h"

#include "dali/common.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"
#include "dali/util/user_stream.h"

#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace retinanetinferencepipeline::inference;

JPEGDecoderPipeline::JPEGDecoderPipeline(const int deviceId,
                                         const int numThreads,
                                         const int batchSize,
                                         const bool pipelineExecution,
                                         const int prefetchQueueDepth,
                                         const bool asyncExecution)
{

    std::string pE_str = pipelineExecution ? "true" : "false";
    std::string aE_str = asyncExecution ? "true" : "false";

    std::cout << "JPEG Pipeline Settings: [Device ID: " << deviceId
              << " numThreads: " << numThreads << ","
              << " batchSize: " << batchSize << ","
              << " pipelineExecution: " << pE_str << ","
              << " prefetchQueueDepth: " << prefetchQueueDepth << ","
              << " asyncExecution: " << aE_str << "]" << std::endl;

    const int seed = -1;
    this->decoderPipeline = new dali::Pipeline(batchSize, numThreads,
                                               deviceId, seed,
                                               pipelineExecution,
                                               prefetchQueueDepth,
                                               asyncExecution); //max_num_stream may become useful here

    //Hardcoded Input node
    const std::string externalInput = "raw_jpegs";
    this->input = {externalInput, "cpu"};
    this->decoderPipeline->AddExternalInput(externalInput);

    this->decoderPipeline->AddOperator(dali::OpSpec("HostDecoder")
                                           .AddArg("device", "cpu")
                                           .AddArg("output_type", dali::DALI_RGB)
                                           .AddInput(this->input.first, this->input.second)
                                           .AddOutput("decoded_images", "cpu"),
                                       "JPEGHostDecoder");

    std::cout << "Setting up JPEG decoder" << std::endl;
    this->outputs = {{"decoded_images", "cpu"}};
}

void JPEGDecoderPipeline::SetPipelineInput(dali::TensorList<dali::CPUBackend>& tl)
{
    this->decoderPipeline->SetExternalInput(this->input.first, tl);
}

void JPEGDecoderPipeline::BuildPipeline()
{
    std::cout << "Building JPEG Pipeline" << std::endl;
    this->decoderPipeline->Build(this->outputs);
}

void JPEGDecoderPipeline::RunPipeline()
{
    this->decoderPipeline->RunCPU();
    this->decoderPipeline->RunGPU();
}

void JPEGDecoderPipeline::GetPipelineOutput(std::vector<dali::TensorList<dali::CPUBackend>*>& tls)
{
    this->decoderPipeline->Outputs(&this->ws);
    tls.push_back(&ws.Output<dali::CPUBackend>(0));
}

std::pair<std::string, std::string> JPEGDecoderPipeline::GetInputNode()
{
    return this->input;
}

std::vector<std::pair<std::string, std::string>> JPEGDecoderPipeline::GetOutputNode()
{
    return this->outputs;
}
