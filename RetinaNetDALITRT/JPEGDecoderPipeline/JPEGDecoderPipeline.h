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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/JPEGDecoderPipeline/JPEGDecoderPipeline.h
 * 
 * Description: DALI pipeline to decode JPEG images
 ***************************************************************************************************/
#pragma once
#include "dali/common.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

#include <string>
#include <utility>
#include <vector>

namespace retinanetinferencepipeline
{
namespace inference
{
class JPEGDecoderPipeline
{
public:
    JPEGDecoderPipeline(const int deviceId = 0,
                        const int numThreads = 1,
                        const int batchSize = 1,
                        const bool pipelineExecution = true,
                        const int prefetchQueueDepth = 2,
                        const bool asyncExecution = true);
    ~JPEGDecoderPipeline() = default;
    void SetPipelineInput(dali::TensorList<dali::CPUBackend>& tl);
    void BuildPipeline();
    void RunPipeline();
    void GetPipelineOutput(std::vector<dali::TensorList<dali::CPUBackend>*>& tls);
    std::pair<std::string, std::string> GetInputNode();
    std::vector<std::pair<std::string, std::string>> GetOutputNode();

private:
    dali::DeviceWorkspace ws;
    dali::Pipeline* decoderPipeline;

    std::pair<std::string, std::string> input;
    std::vector<std::pair<std::string, std::string>> outputs;
};
} //namespace inference
} //namespace retinanetinferencepipeline
