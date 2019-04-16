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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/DALITRTPipeline/DALITRTPipeline.h
 * 
 * Description: Generic DALI Pipeline for doing inference using a TRT Engine
 ***************************************************************************************************/
#pragma once
#include "MultiDeviceInferencePipeline/inference/conf/PipelineSpec/PipelineSpec.h"
#include "MultiDeviceInferencePipeline/inference/preprocessing/preprocessing.h"

#include "dali/common.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

#include <string>
#include <utility>
#include <vector>

namespace multideviceinferencepipeline
{
namespace inference
{
/** 
 * DALITRTPipeline: 
 * A class that manages contruction of a unified inference pipeline
 * 
 * Takes a set of settings and a TensorRT inference engine and creates
 * a unified pipeline that will take a image preprocess and output the 
 * results of inference.  
 *
 **/
class DALITRTPipeline
{
public:
    DALITRTPipeline(const std::string pipelinePrefix,
                    preprocessing::PreprocessingSettings preprocessingSettings,
                    std::string TRTEngineFilePath,
                    std::vector<std::string> pluginPaths,
                    std::vector<std::string> engineInputBindings,
                    std::vector<std::string> engineOutputBindings,
                    const int deviceId = 1,
                    const int DLAcore = -1,
                    const int numThreads = 1,
                    const int batchSize = 1,
                    const bool pipelineExecution = true,
                    const int prefetchQueueDepth = 2,
                    const bool asyncExecution = true);
    DALITRTPipeline(conf::PipelineSpec& spec);
    ~DALITRTPipeline() = default;
    void SetPipelineInput(std::vector<dali::TensorList<dali::CPUBackend>*>& tls);
    void BuildPipeline();
    void RunPipeline();
    void GetPipelineOutput(std::vector<dali::TensorList<dali::GPUBackend>*>& data);
    std::vector<std::pair<std::string, std::string>> GetInputNodes();
    std::vector<std::pair<std::string, std::string>> GetOutputNodes();

private:
    dali::DeviceWorkspace ws;
    dali::Pipeline* inferencePipeline;
    std::string pipelinePrefix;

    std::vector<std::pair<std::string, std::string>> inputs;
    std::vector<std::pair<std::string, std::string>> outputs;
};
} //namespace inference
} //namespace multideviceinferencepipeline
