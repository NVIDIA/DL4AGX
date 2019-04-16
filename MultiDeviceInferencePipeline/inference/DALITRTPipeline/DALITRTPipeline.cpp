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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/DALITRTPipeline/DALITRTPipeline.cpp
 * 
 * Description: Generic DALI Pipeline for doing inference using a TRT Engine
 ***************************************************************************************************/
#include "MultiDeviceInferencePipeline/inference/DALITRTPipeline/DALITRTPipeline.h"
#include "MultiDeviceInferencePipeline/inference/utils/daliUtils.h"

#include "dali/common.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/util/image.h"

#include <cassert>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace multideviceinferencepipeline::inference;

DALITRTPipeline::DALITRTPipeline(conf::PipelineSpec& spec)
{
    std::vector<std::string> in;
    for (auto& i : spec.engineSettings.inputBindings)
    {
        in.push_back(i.first);
    }

    std::vector<std::string> out;
    for (auto& o : spec.engineSettings.outputBindings)
    {
        out.push_back(o.first);
    }

    new (this) DALITRTPipeline(spec.name,
                               spec.preprocessingSettings,
                               spec.enginePath,
                               spec.enginePluginPaths,
                               in,
                               out,
                               spec.deviceId,
                               spec.DLACore,
                               spec.numThreads,
                               spec.batchSize,
                               spec.pipelineExecution,
                               spec.prefetchQueueDepth,
                               spec.asyncExecution);
}

DALITRTPipeline::DALITRTPipeline(const std::string pipelinePrefix,
                                 preprocessing::PreprocessingSettings preprocessingSettings,
                                 std::string TRTEngineFilePath,
                                 std::vector<std::string> pluginPaths,
                                 std::vector<std::string> engineInputBindings,
                                 std::vector<std::string> engineOutputBindings,
                                 const int deviceId,
                                 const int DLACore,
                                 const int numThreads,
                                 const int batchSize,
                                 const bool pipelineExecution,
                                 const int prefetchQueueDepth,
                                 const bool asyncExecution)
{

    this->pipelinePrefix = pipelinePrefix;
    std::string pE_str = pipelineExecution ? "true" : "false";
    std::string aE_str = asyncExecution ? "true" : "false";

    std::cout << pipelinePrefix << " Pipeline Settings: [Device ID: " << deviceId
              << " numThreads: " << numThreads << ","
              << " batchSize: " << batchSize << ","
              << " pipelineExecution: " << pE_str << ","
              << " prefetchQueueDepth: " << prefetchQueueDepth << ","
              << " asyncExecution: " << aE_str << "]" << std::endl;

    const int seed = -1;
    this->inferencePipeline = new dali::Pipeline(batchSize, numThreads,
                                                 deviceId, seed,
                                                 pipelineExecution,
                                                 prefetchQueueDepth,
                                                 asyncExecution); //max_num_stream may become useful here

    //Hardcoded Input node
    const std::string externalInput = "decoded_jpegs";
    this->inputs.push_back({externalInput, "cpu"});
    this->inferencePipeline->AddExternalInput(externalInput);

    //When modifying this code, this is where you register the outputs of preprocessing
    const std::vector<std::pair<std::string, std::string>> preprocessingOutputNodes = {std::make_pair("preprocessed_images", "gpu")};

    std::cout << "Setting up " << pipelinePrefix << " preprocessing steps" << std::endl;
    //Single function to append the preprocessing steps to the pipeline (modify this function in preprocessingPipeline/pipeline.h to change these steps)
    preprocessing::AddOpsToPipeline(this->inferencePipeline, pipelinePrefix,
                                    this->inputs[0], preprocessingOutputNodes,
                                    preprocessingSettings, true);

    //Read in TensorRT Engine
    std::string serializedEngine;
    utils::readSerializedFileToString(TRTEngineFilePath, serializedEngine);

    dali::OpSpec inferOp("TensorRTInfer");

    inferOp.AddArg("device", "gpu")
        .AddArg("inference_batch_size", batchSize)
        .AddArg("engine", serializedEngine)
        .AddArg("plugins", pluginPaths)
        .AddArg("num_outputs", engineOutputBindings.size())
        .AddArg("input_nodes", engineInputBindings)
        .AddArg("output_nodes", engineOutputBindings)
        .AddArg("log_severity", 3);

    if (DLACore >= 0)
    {
        inferOp.AddArg("use_dla_core", DLACore);
    }

    for (auto& in : preprocessingOutputNodes)
    {
        inferOp.AddInput(in.first, "gpu");
    }

    for (auto& out : engineOutputBindings)
    {
        inferOp.AddOutput(out, "gpu");
        this->outputs.push_back({out, "gpu"});
    }

    std::cout << "Registering " << pipelinePrefix << " TensorRT Op" << std::endl;
    this->inferencePipeline->AddOperator(inferOp);
}

void DALITRTPipeline::SetPipelineInput(std::vector<dali::TensorList<dali::CPUBackend>*>& tls)
{
    assert(this->inputs.size() == tls.size());
    for (size_t i = 0; i < tls.size(); i++)
    {
        assert(tls[i] != nullptr);
        this->inferencePipeline->SetExternalInput(this->inputs[i].first, *(tls[i]));
    }
}

void DALITRTPipeline::BuildPipeline()
{
    std::cout << "Building " << this->pipelinePrefix << " Pipeline" << std::endl;
    this->inferencePipeline->Build(this->outputs);
}

void DALITRTPipeline::RunPipeline()
{
    this->inferencePipeline->RunCPU();
    this->inferencePipeline->RunGPU();
}

void DALITRTPipeline::GetPipelineOutput(std::vector<dali::TensorList<dali::GPUBackend>*>& tls)
{
    this->inferencePipeline->Outputs(&this->ws);
    for (size_t i = 0; i < this->outputs.size(); i++)
    {
        tls.push_back(&ws.Output<dali::GPUBackend>(i));
    }
}

std::vector<std::pair<std::string, std::string>> DALITRTPipeline::GetInputNodes()
{
    return this->inputs;
}

std::vector<std::pair<std::string, std::string>> DALITRTPipeline::GetOutputNodes()
{
    return this->outputs;
}
