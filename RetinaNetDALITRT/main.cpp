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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/main.cpp
 * 
 * Description: An application to facitate inference of multiple deep learning models on 
 * multiple heterogenous devices on the DRIVE AGX (and other Xavier Based) Platforms using the 
 * data pipelining and GPU primatives provided by NVIDIA DALI and the Inference Optimization 
 * capabilities of NVIDIA TensorRT 
 ***************************************************************************************************/

#include "common/argparser.h"
#include "common/timers.h"

#include "RetinaNetDALITRT/JPEGDecoderPipeline/JPEGDecoderPipeline.h"

// These dependencies are unchanged from the original app
#include "MultiDeviceInferencePipeline/inference/DALITRTPipeline/DALITRTPipeline.h"
#include "MultiDeviceInferencePipeline/inference/conf/conf.h"
#include "MultiDeviceInferencePipeline/inference/preprocessing/preprocessing.h"
#include "MultiDeviceInferencePipeline/inference/utils/daliUtils.h"

#include "dali/common.h"
#include "dali/pipeline/init.h"
#include "dali/pipeline/operators/op_spec.h"
#include "dali/pipeline/pipeline.h"
#include "dali/plugin/plugin_manager.h"
#include "dali/util/image.h"

#include "opencv2/core/core.hpp"
#include "opencv2/imgcodecs/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <map>
#include <utility>
#include <vector>

using namespace multideviceinferencepipeline::inference;
using namespace retinanetinferencepipeline::inference;

namespace argparser = common::argparser;

const std::string kDET_PIPELINE_NAME = "RetinaNet";

struct Params
{
    std::string confFile;
    std::string trtOpLib;
} gParams;

void printUsage();
void printAverageStdDev(std::string type, std::vector<float>& runtimes);

bool parseArgs(int argc, char* argv[]);
int appMain(conf::ExecutionSettings settings,
            JPEGDecoderPipeline& JPEGPipeline,
            DALITRTPipeline& detPipeline);
int profileApp(conf::ExecutionSettings settings,
               JPEGDecoderPipeline& JPEGPipeline,
               DALITRTPipeline& detPipeline);

int main(int argc, char* argv[])
{
    //Parse Args
    if (!parseArgs(argc, argv))
    {
        return 1;
    }

    std::map<std::string, conf::PipelineSpec> specs;
    conf::ExecutionSettings settings;
    conf::parseInferencePipelineConfFile(gParams.confFile, settings, specs);

    dali::DALIInit(dali::OpSpec("CPUAllocator"),
                   dali::OpSpec("PinnedCPUAllocator"),
                   dali::OpSpec("GPUAllocator"));
    dali::PluginManager::LoadLibrary(gParams.trtOpLib);

    JPEGDecoderPipeline JPEGPipeline{};

    //Create DALI Pipelines for Engine
    if (specs.find(kDET_PIPELINE_NAME) == specs.end())
    {
        std::cerr << "ERROR: Cannot find pipeline spec for the Object Detection \
Pipeline (should be named \""
                  << kDET_PIPELINE_NAME << "\")" << std::endl;
        return 1;
    }
    DALITRTPipeline DetectionPipeline(specs["RetinaNet"]);

    std::cout << "Build DALI pipelines" << std::endl;
    JPEGPipeline.BuildPipeline();
    DetectionPipeline.BuildPipeline();

    if (settings.profile)
    {
        return profileApp(settings,
                          JPEGPipeline,
                          DetectionPipeline);
    }

    return appMain(settings,
                   JPEGPipeline,
                   DetectionPipeline);
}

/**
 * This is function runs inference once on an image, runs postprocessing 
 * and writes it to a file to view 
 **/
int appMain(conf::ExecutionSettings settings,
            JPEGDecoderPipeline& JPEGPipeline,
            DALITRTPipeline& detPipeline)
{
    /// Load JPEG image from file and make a DALI batch (once per pipeline)
    std::cout << "Load JPEG images" << std::endl;
    dali::TensorList<dali::CPUBackend> JPEGBatch;
    utils::makeJPEGBatch(settings.inFiles, &JPEGBatch, settings.batchSize);

    JPEGPipeline.SetPipelineInput(JPEGBatch);
    JPEGPipeline.RunPipeline();

    std::vector<dali::TensorList<dali::CPUBackend>*> detInputBatch;
    JPEGPipeline.GetPipelineOutput(detInputBatch);

    /// Load this image into the pipeline (note there is no cuda memcpy yet as
    /// JPEG decoding is done CPU side, DALI will handle the memcpy between ops
    std::cout << "Load into inference pipelines" << std::endl;
    detPipeline.SetPipelineInput(detInputBatch);

    /// Run the inference pipeline on both the GPU and DLA
    /// While this is done serially in the app context, when the pipelines are built
    /// with AsyncExecution enabled (default), the pipelines themselves will run concurrently
    std::cout << "Starting inference pipelines" << std::endl;
    detPipeline.RunPipeline();

    /// Now setting a blocking call for the pipelines to syncronize the pipeline executions
    std::cout << "Tranfering inference results back to host for postprocessing" << std::endl;
    std::vector<dali::TensorList<dali::GPUBackend>*> detPipelineResults;
    detPipeline.GetPipelineOutput(detPipelineResults);
    /// Copy data back to host
    std::vector<float> detOutput330(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["330"]),
                                    0);
    std::vector<float> detOutput331(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["331"]),
                                    0);
    std::vector<float> detOutput332(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["332"]),
                                    0);
    std::vector<float> detOutput333(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["333"]),
                                    0);
    std::vector<float> detOutput334(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["334"]),
                                    0);
    std::vector<float> detOutput293(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["293"]),
                                    0);
    std::vector<float> detOutput302(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["302"]),
                                    0);
    std::vector<float> detOutput311(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["311"]),
                                    0);
    std::vector<float> detOutput320(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["320"]),
                                    0);
    std::vector<float> detOutput329(conf::bindingSize(settings
                                                      .pipelineBindings[kDET_PIPELINE_NAME]
                                                      .outputBindings["329"]),
                                    0);
    utils::GPUTensorListToRawData<float>(detPipelineResults[0], &detOutput293);
    utils::GPUTensorListToRawData<float>(detPipelineResults[1], &detOutput302);
    utils::GPUTensorListToRawData<float>(detPipelineResults[2], &detOutput311);
    utils::GPUTensorListToRawData<float>(detPipelineResults[3], &detOutput320);
    utils::GPUTensorListToRawData<float>(detPipelineResults[4], &detOutput329);
    utils::GPUTensorListToRawData<float>(detPipelineResults[5], &detOutput330);
    utils::GPUTensorListToRawData<float>(detPipelineResults[6], &detOutput331);
    utils::GPUTensorListToRawData<float>(detPipelineResults[7], &detOutput332);
    utils::GPUTensorListToRawData<float>(detPipelineResults[8], &detOutput333);
    utils::GPUTensorListToRawData<float>(detPipelineResults[9], &detOutput334);

    /// Doing postprocessing on the inference results to generate bounding boxes and a segmentation mask
    /// Overlay on original image

    return 0;
}

int profileApp(conf::ExecutionSettings settings,
               JPEGDecoderPipeline& JPEGPipeline,
               DALITRTPipeline& detPipeline)
{
    auto decodeTimer = common::timers::PreciseCpuTimer();
    auto executionTimer = common::timers::PreciseCpuTimer();
    std::vector<float> decodeRuntimes;
    std::vector<float> executionRuntimes;

    for (uint i = 0; i < settings.iters; i++)
    {
        decodeTimer.start();
        dali::TensorList<dali::CPUBackend> JPEGBatch;
        utils::makeJPEGBatch(settings.inFiles, &JPEGBatch, settings.batchSize);
        JPEGPipeline.SetPipelineInput(JPEGBatch);
        JPEGPipeline.RunPipeline();
        std::vector<dali::TensorList<dali::CPUBackend>*> detInputBatch;
        JPEGPipeline.GetPipelineOutput(detInputBatch);
        detPipeline.SetPipelineInput(detInputBatch);
        decodeTimer.stop();

        executionTimer.start();
        detPipeline.RunPipeline();
        std::vector<dali::TensorList<dali::GPUBackend>*> detPipelineResults;
        detPipeline.GetPipelineOutput(detPipelineResults);
        std::vector<float> detOutput330(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["330"]),
                                        0);
        std::vector<float> detOutput331(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["331"]),
                                        0);
        std::vector<float> detOutput332(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["332"]),
                                        0);
        std::vector<float> detOutput333(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["333"]),
                                        0);
        std::vector<float> detOutput334(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["334"]),
                                        0);
        std::vector<float> detOutput293(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["293"]),
                                        0);
        std::vector<float> detOutput302(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["302"]),
                                        0);
        std::vector<float> detOutput311(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["311"]),
                                        0);
        std::vector<float> detOutput320(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["320"]),
                                        0);
        std::vector<float> detOutput329(conf::bindingSize(settings
                                                          .pipelineBindings[kDET_PIPELINE_NAME]
                                                          .outputBindings["329"]),
                                        0);

        utils::GPUTensorListToRawData<float>(detPipelineResults[0], &detOutput293);
        utils::GPUTensorListToRawData<float>(detPipelineResults[1], &detOutput302);
        utils::GPUTensorListToRawData<float>(detPipelineResults[2], &detOutput311);
        utils::GPUTensorListToRawData<float>(detPipelineResults[3], &detOutput320);
        utils::GPUTensorListToRawData<float>(detPipelineResults[4], &detOutput329);
        utils::GPUTensorListToRawData<float>(detPipelineResults[5], &detOutput330);
        utils::GPUTensorListToRawData<float>(detPipelineResults[6], &detOutput331);
        utils::GPUTensorListToRawData<float>(detPipelineResults[7], &detOutput332);
        utils::GPUTensorListToRawData<float>(detPipelineResults[8], &detOutput333);
        utils::GPUTensorListToRawData<float>(detPipelineResults[9], &detOutput334);
        
        executionTimer.stop();
        
        auto decodeTime = decodeTimer.milliseconds();
        auto executionTime = executionTimer.milliseconds();
        decodeTimer.reset();
        executionTimer.reset();
        decodeRuntimes.push_back(decodeTime);
        executionRuntimes.push_back(executionTime);
        if (decodeRuntimes.size() > settings.timed_iters)
        {
            decodeRuntimes.erase(decodeRuntimes.begin());
        }
        if (executionRuntimes.size() > settings.timed_iters)
        {
            executionRuntimes.erase(executionRuntimes.begin());
        }
        std::cout << "[Host] Decode Time from image on filesystem -> Loaded into CPU Buffer: " << decodeTime << " ms | " << 1000.f / decodeTime << "FPS" << std::endl;
        std::cout << "[Host] Inference Time from cpu buffer -> Inference Results on Host: " << executionTime << " ms | " << 1000.f / executionTime << "FPS" << std::endl;
    }
    printAverageStdDev("Decode", decodeRuntimes);
    printAverageStdDev("Inference", executionRuntimes);
    return 0;
}

void printAverageStdDev(std::string type, std::vector<float>& runtimes)
{
    float avgRuntime = std::accumulate(runtimes.begin(), runtimes.end(), 0.0) / runtimes.size();
    float fps = 1000.f / avgRuntime;
    std::cout << "[" << type << " Stage] Average FPS: " << fps << " (excluding initial outlier runs)" << std::endl;
    std::vector<float> diff(runtimes.size());
    std::transform(runtimes.begin(), runtimes.end(), diff.begin(), [fps](float x) { return (1000.f / x) - fps; });
    float sqSum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
    float stdDev = std::sqrt(sqSum / runtimes.size());
    std::cout << "[" << type << " Stage] Standard Deviation: " << stdDev << std::endl;
}

void printUsage()
{
    std::cout << "\nMandatory params:" << std::endl;
    std::cout << "  --conf=<name>                                     Pipeline configuration file" << std::endl;
    std::cout << "  --trtoplib=<name>                                 DALI/TensorRT Inference Op" << std::endl;
}

bool parseArgs(int argc, char* argv[])
{
    if (argc < 1)
    {
        printUsage();
        return false;
    }

    std::string tmp;
    bool showHelp = false;
    for (int j = 1; j < argc; j++)
    {
        tmp.clear();
        if (argparser::parseBool(argv[j], "help", showHelp, 'h'))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "conf", gParams.confFile))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "trtoplib", gParams.trtOpLib))
        {
            continue;
        }
    }

    if (showHelp)
    {
        printUsage();
        return false;
    }

    return true;
}
