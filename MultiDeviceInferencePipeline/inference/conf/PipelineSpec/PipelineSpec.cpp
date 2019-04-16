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
 * File: DL4AGX/MultiDeviceInferencePipeline/inference/conf/PipelineSpec/PipelineSpec.cpp
 * 
 * Description: Parses a toml file section for DALITRTPipeline settings
 ***************************************************************************************************/
#include "MultiDeviceInferencePipeline/inference/conf/PipelineSpec/PipelineSpec.h"

#include <iostream>

using namespace multideviceinferencepipeline::inference;

conf::PipelineSpec::PipelineSpec(const std::shared_ptr<cpptoml::table>& conf)
{
    this->name = conf->get_as<std::string>("name").value_or("DALITRTPipeline");
    this->deviceId = conf->get_as<int>("device").value_or(0);
    this->DLACore = conf->get_as<int>("dla_core").value_or(-1);

    this->batchSize = conf->get_as<int>("batch_size").value_or(1);
    this->asyncExecution = conf->get_as<bool>("async_execution").value_or(true);
    this->numThreads = conf->get_as<int>("num_threads").value_or(1);

    auto engineConf = conf->get_table("engine");
    auto preprocessingConf = conf->get_table("preprocessing");

    this->enginePath = engineConf->get_as<std::string>("path").value_or("UNKNOWN ENGINE PATH");

    auto resizeConf = preprocessingConf->get_array_of<int64_t>("resize");
    std::vector<int> resize;
    for (const auto& val : *resizeConf)
    {
        resize.push_back((int) val);
    }

    auto meanConf = preprocessingConf->get_array_of<double>("mean");
    std::vector<float> mean;
    for (const auto& val : *meanConf)
    {
        mean.push_back((float) val);
    }

    auto stdDevConf = preprocessingConf->get_array_of<double>("std_dev");
    std::vector<float> stdDev;
    for (const auto& val : *stdDevConf)
    {
        stdDev.push_back((float) val);
    }

    this->preprocessingSettings = {
        resize,
        mean,
        stdDev,
    };

    std::map<std::string, std::tuple<int, int, int>> inputs;
    auto engineInputs = engineConf->get_table_array("inputs");
    for (const auto& in : *engineInputs)
    {
        auto shapeConf = in->get_array_of<int64_t>("shape");
        std::vector<int> shape;
        for (const auto& val : *shapeConf)
        {
            shape.push_back((int) val);
        }
        inputs[in->get_as<std::string>("name").value_or("UNREADABLE NAME")] = std::make_tuple(shape[0], shape[1], shape[2]);
    }

    std::map<std::string, std::tuple<int, int, int>> outputs;
    auto engineOutputs = engineConf->get_table_array("outputs");
    for (const auto& out : *engineOutputs)
    {
        auto shapeConf = out->get_array_of<int64_t>("shape");
        std::vector<int> shape;
        for (const auto& val : *shapeConf)
        {
            shape.push_back((int) val);
        }
        outputs[out->get_as<std::string>("name").value_or("UNREADABLE NAME")] = std::make_tuple(shape[0], shape[1], shape[2]);
    }

    this->engineSettings = {
        inputs,
        outputs};

    auto plugins = engineConf->get_table_array("plugins");
    if (plugins)
    {
        for (const auto& p : *plugins)
        {
            this->enginePluginPaths.push_back(p->get_as<std::string>("path").value_or("UNKNOWN PLUGIN PATH"));
        }
    }
}

void conf::PipelineSpec::printSpec()
{
    auto nameStr = "Name: " + this->name;

    std::stringstream ss;
    ss << "Device ID: " << this->deviceId << "    DLA Core: " << this->DLACore;
    auto deviceStr = ss.str();
    ss.str(std::string());

    ss << "Batch Size: " << this->batchSize << "    Num Threads: " << this->numThreads;
    auto otherSettings = ss.str();
    ss.str(std::string());

    ss << "Execution Mode: " << (this->asyncExecution ? "async" : "sync");
    auto async = ss.str();
    ss.str(std::string());

    ss << "Resize: [" << this->preprocessingSettings.imgDims[0]
       << ", " << this->preprocessingSettings.imgDims[1]
       << ", " << this->preprocessingSettings.imgDims[2] << "]";
    auto resizeStr = ss.str();
    ss.str(std::string());

    ss << "Mean: [" << this->preprocessingSettings.imgMean[0]
       << ", " << this->preprocessingSettings.imgMean[1]
       << ", " << this->preprocessingSettings.imgMean[2] << "]";
    auto meanStr = ss.str();
    ss.str(std::string());

    ss << "Std Dev: [" << this->preprocessingSettings.imgStd[0]
       << ", " << this->preprocessingSettings.imgStd[1]
       << ", " << this->preprocessingSettings.imgStd[2] << "]";
    auto stdStr = ss.str();

    auto path = "Path: " + this->enginePath.substr(0, 30);
    if (this->enginePath.length() > 30)
    {
        path += "...";
    }

    std::vector<std::string> inStrs;
    for (auto const& in : this->engineSettings.inputBindings)
    {
        std::stringstream ss_in;
        ss_in << in.first << ": [" << std::get<0>(in.second)
              << ", " << std::get<1>(in.second)
              << ", " << std::get<2>(in.second) << "]";
        inStrs.push_back(ss_in.str());
    }
    std::vector<std::string> outStrs;
    for (auto const& out : this->engineSettings.outputBindings)
    {
        std::stringstream ss_out;
        ss_out << out.first << ": [" << std::get<0>(out.second)
               << ", " << std::get<1>(out.second)
               << ", " << std::get<2>(out.second) << "]";
        outStrs.push_back(ss_out.str());
    }
    std::vector<std::string> pluginStrs;
    for (auto const& plug : this->enginePluginPaths)
    {
        auto plugPath = "Path: " + plug.substr(0, 30);
        if (plug.length() > 23)
        {
            plugPath += "...";
        }
        pluginStrs.push_back(plugPath);
    }

    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "| Pipeline Spec                                  |" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "| " << nameStr.append(47 - nameStr.length(), ' ') << "|" << std::endl;
    std::cout << "| " << deviceStr.append(47 - deviceStr.length(), ' ') << "|" << std::endl;
    std::cout << "| " << otherSettings.append(47 - otherSettings.length(), ' ') << "|" << std::endl;
    std::cout << "| " << async.append(47 - async.length(), ' ') << "|" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "| Preprocessing:                                 |" << std::endl;
    std::cout << "|   " << resizeStr.append(45 - resizeStr.length(), ' ') << "|" << std::endl;
    std::cout << "|   " << meanStr.append(45 - meanStr.length(), ' ') << "|" << std::endl;
    std::cout << "|   " << stdStr.append(45 - stdStr.length(), ' ') << "|" << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "| Engine:                                        |" << std::endl;
    std::cout << "|   " << path.append(45 - path.length(), ' ') << "|" << std::endl;
    std::cout << "|   Input(s):                                    |" << std::endl;
    for (auto& s : inStrs)
    {
        std::cout << "|     " << s.append(43 - s.length(), ' ') << "|" << std::endl;
    }
    std::cout << "|   Output(s):                                   |" << std::endl;
    for (auto& s : outStrs)
    {
        std::cout << "|     " << s.append(43 - s.length(), ' ') << "|" << std::endl;
    }
    if (this->enginePluginPaths.size() > 0)
    {
        std::cout << "|   Plugin(s):                                   |" << std::endl;
        for (auto& s : pluginStrs)
        {
            std::cout << "|     " << s.append(43 - s.length(), ' ') << "|" << std::endl;
        }
    }
    std::cout << "--------------------------------------------------" << std::endl;
}
