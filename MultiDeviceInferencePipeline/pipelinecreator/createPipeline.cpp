/******************************************************************************
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
 * File: DL4AGX/MultiDeviceInferencePipeline/pipelinecreator/createPipeline.h
 * Description: Application to generate a serialized preprocessing pipeline
 *****************************************************************************/
#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <vector>

#include "common/argparser.h"
#include "common/common.h"
#include "pipeline.h"

namespace argparser = common::argparser;

// Parameters to set
struct Params
{
    std::vector<int> inputDims = std::vector<int>(3, 0);
    std::vector<float> imgMean = std::vector<float>(3, 0.0);
    std::vector<float> imgStd = std::vector<float>(3, 1.0);
    std::string outputPipeline;
    bool cpuMode{false};
} gParams;

void printUsage()
{
    std::cout << "\nInput params:" << std::endl;
    std::cout << "  --outputPipeline                                              Specify output pipeline file" << std::endl;
    std::cout << "  --meanVal                                              Mean values for image preprocessing" << std::endl;
    std::cout << "  --stdVal                                                Std values for image preprocessing" << std::endl;
    std::cout << "  --inputDim                                                                 Input dimension" << std::endl;
    std::cout << "  --cpu                                           Pipeline in CPU mode (Default is GPU mode)" << std::endl;
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
            continue;
        if (argparser::parseString(argv[j], "outputPipeline", gParams.outputPipeline))
            continue;
        if (argparser::parseString(argv[j], "inputDim", tmp))
        {
            std::vector<std::string> inputStrs = argparser::split(tmp, ',');
            if (inputStrs.size() != 3)
            {
                std::cerr << "Invalid input: " << tmp << std::endl;
                std::cerr << "Correct format should be: channel,height,width" << std::endl;
                return false;
            }
            gParams.inputDims[0] = atoi(inputStrs[0].c_str());
            gParams.inputDims[1] = atoi(inputStrs[1].c_str());
            gParams.inputDims[2] = atoi(inputStrs[2].c_str());
            continue;
        }
        if (argparser::parseString(argv[j], "meanVal", tmp))
        {
            std::vector<std::string> inputStrs = argparser::split(tmp, ',');
            if (inputStrs.size() != 3)
            {
                std::cerr << "Invalid input: " << tmp << std::endl;
                std::cerr << "Correct format should be: mean-R,mean-G,mean-B" << std::endl;
                return false;
            }
            gParams.imgMean[0] = atof(inputStrs[0].c_str());
            gParams.imgMean[1] = atof(inputStrs[1].c_str());
            gParams.imgMean[2] = atof(inputStrs[2].c_str());
            continue;
        }
        if (argparser::parseString(argv[j], "stdVal", tmp))
        {
            std::vector<std::string> inputStrs = argparser::split(tmp, ',');
            if (inputStrs.size() != 3)
            {
                std::cerr << "Invalid input: " << tmp << std::endl;
                std::cerr << "Correct format should be: std-R,std-G,std-B" << std::endl;
                return false;
            }
            gParams.imgStd[0] = atof(inputStrs[0].c_str());
            gParams.imgStd[1] = atof(inputStrs[1].c_str());
            gParams.imgStd[2] = atof(inputStrs[2].c_str());
            continue;
        }
        if (argparser::parseBool(argv[j], "cpu", gParams.cpuMode))
            continue;
    }

    if (showHelp)
    {
        printUsage();
        return false;
    }

    return true;
}

int main(int argc, char* argv[])
{
    // Parse command-line arguments.
    if (!parseArgs(argc, argv))
        return 1;

    // initialize DALI pipeline
    std::cout << "Dali pipeline initialization" << std::endl;
    dali::DALIInit(dali::OpSpec("CPUAllocator"),
                   dali::OpSpec("PinnedCPUAllocator"),
                   dali::OpSpec("GPUAllocator"));
    std::string daliPipe = serializePipe(gParams.inputDims, gParams.imgMean, gParams.imgStd, gParams.cpuMode);
    // write to binary file
    std::ofstream outfile(gParams.outputPipeline, std::ofstream::binary);
    if (!outfile)
    {
        std::cerr << "Error writing to the file" << gParams.outputPipeline << std::endl;
    }
    outfile << daliPipe;
    outfile.close();
    std::cout << "Preprocessing pipeline is saved to " << gParams.outputPipeline << std::endl;

    return EXIT_SUCCESS;
}
