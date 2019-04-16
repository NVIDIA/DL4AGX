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
 * Project: MultiDeviceInferencePipeline > enginecreator
 * 
 * File: DL4AGX/MultiDeviceInferencePipeline/enginecreator/createEngine.cpp
 * 
 * Description: Application to convert a UFF file to a TRT Engine
 ***************************************************************************************************/
#include <cassert>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <vector>

// module dependencies
#include "MultiDeviceInferencePipeline/enginecreator/DALIStream/DALIStream.h"
#include "MultiDeviceInferencePipeline/enginecreator/Int8Calibrator/Int8Calibrator.h"

// TRT dependencies
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvUffParser.h"

// utility dependencies
#include "common/common.h"

using namespace multideviceinferencepipeline::enginecreator;
namespace coco = common::datasets::coco;
namespace argparser = common::argparser;

static common::tensorrt::Logger gLogger;

// Default calibration batch size
static constexpr int DEFAULT_CAL_BATCH_SIZE = 50;
static constexpr int DEFAULT_CAL_IMAGES = 500;

std::vector<std::string> gInputs;
std::map<std::string, nvinfer1::Dims3> gInputDimensions;
std::map<std::string, nvinfer1::Dims3> gOutputDimensions;

// Parameters to set
struct Params
{
    std::string uffFile, outputEngine;
    std::string inputBlob, outputBlob;
    int inputC, inputH, inputW;
    int batchSize{DEFAULT_CAL_BATCH_SIZE};

    std::string calibrationCache{"CalibrationTableSSD"};
    std::string calibFolder;           // folder for calibration images
    int calibSize{DEFAULT_CAL_IMAGES}; // number of calibration images
    std::string calibJSON;
    std::string testFolder; // folder for test images
    std::string testJSON;

    std::string evalJSON;
    std::vector<std::string> outputs;
    int device{0}, workspaceSize{4}; // in GB
    bool fp16{false}, int8{false};
    std::string engine;
    std::vector<std::pair<std::string, nvinfer1::Dims3>> uffInputs;
    nvinfer1::DimsNCHW NCHWDims;
    std::string serializedPipe;
    bool runOnDla{false};
    std::string daliPipe;
    bool needPipe{false};
    std::string libPlugin;
    dali::Pipeline* ppreprocessPipe{nullptr};
    dali::DeviceWorkspace preprocessWS;
    int nThreads{1};
    bool asyncDali{true};
    bool pipelinedDali{true};
    int queueDepthDali{2};
    std::vector<std::string> outputBindings;
    std::string trtOpLib;
} gParams;

void postprocess(float* detectionOut,
                 int* keepCount,
                 std::vector<coco::data::image> inputImages,
                 int batchSize,
                 std::vector<coco::results::objectDetection>& output)
{
    int keepTopK = gOutputDimensions[gParams.outputBlob].d[1];
    int detFields = gOutputDimensions[gParams.outputBlob].d[2];
    for (int p = 0; p < batchSize; ++p)
    {
        for (int i = 0; i < (int) (keepCount[p]); ++i)
        {
            float* det = detectionOut + (p * keepTopK + i) * detFields;
            common::datasets::coco::results::objectDetection o;
            o.image_id = inputImages[p].id;
            o.category_id = (int) det[1];
            o.bbox[0] = det[3] * inputImages[p].width;
            o.bbox[1] = det[4] * inputImages[p].height;
            o.bbox[2] = det[5] * inputImages[p].width - det[3] * inputImages[p].width;
            o.bbox[3] = det[6] * inputImages[p].height - det[4] * inputImages[p].height;
            o.score = det[2];
            output.push_back(o);
        }
    }
}

void getJSONData(const std::string& jsonfile, std::vector<coco::data::image>& filelist,
                 int max = std::numeric_limits<int>::max())
{
    // load the json file
    std::ifstream i(jsonfile);
    assert(i.is_open() && "JSON file is not open.");
    json j;
    i >> j;

    int count = 0;
    for (auto& element : j["images"])
    {
        if (count >= max)
            return;
        coco::data::image im = element;
        filelist.push_back(im);
        count++;
    }
}

std::vector<std::pair<int64_t, nvinfer1::DataType>>
calculateBindingBufferSizes(const nvinfer1::ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, nvinfer1::DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        nvinfer1::Dims dims = engine.getBindingDimensions(i);
        nvinfer1::DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = common::tensorrt::volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

nvinfer1::ICudaEngine* uffToTRTModel()
{
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);
    nvinfer1::INetworkDefinition* network = builder->createNetwork();
    nvuffparser::IUffParser* parser = nvuffparser::createUffParser();
    // specify which tensors are outputs
    for (auto& s : gParams.outputs)
    {
        if (!parser->registerOutput(s.c_str()))
        {
            std::cerr << "Failed to register output " << s << std::endl;
            return nullptr;
        }
    }

    // specify which tensors are inputs (and their dimensions)
    for (auto& s : gParams.uffInputs)
    {
        if (!parser->registerInput(s.first.c_str(), s.second, nvuffparser::UffInputOrder::kNCHW))
        {
            std::cerr << "Failed to register input " << s.first << std::endl;
            return nullptr;
        }
    }

    if (!parser->parse(gParams.uffFile.c_str(), *network,
                       gParams.fp16 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT))
        return nullptr;

    for (int i = 0, n = network->getNbInputs(); i < n; i++)
    {
        nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getInput(i)->getDimensions());
        gInputs.push_back(network->getInput(i)->getName());
        gInputDimensions.insert(std::make_pair(network->getInput(i)->getName(), dims));
        std::cout << "Input \"" << network->getInput(i)->getName() << "\": " << dims.d[0] << "x"
                  << dims.d[1] << "x" << dims.d[2] << std::endl;
    }

    for (int i = 0, n = network->getNbOutputs(); i < n; i++)
    {
        nvinfer1::Dims3 dims = static_cast<nvinfer1::Dims3&&>(network->getOutput(i)->getDimensions());
        gOutputDimensions.insert(std::make_pair(network->getOutput(i)->getName(), dims));
        std::cout << "Output \"" << network->getOutput(i)->getName() << "\": " << dims.d[0] << "x"
                  << dims.d[1] << "x" << dims.d[2] << std::endl;
        gParams.outputBindings.push_back(network->getOutput(i)->getName());
    }

    // Build the engine
    builder->setMaxBatchSize(gParams.batchSize);
    builder->setMaxWorkspaceSize((size_t(1024) << 20) * size_t(gParams.workspaceSize));
    builder->setFp16Mode(gParams.fp16);

    DALIStream* calibStream = nullptr;
    Int8EntropyCalibrator* calibrator = nullptr;

    if (gParams.int8)
    {
        std::vector<coco::data::image> calibData;
        getJSONData(gParams.calibJSON, calibData, gParams.calibSize);
        calibStream = new DALIStream(gParams.batchSize, gInputDimensions[gParams.inputBlob], gParams.serializedPipe, calibData, gParams.calibFolder, gParams.ppreprocessPipe, gParams.preprocessWS, gParams.asyncDali);
        calibrator = new Int8EntropyCalibrator(*calibStream, gParams.calibrationCache, gInputDimensions);
        builder->setInt8Mode(true);
        builder->setInt8Calibrator(calibrator);
    }

    if (gParams.runOnDla)
    {
        common::tensorrt::enableDLA(builder, 0);
    }

    nvinfer1::ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
        std::cerr << "could not build engine" << std::endl;

    // Save the engine
    nvinfer1::IHostMemory* trtModelStream{nullptr};
    trtModelStream = engine->serialize();
    std::ofstream savedEngine(gParams.outputEngine);
    savedEngine.write(reinterpret_cast<const char*>(trtModelStream->data()), trtModelStream->size());
    std::cout << "Engine saved to " << gParams.outputEngine << std::endl;
    parser->destroy();
    network->destroy();
    builder->destroy();
    delete calibStream;
    delete calibrator;
    return engine;
}

nvinfer1::ICudaEngine* createEngine()
{
    nvinfer1::ICudaEngine* engine;
    if (!gParams.uffFile.empty())
    {
        if (!gParams.uffFile.empty())
        {
            engine = uffToTRTModel();
        }

        if (!engine)
        {
            std::cerr << "Engine could not be created" << std::endl;
            return nullptr;
        }

        if (!gParams.engine.empty())
        {
            std::ofstream p(gParams.engine);
            if (!p)
            {
                std::cerr << "could not open plan output file" << std::endl;
                return nullptr;
            }
            nvinfer1::IHostMemory* ptr = engine->serialize();
            assert(ptr);
            p.write(reinterpret_cast<const char*>(ptr->data()), ptr->size());
            ptr->destroy();
        }
        return engine;
    }

    // load directly from serialized engine file if deploy not specified
    if (!gParams.engine.empty())
    {
        char* trtModelStream{nullptr};
        size_t size{0};
        std::ifstream file(gParams.engine, std::ios::binary);
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }

        nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
        engine = infer->deserializeCudaEngine(trtModelStream, size, nullptr);
        if (trtModelStream)
        {
            delete[] trtModelStream;
        }
        gInputs.push_back(gParams.inputBlob);
        return engine;
    }

    // complain about empty deploy file
    std::cerr << "Deploy file not specified" << std::endl;
    return nullptr;
}

void printUsage()
{
    std::cout << "\nInput params:" << std::endl;
    std::cout << "  --uffModel                                                                 Input UFF model" << std::endl;
    std::cout << "  --outputEngine                                                  Specify output engine file" << std::endl;
    std::cout << "  --inputBlob                                                                Input blob name" << std::endl;
    std::cout << "  --inputDim                                                                 Input dimension" << std::endl;
    std::cout << "  --outputBlob                                                              Output blob name" << std::endl;
    std::cout << "  --int8                                                                Generate int8 engine" << std::endl;
    std::cout << "  --calibFolder                                                Folder for calibration images" << std::endl;
    std::cout << "  --calibJSON                                                    JSON for calibration images" << std::endl;
    std::cout << "  --testFolder                                                        Folder for test images" << std::endl;
    std::cout << "  --testJSON                                                            JSON for test images" << std::endl;
    std::cout << "  --evalJSON                                               Output evaluation results in JSON" << std::endl;
    std::cout << "  --pipeLine                                             Pipleline to load for preprocessing" << std::endl;
    std::cout << "  --fp16                                                                    Enable fp16 mode" << std::endl;
    std::cout << "  --useDLA                                                              Build engine for DLA" << std::endl;
    std::cout << "  --pipeline                                           Serialized pipeline for preprocessing" << std::endl;
    std::cout << "  --plugin                                                               Lib file for plugin" << std::endl;
    std::cout << "  --workspace                              Workspace size when building engine (default 4GB)" << std::endl;
    std::cout << "  --device                                                     Device ID for GPU (default 0)" << std::endl;
    std::cout << "  --trtoplib                                                          TRT op plugin for DALI" << std::endl;
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
        if (argparser::parseString(argv[j], "uffModel", gParams.uffFile))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "inputBlob", gParams.inputBlob))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "outputBlob", gParams.outputBlob))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "outputEngine", gParams.outputEngine))
        {
            continue;
        }
        if (argparser::parseBool(argv[j], "int8", gParams.int8))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "inputDim", tmp))
        {
            std::vector<std::string> inputStrs = argparser::split(tmp, ',');
            if (inputStrs.size() != 3)
            {
                std::cerr << "Invalid input: " << tmp << std::endl;
                std::cerr << "Correct format should be: channel,height,width" << std::endl;
                return false;
            }
            gParams.inputC = atoi(inputStrs[0].c_str());
            gParams.inputH = atoi(inputStrs[1].c_str());
            gParams.inputW = atoi(inputStrs[2].c_str());
            continue;
        }
        if (argparser::parseInt(argv[j], "batch", gParams.batchSize))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "testJSON", gParams.testJSON))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "testFolder", gParams.testFolder))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "evalJSON", gParams.evalJSON))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "calibJSON", gParams.calibJSON))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "calibFolder", gParams.calibFolder))
        {
            continue;
        }
        if (argparser::parseInt(argv[j], "device", gParams.device))
        {
            continue;
        }
        if (argparser::parseBool(argv[j], "useDLA", gParams.runOnDla))
        {
            continue;
        }
        if (argparser::parseInt(argv[j], "workspace", gParams.workspaceSize))
        {
            continue;
        }
        if (argparser::parseBool(argv[j], "fp16", gParams.fp16))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "pipeline", gParams.daliPipe))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "plugin", gParams.libPlugin))
        {
            continue;
        }
        if (argparser::parseString(argv[j], "calibrationTable", gParams.calibrationCache))
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

int main(int argc, char* argv[])
{
    // Parse command-line arguments.
    if (!parseArgs(argc, argv))
        return 1;

    // load plugin library
    void* dlh = dlopen(gParams.libPlugin.c_str(), RTLD_LAZY);
    assert(nullptr != dlh);

    initLibNvInferPlugins(&gLogger, "");

    gParams.outputs.push_back(gParams.outputBlob);
    gParams.uffInputs.push_back(std::make_pair(gParams.inputBlob, nvinfer1::DimsCHW(gParams.inputC, gParams.inputH, gParams.inputW)));
    gParams.NCHWDims = nvinfer1::DimsNCHW(gParams.batchSize, gParams.inputC, gParams.inputH, gParams.inputW);

    if (gParams.int8 || gParams.testFolder.size() > 0)
        gParams.needPipe = true;
    if (gParams.daliPipe.size() == 0 && gParams.needPipe)
    {
        std::cerr << "Pipeline is needed for calibration or validation with images" << std::endl;
        return false;
    }

    if (gParams.needPipe)
    {
        // initialize DALI pipeline
        dali::DALIInit(dali::OpSpec("CPUAllocator"),
                       dali::OpSpec("PinnedCPUAllocator"),
                       dali::OpSpec("GPUAllocator"));
        std::ifstream infile(gParams.daliPipe, std::ios::binary);
        if (!infile)
        {
            std::cerr << "Error reading the file name " << gParams.daliPipe << std::endl;
        }
        std::stringstream pipe_buffer;
        pipe_buffer << infile.rdbuf();
        gParams.serializedPipe = pipe_buffer.str();
        infile.close();
        std::cout << "DALI pipeline loaded" << std::endl;

        // construct preprocess pipeline
        gParams.ppreprocessPipe = new dali::Pipeline(gParams.serializedPipe, gParams.batchSize, gParams.nThreads, gParams.device, gParams.pipelinedDali, gParams.queueDepthDali, gParams.asyncDali);
        std::vector<std::pair<std::string, std::string>> preprocessOutputs = {{"preprocessed_images", "gpu"}};
        gParams.ppreprocessPipe->Build(preprocessOutputs);
    }

    // set device before building the engine
    cudaSetDevice(gParams.device);

    nvinfer1::ICudaEngine* engine = createEngine();
    if (!engine)
    {
        std::cerr << "Engine could not be created" << std::endl;
        return -1;
    }
    else
    {
        std::cout << "Successfully created engine" << std::endl;
    }
    // exit the program if no inference is needed
    if (gParams.testFolder.empty())
    {
        engine->destroy();
        return EXIT_SUCCESS;
    }

    // populate the list for test input files
    std::vector<coco::data::image> testData;
    getJSONData(gParams.testJSON, testData);
    std::cout << "Total number of test images is " << testData.size() << std::endl;

    std::vector<coco::data::image> images(gParams.batchSize);
    std::vector<coco::results::objectDetection> output;

    // first load TRT plugin for dali
    dali::PluginManager::LoadLibrary(gParams.trtOpLib);
    // get pipe with TRT
    std::string serializedInferencePipe;
    DALIStream::buildInferencePipe(gParams.serializedPipe, gParams.NCHWDims, gParams.outputEngine, gParams.libPlugin, gParams.inputBlob, gParams.outputBindings, serializedInferencePipe, gParams.device, gParams.pipelinedDali, gParams.queueDepthDali, gParams.asyncDali);
    dali::Pipeline inferencePipe(serializedInferencePipe, gParams.batchSize, gParams.nThreads, gParams.device, gParams.pipelinedDali, gParams.queueDepthDali, gParams.asyncDali);
    std::cout << "Inference pipe constructed" << std::endl;
    std::vector<std::pair<std::string, std::string>> outputsInferencePipe;
    for (size_t i = 0; i < gParams.outputBindings.size(); ++i)
    {
        outputsInferencePipe.push_back(make_pair(gParams.outputBindings[i], "gpu"));
    }
    inferencePipe.Build(outputsInferencePipe);
    dali::DeviceWorkspace inferenceWS;

    // Perform batch inference
    std::vector<coco::data::image>::iterator dataIter;
    dataIter = testData.begin();
    std::cout << "In total there will be " << testData.size() / gParams.batchSize << " batches for testing..." << std::endl;
    int batchCount = 0;
    while (dataIter != testData.end())
    {
        int n = 0;
        std::cout << "Doing inference on batch " << batchCount << std::endl;
        std::vector<std::string> jpegNames;
        // gather image names
        while ((n < gParams.batchSize) && (dataIter != testData.end()))
        {
            std::string filename = gParams.testFolder + "/" + (*dataIter).file_name;
            images[n] = (*dataIter);
            n++;
            dataIter++;
            jpegNames.push_back(filename);
        }
        // allocate output data
        std::vector<float> detectionOut(gOutputDimensions[gParams.outputBlob].d[1] * gOutputDimensions[gParams.outputBlob].d[2] * gParams.batchSize, 0);
        std::vector<int> keepCount(gParams.batchSize, 0);
        // do inference
        DALIStream::doInference(detectionOut, keepCount, gParams.NCHWDims, jpegNames, inferencePipe, inferenceWS, gParams.asyncDali);
        // postprocessing
        postprocess(detectionOut.data(), keepCount.data(), images, gParams.batchSize, output);
        batchCount++;
    }

    json joutput(output);
    if (gParams.evalJSON.empty())
    {
        std::cout << "--evalJSON output file is not specified, printing the result to the console."
                  << std::endl;
        std::cout << std::setw(4) << joutput << std::endl;
    }
    else
    {
        // Save the output prediction in COCO JSON format
        std::ofstream o(gParams.evalJSON);
        if (!o.is_open())
        {
            std::cerr << "Could not open " << gParams.evalJSON << " for writing the result"
                      << std::endl;
            return -1;
        }
        else
        {
            o << std::setw(4) << joutput << std::endl;
            o.close();
            std::cout << "Result saved to " << gParams.evalJSON << std::endl;
        }
        o.close();
    }

    if (gParams.needPipe)
    {
        delete gParams.ppreprocessPipe;
    }

    engine->destroy();
    return EXIT_SUCCESS;
}
