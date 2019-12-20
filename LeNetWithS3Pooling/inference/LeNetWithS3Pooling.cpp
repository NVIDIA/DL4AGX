/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "common/argparser.h"
#include "common/tensorrt/buffers.h"
#include "common/tensorrt/utils.h"
#include "common/tensorrt/int8.h"
#include "common/common.h"
#include "common/tensorrt/Logger.h"

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <cuda_runtime_api.h>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

const std::string gSampleName = "TensorRT.sample_onnx_mnist";
std::string inputModel;
std::string inputTensorName;
std::string outputTensorName;

common::tensorrt::Logger gLogger = common::tensorrt::Logger(nvinfer1::ILogger::Severity::kINFO);

struct Params
{
    int batchSize{1};                  //!< Number of inputs in a batch
    int dlaCore{-1};                   //!< Specify the DLA core to run network on.
    bool int8{false};                  //!< Allow runnning the network in Int8 mode.
    bool fp16{false};                  //!< Allow running the network in FP16 mode.
    std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string onnxFileName;
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

//! \brief  The SampleOnnxMNIST class implements the ONNX MNIST sample
//!
//! \details It creates the network using an ONNX model
//!
class LeNetWithS3Pooling
{
    template <typename T>
    using SampleUniquePtr = std::unique_ptr<T, InferDeleter>;

public:
    LeNetWithS3Pooling(const Params& params)
        : mParams(params)
        , mEngine(nullptr)
    {
    }

    //!
    //! \brief Function builds the network engine
    //!
    bool build();

    //!
    //! \brief Runs the TensorRT inference engine for this sample
    //!
    bool infer();

private:
    Params mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

    //!
    //! \brief Parses an ONNX model for MNIST and creates a TensorRT network
    //!
    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
        SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
        SampleUniquePtr<nvonnxparser::IParser>& parser);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const common::tensorrt::BufferManager& buffers);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const common::tensorrt::BufferManager& buffers);
};

//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx MNIST network by parsing the Onnx model and builds
//!          the engine that will be used to run MNIST (mEngine)
//!
//! \return Returns true if the engine was created successfully and false otherwise
//!
bool LeNetWithS3Pooling::build()
{
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder)
    {
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);     
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    auto parser = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser)
    {
        return false;
    }

    auto constructed = this->constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    assert(mOutputDims.nbDims == 2);

    return true;
}

//!
//! \brief Uses a ONNX parser to create the Onnx MNIST Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx MNIST network
//!
//! \param builder Pointer to the engine builder
//!
bool LeNetWithS3Pooling::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(
        common::locateFile(mParams.onnxFileName, mParams.dataDirs).c_str(), static_cast<int>(gLogger.reportableSeverity));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mParams.batchSize);
    config->setMaxWorkspaceSize(1 << 28);
    if (mParams.fp16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    if (mParams.int8)
    {
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        common::tensorrt::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    common::tensorrt::enableDLA(builder.get(), config.get(), mParams.dlaCore);

    return true;
}

//!
//! \brief Runs the TensorRT inference engine for this sample
//!
//! \details This function is the main execution function of the sample. It allocates the buffer,
//!          sets inputs and executes the engine.
//!
bool LeNetWithS3Pooling::infer()
{
    // Create RAII buffer manager object
    common::tensorrt::BufferManager buffers(mEngine, mParams.batchSize);

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Read the input data into the managed buffers
    assert(mParams.inputTensorNames.size() == 1);
    if (!processInput(buffers))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!
bool LeNetWithS3Pooling::processInput(const common::tensorrt::BufferManager& buffers)
{
    const int inputH = mInputDims.d[2];
    const int inputW = mInputDims.d[3];

    std::vector<uint8_t> fileData(inputH * inputW);
    mNumber = 2;
    common::image::PGM::readPGMFile(common::locateFile(std::to_string(mNumber) + ".pgm", mParams.dataDirs), fileData.data(), inputH, inputW);

    stringstream in_ss;
    in_ss << "\n";
    // Print an ascii representation
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "Input:");
    for (int i = 0; i < inputH * inputW; i++)
    {
       in_ss << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % inputW) ? "" : "\n");
    }
    in_ss << std::endl;

    gLogger.log(nvinfer1::ILogger::Severity::kINFO, in_ss.str().c_str());
    
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNames[0]));
    for (int i = 0; i < inputH * inputW; i++)
    {
        hostDataBuffer[i] = 1.0 - float(fileData[i] / 255.0);
    }

    return true;
}

//!
//! \brief Classifies digits and verify result
//!
//! \return whether the classification output matches expectations
//!
bool LeNetWithS3Pooling::verifyOutput(const common::tensorrt::BufferManager& buffers)
{
    const int outputSize = 10;
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNames[0]));
    float val{0.0f};
    int idx{0};

    // Calculate Softmax
    float sum{0.0f};
    for (int i = 0; i < outputSize; i++)
    {
        output[i] = exp(output[i]);
        sum += output[i];
    }

    for (int i = 0; i < outputSize; i++)
    {
        output[i] /= sum;
        val = std::max(val, output[i]);
        if (val == output[i])
        {
            idx = i;
        }
    }

    stringstream ss_orig;
    ss_orig  << "Original number: " << mNumber << std::endl;
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, ss_orig.str().c_str());

    stringstream ss_pred;
    ss_pred  << "Predicted number: " << mNumber << std::endl;
    gLogger.log(nvinfer1::ILogger::Severity::kINFO, ss_pred.str().c_str());

    return idx == mNumber;
}

//!
//! \brief Initializes members of the params struct using the command line args
//!
Params initializeParams(const common::argparser::StandardArgs& args)
{
    Params params;
    if (args.dataDirs.empty()) //!< Use default directories if user hasn't provided directory paths
    {
        params.dataDirs.push_back("");
        params.dataDirs.push_back("./");
        params.dataDirs.push_back("data/");
    }
    else //!< Use the data directory provided by the user
    {
        params.dataDirs = args.dataDirs;
    }
    params.onnxFileName = inputModel;//"mnist_plugin.onnx";
    std::cout << " onnx file is " <<  params.onnxFileName << std::endl;
    params.inputTensorNames.push_back(inputTensorName);
    params.batchSize = 1;
    params.outputTensorNames.push_back(outputTensorName);
    params.dlaCore = args.useDLACore;
    params.int8 = args.runInInt8;
    params.fp16 = args.runInFp16;

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./sample_onnx_mnist <onnx_model_file> <input_tensor_name> <output_tensor_name> <Optional Params>"
        << std::endl;
    std::cout <<"\nOptional Params: \n" << std:: endl;
    std::cout << "--help          Display help information" << std::endl;
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used "
                 "multiple times to add multiple directories. If no data directories are given, the default is to use "
                 "(data/samples/mnist/, data/mnist/)"
              << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, "
                 "where n is the number of DLA engines on the platform."
              << std::endl;
    std::cout << "--int8          Run in Int8 mode." << std::endl;
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}

int main(int argc, char** argv)
{
    common::argparser::StandardArgs args;
    if(argc < 4)
    {
        printHelpInfo();
        return -1;
    }	
    inputModel = argv[1];
    inputTensorName = argv[2];
    outputTensorName = argv[3];
    
    bool argsOK = common::argparser::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Invalid arguments");
        printHelpInfo();
        return -1;
    }
    if (args.help)
    {
        printHelpInfo();
        return 0;
    }

    LeNetWithS3Pooling sample(initializeParams(args));

    gLogger.log(nvinfer1::ILogger::Severity::kINFO, "Building and running a GPU inference engine for LeNet with S3Pooling");

    if (!sample.build())
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Engine failed to build");
        return -1;
    }
    if (!sample.infer())
    {
        gLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to Run Inference");
        return -1;
    }

    return 0;
}
