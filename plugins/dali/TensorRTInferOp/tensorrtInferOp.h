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
 * File: DL4AGX/plugins/dali/TensorRTInferOp/tensorrtInferOp.h
 * Description: Interface of TensorRT Inference Op for DALI
 *************************************************************************/
#pragma once

#include "dali/pipeline/operators/operator.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParserRuntime.h>
#include <cstring>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <map>
#include <utility>
#include <vector>
namespace plugin
{

const int MAX_ALLOWED_TRT_OP_OUTPUT = 16;
const int MAX_ALLOWED_TRT_OP_INPUT = 16;
const int MAX_ALLOWED_DLA_CORE = 2;
const int DEFAULT_DLA_CORE_ID = -1;

typedef struct
{
    // Binding Index
    int binding_index;
    // DataType for each bindings
    ::dali::DALIDataType data_type;
    // Shape required for allocating memory to output tensorlist
    std::vector<::dali::Dims> dali_dimensions;
} BindingParam;

class Logger : public nvinfer1::ILogger
{
public:
    explicit inline Logger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
        : reportableSeverity(severity)
    {
    }

    void log(nvinfer1::ILogger::Severity severity, const char* msg) override
    {
        // suppress messages with severity enum value greater than the reportable
        if (severity > reportableSeverity)
        {
            return;
        }

        switch (severity)
        {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
    nvinfer1::ILogger::Severity reportableSeverity;
};

template <typename Backend>
class TensorRTInfer : public ::dali::Operator<Backend>
{
public:
    explicit inline TensorRTInfer(const ::dali::OpSpec& spec)
        : ::dali::Operator<Backend>(spec)
        , input_blobs_(spec.GetRepeatedArgument<std::string>("input_nodes"))
        , output_blobs_(spec.GetRepeatedArgument<std::string>("output_nodes"))
        , plugins_(spec.GetRepeatedArgument<std::string>("plugins"))
        , trt_batch_size_(spec.GetArgument<::dali::int64>("inference_batch_size"))
        , num_outputs_(spec.GetArgument<int>("num_outputs"))
        , engine_data_(spec.GetArgument<std::string>("engine"))
        , use_dla_core_(spec.GetArgument<int>("use_dla_core"))
    {
        DALI_ENFORCE(use_dla_core_ < MAX_ALLOWED_DLA_CORE,
                     "DLA core id should be less than " + std::to_string(MAX_ALLOWED_DLA_CORE));
        DALI_ENFORCE(input_blobs_.size() > 0, "Input blob is missing");
        DALI_ENFORCE(output_blobs_.size() > 0, "Output blob is missing");
        DALI_ENFORCE(num_outputs_ < MAX_ALLOWED_TRT_OP_OUTPUT,
                     "Number of outputs from op should be less than "
                         + std::to_string(MAX_ALLOWED_TRT_OP_OUTPUT));
        for (auto& s : plugins_)
        {
            void* dlh = dlopen(s.c_str(), RTLD_LAZY);
            DALI_ENFORCE(nullptr != dlh,
                         "Error while performing dlopen on " + s + " library");
        }
        this->trt_logger_ = new Logger((nvinfer1::ILogger::Severity) spec.GetArgument<int>("log_severity"));

        initLibNvInferPlugins(this->trt_logger_, "");
        runtime_ = nvinfer1::createInferRuntime(*trt_logger_);
        if (use_dla_core_ >= 0 && use_dla_core_ < MAX_ALLOWED_DLA_CORE)
        {
            runtime_->setDLACore(use_dla_core_);
        }

        engine_ = runtime_->deserializeCudaEngine(engine_data_.data(), engine_data_.size(), nullptr);
        context_ = engine_->createExecutionContext();

        DALI_ENFORCE((static_cast<size_t>(engine_->getNbBindings()) == (input_blobs_.size() + output_blobs_.size())),
                     "Number of bindings mismatch with the engine");
        DALI_ENFORCE(trt_batch_size_ <= engine_->getMaxBatchSize(),
                     "Batch size is greater than MaxBatch size configured for the engine");

        for (size_t i = 0; i < input_blobs_.size(); i++)
        {
            std::string blob_name = input_blobs_[i];
            BindingParam param;
            param.binding_index = engine_->getBindingIndex(blob_name.c_str());
            binding_param_[blob_name] = param;
        }

        for (size_t i = 0; i < output_blobs_.size(); i++)
        {
            std::string blob_name = output_blobs_[i];
            BindingParam param;
            // Retrieve binding index from engine
            param.binding_index = engine_->getBindingIndex(blob_name.c_str());

            // Retrieve binding Dimensions from engine
            nvinfer1::Dims dimensions = static_cast<nvinfer1::Dims&&>(engine_->getBindingDimensions(param.binding_index));
            std::vector<::dali::Index> dim;
            dim.push_back(trt_batch_size_);
            for (int j = 0; j < dimensions.nbDims; j++)
            {
                dim.push_back(dimensions.d[j]);
            }
            param.dali_dimensions.push_back(dim);
            nvinfer1::DataType datatype = engine_->getBindingDataType(param.binding_index);
            param.data_type = (datatype == nvinfer1::DataType::kHALF) ? ::dali::DALI_FLOAT16 : (datatype == nvinfer1::DataType::kINT32) ? ::dali::DALI_INT32 : ::dali::DALI_FLOAT;
            binding_param_[blob_name] = param;
        }
    }

    ~TensorRTInfer()
    {
        if (engine_)
        {
            engine_->destroy();
            engine_ = nullptr;
        }
        if (runtime_)
        {
            runtime_->destroy();
            runtime_ = nullptr;
        }
        if (context_)
        {
            context_->destroy();
            context_ = nullptr;
        }
    }

protected:
    void RunImpl(::dali::Workspace<Backend>* ws, const int idx) override;

private:
    std::vector<std::string> input_blobs_;
    std::vector<std::string> output_blobs_;
    std::vector<std::string> plugins_;

    int trt_batch_size_;
    int num_outputs_;

    std::string engine_data_;
    nvinfer1::IRuntime* runtime_;
    nvinfer1::ICudaEngine* engine_;
    nvinfer1::IExecutionContext* context_;
    std::map<std::string, BindingParam> binding_param_;
    Logger* trt_logger_;
    int use_dla_core_;
};
} // namespace plugin
