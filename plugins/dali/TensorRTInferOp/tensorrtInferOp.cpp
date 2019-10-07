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
 * File: DL4AGX/plugins/dali/TensorRTInferOp/tensorrtInferOp.cpp
 * Description: Implementation of TensorRT Inference Op for DALI
 *************************************************************************/
#include "tensorrtInferOp.h"
#include <fstream>
#include <sstream>
#include <string>

DALI_SCHEMA(TensorRTInfer)
    .DocStr(R"code(Perform inference over the TensorRT engine.)code")
    .NumInput(1, ::plugin::MAX_ALLOWED_TRT_OP_INPUT)
    .AddOptionalArg("num_outputs",
                    R"code(Number of outputs)code", 1)
    .AddArg("engine",
            R"code(TensorRT engine file to run inference)code", ::dali::DALI_STRING, false)
    .AddArg("input_nodes",
            R"code(Input nodes in the engine)code", ::dali::DALI_STRING_VEC, false)
    .AddArg("output_nodes",
            R"code(Output nodes in the engine)code", ::dali::DALI_STRING_VEC, false)
    .AddOptionalArg("plugins",
                    R"code(Plugin library to load)code",
                    std::vector<std::string>({""}), false)
    .AddOptionalArg("inference_batch_size",
                    R"code(Batch size to run inference)code", 1, false)
    .AddOptionalArg("use_dla_core",
                    R"code(DLA core to run inference upon)code", ::plugin::DEFAULT_DLA_CORE_ID, false)
    .AddOptionalArg("log_severity",
                    R"code(Logging severity for TensorRT)code", (int) nvinfer1::ILogger::Severity::kWARNING, false)
    .OutputFn([](const ::dali::OpSpec spec) -> int {
        return spec.GetArgument<int>("num_outputs");
    });

DALI_REGISTER_OPERATOR(TensorRTInfer, ::plugin::TensorRTInfer<::dali::GPUBackend>, ::dali::GPU);

namespace plugin
{
template <>
//void TensorRTInfer<::dali::GPUBackend>::RunImpl(::dali::DeviceWorkspace* ws, const int idx)
void TensorRTInfer<::dali::GPUBackend>::RunImpl(::dali::DeviceWorkspace& ws)
{
    int num_bindings = engine_->getNbBindings();
    std::string blob_name;
    std::vector<const void*> input_buffers(input_blobs_.size());
    // Input output buffer
    std::vector<void*> io_buffers(num_bindings);
    // Copy the address of data for each input bindings from engine
    for (size_t i = 0; i < input_blobs_.size(); i++)
    {
        // Assign the first tensor address as the binding address to run for >1 batch size
        input_buffers[binding_param_[input_blobs_[i]].binding_index] = ws.Input<::dali::GPUBackend>(i).raw_tensor(0);
    }
    // Copy the input bindings for TensorRT
    std::memcpy(io_buffers.data(), input_buffers.data(), sizeof(void*) * input_blobs_.size());
    for (size_t i = 0; i < output_blobs_.size(); i++)
    {
        blob_name = output_blobs_[i];
        // Output tensorlist freom DALI pipeline
        auto& output = ws.Output<::dali::GPUBackend>(i);
        // Allocate memory for output tensorrlist and set datatype
        output.Resize(binding_param_[blob_name].dali_dimensions);
        ::dali::TypeInfo type;
        switch (binding_param_[blob_name].data_type)
        {
        case ::dali::DALI_FLOAT16:
            type = ::dali::TypeInfo::Create<::dali::float16>();
            break;
        case ::dali::DALI_INT32:
            type = ::dali::TypeInfo::Create<::dali::int32>();
            break;
        default:
            type = ::dali::TypeInfo::Create<float>();
        }
        output.set_type(type);
        // Binding address points to the address of the output binding batch
        io_buffers[binding_param_[blob_name].binding_index] = output.raw_mutable_tensor(0);
    }
    // Run inference
    context_->enqueue(trt_batch_size_, io_buffers.data(), ws.stream(), nullptr);
}
}
