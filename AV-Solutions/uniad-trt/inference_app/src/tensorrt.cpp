/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt.hpp"
#include "check.hpp"

namespace TensorRT {

static class Logger : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR) {
      std::cerr << "[NVINFER ERROR]: " << msg << std::endl;
    }
  }
} gLogger_;

static std::string format_shape(const nvinfer1::Dims &shape) {
  char buf[200] = {0};
  char *p = buf;
  for (int i = 0; i < shape.nbDims; ++i) {
    if (i + 1 < shape.nbDims)
      p += sprintf(p, "%d x ", (int)shape.d[i]);
    else
      p += sprintf(p, "%d", (int)shape.d[i]);
  }
  return buf;
}

static std::vector<uint8_t> load_file(const std::string &file) {
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) return {};

  in.seekg(0, std::ios::end);
  size_t length = in.tellg();

  std::vector<uint8_t> data;
  if (length > 0) {
    in.seekg(0, std::ios::beg);
    data.resize(length);

    in.read((char *)&data[0], length);
  }
  in.close();
  return data;
}

static const char *data_type_string(nvinfer1::DataType dt) {
  switch (dt) {
    case nvinfer1::DataType::kFLOAT:
      return "float32";
    case nvinfer1::DataType::kHALF:
      return "float16";
    case nvinfer1::DataType::kINT8:
      return "int8";
    case nvinfer1::DataType::kINT32: 
      return "int32";
    case nvinfer1::DataType::kBOOL: 
      return "bool";
    case nvinfer1::DataType::kUINT8: 
      return "uint8";
#if NV_TENSORRT_MAJOR >= 10
    case nvinfer1::DataType::kFP8: 
      return "fp8";
    case nvinfer1::DataType::kBF16: 
      return "bf16";
    case nvinfer1::DataType::kINT64: 
      return "int64";
    case nvinfer1::DataType::kINT4:
      return "int4";
#endif
    default:
      return "Unknow";
  }
}

template <typename _T>
static void destroy_pointer(_T *ptr) {
  if (ptr) delete ptr;
}

DDSOutputAllocator::DDSOutputAllocator(
  std::string const& tensorName, void* currentMemory
) : mTensorName(tensorName)
  , mCurrentMemory(currentMemory) {}

DDSOutputAllocator::~DDSOutputAllocator() {}

void* DDSOutputAllocator::reallocateOutput(
  char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment) noexcept
{
  Assertf(currentMemory == mCurrentMemory, "Memory address not matched: %s\n", tensorName);
  Assertf(tensorName == mTensorName, "Tensor name not matched: %s\n", tensorName);
  Assertf(!reallocateOutputCalled, "Duplicate call to reallocateOutput: %s\n", tensorName);
  reallocateOutputCalled = true;
  outputPtr = mCurrentMemory;
  return outputPtr;
}

void* DDSOutputAllocator::reallocateOutputAsync(
  char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream) noexcept {
  Assertf(currentMemory == mCurrentMemory, "Memory address not matched: %s\n", tensorName);
  Assertf(tensorName == mTensorName, "Tensor name not matched: %s\n", tensorName);
  Assertf(!reallocateOutputCalled, "Duplicate call to reallocateOutput: %s\n", tensorName);
  reallocateOutputCalled = true;
  outputPtr = currentMemory;
  return outputPtr;
}

void DDSOutputAllocator::notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept {
  Assertf(reallocateOutputCalled, "TensorRT must invoke reallocateOutput first: %s\n", tensorName);
  Assertf(tensorName == mTensorName, "Tensor name not matched: %s\n", tensorName);
  Assertf(!notifyShapeCalled, "Duplicate call to notifyShape: %s\n", tensorName);
  notifyShapeCalled = true;
  outputDims = dims;
}

class __native_engine_context {
 public:
  virtual ~__native_engine_context() { destroy(); }

  bool construct(const void *pdata, size_t size, const char *message_name) {
    destroy();

    if (pdata == nullptr || size == 0) {
      printf("Construct for empty data found.\n");
      return false;
    }

    runtime_ = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger_), destroy_pointer<nvinfer1::IRuntime>);
    if (runtime_ == nullptr) {
      printf("Failed to create tensorRT runtime: %s.\n", message_name);
      return false;
    }

    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(pdata, size),
                                                     destroy_pointer<nvinfer1::ICudaEngine>);
    if (engine_ == nullptr) {
      printf("Failed to deserialize engine: %s\n", message_name);
      return false;
    }

    context_ = std::shared_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext(),
                                                            destroy_pointer<nvinfer1::IExecutionContext>);
    if (context_ == nullptr) {
      printf("Failed to create execution context: %s\n", message_name);
      return false;
    }
    return context_ != nullptr;
  }

 private:
  void destroy() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
  }

 public:
  std::shared_ptr<nvinfer1::IExecutionContext> context_;
  std::shared_ptr<nvinfer1::ICudaEngine> engine_;
  std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
};

class EngineImplement : public Engine {
 public:
  std::shared_ptr<__native_engine_context> context_;
  std::unordered_map<std::string, int> binding_name_to_index_;

  virtual ~EngineImplement() = default;

  bool construct(const void *data, size_t size, const char *message_name) {
    context_ = std::make_shared<__native_engine_context>();
    if (!context_->construct(data, size, message_name)) {
      return false;
    }

    setup();
    return true;
  }

  bool load(const std::string &file) {
    auto data = load_file(file);
    if (data.empty()) {
      printf("An empty file has been loaded. Please confirm your file path: %s\n", file.c_str());
      return false;
    }
    return this->construct(data.data(), data.size(), file.c_str());
  }

  void setup() {
    auto engine = this->context_->engine_;
    int nbBindings = engine->getNbIOTensors();

    binding_name_to_index_.clear();
    for (int i = 0; i < nbBindings; ++i) {
      const char *bindingName = engine->getIOTensorName(i);
      binding_name_to_index_[bindingName] = i;
    }
  }

  virtual int index(const std::string &name) override {
    auto iter = binding_name_to_index_.find(name);
    Assertf(iter != binding_name_to_index_.end(), "Can not found the binding name: %s", name.c_str());
    return iter->second;
  }

  virtual bool forward(
    const std::unordered_map<std::string, const void *> &bindings,
    std::unordered_map<std::string, std::vector<TRT_INT_TYPE>> &DDSOutputShapes,
    void *stream,
    bool enable_timer,
    nv::EventTimer &timer_
  ) override {
    cudaStream_t _stream = static_cast<cudaStream_t>(stream);
    auto engine = this->context_->engine_;
    auto context = this->context_->context_;
    std::unordered_map<char const *, std::unique_ptr<DDSOutputAllocator>> allocatorMap;
    for(int ibinding = 0; ibinding < engine->getNbIOTensors(); ++ibinding){
      auto tensor_name = engine->getIOTensorName(ibinding);
      auto binding_iter = bindings.find(tensor_name);
      if(binding_iter == bindings.end()){
        printf("Failed to set the tensor address, can not found tensor %s in bindings provided.", tensor_name);
        return false;
      }
      if(!context->setTensorAddress(tensor_name, (void*)binding_iter->second)){
        printf("Failed to set tensor address for tensor %s\n", tensor_name);
        return false;
      }
      if (!this->is_input(tensor_name) && is_dynamic_dim(engine->getTensorShape(tensor_name))){
        allocatorMap[tensor_name] = std::make_unique<DDSOutputAllocator>(
          tensor_name, (void*)binding_iter->second
        );
        context->setOutputAllocator(tensor_name, allocatorMap.at(tensor_name).get());
        Assertf(context->getOutputAllocator(tensor_name) == allocatorMap.at(tensor_name).get(), "Failed to set output allocator: %s\n", tensor_name);
      }
    }
    if (enable_timer) timer_.start(_stream);
    auto enqueueResult = context->enqueueV3(_stream);
    if (enable_timer) timer_.stop("enqueueV3");
    for (auto iter=allocatorMap.begin(); iter!=allocatorMap.end(); ++iter) {
      auto dim = iter->second->outputDims;
      DDSOutputShapes[iter->first] = std::vector<TRT_INT_TYPE>(dim.d, dim.d + dim.nbDims);
    }
    return enqueueResult;
  }

  virtual std::vector<TRT_INT_TYPE> run_dims(const std::string &name) override {
    auto context = this->context_->context_;
    auto dim = context->getTensorShape(name.c_str());
    return std::vector<TRT_INT_TYPE>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<TRT_INT_TYPE> run_dims(int ibinding) override {
    return this->run_dims(this->context_->engine_->getIOTensorName(ibinding));
  }

  virtual std::vector<TRT_INT_TYPE> static_dims(const std::string &name) override {
    auto engine = this->context_->engine_;
    auto dim = engine->getTensorShape(name.c_str());
    return std::vector<TRT_INT_TYPE>(dim.d, dim.d + dim.nbDims);
  }

  virtual std::vector<TRT_INT_TYPE> static_dims(int ibinding) override {
    return this->static_dims(this->context_->engine_->getIOTensorName(ibinding));
  }

  virtual int num_bindings() override { return this->context_->engine_->getNbIOTensors(); }

  virtual std::string get_binding_name(int ibinding) override { return this->context_->engine_->getIOTensorName(ibinding); }

  virtual bool is_input(int ibinding) override { 
    auto engine = this->context_->engine_;
    return engine->getTensorIOMode(engine->getIOTensorName(ibinding)) == nvinfer1::TensorIOMode::kINPUT; 
  }
  virtual bool is_input(const std::string &name) override { 
    auto engine = this->context_->engine_;
    return engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT; 
  }

  virtual bool set_run_dims(const std::string &name, const std::vector<TRT_INT_TYPE> &dims) override {
    nvinfer1::Dims d;
    memcpy(d.d, dims.data(), sizeof(TRT_INT_TYPE) * dims.size());
    d.nbDims = dims.size();
    auto context = this->context_->context_;
    return context->setInputShape(name.c_str(), d);
  }

  virtual bool set_run_dims(int ibinding, const std::vector<TRT_INT_TYPE> &dims) override {
    return this->set_run_dims(this->context_->engine_->getIOTensorName(ibinding), dims);
  }

  virtual int numel(const std::string &name) override { return numel(index(name)); }

  virtual int numel(int ibinding) override {
    auto dim = this->run_dims(ibinding);
    return std::accumulate(dim.begin(), dim.end(), 1, std::multiplies<TRT_INT_TYPE>());
  }

  virtual DType dtype(const std::string &name) override { return dtype(index(name)); }

  virtual DType dtype(int ibinding) override { 
    auto engine = this->context_->engine_;
    auto dtype = engine->getTensorDataType(engine->getIOTensorName(ibinding));
    switch(dtype){
      case nvinfer1::DataType::kFLOAT: return DType::FLOAT;
      case nvinfer1::DataType::kHALF: return DType::HALF;
      case nvinfer1::DataType::kINT8: return DType::INT8;
      case nvinfer1::DataType::kINT32: return DType::INT32;
      case nvinfer1::DataType::kBOOL: return DType::BOOL;
      case nvinfer1::DataType::kUINT8: return DType::UINT8;
#if NV_TENSORRT_MAJOR >= 10
      case nvinfer1::DataType::kFP8: return DType::FP8;
      case nvinfer1::DataType::kBF16: return DType::BF16;
      case nvinfer1::DataType::kINT64: return DType::INT64;
      case nvinfer1::DataType::kINT4: return DType::INT4;
#endif
      default: return DType::NONE;
    }
  }

  virtual bool has_dynamic_dim() override {
    // check if any input or output bindings have dynamic shapes
    // code from ChatGPT
    int numBindings = this->num_bindings();
    for (int i = 0; i < numBindings; ++i) {
      auto dims = this->static_dims(i);
      for (size_t j = 0; j < dims.size(); ++j) {
        if (dims[j] == -1) return true;
      }
    }
    return false;
  }

  virtual bool is_dynamic_dim(nvinfer1::Dims dims) override {
    for (size_t j = 0; j < dims.nbDims; ++j) {
      if (dims.d[j] == -1) return true;
    }
    return false;
  }

  virtual void print(const char *name) override {
    printf("------------------------------------------------------\n");
    printf("%s is %s model\n", name, has_dynamic_dim() ? "Dynamic Shape" : "Static Shape");

    int num_input = 0;
    int num_output = 0;
    auto engine = this->context_->engine_;
    for (int i = 0; i < this->num_bindings(); ++i) {
      if (this->is_input(i))
        num_input++;
      else
        num_output++;
    }

    printf("Inputs: %d\n", num_input);
    for (int i = 0; i < num_input; ++i) {
      auto name = engine->getIOTensorName(i);
      auto dim = engine->getTensorShape(name);
      auto dtype = engine->getTensorDataType(name);
      printf("\t%d.%s : {%s} [%s]\n", i, name, format_shape(dim).c_str(), data_type_string(dtype));
    }

    printf("Outputs: %d\n", num_output);
    for (int i = 0; i < num_output; ++i) {
      auto name = engine->getIOTensorName(i + num_input);
      auto dim = engine->getTensorShape(name);
      auto dtype = engine->getTensorDataType(name);
      printf("\t%d.%s : {%s} [%s]\n", i, name, format_shape(dim).c_str(), data_type_string(dtype));
    }
    printf("------------------------------------------------------\n");
  }
};

std::shared_ptr<Engine> load(const std::string &file) {
  std::shared_ptr<EngineImplement> impl(new EngineImplement());
  if (!impl->load(file)) impl.reset();
  return impl;
}

};  // namespace TensorRT
