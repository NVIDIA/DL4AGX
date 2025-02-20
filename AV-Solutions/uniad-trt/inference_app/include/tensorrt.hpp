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

#ifndef __TENSORRT_HPP__
#define __TENSORRT_HPP__

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>
#include <string.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include "NvInfer.h"
#include "timer.hpp"
#include "NvInferRuntime.h"
#include <NvInferRuntimeBase.h>

typedef std::conditional<NV_TENSORRT_MAJOR<10, int32_t, int64_t>::type TRT_INT_TYPE;

namespace TensorRT {

enum class DType : int { FLOAT = 0, HALF = 1, INT8 = 2, INT32 = 3, BOOL = 4, UINT8 = 5, FP8 = 6, BF16 = 7, INT64 = 8, INT4 = 9, NONE=-1 };

class DDSOutputAllocator : public nvinfer1::IOutputAllocator {
 public:
  DDSOutputAllocator(
    std::string const& tensorName, void* currentMemory
  );
  ~DDSOutputAllocator() override;
  void* reallocateOutput(
    char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment
  ) noexcept override;
  void* reallocateOutputAsync(
    char const* tensorName, void* currentMemory, uint64_t size, uint64_t alignment, cudaStream_t stream
  ) noexcept override;
  void notifyShape(char const* tensorName, nvinfer1::Dims const& dims) noexcept override;
  bool reallocateOutputCalled{false};
  bool notifyShapeCalled{false};
  nvinfer1::Dims outputDims{-1, {}};
 private:
  void* outputPtr{};
  uint64_t outputSize{0};
  std::string const mTensorName;
  void* mCurrentMemory{};
}; // class DDSOutputAllocator  

class Engine {
 public:
  virtual ~Engine() = default;
  virtual bool forward(
    const std::unordered_map<std::string, const void *> &bindings,
    std::unordered_map<std::string, std::vector<TRT_INT_TYPE>> &DDSOutputShapes,
    void *stream,
    bool enable_timer,
    nv::EventTimer &timer_
  ) = 0;
  virtual int index(const std::string &name) = 0;
  virtual std::vector<TRT_INT_TYPE> run_dims(const std::string &name) = 0;
  virtual std::vector<TRT_INT_TYPE> run_dims(int ibinding) = 0;
  virtual std::vector<TRT_INT_TYPE> static_dims(const std::string &name) = 0;
  virtual std::vector<TRT_INT_TYPE> static_dims(int ibinding) = 0;
  virtual int numel(const std::string &name) = 0;
  virtual int numel(int ibinding) = 0;
  virtual int num_bindings() = 0;
  virtual bool is_input(int ibinding) = 0;
  virtual bool is_input(const std::string &name) = 0;
  virtual std::string get_binding_name(int ibinding) = 0;
  virtual bool set_run_dims(const std::string &name, const std::vector<TRT_INT_TYPE> &dims) = 0;
  virtual bool set_run_dims(int ibinding, const std::vector<TRT_INT_TYPE> &dims) = 0;
  virtual DType dtype(const std::string &name) = 0;
  virtual DType dtype(int ibinding) = 0;
  virtual bool has_dynamic_dim() = 0;
  virtual bool is_dynamic_dim(nvinfer1::Dims dims) = 0;
  virtual void print(const char *name = "TensorRT-Engine") = 0;
};

std::shared_ptr<Engine> load(const std::string &file);
};  // namespace TensorRT

#endif  // __TENSORRT_HPP__