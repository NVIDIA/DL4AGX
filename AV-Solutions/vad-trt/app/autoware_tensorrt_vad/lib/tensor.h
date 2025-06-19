/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <memory>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

namespace nv {

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(float) / 2;
    case nvinfer1::DataType::kINT8:
      return sizeof(int8_t);
    case nvinfer1::DataType::kINT32:
      return sizeof(int32_t);
    case nvinfer1::DataType::kBOOL:
      return sizeof(bool);
    case nvinfer1::DataType::kUINT8:
      return sizeof(uint8_t);
    case nvinfer1::DataType::kFP8:
      return sizeof(float) / 4;
    default:
      return 0;
  }
}

struct Tensor {
  std::string name;
  void* ptr;
  nvinfer1::Dims dim;
  int32_t volume = 1;
  nvinfer1::DataType dtype;
  // nvinfer1::TensorIOMode iomode;

  Tensor(std::string name, nvinfer1::Dims dim, nvinfer1::DataType dtype): 
    name(name), dim(dim), dtype(dtype) 
  {
    if( dim.nbDims == 0 ) {
      volume = 0;
    } else {
      volume = 1;
      for(int i=0; i<dim.nbDims; i++) {
          volume *= dim.d[i];
      }
    }
    cudaMalloc(&ptr, volume * getElementSize(dtype));
  }

  int32_t nbytes() {
    return volume * getElementSize(dtype);
  }

  template<class Dtype=float>
  void load(const std::vector<float>& data, cudaStream_t stream = 0) {
    if (static_cast<int32_t>(data.size()) != volume) {
      std::cerr << "Data size mismatch: expected " << volume << ", got " << data.size() << std::endl;
      return;
    }
    
    size_t dsize = volume * getElementSize(dtype);
    
    if (dtype == nvinfer1::DataType::kFLOAT) {
      // Direct copy for float data
      cudaMemcpyAsync(ptr, data.data(), dsize, cudaMemcpyHostToDevice, stream);
    } else {
      // Type conversion needed
      std::vector<char> buffer(dsize);
      Dtype* dbuffer = reinterpret_cast<Dtype*>(buffer.data());
      
      for (int i = 0; i < volume; i++) {
        dbuffer[i] = static_cast<Dtype>(data[i]);
      }
      
      cudaMemcpyAsync(ptr, buffer.data(), dsize, cudaMemcpyHostToDevice, stream);
    }
  }

  template<class T>
  std::vector<T> cpu() {
    std::vector<T> buffer(volume);
    cudaMemcpy(buffer.data(), ptr, volume * sizeof(T), cudaMemcpyDeviceToHost);
    return buffer;
  }

  std::vector<char> load_ref(std::string fname) {
    size_t bsize = volume * sizeof(float);
    std::vector<char> buffer(bsize);
    std::ifstream file_(fname, std::ios::binary);
    file_.read(buffer.data(), bsize);
    return buffer;
  }
}; // struct Tensor

using TensorMap = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

} // namespace nv

std::ostream& operator<<(std::ostream& os, nv::Tensor& t);

#endif // _TENSOR_H_
