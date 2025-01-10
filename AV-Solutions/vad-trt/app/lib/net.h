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

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <unordered_map>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include "tensor.h"

namespace nv {
using namespace nvinfer1;

struct Net {
  ICudaEngine* engine;
  IExecutionContext* context;
  nv::TensorMap bindings;
  bool use_cuda_graph = false;
  cudaGraph_t graph;
  cudaGraphExec_t graph_exec;

  Net(
    std::string engine_path, 
    IRuntime* runtime, 
    nv::TensorMap& ext
  ) {
    std::ifstream engine_file(engine_path, std::ios::binary);
    if (!engine_file) {
      throw std::runtime_error("Error opening engine file: " + engine_path);
    }
    engine_file.seekg(0, engine_file.end);
    long int fsize = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);

    // Read the engine file into a buffer
    std::vector<char> engineData(fsize);

    engine_file.read(engineData.data(), fsize);
    engine = runtime->deserializeCudaEngine(engineData.data(), fsize);
    context = engine->createExecutionContext(); 

    int nb = engine->getNbIOTensors();  

    for( int n=0; n<nb; n++ ) {
      std::string name = engine->getIOTensorName(n);
      Dims d = engine->getTensorShape(name.c_str());      
      DataType dtype = engine->getTensorDataType(name.c_str());
      if( ext.find(name) != ext.end() ) {
        // use external memory
        context->setTensorAddress(name.c_str(), ext[name]->ptr);
      } else {
        bindings[name] = std::make_shared<Tensor>(name, d, dtype);
        // bindings[name]->iomode = engine->getTensorIOMode(name.c_str());
        std::cout << *(bindings[name]) << std::endl;
        context->setTensorAddress(name.c_str(), bindings[name]->ptr);
      }      
    }
  }

  void Enqueue(cudaStream_t stream) {
    if( this->use_cuda_graph ) {
      cudaGraphLaunch(graph_exec, stream);
    } else {
      context->enqueueV3(stream);
    }  
  }

  ~Net() {
  }

  void EnableCudaGraph(cudaStream_t stream) {    
    // run first time to avoid allocation
    this->Enqueue(stream);
    cudaStreamSynchronize(stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    this->Enqueue(stream);
    cudaStreamEndCapture(stream, &graph);
    this->use_cuda_graph = true;
#if CUDART_VERSION < 12000
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
#else
    cudaGraphInstantiate(&graph_exec, graph, 0);
#endif
  }
}; // class Net

void validate(Net& net, std::string frame_dir, std::string key) {
  auto ref = net.bindings[key]->load_ref(frame_dir + key + ".bin");
  float* f_ref = reinterpret_cast<float*>(ref.data());
  auto fresult = net.bindings[key]->cpu<float>();
  std::cout << key << std::endl;
  for(int n=0; n<16; n++ ) {
      printf("%6.4f ", fresult[n]);
  }
  printf("\n");
  for(int n=0; n<16; n++ ) {
      printf("%6.4f ", f_ref[n]);
  }
  printf("\n");
}

} // namespace nv
