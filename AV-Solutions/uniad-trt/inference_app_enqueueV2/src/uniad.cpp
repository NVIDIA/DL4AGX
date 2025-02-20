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

#include "uniad.hpp"

namespace UniAD {

KernelImplement::~KernelImplement() {
  // free all allocated device mem and host mem
  for (void* _ptr : inputs_device_) {
    if (_ptr) checkRuntime(cudaFree(_ptr));
  }
  for (void* _ptr : outputs_device_) {
    if (_ptr) checkRuntime(cudaFree(_ptr));
  }
  for (void* _ptr : inputs_host_) {
    if (_ptr) checkRuntime(cudaFreeHost(_ptr));
  }
  for (void* _ptr : outputs_host_) {
    if (_ptr) checkRuntime(cudaFreeHost(_ptr));
  }
}
int KernelImplement::init(const KernelParams& param) {
  param_ = param;
  // deserialization trt engine
  engine_ = TensorRT::load(param_.trt_engine);
  if (engine_ == nullptr) {
    printf("[ERROR] Can not load TRT engine at %s.\n", param.trt_engine.c_str());
    return -1;
  }
  // malloc the device and host memory, need to be free in deconstructor
  int _shape_product;
  size_t _dsize;
  for (int ib=0; ib<engine_->num_bindings(); ++ib) {
    void *_ptr_device=nullptr;
    void *_ptr_host=nullptr;
    std::vector<int> _shape;
    const std::string bindingName = engine_->get_binding_name(ib);
    if (engine_->is_input(ib)) {
      ++param_.num_inputs;
      // prepare input
      _shape = param_.input_max_shapes[bindingName];
      _shape_product = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<int>());
      _dsize = sizeof(engine_->dtype(bindingName));
      checkRuntime(cudaMalloc(&_ptr_device, _shape_product * _dsize));
      checkRuntime(cudaMallocHost(&_ptr_host, _shape_product * _dsize));
      inputs_device_.push_back(_ptr_device);
      inputs_host_.push_back(_ptr_host);
    } else {
      // prepare output
      _shape = param_.output_max_shapes[bindingName];
      _shape_product = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<int>());
      _dsize = sizeof(engine_->dtype(bindingName));
      checkRuntime(cudaMalloc(&_ptr_device, _shape_product * _dsize));
      checkRuntime(cudaMallocHost(&_ptr_host, _shape_product * _dsize));
      outputs_device_.push_back(_ptr_device);
      outputs_host_.push_back(_ptr_host);
    }
    bindings_.push_back(_ptr_device);
  }
  return 0;
}
void KernelImplement::forward_timer(const UniAD::KernelInput& inputs, UniAD::KernelOutput& outputs, void *stream, bool enable_timer) {
  cudaStream_t _stream = static_cast<cudaStream_t>(stream);
  std::vector<float> times;
  int _shape_product;
  size_t _dsize;
  // copy the cuda memory from host to device
  for (int ib=0; ib<engine_->num_bindings(); ++ib) {
    if (engine_->is_input(ib)) {
      // prepare input
      const std::string bindingName = engine_->get_binding_name(ib);
      std::vector<int> _shape = inputs.input_shapes.at(bindingName);
      _shape_product = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<int>());
      _dsize = sizeof(engine_->dtype(bindingName));
      checkRuntime(cudaMemcpyAsync(inputs_host_[ib], inputs.data_ptrs.at(bindingName), _shape_product * _dsize, cudaMemcpyHostToHost, _stream));
      checkRuntime(cudaMemcpyAsync(inputs_device_[ib], inputs_host_[ib], _shape_product * _dsize, cudaMemcpyHostToDevice, _stream));
      // set dynamic shape
      engine_->set_run_dims(bindingName, _shape);
    }
  }
  if (enable_timer) timer_.start(_stream);
  engine_->forward(bindings_, _stream);
  if (enable_timer) times.emplace_back(timer_.stop("Inference"));
  // copy the output back to host and post process
  for (int ib=0; ib<engine_->num_bindings(); ++ib) {
    if (!engine_->is_input(ib)) {
      // load output
      const std::string bindingName = engine_->get_binding_name(ib);
      std::vector<int> _shape = engine_->run_dims(bindingName);
      outputs.output_shapes[bindingName] = _shape;
      _shape_product = std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<int>());
      if (outputs.output_dsizes.find(bindingName) != outputs.output_dsizes.cend()) _dsize = outputs.output_dsizes[bindingName];
      else {
        _dsize = sizeof(engine_->dtype(bindingName));
        outputs.output_dsizes[bindingName] = _dsize;
      }
      
      if (_shape_product > 0){
        checkRuntime(cudaMemcpyAsync(outputs_host_[ib-param_.num_inputs], outputs_device_[ib-param_.num_inputs], _shape_product * _dsize, cudaMemcpyDeviceToHost, _stream));
        checkRuntime(cudaMemcpyAsync(outputs.data_ptrs[bindingName], outputs_host_[ib-param_.num_inputs], _shape_product * _dsize, cudaMemcpyHostToHost, _stream));
      }
    }
  }
  return;
}
void KernelImplement::forward_one_frame(const UniAD::KernelInput& inputs, UniAD::KernelOutput& outputs, bool enable_timer, void *stream) {
  this->forward_timer(inputs, outputs, stream, enable_timer);
  return;
}
void KernelImplement::print_info() {
  printf("UniAD TRT engine.\n");
  engine_->print("UniAD");
}
}; // namespace UniAD