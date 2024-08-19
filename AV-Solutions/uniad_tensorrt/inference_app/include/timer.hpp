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

#ifndef __TIMER_HPP__
#define __TIMER_HPP__
#include "check.hpp"

namespace nv {

class EventTimer {
 public:
  EventTimer() {
    checkRuntime(cudaEventCreate(&begin_));
    checkRuntime(cudaEventCreate(&end_));
  }

  virtual ~EventTimer() {
    checkRuntime(cudaEventDestroy(begin_));
    checkRuntime(cudaEventDestroy(end_));
  }

  void start(cudaStream_t stream) { checkRuntime(cudaEventRecord(begin_, stream)); }

  float stop(const std::string& prefix = "timer") {
    float times = 0;
    checkRuntime(cudaEventRecord(end_, stream_));
    checkRuntime(cudaEventSynchronize(end_));
    checkRuntime(cudaEventElapsedTime(&times, begin_, end_));
    printf("[timer:  %s]: \t%.5f ms\n", prefix.c_str(), times);
    return times;
  }

 private:
  cudaStream_t stream_ = nullptr;
  cudaEvent_t begin_ = nullptr, end_ = nullptr;
};

};  // namespace nv

#endif  // __TIMER_HPP__