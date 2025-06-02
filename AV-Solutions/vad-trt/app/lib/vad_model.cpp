// Copyright 2025 Shin-kyoto.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/tensorrt_vad/vad_trt.hpp"

#include <chrono>
#include <optional>
#include <string>

namespace autoware::tensorrt_vad
{

// VadModelクラスの実装
VadModel::VadModel()
{
}

VadModel::~VadModel()
{
  // TODO(Shin-kyoto): TensorRTリソースのクリーンアップを実装
  // 実際の実装では以下のような処理を行う：
  // if (context_) {
  //   context_->destroy();
  //   context_ = nullptr;
  // }
  // if (engine_) {
  //   engine_->destroy();
  //   engine_ = nullptr;
  // }
  // if (runtime_) {
  //   runtime_->destroy();
  //   runtime_ = nullptr;
  // }
  // if (stream_) {
  //   cudaStreamDestroy(stream_);
  //   stream_ = nullptr;
  // }
}

// std::optional<VadOutputData> VadModel::infer(const VadInputData & input)
// {
//     nets["backbone"]->bindings["img"]->load(input.camera_images_, stream);

//     if (is_first_frame) {
//         nets["head_no_prev"]->bindings["img_metas.0[shift]"]->load(input.shift_, stream);
//         nets["head_no_prev"]->bindings["img_metas.0[lidar2img]"]->load(input.lidar2img_, stream);
//         nets["head_no_prev"]->bindings["img_metas.0[can_bus]"]->load(input.can_bus_, stream);
//         nets["head_no_prev"]->Enqueue(stream);
//     }




//     VadOutputData output;

//     return output;
// }
}  // namespace autoware::tensorrt_vad
