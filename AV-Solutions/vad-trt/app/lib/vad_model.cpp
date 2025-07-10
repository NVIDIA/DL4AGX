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

#include "vad_model.hpp"

namespace autoware::tensorrt_vad {

std::vector<std::vector<std::vector<std::vector<float>>>> postprocess_traj_preds(
    const std::vector<float>& all_traj_preds_flat) {
  const int32_t num_objects = 900;
  const int32_t num_fut_modes = 6;
  const int32_t num_fut_ts = 6;
  const int32_t traj_coords = 2;
  std::vector<std::vector<std::vector<std::vector<float>>>> traj_preds;
  traj_preds.resize(num_objects);
  for (int32_t obj = 0; obj < num_objects; ++obj) {
    traj_preds[obj].resize(num_fut_modes);
    for (int32_t fut_mode = 0; fut_mode < num_fut_modes; ++fut_mode) {
      traj_preds[obj][fut_mode].resize(num_fut_ts);
      for (int32_t ts = 0; ts < num_fut_ts; ++ts) {
        traj_preds[obj][fut_mode][ts].resize(traj_coords);
        int32_t idx_occupied = obj * num_fut_modes * num_fut_ts * traj_coords;
        int32_t idx_flat = idx_occupied + fut_mode * num_fut_ts * traj_coords + ts * traj_coords;
        traj_preds[obj][fut_mode][ts][0] = all_traj_preds_flat[idx_flat];
        traj_preds[obj][fut_mode][ts][1] = all_traj_preds_flat[idx_flat + 1];
      }
    }
  }
  return traj_preds;
}

std::vector<std::vector<float>> postprocess_traj_cls_scores(
    const std::vector<float>& all_traj_cls_scores_flat) {
  const int32_t num_objects = 900;
  const int32_t num_fut_modes = 6;
  std::vector<std::vector<float>> traj_cls_scores;
  traj_cls_scores.resize(num_objects);
  for (int32_t obj = 0; obj < num_objects; ++obj) {
    traj_cls_scores[obj].resize(num_fut_modes);
    for (int32_t fut_mode = 0; fut_mode < num_fut_modes; ++fut_mode) {
      int32_t idx_occupied = obj * num_fut_modes;
      int32_t idx_flat = idx_occupied + fut_mode;
      traj_cls_scores[obj][fut_mode] = all_traj_cls_scores_flat[idx_flat];
    }
  }
  return traj_cls_scores;
}

std::vector<std::vector<float>> postprocess_bbox_preds(
    const std::vector<float>& all_bbox_preds_flat) {
  const int32_t num_objects = 900;
  const int32_t bbox_features = 10;
  std::vector<std::vector<float>> bbox_preds;
  bbox_preds.resize(num_objects);
  for (int32_t obj = 0; obj < num_objects; ++obj) {
    bbox_preds[obj].resize(bbox_features);
    for (int32_t feat = 0; feat < bbox_features; ++feat) {
      int32_t idx_occupied = obj * bbox_features;
      int32_t idx_flat = idx_occupied + feat;
      bbox_preds[obj][feat] = all_bbox_preds_flat[idx_flat];
    }
  }
  return bbox_preds;
}

} // namespace autoware::tensorrt_vad
