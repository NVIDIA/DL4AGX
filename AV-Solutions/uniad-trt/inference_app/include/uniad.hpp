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

#ifndef __UNIAD_HPP__
#define __UNIAD_HPP__

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <assert.h>
#include "dtype.hpp"
#include <numeric>
#include "uniad.hpp"
#include "check.hpp"
#include "timer.hpp"
#include "tensorrt.hpp"
#include "NvInfer.h"
#include "NvInferRuntime.h"

#define TRACK_INS_MAX 1200
#define TRACK_MAX 200
#define TRACK_INS_MIN 901

namespace UniAD {

struct KernelParams {
  std::string trt_engine;
  int num_inputs = 0;
  std::unordered_map<std::string, std::vector<int>> input_max_shapes = {
    {"prev_track_intances0", {TRACK_INS_MAX, 512}},
    {"prev_track_intances1", {TRACK_INS_MAX, 3}},
    {"prev_track_intances3", {TRACK_INS_MAX}},
    {"prev_track_intances4", {TRACK_INS_MAX}},
    {"prev_track_intances5", {TRACK_INS_MAX}},
    {"prev_track_intances6", {TRACK_INS_MAX}},
    {"prev_track_intances8", {TRACK_INS_MAX}},
    {"prev_track_intances9", {TRACK_INS_MAX, 10}},
    {"prev_track_intances11", {TRACK_INS_MAX, 4, 256}},
    {"prev_track_intances12", {TRACK_INS_MAX, 4}},
    {"prev_track_intances13", {TRACK_INS_MAX}},
    {"prev_timestamp", {1}},
    {"prev_l2g_r_mat", {1, 3, 3}},
    {"prev_l2g_t", {1, 3}},
    {"prev_bev", {2500, 1, 256}},
    {"timestamp", {1}},
    {"l2g_r_mat", {1, 3, 3}},
    {"l2g_t", {1, 3}},
    {"img", {1, 6, 3, 256, 416}},
    {"img_metas_can_bus", {18}},
    {"img_metas_lidar2img", {1, 6, 4, 4}},
    {"command", {1}},
    {"use_prev_bev", {1}},
    {"max_obj_id", {1}}
  };
  std::unordered_map<std::string, std::size_t> input_sizes = {
    {"prev_track_intances0", sizeof(float)},
    {"prev_track_intances1", sizeof(float)},
    {"prev_track_intances3", sizeof(int32_t)},
    {"prev_track_intances4", sizeof(int32_t)},
    {"prev_track_intances5", sizeof(int32_t)},
    {"prev_track_intances6", sizeof(float)},
    {"prev_track_intances8", sizeof(float)},
    {"prev_track_intances9", sizeof(float)},
    {"prev_track_intances11", sizeof(float)},
    {"prev_track_intances12", sizeof(int32_t)},
    {"prev_track_intances13", sizeof(float)},
    {"prev_timestamp", sizeof(float)},
    {"prev_l2g_r_mat", sizeof(float)},
    {"prev_l2g_t", sizeof(float)},
    {"prev_bev", sizeof(float)},
    {"timestamp", sizeof(float)},
    {"l2g_r_mat", sizeof(float)},
    {"l2g_t", sizeof(float)},
    {"img", sizeof(float)},
    {"img_metas_can_bus", sizeof(float)},
    {"img_metas_lidar2img", sizeof(float)},
    {"command", sizeof(float)},
    {"use_prev_bev", sizeof(int32_t)},
    {"max_obj_id", sizeof(int32_t)}
  };
  std::unordered_map<std::string, std::vector<int>> output_max_shapes = {
    {"prev_track_intances0_out", {TRACK_INS_MAX, 512}},
    {"prev_track_intances1_out", {TRACK_INS_MAX, 3}},
    {"prev_track_intances3_out", {TRACK_INS_MAX}},
    {"prev_track_intances4_out", {TRACK_INS_MAX}},
    {"prev_track_intances5_out", {TRACK_INS_MAX}},
    {"prev_track_intances6_out", {TRACK_INS_MAX}},
    {"prev_track_intances8_out", {TRACK_INS_MAX}},
    {"prev_track_intances9_out", {TRACK_INS_MAX, 10}},
    {"prev_track_intances11_out", {TRACK_INS_MAX, 4, 256}},
    {"prev_track_intances12_out", {TRACK_INS_MAX, 4}},
    {"prev_track_intances13_out", {TRACK_INS_MAX}},
    {"prev_timestamp_out", {1}},
    {"prev_l2g_r_mat_out", {1, 3, 3}},
    {"prev_l2g_t_out", {1, 3}},
    {"bev_embed", {2500, 1, 256}},
    {"bboxes_dict_bboxes", {TRACK_MAX, 9}},
    {"scores", {TRACK_MAX}},
    {"labels", {TRACK_MAX}},
    {"bbox_index", {TRACK_MAX}},
    {"obj_idxes", {TRACK_MAX}},
    {"max_obj_id_out", {1}},
    {"outs_planning", {1, 6, 2}}
  };
};

struct KernelInput {
  std::vector<float> prev_track_intances0 = std::vector<float>(TRACK_INS_MAX*512, 0);
  std::vector<float> prev_track_intances1 = std::vector<float>(TRACK_INS_MAX*3, 0);
  std::vector<int32_t> prev_track_intances3 = std::vector<int32_t>(TRACK_INS_MAX, 0);
  std::vector<int32_t> prev_track_intances4 = std::vector<int32_t>(TRACK_INS_MAX, 0);
  std::vector<int32_t> prev_track_intances5 = std::vector<int32_t>(TRACK_INS_MAX, 0);
  std::vector<float> prev_track_intances6 = std::vector<float>(TRACK_INS_MAX, 0);
  std::vector<float> prev_track_intances8 = std::vector<float>(TRACK_INS_MAX, 0);
  std::vector<float> prev_track_intances9 = std::vector<float>(TRACK_INS_MAX*10, 0);
  std::vector<float> prev_track_intances11 = std::vector<float>(TRACK_INS_MAX*4*256, 0);
  std::vector<int32_t> prev_track_intances12 = std::vector<int32_t>(TRACK_INS_MAX*4, 0);
  std::vector<float> prev_track_intances13 = std::vector<float>(TRACK_INS_MAX, 0);
  std::vector<float> prev_timestamp = std::vector<float>(1, 0);
  std::vector<float> prev_l2g_r_mat = std::vector<float>(1*3*3, 0);
  std::vector<float> prev_l2g_t = std::vector<float>(1*3, 0);
  std::vector<float> prev_bev = std::vector<float>(2500*1*256, 0);
  std::vector<float> timestamp = std::vector<float>(1);
  std::vector<float> l2g_r_mat = std::vector<float>(1*3*3);
  std::vector<float> l2g_t = std::vector<float>(1*3);
  std::vector<float> img = std::vector<float>(1*6*3*256*416);
  std::vector<float> img_metas_can_bus = std::vector<float>(18);
  std::vector<float> img_metas_lidar2img = std::vector<float>(1*6*4*4);
  std::vector<float> command = std::vector<float>(1);
  std::vector<int32_t> use_prev_bev = std::vector<int32_t>(1, 0);
  std::vector<int32_t> max_obj_id = std::vector<int32_t>(1, 0);

  std::unordered_map<std::string, void*> data_ptrs = {
    {"prev_track_intances0", prev_track_intances0.data()},
    {"prev_track_intances1", prev_track_intances1.data()},
    {"prev_track_intances3", prev_track_intances3.data()},
    {"prev_track_intances4", prev_track_intances4.data()},
    {"prev_track_intances5", prev_track_intances5.data()},
    {"prev_track_intances6", prev_track_intances6.data()},
    {"prev_track_intances8", prev_track_intances8.data()},
    {"prev_track_intances9", prev_track_intances9.data()},
    {"prev_track_intances11", prev_track_intances11.data()},
    {"prev_track_intances12", prev_track_intances12.data()},
    {"prev_track_intances13", prev_track_intances13.data()},
    {"prev_timestamp", prev_timestamp.data()},
    {"prev_l2g_r_mat", prev_l2g_r_mat.data()},
    {"prev_l2g_t", prev_l2g_t.data()},
    {"prev_bev", prev_bev.data()},
    {"timestamp", timestamp.data()},
    {"l2g_r_mat", l2g_r_mat.data()},
    {"l2g_t", l2g_t.data()},
    {"img", img.data()},
    {"img_metas_can_bus", img_metas_can_bus.data()},
    {"img_metas_lidar2img", img_metas_lidar2img.data()},
    {"command", command.data()},
    {"use_prev_bev", use_prev_bev.data()},
    {"max_obj_id", max_obj_id.data()}
  };
  std::unordered_map<std::string, std::vector<int>> input_shapes = {
    {"prev_track_intances0", {TRACK_INS_MIN, 512}},
    {"prev_track_intances1", {TRACK_INS_MIN, 3}},
    {"prev_track_intances3", {TRACK_INS_MIN}},
    {"prev_track_intances4", {TRACK_INS_MIN}},
    {"prev_track_intances5", {TRACK_INS_MIN}},
    {"prev_track_intances6", {TRACK_INS_MIN}},
    {"prev_track_intances8", {TRACK_INS_MIN}},
    {"prev_track_intances9", {TRACK_INS_MIN, 10}},
    {"prev_track_intances11", {TRACK_INS_MIN, 4, 256}},
    {"prev_track_intances12", {TRACK_INS_MIN, 4}},
    {"prev_track_intances13", {TRACK_INS_MIN}},
    {"prev_timestamp", {1}},
    {"prev_l2g_r_mat", {1, 3, 3}},
    {"prev_l2g_t", {1, 3}},
    {"prev_bev", {2500, 1, 256}},
    {"use_prev_bev", {1}},
    {"max_obj_id", {1}}
  };
};

struct KernelOutput {
  std::vector<float> prev_track_intances0_out = std::vector<float>(TRACK_INS_MAX*512);
  std::vector<float> prev_track_intances1_out = std::vector<float>(TRACK_INS_MAX*3);
  std::vector<int32_t> prev_track_intances3_out = std::vector<int32_t>(TRACK_INS_MAX);
  std::vector<int32_t> prev_track_intances4_out = std::vector<int32_t>(TRACK_INS_MAX);
  std::vector<int32_t> prev_track_intances5_out = std::vector<int32_t>(TRACK_INS_MAX);
  std::vector<float> prev_track_intances6_out = std::vector<float>(TRACK_INS_MAX);
  std::vector<float> prev_track_intances8_out = std::vector<float>(TRACK_INS_MAX);
  std::vector<float> prev_track_intances9_out = std::vector<float>(TRACK_INS_MAX*10);
  std::vector<float> prev_track_intances11_out = std::vector<float>(TRACK_INS_MAX*4*256);
  std::vector<int32_t> prev_track_intances12_out = std::vector<int32_t>(TRACK_INS_MAX*4);
  std::vector<float> prev_track_intances13_out = std::vector<float>(TRACK_INS_MAX);
  std::vector<float> prev_timestamp_out = std::vector<float>(1);
  std::vector<float> prev_l2g_r_mat_out = std::vector<float>(1*3*3);
  std::vector<float> prev_l2g_t_out = std::vector<float>(1*3);
  std::vector<float> bev_embed = std::vector<float>(2500*1*256);
  std::vector<float> bboxes_dict_bboxes = std::vector<float>(TRACK_MAX*9);
  std::vector<float> scores = std::vector<float>(TRACK_MAX);
  std::vector<int32_t> labels = std::vector<int32_t>(TRACK_MAX);
  std::vector<int32_t> bbox_index = std::vector<int32_t>(TRACK_MAX);
  std::vector<int32_t> obj_idxes = std::vector<int32_t>(TRACK_MAX);
  std::vector<int32_t> max_obj_id_out = std::vector<int32_t>(1);
  std::vector<float> outs_planning = std::vector<float>(1*6*2);

  std::unordered_map<std::string, void*> data_ptrs = {
    {"prev_track_intances0_out", prev_track_intances0_out.data()},
    {"prev_track_intances1_out", prev_track_intances1_out.data()},
    {"prev_track_intances3_out", prev_track_intances3_out.data()},
    {"prev_track_intances4_out", prev_track_intances4_out.data()},
    {"prev_track_intances5_out", prev_track_intances5_out.data()},
    {"prev_track_intances6_out", prev_track_intances6_out.data()},
    {"prev_track_intances8_out", prev_track_intances8_out.data()},
    {"prev_track_intances9_out", prev_track_intances9_out.data()},
    {"prev_track_intances11_out", prev_track_intances11_out.data()},
    {"prev_track_intances12_out", prev_track_intances12_out.data()},
    {"prev_track_intances13_out", prev_track_intances13_out.data()},
    {"prev_timestamp_out", prev_timestamp_out.data()},
    {"prev_l2g_r_mat_out", prev_l2g_r_mat_out.data()},
    {"prev_l2g_t_out", prev_l2g_t_out.data()},
    {"bboxes_dict_bboxes", bboxes_dict_bboxes.data()},
    {"scores", scores.data()},
    {"labels", labels.data()},
    {"bbox_index", bbox_index.data()},
    {"obj_idxes", obj_idxes.data()},
    {"max_obj_id_out", max_obj_id_out.data()},
    {"bev_embed", bev_embed.data()},
    {"outs_planning", outs_planning.data()}
  };
  std::unordered_map<std::string, std::vector<int>> output_shapes;
  std::unordered_map<std::string, size_t> output_dsizes;
};

class Kernel {
 public:
  virtual ~Kernel() {};
  virtual int init(const KernelParams& param) = 0;
  virtual void forward_timer(const UniAD::KernelInput& inputs, UniAD::KernelOutput& outputs, void *stream, bool enable_timer) = 0;
  virtual void forward_one_frame(const UniAD::KernelInput& inputs, UniAD::KernelOutput& outputs, bool enable_timer, void *stream) = 0;
  virtual void print_info() = 0;
};

class KernelImplement : public Kernel {
public:
  virtual ~KernelImplement();
  virtual int init(const KernelParams& param);
  virtual void forward_timer(const UniAD::KernelInput& inputs, UniAD::KernelOutput& outputs, void *stream, bool enable_timer);
  virtual void forward_one_frame(const UniAD::KernelInput& inputs, UniAD::KernelOutput& outputs, bool enable_timer, void *stream);
  virtual void print_info();
private:
  KernelParams param_;
  nv::EventTimer timer_;
  std::shared_ptr<TensorRT::Engine> engine_;
  std::vector<const void *> bindings_;
  std::vector<void *> inputs_device_;
  std::vector<void *> outputs_device_;
  std::vector<void *> inputs_host_;
  std::vector<void *> outputs_host_;
};

}; //namespace UniAD
#endif  // __UNIAD_HPP__
