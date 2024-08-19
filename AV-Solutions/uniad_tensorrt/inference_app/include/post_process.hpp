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

#include <cuda_runtime.h>
#include <string.h>
#include <unordered_map>
#include <vector>
#include <utility>
#include <fstream>
#include <numeric>
#include <sys/stat.h>
#include "uniad.hpp"

std::vector<std::pair<float, float>> decode_planning_traj(const UniAD::KernelOutput& output_instance) {
    std::vector<std::pair<float, float>> planning_traj;
    for (size_t i=0; i<output_instance.outs_planning.size(); i+=2) {
        planning_traj.push_back({output_instance.outs_planning[i], output_instance.outs_planning[i+1]});
    }
    // TODO: collision optimization
    return planning_traj;
}

std::string decode_command(const UniAD::KernelInput& input_instance) {
    std::unordered_map<int, std::string> command_map = {
        {0, "TURN RIGHT"},
        {1, "TURN LEFT"},
        {2, "KEEP FORWARD"}
    };
    int command_idx = (int)(input_instance.command[0]);
    return command_map[command_idx];
}

std::vector<std::vector<float>> decode_bbox(const UniAD::KernelOutput& output_instance) {
    std::vector<std::vector<float>> pred_bbox;
    int total_number_bbox = output_instance.output_shapes.at("scores")[0];
    for (size_t i=0; i<total_number_bbox; ++i) {
        if (output_instance.scores[i] < 0.25) continue;
        pred_bbox.push_back({
            output_instance.bboxes_dict_bboxes[9*i+0], // x
            output_instance.bboxes_dict_bboxes[9*i+1], // y
            output_instance.bboxes_dict_bboxes[9*i+2] + output_instance.bboxes_dict_bboxes[9*i+5]/2., // z
            output_instance.bboxes_dict_bboxes[9*i+3], // w
            output_instance.bboxes_dict_bboxes[9*i+4], // l
            output_instance.bboxes_dict_bboxes[9*i+5], // h
            output_instance.bboxes_dict_bboxes[9*i+6], // yaw
            output_instance.bboxes_dict_bboxes[9*i+7], // vx
            output_instance.bboxes_dict_bboxes[9*i+8], // vy
            output_instance.labels[i], // label
            output_instance.scores[i], // scores
        });
    }
    return pred_bbox;
}