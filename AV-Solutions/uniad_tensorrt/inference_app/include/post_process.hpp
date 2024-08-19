/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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