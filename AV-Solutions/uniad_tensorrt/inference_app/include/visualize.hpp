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

#ifndef __VISUALIZE_HPP__
#define __VISUALIZE_HPP__

#include <memory>
#include <string>
#include <vector>
#include <utility>
#include "dtype.hpp"

namespace nv {

struct Position {
  float x, y, z;
};

struct Size {
  float w, l, h;  // x, y, z
};

struct Velocity {
  float vx, vy;
};

struct Prediction {
  Position position;
  Size size;
  Velocity velocity;
  float z_rotation;
  float score;
  int id;
};

struct NameAndColor {
  std::string name;
  unsigned char r, g, b;
};

/////////////////////////////////////////////////////////////
// Used to plot the detection results on the image
//
struct ImageArtistParameter {
  int image_width;
  int image_stride;
  int image_height;
  int num_camera;
  std::vector<nvtype::Float4> viewport_nx4x4;
  std::vector<NameAndColor> classes;
};

class ImageArtist {
 public:
  virtual void draw_prediction(int camera_index, const std::vector<Prediction>& predictions, bool flipx) = 0;
  virtual void draw_prediction(int camera_index, const std::vector<std::vector<float>>& predictions, bool flipx) = 0;
  virtual void draw_planning_traj(int camera_index, const std::vector<std::pair<float, float>>& planning_traj, bool flipx) = 0;
  virtual void apply(unsigned char* image_rgb_device, void* stream) = 0;
};

std::shared_ptr<ImageArtist> create_image_artist(const ImageArtistParameter& param);

/////////////////////////////////////////////////////////////
// Used to render point cloud to image
//
struct BEVArtistParameter {
  int image_width;
  int image_stride;
  int image_height;
  float cx, cy, norm_size;
  float rotate_x;
  std::vector<NameAndColor> classes;
};

class BEVArtist {
 public:
  virtual void draw_lidar_points(const nvtype::half* points_device, unsigned int number_of_points) = 0;
  virtual void draw_prediction(const std::vector<Prediction>& predictions, bool take_title) = 0;
  virtual void draw_prediction(const std::vector<std::vector<float>>& predictions, bool take_title) = 0;
  virtual void draw_ego() = 0;
  virtual void draw_circle(int x, int y, int r) = 0;
  virtual void draw_planning_traj(const std::vector<std::pair<float, float>>& planning_traj, std::string& command) = 0;
  virtual void apply(unsigned char* image_rgb_device, void* stream) = 0;
};

std::shared_ptr<BEVArtist> create_bev_artist(const BEVArtistParameter& param);

/////////////////////////////////////////////////////////////
// Used to stitch all images and point clouds
//
struct SceneArtistParameter {
  int width;
  int stride;
  int height;
  unsigned char* image_device;
};

class SceneArtist {
 public:
  virtual void resize_to(const unsigned char* image_device, int x0, int y0, int x1, int y1, int image_width, int image_stride,
                         int image_height, float alpha, void* stream) = 0;

  virtual void flipx(const unsigned char* image_device, int image_width, int image_stride, int image_height,
                     unsigned char* output_device, int output_stride, void* stream) = 0;
};

std::shared_ptr<SceneArtist> create_scene_artist(const SceneArtistParameter& param);

};  // namespace nv

#endif  // __VISUALIZE_HPP__