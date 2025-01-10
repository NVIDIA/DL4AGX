/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Modified from https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/blob/master/CUDA-BEVFusion/src/common/visualize.hpp
// Add functions to visualize BBOX predictions and planning trajectories in the ImageArtist and BEVArtist

#ifndef __VISUALIZE_HPP__
#define __VISUALIZE_HPP__

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "check.hpp"

namespace nvtype {

struct Int2 {
  int x, y;

  Int2() = default;
  Int2(int x, int y = 0) : x(x), y(y) {}
};

struct Int3 {
  int x, y, z;

  Int3() = default;
  Int3(int x, int y = 0, int z = 0) : x(x), y(y), z(z) {}
};

struct Float2 {
  float x, y;

  Float2() = default;
  Float2(float x, float y = 0) : x(x), y(y) {}
};

struct Float3 {
  float x, y, z;

  Float3() = default;
  Float3(float x, float y = 0, float z = 0) : x(x), y(y), z(z) {}
};

struct Float4 {
  float x, y, z, w;

  Float4() = default;
  Float4(float x, float y = 0, float z = 0, float w = 0) : x(x), y(y), z(z), w(w) {}
};

// It is only used to specify the type only, while hoping to avoid header file contamination.
typedef struct {
  unsigned short __x;
} half;

};  // namespace nvtype

namespace nv {

#define LINEAR_LAUNCH_THREADS 512
#define cuda_linear_index (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_x (blockDim.x * blockIdx.x + threadIdx.x)
#define cuda_2d_y (blockDim.y * blockIdx.y + threadIdx.y)
#define divup(a, b) ((static_cast<int>(a) + static_cast<int>(b) - 1) / static_cast<int>(b))

#ifdef CUDA_DEBUG
#define cuda_linear_launch(kernel, stream, num, ...)                                   \
  do {                                                                                 \
    size_t __num__ = (size_t)(num);                                                    \
    size_t __blocks__ = (__num__ + LINEAR_LAUNCH_THREADS - 1) / LINEAR_LAUNCH_THREADS; \
    kernel<<<__blocks__, LINEAR_LAUNCH_THREADS, 0, stream>>>(__num__, __VA_ARGS__);    \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);             \
    nv::check_runtime(cudaStreamSynchronize(stream), #kernel, __LINE__, __FILE__);     \
  } while (false)

#define cuda_2d_launch(kernel, stream, nx, ny, ...)                                \
  do {                                                                             \
    dim3 __threads__(32, 32);                                                      \
    dim3 __blocks__(divup(nx, 32), divup(ny, 32));                                 \
    kernel<<<__blocks__, __threads__, 0, stream>>>(nx, ny, __VA_ARGS__);           \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);         \
    nv::check_runtime(cudaStreamSynchronize(stream), #kernel, __LINE__, __FILE__); \
  } while (false)
#else  // CUDA_DEBUG
#define cuda_linear_launch(kernel, stream, num, ...)                                \
  do {                                                                              \
    size_t __num__ = (size_t)(num);                                                 \
    size_t __blocks__ = divup(__num__, LINEAR_LAUNCH_THREADS);                      \
    kernel<<<__blocks__, LINEAR_LAUNCH_THREADS, 0, stream>>>(__num__, __VA_ARGS__); \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);          \
  } while (false)

#define cuda_2d_launch(kernel, stream, nx, ny, nz, ...)                      \
  do {                                                                       \
    dim3 __threads__(32, 32);                                                \
    dim3 __blocks__(divup(nx, 32), divup(ny, 32), nz);                       \
    kernel<<<__blocks__, __threads__, 0, stream>>>(nx, ny, nz, __VA_ARGS__); \
    nv::check_runtime(cudaPeekAtLastError(), #kernel, __LINE__, __FILE__);   \
  } while (false)
#endif  // CUDA_DEBUG

static inline std::string format(const char* fmt, ...) {
  char buffer[2048];
  va_list vl;
  va_start(vl, fmt);
  vsnprintf(buffer, sizeof(buffer), fmt, vl);
  return buffer;
}

struct Position {
  float x, y, z;
};

struct Size {
  float w, l, h;
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
  std::string font_path;
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
  std::string font_path;

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
  std::string font_path;
  virtual void resize_to(const unsigned char* image_device, int x0, int y0, int x1, int y1, int image_width, int image_stride,
                         int image_height, float alpha, void* stream) = 0;

  virtual void flipx(const unsigned char* image_device, int image_width, int image_stride, int image_height,
                     unsigned char* output_device, int output_stride, void* stream) = 0;
};

std::shared_ptr<SceneArtist> create_scene_artist(const SceneArtistParameter& param);

struct VisualizeFrame {
  int fid;
  int n_cam = 6;
  int cmd;  // Right, Left, Straight

  // 6 * 4 * 4
  std::vector<float> img_metas_lidar2img;
  
  // N * 10 (x, y, z, w, l, h, yaw, vx, vy, label, score)
  std::vector<std::vector<float>> det;

  // 6, 2 (after cumsum, in absolute coord)
  std::vector<float> planning;
};

void visualize(
  const std::vector<unsigned char*> images, 
  const VisualizeFrame& frame,
  const std::string& font_path,
  const std::string& save_path,
  cudaStream_t stream
);

};  // namespace nv

#endif  // __VISUALIZE_HPP__