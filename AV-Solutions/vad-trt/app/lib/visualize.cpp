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

#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <unordered_map>
#include "visualize.hpp"

namespace nv {

std::vector<std::pair<float, float>> decode_planning_traj(
  const VisualizeFrame& frame
) {
  std::vector<std::pair<float, float>> planning_traj;
  for (size_t i=0; i<frame.planning.size(); i+=2) {
    planning_traj.push_back({frame.planning[i], frame.planning[i+1]});
  }
  // TODO: collision optimization
  return planning_traj;
}

std::string decode_command(const VisualizeFrame& frame) {
  std::unordered_map<int, std::string> command_map = {
    {0, "TURN RIGHT"},
    {1, "TURN LEFT"},
    {2, "KEEP FORWARD"}
  };
  return command_map[frame.cmd];
}

void visualize(
  const std::vector<unsigned char*> images, 
  const VisualizeFrame& frame,
  const std::string& font_path,
  const std::string& save_path,
  cudaStream_t stream
) {
  std::vector<std::pair<float, float>> planning_traj = decode_planning_traj(frame);
  std::string command = decode_command(frame); 

  int lidar_size = 1200;
  int content_width = lidar_size + 900;
  nv::ImageArtistParameter image_artist_param;
  image_artist_param.num_camera = images.size();
  image_artist_param.image_width = 1600;
  image_artist_param.image_height = 900;
  image_artist_param.image_stride = image_artist_param.image_width * 3;

  for (size_t i=0; i<frame.img_metas_lidar2img.size(); i+=4) {
    nvtype::Float4 transform_vec(
      frame.img_metas_lidar2img[i], 
      frame.img_metas_lidar2img[i+1],
      frame.img_metas_lidar2img[i+2], 
      frame.img_metas_lidar2img[i+3]);
    image_artist_param.viewport_nx4x4.push_back(transform_vec);
  }

  int gap = 0;
  int camera_width = content_width / 3;
  int camera_height = static_cast<float>(camera_width / (float)image_artist_param.image_width * image_artist_param.image_height);
  int content_height = 2 * camera_height + 3 * content_width / 4;

  nv::SceneArtistParameter scene_artist_param;
  scene_artist_param.width = content_width;
  scene_artist_param.height = content_height;
  scene_artist_param.stride = scene_artist_param.width * 3;

  size_t scene_device_nelem = scene_artist_param.height * scene_artist_param.width * 3;
  void* scene_device_dptr;
  cudaMalloc(&scene_device_dptr, scene_device_nelem);
  cudaMemset(scene_device_dptr, 0x33, scene_device_nelem);

  scene_artist_param.image_device = (unsigned char*)scene_device_dptr;
  auto scene = nv::create_scene_artist(scene_artist_param);
  scene->font_path = font_path;

  nv::BEVArtistParameter bev_artist_param;
  bev_artist_param.image_width = content_width;
  bev_artist_param.image_height = content_height;
  bev_artist_param.rotate_x = 70.0f;
  bev_artist_param.norm_size = lidar_size * 0.5f;
  bev_artist_param.cx = content_width * 0.5f;
  bev_artist_param.cy = content_height * 0.5f + camera_height;
  bev_artist_param.image_stride = scene_artist_param.stride;

  auto bev_visualizer = nv::create_bev_artist(bev_artist_param);
  bev_visualizer->font_path = font_path;
  bev_visualizer->draw_ego();

  // range marker
  for (int r=15; r<=60; r+=15) 
    bev_visualizer->draw_circle(0, 0, r);

  // draw in bev view
  bev_visualizer->draw_planning_traj(planning_traj, command);
  bev_visualizer->draw_prediction(frame.det, true);
  bev_visualizer->apply((unsigned char*)scene_device_dptr, stream);

  int offset_cameras[][3] = {
    {camera_width, 0, 0},
    {camera_width*2, 0, 0},
    {0, 0, 0},
    {camera_width, camera_height, 1},
    {0, camera_height, 1},
    {camera_width*2, camera_height, 1}};

  auto visualizer = nv::create_image_artist(image_artist_param);
  visualizer->font_path = font_path;

  // draw in camera view
  size_t device_image_nelem = 900 * 1600 * 3;
  void* device_image_dptr;
  void* device_image_clone_dptr;
  cudaMalloc(&device_image_dptr, device_image_nelem);
  cudaMalloc(&device_image_clone_dptr, device_image_nelem);

  for (size_t icamera = 0; icamera < images.size(); ++icamera) {
    int ox = offset_cameras[icamera][0];
    int oy = offset_cameras[icamera][1];
    bool xflip = static_cast<bool>(offset_cameras[icamera][2]);
    visualizer->draw_prediction(icamera, frame.det, xflip);
    visualizer->draw_planning_traj(icamera, planning_traj, xflip);

    cudaMemcpy(device_image_dptr, images[icamera], device_image_nelem, cudaMemcpyHostToDevice);

    if (xflip) {
      cudaMemcpy(device_image_clone_dptr, images[icamera], device_image_nelem, cudaMemcpyHostToDevice);      
      scene->flipx(
        (unsigned char*)device_image_clone_dptr, 
        1600, 1600 * 3, 900, 
        (unsigned char*)device_image_dptr,
        1600 * 3, stream);
      checkRuntime(cudaStreamSynchronize(stream));
    }
    visualizer->apply((unsigned char*)device_image_dptr, stream);

    scene->resize_to(
      (unsigned char*)device_image_dptr, 
      ox, oy, ox + camera_width, oy + camera_height, 1600,
      1600 * 3, 900, 0.8f, stream);
    checkRuntime(cudaStreamSynchronize(stream));
  }

  unsigned char* scene_device_hptr = new unsigned char[scene_device_nelem];
  cudaMemcpy(scene_device_hptr, scene_device_dptr, scene_device_nelem, cudaMemcpyDeviceToHost);
  stbi_write_jpg(
    save_path.c_str(), 
    scene_artist_param.width, scene_artist_param.height, 3,
    scene_device_hptr, 100);

  cudaFree(device_image_dptr);
  cudaFree(device_image_clone_dptr);
  cudaFree(scene_device_dptr);
  delete[] scene_device_hptr;
}

} // namespace nv
