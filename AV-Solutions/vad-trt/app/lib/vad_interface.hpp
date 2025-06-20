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

#ifndef AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_

#include <vector>

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"

namespace autoware::tensorrt_vad
{

struct VadTopicData
{
  // このデータセットの基準となるタイムスタンプ
  rclcpp::Time stamp;

  // 複数のカメラからの画像データ。
  // launchファイルでリマップされた ~/input/image0, ~/input/image1, ... に対応します。
  // vectorのインデックスが autoware_camera_id となります。
  std::vector<sensor_msgs::msg::Image::ConstSharedPtr> images;

  // 上記の各画像に対応するカメラのキャリブレーション情報
  std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> camera_infos;

  // 静的な座標変換情報（例: lidar_to_camera）
  tf2_msgs::msg::TFMessage::ConstSharedPtr tf_static;

  // 車両の運動状態データ（/localization/kinematic_state などから）
  nav_msgs::msg::Odometry::ConstSharedPtr kinematic_state;

  // IMUデータ（/sensing/imu/tamagawa/imu_raw などから）
  sensor_msgs::msg::Imu::ConstSharedPtr imu_raw;

  // フレームが完成しているかをチェック
  bool is_complete() const {
    if (images.size() != 6 || camera_infos.size() != 6) return false;
    
    for (int i = 0; i < 6; ++i) {
      if (!images[i] || !camera_infos[i]) return false;
    }
    
    return tf_static != nullptr && 
           kinematic_state != nullptr && 
           imu_raw != nullptr;
  }
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_
