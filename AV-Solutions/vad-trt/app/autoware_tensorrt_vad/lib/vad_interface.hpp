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
#include <unordered_map>
#include <optional>
#include <tuple>

#include "vad_interface_config.hpp"

#include "rclcpp/time.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "tf2_ros/buffer.h"
#include <tf2_eigen/tf2_eigen.hpp>

#include "vad_model.hpp" 

namespace autoware::tensorrt_vad
{

struct VadInputTopicData
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
    
    for (int32_t i = 0; i < 6; ++i) {
      if (!images[i] || !camera_infos[i]) return false;
    }
    
    return tf_static != nullptr && 
           kinematic_state != nullptr && 
           imu_raw != nullptr;
  }
};

// 各process_*メソッドの戻り値となるデータ構造
using CameraImagesData = std::vector<float>;
using ShiftData = std::vector<float>;
using Lidar2ImgData = std::vector<float>;
using CanBusData = std::vector<float>;

/**
 * @class VadInterface
 * @brief ROS topicデータをVADの入力形式に変換するインターフェース
 */

class VadInterface {
public:
  explicit VadInterface(const VadInterfaceConfig& config, 
                        std::shared_ptr<tf2_ros::Buffer> tf_buffer);

  VadInputData convert_input(const VadInputTopicData & vad_input_topic_data, const std::vector<float> & prev_can_bus = {});
  Lidar2ImgData process_lidar2img(
    const tf2_msgs::msg::TFMessage::ConstSharedPtr & tf_static,
    const std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> & camera_infos,
    float scale_width, float scale_height) const;

  CanBusData process_can_bus(
    const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
    const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw,
    const std::vector<float> & prev_can_bus) const;

  ShiftData process_shift(
    const std::vector<float> & can_bus,
    const std::vector<float> & prev_can_bus) const;

  CameraImagesData process_image(
    const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> & images) const;


private:
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  int32_t target_image_width_, target_image_height_;
  int32_t input_image_width_, input_image_height_;
  std::array<float, 6> point_cloud_range_;
  int32_t bev_h_, bev_w_;
  float default_patch_angle_;
  int32_t default_command_;
  std::vector<float> default_shift_;
  std::array<float, 3> image_normalization_param_mean_;
  std::array<float, 3> image_normalization_param_std_;
  Eigen::Matrix4f vad2base_;
  Eigen::Matrix4f base2vad_;
  std::unordered_map<int32_t, int32_t> autoware_to_vad_camera_mapping_;

  // --- 内部処理関数 ---
  std::optional<Eigen::Matrix4f> lookup_base_to_camera_rt(tf2_ros::Buffer & buffer, int32_t autoware_camera_id) const;
  Eigen::Matrix4f create_viewpad(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info) const;
  Eigen::Matrix4f apply_scaling(const Eigen::Matrix4f & lidar2img, float scale_width, float scale_height) const;
  std::vector<float> matrix_to_flat(const Eigen::Matrix4f & matrix) const;

  std::vector<float> normalize_image(unsigned char *image_data, int32_t width, int32_t height) const;
  
  std::vector<float> calculate_can_bus(
    const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
    const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw,
    const std::vector<float> & prev_can_bus) const;
  
  std::vector<float> calculate_shift(
    const std::vector<float> & can_bus,
    const std::vector<float> & prev_can_bus) const;

  std::tuple<float, float, float> aw2vad_xyz(float aw_x, float aw_y, float aw_z) const;
  Eigen::Quaternionf aw2vad_quaternion(const Eigen::Quaternionf & q_aw) const;
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_INTERFACE_HPP_
