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

#ifndef AUTOWARE_TENSORRT_VAD_VAD_NODE_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_NODE_HPP_

#include "autoware/tensorrt_vad/vad_model.hpp"
#include "ros_vad_logger.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <cmath>
#include <memory>

// Forward declarations for future use
// #include <cuda_runtime.h>
// #include <NvInfer.h>
// #include <stb/stb_image.h>

namespace autoware::tensorrt_vad
{

class VadNode : public rclcpp::Node
{
public:
  explicit VadNode(const rclcpp::NodeOptions & options);

private:
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr msg, std::size_t camera_id);
  void camera_info_callback(const sensor_msgs::msg::CameraInfo & msg, std::size_t camera_id);

  std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> image_subs_{};
  std::vector<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> camera_info_subs_{};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_{tf_buffer_};

  // VAD model - 具体的なロガー型を指定
  std::unique_ptr<VadModel<RosVadLogger>> vad_model_ptr_{};

  // VAD input topic
  // std::unique_ptr<VadTopicData> vad_topic_data_ptr_{};

  // VAD interface
  // std::unique_ptr<VadInterface> vad_interface_ptr_{};

  // Publishers
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr trajectory_pub_{nullptr};
  // rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr detected_objects_pub_{nullptr};

  // 推論を実行するメソッド
  // std::tuple<std::optional<autoware_planning_msgs::msg::Trajectory>, std::optional<autoware_perception_msgs::msg::DetectedObjects> > execute_inference(const VadTopicData & vad_topic_data);
};
}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_NODE_HPP_
