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

#include "autoware/tensorrt_vad/vad_node.hpp"

#include "autoware/tensorrt_vad/utils.hpp"

#include <rclcpp_components/register_node_macro.hpp>

namespace autoware::tensorrt_vad
{

// ヘルパー関数：yaw角からクォータニオンを作成
geometry_msgs::msg::Quaternion calculate_quaternion_from_yaw(double yaw)
{
  geometry_msgs::msg::Quaternion q{};
  q.x = 0.0;
  q.y = 0.0;
  q.z = std::sin(yaw * 0.5);
  q.w = std::cos(yaw * 0.5);
  return q;
}

VadNode::VadNode(const rclcpp::NodeOptions & options) : Node("vad_node", options), tf_buffer_(this->get_clock())
{
  // Publishers の初期化
  trajectory_pub_ = this->create_publisher<autoware_planning_msgs::msg::Trajectory>("~/output/trajectory", rclcpp::QoS(1));

  // VADモデルの初期化 - 一時的にコメントアウト
  // TODO: 適切な設定ファイルとロガーを用意してから初期化
  // auto logger = std::make_shared<RosVadLogger>();
  // VadConfig config;
  // vad_model_ptr_ = std::make_unique<VadModel<RosVadLogger>>(config, logger);

  RCLCPP_INFO(this->get_logger(), "VAD Node initialized");
}

// std::tuple<std::optional<autoware_planning_msgs::msg::Trajectory>, std::optional<autoware_perception_msgs::msg::DetectedObjects> > execute_inference(const VadTopicData vad_topic_data)
// {
//   // VadInterfaceを通じてVadInputDataに変換
//   // scalingされた状態の画像を含む
//   const auto vad_input = vad_interface_ptr_->convert_ros_to_vad_input(vad_topic_data);

//   // VadModelで推論実行
//   const auto vad_output = vad_model_ptr_->infer(vad_input);

//   // VadInterfaceを通じてROS型に変換
//   return vad_interface_ptr_->convert_vad_output_to_ros(vad_output);
}  // namespace autoware::tensorrt_vad

// Register the component with the ROS2 component system
// NOLINTNEXTLINE(readability-identifier-naming,cppcoreguidelines-avoid-non-const-global-variables)
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::tensorrt_vad::VadNode)

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<autoware::tensorrt_vad::VadNode>(rclcpp::NodeOptions{});
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
