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

#ifndef AUTOWARE_TENSORRT_VAD_ROS_VAD_LOGGER_HPP_
#define AUTOWARE_TENSORRT_VAD_ROS_VAD_LOGGER_HPP_

#include "autoware/tensorrt_vad/vad_model.hpp"
#include <rclcpp/rclcpp.hpp>

namespace autoware::tensorrt_vad
{

/**
 * @brief ROS2用のVadLogger実装
 * 
 * 使用例:
 * auto ros_logger = std::make_shared<RosVadLogger>(node);
 * VadConfig config;
 * // configを設定...
 * VadModel model(config, ros_logger);
 */
class RosVadLogger : public VadLogger {
private:
    rclcpp::Logger logger_;

public:
    explicit RosVadLogger(rclcpp::Node::SharedPtr node) 
        : logger_(node->get_logger()) {}
    
    explicit RosVadLogger(const rclcpp::Logger& logger) 
        : logger_(logger) {}

    void debug(const std::string& message) override {
        RCLCPP_DEBUG(logger_, "%s", message.c_str());
    }

    void info(const std::string& message) override {
        RCLCPP_INFO(logger_, "%s", message.c_str());
    }

    void warn(const std::string& message) override {
        RCLCPP_WARN(logger_, "%s", message.c_str());
    }

    void error(const std::string& message) override {
        RCLCPP_ERROR(logger_, "%s", message.c_str());
    }
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_ROS_VAD_LOGGER_HPP_
