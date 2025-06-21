/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include <cmath>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <string.h>
#include <vector>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include <Eigen/Dense>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/imu.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

#include "ros_vad_logger.hpp"
#include "vad_interface.hpp"
#include "vad_model.hpp"
#include "visualize.hpp"

#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

// Convert Autoware coordinates to nuScenes coordinates
std::pair<float, float> aw2ns_xy(float aw_x, float aw_y) {
  float ns_x = -aw_y;
  float ns_y = aw_x;
  return {ns_x, ns_y};
}

// Convert a quaternion from Autoware to nuScenes coordinate system
Eigen::Quaternionf aw2ns_quaternion(const Eigen::Quaternionf &q_aw) {
  // Create a -90-degree rotation around Z-axis (Autoware -> nuScenes)
  Eigen::Quaternionf q_rotation(
      Eigen::AngleAxisf(-M_PI / 2, Eigen::Vector3f::UnitZ()));

  // Apply the rotation
  return q_rotation * q_aw;
}

unsigned char *
convert_normalized_to_rgb(const std::vector<float> &normalized_data) {
  const int src_width = 640;
  const int src_height = 384;
  const int dst_width = 1600;
  const int dst_height = 960;

  // 正規化の逆変換用パラメータ
  float mean[3] = {103.530f, 116.280f, 123.675f};
  float std[3] = {1.0f, 1.0f, 1.0f};

  // まず元の正規化データから一時的なRGB画像を作成
  const int channels = 3;
  unsigned char *temp_rgb =
      new unsigned char[src_width * src_height * channels];

  // CHW形式（正規化）からHWC形式（RGB）に変換
  for (int h = 0; h < src_height; ++h) {
    for (int w = 0; w < src_width; ++w) {
      for (int c = 0; c < channels; ++c) {
        // CHW形式でのソースインデックス
        int src_idx = c * src_height * src_width + h * src_width + w;

        // HWC形式（RGB）でのデスティネーションインデックス
        int dst_idx =
            (h * src_width + w) * channels + (2 - c); // BGRからRGBに変換

        // 正規化を逆にして、unsigned charに変換
        float pixel_value = normalized_data[src_idx] * std[c] + mean[c];
        temp_rgb[dst_idx] = static_cast<unsigned char>(
            std::max(0.0f, std::min(255.0f, pixel_value)));
      }
    }
  }

  // 拡大後のRGB画像用にメモリを確保
  unsigned char *rgb_data =
      new unsigned char[dst_width * dst_height * channels];

  // stbir_resize_uint8を使用してリサイズ
  int resize_result =
      stbir_resize_uint8(temp_rgb, src_width, src_height, 0, rgb_data,
                         dst_width, dst_height, 0, channels);

  // 一時メモリを解放
  delete[] temp_rgb;

  // リサイズ失敗の場合のエラーハンドリング
  if (!resize_result) {
    std::cerr << "画像のリサイズに失敗しました" << std::endl;
    delete[] rgb_data;
    return nullptr;
  }

  return rgb_data;
}

class VADNode : public rclcpp::Node {
public:
  VADNode(const std::vector<std::string> &yaml_config_paths)
      : Node("vad_node", createNodeOptions(yaml_config_paths)) {
    // Publishers
    trajectory_publisher_ =
        this->create_publisher<autoware_planning_msgs::msg::Trajectory>(
            "/planning/vad/trajectory", rclcpp::QoS(1));

    objects_publisher_ =
        this->create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
            "/perception/object_recognition/detection/vad/objects",
            rclcpp::QoS(1));

    // Subscribers for each camera
    for (int32_t i = 0; i < 6; ++i) {
      auto callback =
          [this, i](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
            this->onImageReceived(msg, i);
          };

      camera_subscribers_.push_back(
          this->create_subscription<sensor_msgs::msg::CompressedImage>(
              "/sensing/camera/camera" + std::to_string(i) +
                  "/image/compressed",
              rclcpp::QoS(1), callback));
    }

    // VadConfigを読み込み
    loadVadConfig();

    RCLCPP_INFO(this->get_logger(), "VAD Node has been initialized");
  }

  // VadConfigを取得する関数
  const autoware::tensorrt_vad::VadConfig &getVadConfig() const {
    return vad_config_;
  }

  void publishTrajectory(const std::vector<float> &planning) {
    auto trajectory_msg =
        std::make_unique<autoware_planning_msgs::msg::Trajectory>();

    for (size_t i = 0; i < planning.size(); i += 2) {
      autoware_planning_msgs::msg::TrajectoryPoint point;

      point.pose.position.x = planning[i + 1];
      point.pose.position.y = -planning[i];
      point.pose.position.z = 0.0;

      if (i + 2 < planning.size()) {
        float ns_dx = planning[i + 2] - planning[i];
        float ns_dy = planning[i + 3] - planning[i + 1];
        float aw_dx = ns_dy;  // Autowareの座標系に変換
        float aw_dy = -ns_dx; // Autowareの座標系に変換
        float yaw = std::atan2(aw_dy, aw_dx);
        point.pose.orientation = createQuaternionFromYaw(yaw);
      }

      point.longitudinal_velocity_mps = 0.0;
      point.lateral_velocity_mps = 0.0;
      point.acceleration_mps2 = 0.0;
      point.heading_rate_rps = 0.0;

      trajectory_msg->points.push_back(point);
    }

    trajectory_msg->header.stamp = this->now();
    trajectory_msg->header.frame_id = "map";

    trajectory_publisher_->publish(std::move(trajectory_msg));
  }

  void
  publishDetectedObjects(const std::vector<std::vector<float>> &detections) {
    auto objects_msg =
        std::make_unique<autoware_perception_msgs::msg::DetectedObjects>();
    objects_msg->header.stamp = this->now();
    objects_msg->header.frame_id = "map";

    for (const auto &det : detections) {
      // det format: x, y, z, w, l, h, yaw, vx, vy, label, score
      autoware_perception_msgs::msg::DetectedObject object;

      object.kinematics.pose_with_covariance.pose.position.x = det[0];
      object.kinematics.pose_with_covariance.pose.position.y = det[1];
      object.kinematics.pose_with_covariance.pose.position.z = det[2];

      object.kinematics.pose_with_covariance.pose.orientation =
          createQuaternionFromYaw(det[6]);

      object.shape.dimensions.x = det[3]; // width
      object.shape.dimensions.y = det[4]; // length
      object.shape.dimensions.z = det[5]; // height

      object.kinematics.twist_with_covariance.twist.linear.x = det[7]; // vx
      object.kinematics.twist_with_covariance.twist.linear.y = det[8]; // vy

      autoware_perception_msgs::msg::ObjectClassification classification;
      classification.label = static_cast<uint8_t>(det[9]);
      classification.probability = det[10];
      object.classification.push_back(classification);

      objects_msg->objects.push_back(object);
    }

    objects_publisher_->publish(std::move(objects_msg));
  }

private:
  rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr
      trajectory_publisher_;
  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr
      objects_publisher_;
  std::vector<
      rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr>
      camera_subscribers_;

  // VadConfig
  autoware::tensorrt_vad::VadConfig vad_config_;

  // yamlファイルのパスからNodeOptionsを作成する静的関数
  static rclcpp::NodeOptions
  createNodeOptions(const std::vector<std::string> &yaml_config_paths) {
    rclcpp::NodeOptions node_options;
    std::vector<std::string> ros_args = {"--ros-args"};
    for (const auto &yaml_path : yaml_config_paths) {
      ros_args.push_back("--params-file");
      ros_args.push_back(yaml_path);
    }
    node_options.arguments(ros_args);
    return node_options;
  }

  void loadVadConfig() {
    // このノード自体からパラメータを読み込み
    vad_config_.plugins_path =
        this->declare_parameter<std::string>("model_params.plugins_path", "");
    vad_config_.warm_up_num =
        this->declare_parameter<int>("model_params.warm_up_num", 20);

    // ネットワーク設定の読み込み
    loadNetConfigs();
  }

  void loadNetConfigs() {
    // 階層構造でネットワーク設定を読み込み

    // backbone設定
    autoware::tensorrt_vad::NetConfig backbone_config;
    backbone_config.name = this->declare_parameter<std::string>(
        "model_params.nets.backbone.name", "backbone");
    backbone_config.engine_file = this->declare_parameter<std::string>(
        "model_params.nets.backbone.engine_file", "");
    backbone_config.use_graph = this->declare_parameter<bool>(
        "model_params.nets.backbone.use_graph", true);

    // head設定
    autoware::tensorrt_vad::NetConfig head_config;
    head_config.name = this->declare_parameter<std::string>(
        "model_params.nets.head.name", "head");
    head_config.engine_file = this->declare_parameter<std::string>(
        "model_params.nets.head.engine_file", "");
    head_config.use_graph =
        this->declare_parameter<bool>("model_params.nets.head.use_graph", true);

    // head inputsの読み込み
    std::string input_feature = this->declare_parameter<std::string>(
        "model_params.nets.head.inputs.input_feature", "mlvl_feats.0");
    std::string net_param = this->declare_parameter<std::string>(
        "model_params.nets.head.inputs.net", "backbone");
    std::string name_param = this->declare_parameter<std::string>(
        "model_params.nets.head.inputs.name", "out.0");
    head_config.inputs[input_feature]["net"] = net_param;
    head_config.inputs[input_feature]["name"] = name_param;

    // head_no_prev設定
    autoware::tensorrt_vad::NetConfig head_no_prev_config;
    head_no_prev_config.name = this->declare_parameter<std::string>(
        "model_params.nets.head_no_prev.name", "head_no_prev");
    head_no_prev_config.engine_file = this->declare_parameter<std::string>(
        "model_params.nets.head_no_prev.engine_file", "");
    head_no_prev_config.use_graph = this->declare_parameter<bool>(
        "model_params.nets.head_no_prev.use_graph", true);

    // head_no_prev inputsの読み込み
    std::string input_feature_no_prev = this->declare_parameter<std::string>(
        "model_params.nets.head_no_prev.inputs.input_feature", "mlvl_feats.0");
    std::string net_param_no_prev = this->declare_parameter<std::string>(
        "model_params.nets.head_no_prev.inputs.net", "backbone");
    std::string name_param_no_prev = this->declare_parameter<std::string>(
        "model_params.nets.head_no_prev.inputs.name", "out.0");
    head_no_prev_config.inputs[input_feature_no_prev]["net"] =
        net_param_no_prev;
    head_no_prev_config.inputs[input_feature_no_prev]["name"] =
        name_param_no_prev;

    vad_config_.nets_config.push_back(backbone_config);
    vad_config_.nets_config.push_back(head_config);
    vad_config_.nets_config.push_back(head_no_prev_config);
  }

  void onImageReceived(const sensor_msgs::msg::CompressedImage::SharedPtr msg,
                       int32_t camera_index) {
    // 現状は受信のみ
    RCLCPP_DEBUG(this->get_logger(), "Received image from camera %d",
                 camera_index);
  }

  geometry_msgs::msg::Quaternion createQuaternionFromYaw(double yaw) {
    geometry_msgs::msg::Quaternion q;
    q.x = 0.0;
    q.y = 0.0;
    q.z = std::sin(yaw * 0.5);
    q.w = std::cos(yaw * 0.5);
    return q;
  }
};

// バイナリ画像データをROS
// CompressedImageに変換し、その後TensorRTのバインディングに渡す関数
std::vector<float>
processImageForInference(const std::vector<std::vector<float>> &frame_images) {

  // 画像データを連結
  std::vector<float> concatenated_data;
  size_t single_camera_size = 3 * 384 * 640;
  concatenated_data.reserve(single_camera_size * 6);

  // カメラの順序: {0, 1, 2, 3, 4, 5}
  for (int camera_idx = 0; camera_idx < 6; ++camera_idx) {
    const auto &img_data = frame_images[camera_idx];
    if (img_data.size() != single_camera_size) {
      throw std::runtime_error("画像サイズが不正です: " +
                               std::to_string(camera_idx));
    }
    concatenated_data.insert(concatenated_data.end(), img_data.begin(),
                             img_data.end());
  }

  return concatenated_data;
}

class Logger : public nvinfer1::ILogger {
public:
  void log(nvinfer1::ILogger::Severity severity,
           const char *msg) noexcept override {
    // Only print error messages
    if (severity == nvinfer1::ILogger::Severity::kERROR) {
      std::cerr << msg << std::endl;
    }
  }
};

Logger gLogger;

class EventTimer {
public:
  EventTimer() {
    cudaEventCreate(&begin_);
    cudaEventCreate(&end_);
  }

  virtual ~EventTimer() {
    cudaEventDestroy(begin_);
    cudaEventDestroy(end_);
  }

  void start(cudaStream_t stream) { cudaEventRecord(begin_, stream); }

  void end(cudaStream_t stream) { cudaEventRecord(end_, stream); }

  float report(const std::string &prefix = "timer") {
    float times = 0;
    cudaEventSynchronize(end_);
    cudaEventElapsedTime(&times, begin_, end_);
    printf("[TIMER:  %s]: \t%.5f ms\n", prefix.c_str(), times);
    return times;
  }

private:
  cudaEvent_t begin_ = nullptr, end_ = nullptr;
};

std::vector<std::vector<float>>
load_image_from_rosbag_single_frame(const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> &images, int frame_id) {

  std::vector<std::vector<float>> frame_images;
  frame_images.resize(6); // VADカメラ順序で初期化

  // AutowareカメラインデックスからVADカメラインデックスへのマッピング
  std::unordered_map<int, int> autoware_to_vad = {
      {0, 0}, // FRONT
      {4, 1}, // FRONT_RIGHT
      {2, 2}, // FRONT_LEFT
      {1, 3}, // BACK
      {3, 4}, // BACK_LEFT
      {5, 5}  // BACK_RIGHT
  };

  // 目標の画像サイズを定義
  const int32_t target_width = 640;
  const int32_t target_height = 384;

  // 正規化のパラメータ
  float mean[3] = {103.530f, 116.280f, 123.675f};
  float std[3] = {1.0f, 1.0f, 1.0f};

  // 各カメラの画像を処理
  for (int autoware_idx = 0; autoware_idx < 6; ++autoware_idx) {
    const auto &image_msg = images[autoware_idx];

    // 画像データの処理
    int32_t width, height, channels;
    unsigned char *image_data = stbi_load_from_memory(
        image_msg->data.data(), static_cast<int>(image_msg->data.size()),
        &width, &height, &channels, STBI_rgb); // RGBとして読み込む

    // サイズが目標と違う場合はリサイズする
    unsigned char *resized_data = nullptr;
    if (width != target_width || height != target_height) {
      // stb_image_resizeを使用してリサイズ
      resized_data =
          (unsigned char *)malloc(target_width * target_height * channels);

      // stb_image_resizeを使ってリサイズ
      int resize_result =
          stbir_resize_uint8(image_data, width, height, 0, resized_data,
                              target_width, target_height, 0, channels);

      // 元のデータを解放し、リサイズしたデータを使用
      stbi_image_free(image_data);
      image_data = resized_data;
      width = target_width;
      height = target_height;
    }

    // BGRの順で処理
    std::vector<float> normalized_image_data(width * height * 3);
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          int32_t src_idx = (h * width + w) * 3 + (2 - c); // BGR -> RGB
          int32_t dst_idx = c * height * width + h * width + w; // CHW形式
          float pixel_value = static_cast<float>(image_data[src_idx]);
          normalized_image_data[dst_idx] = (pixel_value - mean[c]) / std[c];
        }
      }
    }

    // VADカメラ順序で格納
    int vad_idx = autoware_to_vad[autoware_idx];
    frame_images[vad_idx] = normalized_image_data;
  }

  return frame_images;
}

// 元の関数は後方互換性のために残す
std::unordered_map<int, std::vector<std::vector<float>>>
load_image_from_rosbag(const std::vector<autoware::tensorrt_vad::VadTopicData>
                           &vad_topic_data_list) {

  std::unordered_map<int, std::vector<std::vector<float>>>
      subscribed_image_dict;

  for (size_t frame_id = 0; frame_id < vad_topic_data_list.size();
       ++frame_id) {
    subscribed_image_dict[frame_id + 1] = load_image_from_rosbag_single_frame(vad_topic_data_list[frame_id].images, frame_id);
  }

  return subscribed_image_dict;
}

void compare_with_reference(
    const std::vector<std::vector<float>> &subscribed_images,
    const std::string &reference_path) {

  // リファレンス画像データの読み込み
  std::vector<float> reference_data;
  std::ifstream ref_file(reference_path, std::ios::binary);
  if (!ref_file) {
    throw std::runtime_error("リファレンスファイルを開けません: " +
                             reference_path);
  }

  // ファイルサイズを取得
  ref_file.seekg(0, std::ios::end);
  size_t file_size = ref_file.tellg();
  ref_file.seekg(0, std::ios::beg);

  // メモリを確保してデータを読み込み
  reference_data.resize(file_size / sizeof(float));
  ref_file.read(reinterpret_cast<char *>(reference_data.data()), file_size);

  // リファレンスデータを6カメラ×(3, 384, 640)の形式に変換
  size_t single_camera_size = 3 * 384 * 640;
  if (reference_data.size() != single_camera_size * 6) {
    throw std::runtime_error("リファレンスデータのサイズが不正です");
  }

  // 各カメラの画像データを比較
  for (size_t cam_idx = 0; cam_idx < 6; ++cam_idx) {
    const auto &subscribed_img = subscribed_images[cam_idx];

    // サイズチェック
    if (subscribed_img.size() != single_camera_size) {
      throw std::runtime_error("購読画像のサイズが不正です: カメラ " +
                               std::to_string(cam_idx));
    }

    // 画素値の比較
    size_t start_idx = cam_idx * single_camera_size;
    float max_diff = 0.0f;
    int32_t diff_count = 0;
    constexpr float tolerance = 38.6f;

    for (size_t i = 0; i < single_camera_size; ++i) {
      float diff = std::abs(subscribed_img[i] - reference_data[start_idx + i]);
      if (diff > tolerance) {
        diff_count++;
        max_diff = std::max(max_diff, diff);
      }
    }

    if (diff_count > 0) {
      RCLCPP_WARN(rclcpp::get_logger("compare_images"),
                  "カメラ %zu: 許容誤差を超える差異が %d 箇所で検出されました "
                  "(最大差: %f)",
                  cam_idx, diff_count, max_diff);
    }
  }
}

std::vector<float> calculateShift(float delta_x, float delta_y,
                                  float patch_angle_rad) {
  const float point_cloud_range[] = {-15.0, -30.0, -2.0, 15.0, 30.0, 2.0};
  const int32_t bev_h_ = 100;
  const int32_t bev_w_ = 100;

  float real_w = point_cloud_range[3] - point_cloud_range[0];
  float real_h = point_cloud_range[4] - point_cloud_range[1];
  float grid_length[] = {real_h / bev_h_, real_w / bev_w_};

  float ego_angle = patch_angle_rad / M_PI * 180.0;
  float grid_length_y = grid_length[0];
  float grid_length_x = grid_length[1];

  float translation_length = std::sqrt(delta_x * delta_x + delta_y * delta_y);
  float translation_angle = std::atan2(delta_y, delta_x) / M_PI * 180.0;
  float bev_angle = ego_angle - translation_angle;

  float shift_y = translation_length * std::cos(bev_angle / 180.0 * M_PI) /
                  grid_length_y / bev_h_;
  float shift_x = translation_length * std::sin(bev_angle / 180.0 * M_PI) /
                  grid_length_x / bev_w_;

  return {shift_x, shift_y};
}

std::pair<std::vector<float>, std::vector<float>>
load_can_bus_shift_from_rosbag_single_frame(
    const nav_msgs::msg::Odometry::ConstSharedPtr &kinematic_state,
    const sensor_msgs::msg::Imu::ConstSharedPtr &imu_raw,
    int frame_id,
    const std::vector<float> &prev_can_bus = {}) {

  // default patch_angle
  float default_patch_angle = -1.0353195667266846f;

  std::vector<float> can_bus(18, 0.0f);
  std::vector<float> shift(2, 0.0f);

  // Apply Autoware to nuScenes coordinate transformation to position
  auto [ns_x, ns_y] =
      aw2ns_xy(kinematic_state->pose.pose.position.x,
                kinematic_state->pose.pose.position.y);

  std::vector<float> translation = {
      ns_x, ns_y,
      static_cast<float>(
          kinematic_state->pose.pose.position.z)};

  // Apply Autoware to nuScenes coordinate transformation to orientation
  Eigen::Quaternionf q_aw(
      kinematic_state->pose.pose.orientation.w,
      kinematic_state->pose.pose.orientation.x,
      kinematic_state->pose.pose.orientation.y,
      kinematic_state->pose.pose.orientation.z);

  Eigen::Quaternionf q_ns = aw2ns_quaternion(q_aw);

  std::vector<float> rotation = {q_ns.x(), q_ns.y(), q_ns.z(), q_ns.w()};

  // Apply Autoware to nuScenes coordinate transformation to velocity
  auto [ns_vx, ns_vy] =
      aw2ns_xy(kinematic_state->twist.twist.linear.x,
                kinematic_state->twist.twist.linear.y);

  std::vector<float> velocity = {
      ns_vx, ns_vy,
      static_cast<float>(
          kinematic_state->twist.twist.linear.z)};

  // Apply Autoware to nuScenes coordinate transformation to angular
  // velocity
  auto [ns_wx, ns_wy] =
      aw2ns_xy(kinematic_state->twist.twist.angular.x,
                kinematic_state->twist.twist.angular.y);

  std::vector<float> angular_velocity = {
      ns_wx, ns_wy,
      static_cast<float>(
          kinematic_state->twist.twist.angular.z)};

  // Apply Autoware to nuScenes coordinate transformation to acceleration
  auto [ns_ax, ns_ay] =
      aw2ns_xy(imu_raw->linear_acceleration.x,
                imu_raw->linear_acceleration.y);

  std::vector<float> acceleration = {
      ns_ax, ns_ay,
      static_cast<float>(imu_raw->linear_acceleration.z)};

  // can_busデータの構築（18次元ベクトル）

  // translation (0:3)
  std::copy(translation.begin(), translation.end(), can_bus.begin());

  // rotation (3:7)
  std::copy(rotation.begin(), rotation.end(), can_bus.begin() + 3);

  // acceleration (7:10)
  std::copy(acceleration.begin(), acceleration.end(), can_bus.begin() + 7);

  // angular velocity (10:13)
  std::copy(angular_velocity.begin(), angular_velocity.end(),
            can_bus.begin() + 10);

  // velocity (13:16)
  std::copy(velocity.begin(), velocity.begin() + 2, can_bus.begin() + 13);
  can_bus[15] = 0.0f; // z方向の速度は0とする

  // patch_angle[rad]の計算 (16)
  double yaw = std::atan2(
      2.0 * (can_bus[6] * can_bus[5] + can_bus[3] * can_bus[4]),
      1.0 - 2.0 * (can_bus[4] * can_bus[4] + can_bus[5] * can_bus[5]));
  if (yaw < 0)
    yaw += 2 * M_PI;
  can_bus[16] = static_cast<float>(yaw);

  // patch_angle[deg]の計算 (17)
  if (frame_id > 0 && !prev_can_bus.empty()) {
    float prev_angle = prev_can_bus[16];
    can_bus[17] = (yaw - prev_angle) * 180.0f / M_PI;
  } else {
    can_bus[17] = default_patch_angle; // 最初のフレームのデフォルト値
  }

  // シフトデータの計算
  if (frame_id > 0 && !prev_can_bus.empty()) {
    float delta_x = translation[0] - prev_can_bus[0];
    float delta_y = translation[1] - prev_can_bus[1];

    shift = calculateShift(delta_x, delta_y, yaw);
  } else {
    shift = {0.0f, 0.0f};
  }

  return std::make_pair(can_bus, shift);
}

// 元の関数は後方互換性のために残す
std::tuple<std::unordered_map<int, std::vector<float>>,
           std::unordered_map<int, std::vector<float>>>
load_can_bus_shift_from_rosbag(
    const std::vector<autoware::tensorrt_vad::VadTopicData>
        &vad_topic_data_list) {
  std::unordered_map<int, std::vector<float>> can_bus_dict;
  std::unordered_map<int, std::vector<float>> shift_dict;

  std::vector<float> prev_can_bus;
  for (size_t frame_id = 0; frame_id < vad_topic_data_list.size();
       ++frame_id) {
    auto [can_bus, shift] = load_can_bus_shift_from_rosbag_single_frame(
        vad_topic_data_list[frame_id].kinematic_state,
        vad_topic_data_list[frame_id].imu_raw,
        frame_id, prev_can_bus);
    
    can_bus_dict[frame_id + 1] = can_bus;
    shift_dict[frame_id + 1] = shift;
    prev_can_bus = can_bus;
  }

  return std::make_tuple(can_bus_dict, shift_dict);
}

std::optional<int> extract_autoware_camera_id(
    const std::string &topic_name,
    const std::unordered_map<int, int> &vad_to_autoware_camera) {

  for (const auto &[vad_id, autoware_id] : vad_to_autoware_camera) {
    std::string camera_topic =
        "/sensing/camera/camera" + std::to_string(autoware_id) + "/camera_info";
    if (topic_name == camera_topic) {
      return autoware_id;
    }
  }
  return std::nullopt;
}

std::vector<float> load_lidar2img_from_rosbag_single_frame(
    const tf2_msgs::msg::TFMessage::ConstSharedPtr &tf_static,
    const std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> &camera_infos,
    int frame_id,
    float scale_width, float scale_height) {
  
  std::vector<float> frame_lidar2img(16 * 6, 0.0f); // 6カメラ分のスペースを確保

  // AutowareカメラインデックスからVADカメラインデックスへのマッピング
  std::unordered_map<int, int> autoware_to_vad = {
      {0, 0}, // FRONT
      {4, 1}, // FRONT_RIGHT
      {2, 2}, // FRONT_LEFT
      {1, 3}, // BACK
      {3, 4}, // BACK_LEFT
      {5, 5}  // BACK_RIGHT
  };

  // 各カメラのTF変換を処理
  for (const auto &transform : tf_static->transforms) {
    std::string child_frame_id = transform.child_frame_id;

    if (child_frame_id.find("camera") != std::string::npos &&
        child_frame_id.find("/camera_optical_link") != std::string::npos) {
      // Autowareカメラ名からカメラIDを抽出
      int32_t autoware_camera_id = std::stoi(
          child_frame_id.substr(child_frame_id.find("camera") + 6, 1));

      // カメラの内部パラメータを確認
      if (autoware_camera_id >= 0 && autoware_camera_id < 6 &&
          camera_infos[autoware_camera_id]) {

        // カメラ行列Kを3x3行列として抽出
        Eigen::Matrix3f k_matrix;
        const auto &camera_info = camera_infos[autoware_camera_id];
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            k_matrix(i, j) = camera_info->k[i * 3 + j];
          }
        }

        // 変換行列の構築
        Eigen::Vector3f aw_translation(transform.transform.translation.x,
                                        transform.transform.translation.y,
                                        transform.transform.translation.z);

        // Apply Autoware to nuScenes coordinate transformation to
        // translation
        auto [ns_x, ns_y] = aw2ns_xy(aw_translation[0], aw_translation[1]);
        Eigen::Vector3f ns_translation(ns_x, ns_y, aw_translation[2]);

        Eigen::Quaternionf q_aw(
            transform.transform.rotation.w, transform.transform.rotation.x,
            transform.transform.rotation.y, transform.transform.rotation.z);

        // Apply Autoware to nuScenes coordinate transformation to
        // quaternion
        Eigen::Quaternionf q_ns = aw2ns_quaternion(q_aw);

        // lidar2cam_rtを構築 (nuScenes座標系)
        Eigen::Matrix4f lidar2cam_rt = Eigen::Matrix4f::Identity();
        lidar2cam_rt.block<3, 3>(0, 0) = q_ns.toRotationMatrix();
        lidar2cam_rt.block<3, 1>(0, 3) = ns_translation;

        // lidar2cam_rt.Tを計算
        Eigen::Matrix4f lidar2cam_rt_T = lidar2cam_rt.transpose();

        // viewpadを作成
        Eigen::Matrix4f viewpad = Eigen::Matrix4f::Zero();
        viewpad.block<3, 3>(0, 0) = k_matrix;
        viewpad(3, 3) = 1.0f;

        // lidar2img = viewpad @ lidar2cam_rt.T を計算
        Eigen::Matrix4f lidar2img = viewpad * lidar2cam_rt_T;

        // スケーリングを適用
        Eigen::Matrix4f scale_matrix = Eigen::Matrix4f::Identity();
        scale_matrix(0, 0) = scale_width;
        scale_matrix(1, 1) = scale_height;

        lidar2img = scale_matrix * lidar2img;

        // 結果を格納
        std::vector<float> lidar2img_flat(16);
        int32_t k = 0;
        for (int i = 0; i < 4; ++i) {
          for (int j = 0; j < 4; ++j) {
            lidar2img_flat[k++] = lidar2img(i, j);
          }
        }

        // lidar2imgの計算後、VADカメラIDの位置に格納
        int vad_camera_id = autoware_to_vad[autoware_camera_id];
        if (vad_camera_id >= 0 && vad_camera_id < 6) {
          std::copy(lidar2img_flat.begin(), lidar2img_flat.end(),
                    frame_lidar2img.begin() + vad_camera_id * 16);
        }
      }
    }
  }

  return frame_lidar2img;
}

// 元の関数は後方互換性のために残す
std::unordered_map<int, std::vector<float>> load_lidar2img_from_rosbag(
    const std::vector<autoware::tensorrt_vad::VadTopicData>
        &vad_topic_data_list,
    float scale_width, float scale_height) {
  std::unordered_map<int, std::vector<float>> result;

  for (size_t frame_id = 0; frame_id < vad_topic_data_list.size();
       ++frame_id) {
    result[frame_id + 1] = load_lidar2img_from_rosbag_single_frame(
        vad_topic_data_list[frame_id].tf_static,
        vad_topic_data_list[frame_id].camera_infos,
        frame_id, scale_width, scale_height);
  }

  return result;
}

void compare_with_reference_lidar2img(const std::vector<float> &lidar2img_data,
                                      const std::string &reference_file_path) {
  std::ifstream file(reference_file_path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    std::cerr << "参照ファイルを開けませんでした: " << reference_file_path
              << std::endl;
    throw std::runtime_error("参照ファイルを開けませんでした");
  }

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  if (size != lidar2img_data.size() * sizeof(float)) {
    std::cerr << "参照ファイルのサイズが一致しません: " << reference_file_path
              << std::endl;
    std::cerr << "期待されるサイズ: " << lidar2img_data.size()
              << " バイト, 実際のサイズ: " << size << " バイト" << std::endl;
    throw std::runtime_error("参照ファイルのサイズが一致しません");
  }

  std::vector<float> reference_data(lidar2img_data.size());
  if (!file.read(reinterpret_cast<char *>(reference_data.data()), size)) {
    std::cerr << "参照ファイルの読み込みに失敗しました: " << reference_file_path
              << std::endl;
    throw std::runtime_error("参照ファイルの読み込みに失敗しました");
  }

  for (size_t i = 0; i < lidar2img_data.size(); ++i) {
    if (std::abs(lidar2img_data[i] - reference_data[i]) > 1e-3) {
      std::cerr << "値が一致しません: " << reference_file_path
                << " のインデックス " << i << std::endl;
      std::cerr << "binファイルの値: " << reference_data[i]
                << ", lidar2imgの値: " << lidar2img_data[i] << std::endl;
      throw std::runtime_error("値が一致しません");
    }
  }

  std::cout << "lidar2imgデータは参照ファイルと一致しています: "
            << reference_file_path << std::endl;
}

std::vector<autoware::tensorrt_vad::VadTopicData>
extract_vad_topic_data_from_rosbag(const std::string &bag_path) {

  std::vector<autoware::tensorrt_vad::VadTopicData> vad_topic_data_list;

  try {
    // ROSバッグの設定
    rosbag2_storage::StorageOptions storage_options;
    storage_options.uri = bag_path;
    storage_options.storage_id = "sqlite3";
    rosbag2_cpp::ConverterOptions converter_options;
    converter_options.input_serialization_format = "cdr";
    converter_options.output_serialization_format = "cdr";
    rosbag2_cpp::readers::SequentialReader reader;
    reader.open(storage_options, converter_options);

    autoware::tensorrt_vad::VadTopicData current_frame;
    bool frame_started = false;

    while (reader.has_next()) {
      auto bag_message = reader.read_next();

      // 画像トピックの処理
      for (int autoware_idx = 0; autoware_idx < 6; ++autoware_idx) {
        std::string image_topic = "/sensing/camera/camera" +
                                  std::to_string(autoware_idx) +
                                  "/image_rect_color/compressed";
        if (bag_message->topic_name == image_topic) {
          auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
          rclcpp::SerializedMessage serialized_msg(
              *bag_message->serialized_data);
          rclcpp::Serialization<sensor_msgs::msg::CompressedImage>()
              .deserialize_message(&serialized_msg, msg.get());

          // CompressedImageをImageに変換
          auto image_msg = std::make_shared<sensor_msgs::msg::Image>();
          image_msg->header = msg->header;
          image_msg->height = msg->format.find("height=") != std::string::npos
                                  ? std::stoi(msg->format.substr(
                                        msg->format.find("height=") + 7,
                                        msg->format.find(";") -
                                            msg->format.find("height=") - 7))
                                  : 0;
          image_msg->width =
              msg->format.find("width=") != std::string::npos
                  ? std::stoi(msg->format.substr(
                        msg->format.find("width=") + 6,
                        msg->format.find(";") - msg->format.find("width=") - 6))
                  : 0;
          image_msg->encoding = "bgr8";
          image_msg->data = msg->data;
          image_msg->step = image_msg->width * 3;

          if (!frame_started) {
            current_frame.stamp = msg->header.stamp;
            current_frame.images.resize(6);
            current_frame.camera_infos.resize(6);
            frame_started = true;
          }

          current_frame.images[autoware_idx] = image_msg;
        }
      }

      // カメラ情報トピックの処理
      for (int autoware_idx = 0; autoware_idx < 6; ++autoware_idx) {
        std::string camera_info_topic = "/sensing/camera/camera" +
                                        std::to_string(autoware_idx) +
                                        "/camera_info";
        if (bag_message->topic_name == camera_info_topic) {
          auto msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
          rclcpp::SerializedMessage serialized_msg(
              *bag_message->serialized_data);
          rclcpp::Serialization<sensor_msgs::msg::CameraInfo>()
              .deserialize_message(&serialized_msg, msg.get());

          if (!frame_started) {
            current_frame.stamp = msg->header.stamp;
            current_frame.images.resize(6);
            current_frame.camera_infos.resize(6);
            frame_started = true;
          }

          current_frame.camera_infos[autoware_idx] = msg;
        }
      }

      // TF staticトピックの処理
      if (bag_message->topic_name == "/tf_static") {
        auto msg = std::make_shared<tf2_msgs::msg::TFMessage>();
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
        rclcpp::Serialization<tf2_msgs::msg::TFMessage>().deserialize_message(
            &serialized_msg, msg.get());

        if (!frame_started) {
          current_frame.stamp = rclcpp::Time(0);
          current_frame.images.resize(6);
          current_frame.camera_infos.resize(6);
          frame_started = true;
        }

        current_frame.tf_static = msg;
      }

      // 運動状態トピックの処理
      if (bag_message->topic_name == "/localization/kinematic_state") {
        auto msg = std::make_shared<nav_msgs::msg::Odometry>();
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
        rclcpp::Serialization<nav_msgs::msg::Odometry>().deserialize_message(
            &serialized_msg, msg.get());

        if (!frame_started) {
          current_frame.stamp = msg->header.stamp;
          current_frame.images.resize(6);
          current_frame.camera_infos.resize(6);
          frame_started = true;
        }

        current_frame.kinematic_state = msg;
      }

      // IMUトピックの処理
      if (bag_message->topic_name == "/sensing/imu/tamagawa/imu_raw") {
        auto msg = std::make_shared<sensor_msgs::msg::Imu>();
        rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
        rclcpp::Serialization<sensor_msgs::msg::Imu>().deserialize_message(
            &serialized_msg, msg.get());

        if (!frame_started) {
          current_frame.stamp = msg->header.stamp;
          current_frame.images.resize(6);
          current_frame.camera_infos.resize(6);
          frame_started = true;
        }

        current_frame.imu_raw = msg;
      }

      // フレームが完成したらリストに追加
      if (frame_started && current_frame.is_complete()) {
        vad_topic_data_list.push_back(current_frame);

        // 次のフレームの準備
        current_frame = autoware::tensorrt_vad::VadTopicData();
        frame_started = false;
      }
    }

  } catch (const std::exception &e) {
    std::cerr << "ROSバッグの読み込み中にエラーが発生: " << e.what()
              << std::endl;
    throw;
  }

  return vad_topic_data_list;
}

int main(int argc, char **argv) {
  // ROSの初期化
  rclcpp::init(argc, argv);

  printf("nvinfer: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR,
         NV_TENSORRT_PATCH);
  cudaSetDevice(0);

  const std::string config = argv[1];
  fs::path cfg_pth = config;
  fs::path cfg_dir = cfg_pth.parent_path();
  printf("[INFO] setting up from %s\n", config.c_str());
  printf("[INFO] assuming data dir is %s\n", cfg_dir.string().c_str());

  std::ifstream f(config);
  json cfg = json::parse(f);

  // jsonからyamlパスを読み込み
  std::vector<std::string> yaml_config_paths;
  for (const auto &yaml_path : cfg["yaml_config_paths"]) {
    yaml_config_paths.push_back(yaml_path.get<std::string>());
  }

  // VADNodeを作成（yamlパスを渡す）
  auto node = std::make_shared<VADNode>(yaml_config_paths);

  // VADNodeからVadConfigを取得
  const auto &vad_config = node->getVadConfig();

  // VadModelを初期化
  auto ros_logger =
      std::make_shared<autoware::tensorrt_vad::RosVadLogger>(node);
  autoware::tensorrt_vad::VadModel vad_model(vad_config, ros_logger);

  EventTimer timer;
  std::string data_dir = cfg_dir.string() + "/data/";
  int32_t n_frames = cfg["n_frames"];
  printf("[INFO] n_frames=%d\n", n_frames);
  std::vector<float> lidar2img;

  // commandパラメータを読み込む
  int32_t default_command =
      node->declare_parameter<int>("model_params.default_command", 0);
  printf("[INFO] default_command=%d\n", default_command);

  auto vad_topic_data_list = extract_vad_topic_data_from_rosbag(
      "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/"
      "app/demo/rosbag/output_bag/");
  
  int32_t input_image_width = cfg["input_image_width"];
  int32_t input_image_height = cfg["input_image_hight"];
  float scale_width = 640.0f / static_cast<float>(input_image_width);
  float scale_height = 384.0f / static_cast<float>(input_image_height);
  
  // フレームごとに処理
  std::vector<float> prev_can_bus;
  for (int frame_id = 1; frame_id <= vad_topic_data_list.size(); frame_id++) {
    std::string frame_dir = data_dir + std::to_string(frame_id) + "/";

    // 各フレームのデータを個別に読み込み
    auto frame_images = load_image_from_rosbag_single_frame(
        vad_topic_data_list[frame_id - 1].images, frame_id - 1);
    
    auto [frame_can_bus, frame_shift] = load_can_bus_shift_from_rosbag_single_frame(
        vad_topic_data_list[frame_id - 1].kinematic_state,
        vad_topic_data_list[frame_id - 1].imu_raw,
        frame_id - 1, prev_can_bus);
    
    auto frame_lidar2img = load_lidar2img_from_rosbag_single_frame(
        vad_topic_data_list[frame_id - 1].tf_static,
        vad_topic_data_list[frame_id - 1].camera_infos,
        frame_id - 1, scale_width, scale_height);

    // 前フレームのcan_busデータを更新
    prev_can_bus = frame_can_bus;

    // 画像をconcatenate
    auto image_data = processImageForInference(frame_images);
    autoware::tensorrt_vad::VadInputData vad_input_data{
        image_data,      // camera_images_
        frame_shift,     // shift_
        frame_lidar2img, // lidar2img_
        frame_can_bus,   // can_bus_
        default_command  // command_
    };

    // VadModelのinfer関数を使用
    auto inference_result = vad_model.infer(vad_input_data);
    if (!inference_result.has_value()) {
      std::cerr << "Inference failed for frame " << frame_id << std::endl;
      continue;
    }

    cudaStreamSynchronize(vad_model.stream_);

    auto vad_output_data = inference_result.value();

    std::string viz_dir = cfg["viz"];
    viz_dir = cfg_dir.string() + "/" + viz_dir;

    std::vector<unsigned char *> images;

    // 各カメラからの画像をRGBに変換
    for (int cam_idx = 0; cam_idx < 6; ++cam_idx) {
      const auto &normalized_img = frame_images[cam_idx];
      unsigned char *rgb_img = convert_normalized_to_rgb(normalized_img);
      images.push_back(rgb_img);
    }
    std::string font_path =
        cfg_dir.string() + "/" + cfg["font_path"].get<std::string>();

    nv::VisualizeFrame frame;
    frame.cmd =
        vad_input_data
            .command_; // VadInputDataのcommand_の値（2 = "KEEP FORWARD"）を使用

    frame.img_metas_lidar2img = vad_model.nets_["head"]
                                    ->bindings["img_metas.0[lidar2img]"]
                                    ->cpu<float>();

    // pred -> frame.planning
    frame.planning = vad_output_data.predicted_trajectory_;
    node->publishTrajectory(frame.planning);
    printf("publish trajectory");
    rclcpp::spin_some(node);

    std::vector<float> bbox_preds =
        vad_model.nets_["head"]->bindings["out.all_bbox_preds"]->cpu<float>();
    std::vector<float> cls_scores =
        vad_model.nets_["head"]->bindings["out.all_cls_scores"]->cpu<float>();

    // det to frame.det
    constexpr int32_t N_MAX_DET = 300;
    for (int32_t d = 0; d < N_MAX_DET; d++) {
      // 3, 1, 100, 10
      std::vector<float> box_score(cls_scores.begin() + d * 10,
                                   cls_scores.begin() + d * 10 + 10);
      float max_score = -1;
      int32_t max_label = -1;
      for (int32_t l = 0; l < 10; l++) {
        // sigmoid
        float this_score = 1.0f / (1.0f + std::exp(-box_score[l]));
        if (this_score > max_score) {
          max_score = this_score;
          max_label = l;
        }
      }
      if (max_score > 0.35) {
        // from: cx, cy, w, l, cz, h, sin, cos, vx, vy
        //   to:  x,  y, z, w,  l, h, yaw, vx, vy, label, score
        std::vector<float> raw(bbox_preds.begin() + d * 10,
                               bbox_preds.begin() + d * 10 + 10);
        std::vector<float> ret(11);
        ret[0] = raw[0];
        ret[1] = raw[1];
        ret[2] = raw[4];
        ret[3] = std::exp(raw[2]);
        ret[4] = std::exp(raw[3]);
        ret[5] = std::exp(raw[5]);
        ret[6] = std::atan2(raw[6], raw[7]);
        ret[7] = raw[8];
        ret[8] = raw[9];
        ret[9] = (float)max_label;
        ret[10] = max_score;
        frame.det.push_back(ret);
      }
    }
    node->publishDetectedObjects(frame.det);
    nv::visualize(images, frame, font_path,
                  viz_dir + "/" + std::to_string(frame_id) + ".jpg",
                  vad_model.stream_);

    printf("[INFO] %d, cmd=%d finished\n", frame_id, frame.cmd);
  }

  int32_t perf_loop = cfg.value("perf_loop", 0);
  if (perf_loop > 0) {
    printf("[INFO] running %d rounds of perf_loop\n", perf_loop);
  }
  for (int32_t i = 0; i < perf_loop; i++) {
    timer.start(vad_model.stream_);
    vad_model.nets_["backbone"]->Enqueue(vad_model.stream_);
    vad_model.nets_["head"]->Enqueue(vad_model.stream_);
    timer.end(vad_model.stream_);
    cudaStreamSynchronize(vad_model.stream_);
    timer.report("vad-trt");
  }

  cudaStreamSynchronize(vad_model.stream_);

  // ROSのシャットダウン
  rclcpp::shutdown();
  // dlclose(so_handle);
  return 0;
}
