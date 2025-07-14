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

#include "autoware/tensorrt_vad/ros_vad_logger.hpp"
#include "autoware/tensorrt_vad/vad_interface.hpp"
#include "autoware/tensorrt_vad/vad_model.hpp"
#include "visualize.hpp"

#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>

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
  autoware::tensorrt_vad::VadInterfaceConfig vad_interface_config_;
  std::vector<float> prev_can_bus_;

  VADNode(const std::vector<std::string> &yaml_config_paths)
      : Node("vad_node", createNodeOptions(yaml_config_paths)),
        vad_interface_config_(
          declare_parameter<int32_t>("interface_params.input_image_width"),
          declare_parameter<int32_t>("interface_params.input_image_height"),
          declare_parameter<int32_t>("interface_params.target_image_width"),
          declare_parameter<int32_t>("interface_params.target_image_height"),
          declare_parameter<std::vector<double>>("interface_params.point_cloud_range"),
          declare_parameter<int32_t>("interface_params.bev_h"),
          declare_parameter<int32_t>("interface_params.bev_w"),
          declare_parameter<double>("interface_params.default_patch_angle"),
          declare_parameter<int32_t>("model_params.default_command"),
          declare_parameter<std::vector<double>>("interface_params.default_shift"),
          declare_parameter<std::vector<double>>("interface_params.image_normalization_param_mean"),
          declare_parameter<std::vector<double>>("interface_params.image_normalization_param_std"),
          declare_parameter<std::vector<double>>("interface_params.vad2base"),
          declare_parameter<std::vector<int64_t>>("interface_params.autoware_to_vad_camera_mapping")
        )
  {

    std::vector<double> default_can_bus = this->declare_parameter<std::vector<double>>("interface_params.default_can_bus");
    // default_can_bus: copy and convert
    prev_can_bus_.clear();
    for (auto v : default_can_bus) prev_can_bus_.push_back(static_cast<float>(v));
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

std::vector<autoware::tensorrt_vad::VadInputTopicData>
extract_vad_topic_data_from_rosbag(const std::string &bag_path, std::shared_ptr<tf2_ros::Buffer> tf_buffer) {

  std::vector<autoware::tensorrt_vad::VadInputTopicData> vad_topic_data_list;

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

    autoware::tensorrt_vad::VadInputTopicData current_frame;
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

        // base_link -> camera の変換をバッファに登録
        for (const auto &transform : msg->transforms) {
          tf_buffer->setTransform(transform, "default_authority", true);
        }
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
        current_frame = autoware::tensorrt_vad::VadInputTopicData();
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


  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto vad_topic_data_list = extract_vad_topic_data_from_rosbag(
      "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/"
      "app/demo/rosbag/output_bag/", tf_buffer);
  // Read visualization configuration
  float lidar_z_compensation = cfg["lidar_z_compensation"].get<float>();
  float init_lidar_y = cfg["visualization"]["init_lidar_y"].get<float>();
  float ground_height = cfg["visualization"]["ground_height"].get<float>();

  autoware::tensorrt_vad::VadInterface vad_interface(node->vad_interface_config_, tf_buffer);

  
  // フレームごとに処理
  std::vector<float> prev_can_bus = node->prev_can_bus_;
  for (int32_t frame_id = 1; frame_id <= vad_topic_data_list.size(); frame_id++) {
    std::string frame_dir = data_dir + std::to_string(frame_id) + "/";

    // VadInterfaceを使用してVadInputTopicDataをVadInputDataに変換（古いprev_can_busを使用）
    auto vad_input_data = vad_interface.convert_input(
        vad_topic_data_list[frame_id - 1],
        prev_can_bus);

    // 前フレームのcan_busデータを更新（次のフレーム用）
    prev_can_bus = vad_input_data.can_bus_;

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

    // VadInterfaceから個別のカメラ画像を取得するために、
    // concatenatedデータから個別カメラ画像を抽出
    size_t single_camera_size = 3 * 384 * 640;
    for (int32_t cam_idx = 0; cam_idx < 6; ++cam_idx) {
      // 個別カメラ画像データを抽出
      std::vector<float> individual_camera_data(
          vad_input_data.camera_images_.begin() + cam_idx * single_camera_size,
          vad_input_data.camera_images_.begin() + (cam_idx + 1) * single_camera_size);
      
      unsigned char *rgb_img = convert_normalized_to_rgb(individual_camera_data);
      images.push_back(rgb_img);
    }
    std::string font_path =
        cfg_dir.string() + "/" + cfg["font_path"].get<std::string>();

    nv::VisualizeFrame frame;
    frame.cmd = vad_input_data.command_;

    frame.img_metas_lidar2img = vad_input_data.lidar2img_;

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
        ret[2] = raw[4] + lidar_z_compensation;  // postprocessでlidar_z_compensationを適用
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
    // if frame is 13, print frame.det
    if (frame_id == 13) {
      for (const auto &det : frame.det) {
        std::cout << "det: " << det[0] << ", " << det[1] << ", " << det[2] << ", " << det[3] << ", " << det[4] << ", " << det[5] << ", " << det[6] << ", " << det[7] << ", " << det[8] << ", " << det[9] << ", " << det[10] << std::endl;
      }
    }
    node->publishDetectedObjects(frame.det);
    nv::visualize(images, frame, font_path,
                  viz_dir + "/" + std::to_string(frame_id) + ".jpg",
                  vad_model.stream_,
                  init_lidar_y, ground_height);

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
