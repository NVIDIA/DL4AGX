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

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <string.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <numeric>
#include <cmath>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <optional>

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb/stb_image_resize.h>

#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>
#include <Eigen/Dense>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>

#include <rclcpp/serialization.hpp>
#include <rclcpp/serialized_message.hpp>

#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

using json = nlohmann::json;
namespace fs = std::filesystem;

#include "net.h"
#include "visualize.hpp"  
#include "vad_model.hpp"

#include <rclcpp/rclcpp.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>

// Convert Autoware coordinates to nuScenes coordinates
std::pair<float, float> aw2ns_xy(float aw_x, float aw_y) {
    float ns_x = -aw_y;
    float ns_y = aw_x;
    return {ns_x, ns_y};
}

// Convert a quaternion from Autoware to nuScenes coordinate system
Eigen::Quaternionf aw2ns_quaternion(const Eigen::Quaternionf& q_aw) {
    // Create a -90-degree rotation around Z-axis (Autoware -> nuScenes)
    Eigen::Quaternionf q_rotation(Eigen::AngleAxisf(-M_PI/2, Eigen::Vector3f::UnitZ()));
    
    // Apply the rotation
    return q_rotation * q_aw;
}

class VADNode : public rclcpp::Node 
{
public:
    VADNode() 
        : Node("vad_node") 
    {
        // Publishers
        trajectory_publisher_ = this->create_publisher<autoware_planning_msgs::msg::Trajectory>(
            "/planning/vad/trajectory", 
            rclcpp::QoS(1));

        objects_publisher_ = this->create_publisher<autoware_perception_msgs::msg::DetectedObjects>(
            "/perception/object_recognition/detection/vad/objects",
            rclcpp::QoS(1));

        // Subscribers for each camera
        for (int32_t i = 0; i < 6; ++i) {
            auto callback = [this, i](const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
                this->onImageReceived(msg, i);
            };

            camera_subscribers_.push_back(
                this->create_subscription<sensor_msgs::msg::CompressedImage>(
                    "/sensing/camera/camera" + std::to_string(i) + "/image/compressed",
                    rclcpp::QoS(1),
                    callback
                )
            );
        }

        RCLCPP_INFO(this->get_logger(), "VAD Node has been initialized");
    }

    void publishTrajectory(const std::vector<float>& planning) 
    {
        auto trajectory_msg = std::make_unique<autoware_planning_msgs::msg::Trajectory>();
        
        for (size_t i = 0; i < planning.size(); i += 2) 
        {
            autoware_planning_msgs::msg::TrajectoryPoint point;
            
            point.pose.position.x = planning[i];
            point.pose.position.y = planning[i + 1];
            point.pose.position.z = 0.0;

            if (i + 2 < planning.size()) {
                float dx = planning[i + 2] - planning[i];
                float dy = planning[i + 3] - planning[i + 1];
                float yaw = std::atan2(dy, dx);
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

    void publishDetectedObjects(const std::vector<std::vector<float>>& detections)
    {
        auto objects_msg = std::make_unique<autoware_perception_msgs::msg::DetectedObjects>();
        objects_msg->header.stamp = this->now();
        objects_msg->header.frame_id = "map";

        for (const auto& det : detections)
        {
            // det format: x, y, z, w, l, h, yaw, vx, vy, label, score
            autoware_perception_msgs::msg::DetectedObject object;

            object.kinematics.pose_with_covariance.pose.position.x = det[0];
            object.kinematics.pose_with_covariance.pose.position.y = det[1];
            object.kinematics.pose_with_covariance.pose.position.z = det[2];

            object.kinematics.pose_with_covariance.pose.orientation = createQuaternionFromYaw(det[6]);

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
    rclcpp::Publisher<autoware_planning_msgs::msg::Trajectory>::SharedPtr trajectory_publisher_;
    rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_publisher_;
    std::vector<rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr> camera_subscribers_;

    void onImageReceived(const sensor_msgs::msg::CompressedImage::SharedPtr msg, int32_t camera_index)
    {
        // 現状は受信のみ
        RCLCPP_DEBUG(this->get_logger(), "Received image from camera %d", camera_index);
    }

    geometry_msgs::msg::Quaternion createQuaternionFromYaw(double yaw) 
    {
        geometry_msgs::msg::Quaternion q;
        q.x = 0.0;
        q.y = 0.0;
        q.z = std::sin(yaw * 0.5);
        q.w = std::cos(yaw * 0.5);
        return q;
    }
};

// バイナリ画像データをROS CompressedImageに変換し、その後TensorRTのバインディングに渡す関数
std::vector<float> processImageForInference(
    const std::vector<std::vector<float>>& frame_images) {
    
    // 画像データを連結
    std::vector<float> concatenated_data;
    size_t single_camera_size = 3 * 384 * 640;
    concatenated_data.reserve(single_camera_size * 6);

    // カメラの順序: {0, 1, 2, 3, 4, 5}
    for (int camera_idx = 0; camera_idx < 6; ++camera_idx) {
        const auto& img_data = frame_images[camera_idx];
        if (img_data.size() != single_camera_size) {
            throw std::runtime_error("画像サイズが不正です: " + std::to_string(camera_idx));
        }
        concatenated_data.insert(concatenated_data.end(), img_data.begin(), img_data.end());
    }

    return concatenated_data;
}


class Logger : public nvinfer1::ILogger {
public:
	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
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

  float report(const std::string& prefix = "timer") {
    float times = 0;    
    cudaEventSynchronize(end_);
    cudaEventElapsedTime(&times, begin_, end_);
    printf("[TIMER:  %s]: \t%.5f ms\n", prefix.c_str(), times);
    return times;
  }

private:
  cudaEvent_t begin_ = nullptr, end_ = nullptr;
};

void releaseNetwork(std::unordered_map<std::string, std::shared_ptr<nv::Net>>& nets, 
                   const std::string& name) {
    if (nets.find(name) != nets.end()) {
        // まずbindingsをクリア
        nets[name]->bindings.clear();
        cudaDeviceSynchronize();
        
        // 次にNetオブジェクトを解放
        nets[name].reset();
        nets.erase(name);
        cudaDeviceSynchronize();
    }
}

void loadHeadEngine(
    std::unordered_map<std::string, std::shared_ptr<nv::Net>>& nets,
    const json& cfg,
    const std::string& cfg_dir,
    nvinfer1::IRuntime* runtime,
    cudaStream_t stream) {
    
    auto head_engine = std::find_if(cfg["nets"].begin(), cfg["nets"].end(),
        [](const json& engine) { return engine["name"] == "head"; });
    
    std::string eng_file = (*head_engine)["file"];
    std::string eng_pth = cfg_dir + "/" + eng_file;
    printf("-> loading head engine: %s\n", eng_pth.c_str());
    
    std::unordered_map<std::string, std::shared_ptr<nv::Tensor>> ext;
    auto inputs = (*head_engine)["inputs"];
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
        std::string k = it.key();
        auto ext_map = it.value();      
        std::string ext_net = ext_map["net"];
        std::string ext_name = ext_map["name"];
        printf("%s <- %s[%s]\n", k.c_str(), ext_net.c_str(), ext_name.c_str());
        ext[k] = nets[ext_net]->bindings[ext_name];
    }

    nets["head"] = std::make_shared<nv::Net>(eng_pth, runtime, ext);

    bool use_graph = (*head_engine)["use_graph"];
    if (use_graph) {
        nets["head"]->EnableCudaGraph(stream);
    }
}

std::unordered_map<int, std::vector<std::vector<float>>> load_image_from_rosbag(
    const std::string& bag_path, int32_t n_frames) {
    
    std::unordered_map<int, std::vector<std::vector<float>>> subscribed_image_dict;
    std::unordered_map<int, std::vector<float>> frame_images_dict;
    
    // AutowareカメラインデックスからVADカメラインデックスへのマッピング
    std::unordered_map<int, int> autoware_to_vad = {
        {0, 0},  // FRONT
        {4, 1},  // FRONT_RIGHT
        {2, 2},  // FRONT_LEFT
        {1, 3},  // BACK
        {3, 4},  // BACK_LEFT
        {5, 5}   // BACK_RIGHT
    };
    
    // 目標の画像サイズを定義
    const int32_t target_width = 640;
    const int32_t target_height = 384;
    
    try {
        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = bag_path;
        storage_options.storage_id = "sqlite3";
        rosbag2_cpp::ConverterOptions converter_options;
        converter_options.input_serialization_format = "cdr";
        converter_options.output_serialization_format = "cdr";
        rosbag2_cpp::readers::SequentialReader reader;
        reader.open(storage_options, converter_options);
        
        int32_t current_frame_id = 1;
        
        while (reader.has_next() && current_frame_id <= n_frames) {
            auto bag_message = reader.read_next();
            
            for (int autoware_idx = 0; autoware_idx < 6; ++autoware_idx) {
                std::string topic = "/sensing/camera/camera" + std::to_string(autoware_idx) + "/image_rect_color/compressed";
                if (bag_message->topic_name == topic) {
                    auto msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
                    rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                    rclcpp::Serialization<sensor_msgs::msg::CompressedImage>().deserialize_message(
                        &serialized_msg, msg.get());
                    
                    int32_t vad_idx = autoware_to_vad[autoware_idx];
                    
                    // 画像データの処理
                    int32_t width, height, channels;
                    unsigned char* image_data = stbi_load_from_memory(
                        msg->data.data(), static_cast<int>(msg->data.size()),
                        &width, &height, &channels, STBI_rgb); // RGBとして読み込む
                    
                    if (image_data == nullptr) {
                        std::cerr << "Failed to load image data" << std::endl;
                        continue;
                    }
                    
                    // サイズが目標と違う場合はリサイズする
                    unsigned char* resized_data = nullptr;
                    if (width != target_width || height != target_height) {
                        // RCLCPP_INFO(rclcpp::get_logger("load_image_from_rosbag"),
                        //     "カメラ %d の画像をリサイズします: %dx%d から %dx%d へ",
                        //     autoware_idx, width, height, target_width, target_height);
                        
                        // stb_image_resizeを使用してリサイズ
                        resized_data = (unsigned char*)malloc(target_width * target_height * channels);
                        if (!resized_data) {
                            std::cerr << "リサイズ用のメモリ確保に失敗しました" << std::endl;
                            stbi_image_free(image_data);
                            continue;
                        }
                        
                        // stb_image_resizeを使ってリサイズ
                        int resize_result = stbir_resize_uint8(
                            image_data, width, height, 0,
                            resized_data, target_width, target_height, 0,
                            channels);
                            
                        if (!resize_result) {
                            std::cerr << "画像のリサイズに失敗しました" << std::endl;
                            free(resized_data);
                            stbi_image_free(image_data);
                            continue;
                        }
                        
                        // 元のデータを解放し、リサイズしたデータを使用
                        stbi_image_free(image_data);
                        image_data = resized_data;
                        width = target_width;
                        height = target_height;
                    }
                    
                    // 正規化のパラメータ
                    float mean[3] = {103.530f, 116.280f, 123.675f};
                    float std[3] = {1.0f, 1.0f, 1.0f};
                    
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
                    
                    // 画像データを解放
                    if (image_data == resized_data) {
                        free(resized_data);
                    } else {
                        stbi_image_free(image_data);
                    }
                    
                    // 正規化された画像データを保存
                    frame_images_dict[vad_idx] = normalized_image_data;
                    
                    // 6画像がそろった場合
                    if (frame_images_dict.size() == 6) {
                        std::vector<std::vector<float>> frame_images;
                        frame_images.reserve(6);
                        
                        // 各カメラの画像を正しい順序で保存
                        for (int i = 0; i < 6; ++i) {
                            if (frame_images_dict.find(i) == frame_images_dict.end()) {
                                std::cerr << "カメラ " << i << " の画像が見つかりません" << std::endl;
                                throw std::runtime_error("Missing camera image");
                            }
                            frame_images.push_back(frame_images_dict[i]);
                        }
                        
                        subscribed_image_dict[current_frame_id] = frame_images;
                        
                        // frame_images_dictをクリア
                        frame_images_dict.clear();
                        current_frame_id++;
                    }
                }
            }
        }
                
    } catch (const std::exception& e) {
        std::cerr << "Error in load_image_from_rosbag: " << e.what() << std::endl;
        throw;
    }
    
    // すべてのフレームが揃っているか確認
    for (int frame_id = 1; frame_id <= n_frames; ++frame_id) {
        if (subscribed_image_dict.find(frame_id) == subscribed_image_dict.end()) {
            std::cerr << "Frame " << frame_id << " not found in dictionary" << std::endl;
            throw std::runtime_error("Frame not found in dictionary");
        }
    }
    
    return subscribed_image_dict;
}

void compare_with_reference(
    const std::vector<std::vector<float>>& subscribed_images,
    const std::string& reference_path) {
    
    // リファレンス画像データの読み込み
    std::vector<float> reference_data;
    std::ifstream ref_file(reference_path, std::ios::binary);
    if (!ref_file) {
        throw std::runtime_error("リファレンスファイルを開けません: " + reference_path);
    }

    // ファイルサイズを取得
    ref_file.seekg(0, std::ios::end);
    size_t file_size = ref_file.tellg();
    ref_file.seekg(0, std::ios::beg);

    // メモリを確保してデータを読み込み
    reference_data.resize(file_size / sizeof(float));
    ref_file.read(reinterpret_cast<char*>(reference_data.data()), file_size);

    // リファレンスデータを6カメラ×(3, 384, 640)の形式に変換
    size_t single_camera_size = 3 * 384 * 640;
    if (reference_data.size() != single_camera_size * 6) {
        throw std::runtime_error("リファレンスデータのサイズが不正です");
    }

    // 各カメラの画像データを比較
    for (size_t cam_idx = 0; cam_idx < 6; ++cam_idx) {
        const auto& subscribed_img = subscribed_images[cam_idx];
        
        // サイズチェック
        if (subscribed_img.size() != single_camera_size) {
            throw std::runtime_error("購読画像のサイズが不正です: カメラ " + std::to_string(cam_idx));
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
                "カメラ %zu: 許容誤差を超える差異が %d 箇所で検出されました (最大差: %f)",
                cam_idx, diff_count, max_diff);
        }
    }
}


std::vector<float> calculateShift(float delta_x, float delta_y, float patch_angle_rad) {
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
    
    float shift_y = translation_length * std::cos(bev_angle / 180.0 * M_PI) / grid_length_y / bev_h_;
    float shift_x = translation_length * std::sin(bev_angle / 180.0 * M_PI) / grid_length_x / bev_w_;
    
    return {shift_x, shift_y};
}

std::tuple<std::unordered_map<int, std::vector<float>>, std::unordered_map<int, std::vector<float>>> 
load_can_bus_shift_from_rosbag(const std::string& bag_path, int32_t n_frames) {
    std::unordered_map<int, std::vector<float>> can_bus_dict;
    std::unordered_map<int, std::vector<float>> shift_dict;
    int32_t current_frame_id = 1;
    
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

        // フレームごとのデータを一時保存する構造体
        struct FrameData {
            bool has_kinematic = false;
            bool has_imu = false;
            std::vector<float> translation;
            std::vector<float> rotation;
            std::vector<float> acceleration;
            std::vector<float> angular_velocity;
            std::vector<float> velocity;
        };
        
        FrameData current_frame;
        
        while (reader.has_next() && current_frame_id <= n_frames) {
            auto bag_message = reader.read_next();
            
            if (bag_message->topic_name == "/localization/kinematic_state") {
                auto msg = std::make_shared<nav_msgs::msg::Odometry>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<nav_msgs::msg::Odometry>().deserialize_message(
                    &serialized_msg, msg.get());
                
                current_frame.has_kinematic = true;
                
                // Apply Autoware to nuScenes coordinate transformation to position
                auto [ns_x, ns_y] = aw2ns_xy(msg->pose.pose.position.x, msg->pose.pose.position.y);
                
                current_frame.translation.clear();
                current_frame.translation.push_back(ns_x);
                current_frame.translation.push_back(ns_y);
                current_frame.translation.push_back(static_cast<float>(msg->pose.pose.position.z));

                // Apply Autoware to nuScenes coordinate transformation to orientation
                Eigen::Quaternionf q_aw(
                    msg->pose.pose.orientation.w,
                    msg->pose.pose.orientation.x,
                    msg->pose.pose.orientation.y,
                    msg->pose.pose.orientation.z
                );
                
                Eigen::Quaternionf q_ns = aw2ns_quaternion(q_aw);
                
                current_frame.rotation.clear();
                current_frame.rotation.push_back(q_ns.x());
                current_frame.rotation.push_back(q_ns.y());
                current_frame.rotation.push_back(q_ns.z());
                current_frame.rotation.push_back(q_ns.w());

                // Apply Autoware to nuScenes coordinate transformation to velocity
                auto [ns_vx, ns_vy] = aw2ns_xy(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
                
                current_frame.velocity.clear();
                current_frame.velocity.push_back(ns_vx);
                current_frame.velocity.push_back(ns_vy);
                current_frame.velocity.push_back(static_cast<float>(msg->twist.twist.linear.z));

                // Apply Autoware to nuScenes coordinate transformation to angular velocity
                auto [ns_wx, ns_wy] = aw2ns_xy(msg->twist.twist.angular.x, msg->twist.twist.angular.y);
                
                current_frame.angular_velocity.clear();
                current_frame.angular_velocity.push_back(ns_wx);
                current_frame.angular_velocity.push_back(ns_wy);
                current_frame.angular_velocity.push_back(static_cast<float>(msg->twist.twist.angular.z));
            }
            else if (bag_message->topic_name == "/sensing/imu/tamagawa/imu_raw") {
                auto msg = std::make_shared<sensor_msgs::msg::Imu>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<sensor_msgs::msg::Imu>().deserialize_message(
                    &serialized_msg, msg.get());
                
                current_frame.has_imu = true;
                
                // Apply Autoware to nuScenes coordinate transformation to acceleration
                auto [ns_ax, ns_ay] = aw2ns_xy(msg->linear_acceleration.x, msg->linear_acceleration.y);
                
                current_frame.acceleration.clear();
                current_frame.acceleration.push_back(ns_ax);
                current_frame.acceleration.push_back(ns_ay);
                current_frame.acceleration.push_back(static_cast<float>(msg->linear_acceleration.z));
            }
            
            // kinematicとimuの両方のデータが揃った場合
            if (current_frame.has_kinematic && current_frame.has_imu) {
                // can_busデータの構築（18次元ベクトル）
                std::vector<float> can_bus(18, 0.0f);
                
                // translation (0:3)
                std::copy(current_frame.translation.begin(), current_frame.translation.end(), can_bus.begin());
                
                // rotation (3:7)
                std::copy(current_frame.rotation.begin(), current_frame.rotation.end(), can_bus.begin() + 3);
                
                // acceleration (7:10)
                std::copy(current_frame.acceleration.begin(), current_frame.acceleration.end(), can_bus.begin() + 7);
                
                // angular velocity (10:13)
                std::copy(current_frame.angular_velocity.begin(), current_frame.angular_velocity.end(), can_bus.begin() + 10);
                
                // velocity (13:16)
                std::copy(current_frame.velocity.begin(), current_frame.velocity.begin() + 2, can_bus.begin() + 13);
                can_bus[15] = 0.0f;  // z方向の速度は0とする
                
                // patch_angle[rad]の計算 (16)
                double yaw = std::atan2(
                    2.0 * (can_bus[6] * can_bus[5] + can_bus[3] * can_bus[4]),
                    1.0 - 2.0 * (can_bus[4] * can_bus[4] + can_bus[5] * can_bus[5])
                );
                if (yaw < 0) yaw += 2 * M_PI;
                can_bus[16] = static_cast<float>(yaw);
                
                // patch_angle[deg]の計算 (17)
                if (current_frame_id > 1 && can_bus_dict.find(current_frame_id - 1) != can_bus_dict.end()) {
                    float prev_angle = can_bus_dict[current_frame_id - 1][16];
                    can_bus[17] = (yaw - prev_angle) * 180.0f / M_PI;
                } else {
                    can_bus[17] = -1.0353195667266846f;  // 最初のフレームのデフォルト値
                }
                
                can_bus_dict[current_frame_id] = can_bus;
                
                // シフトデータの計算
                if (current_frame_id > 1) {
                    const auto& prev_translation = can_bus_dict[current_frame_id - 1];
                    float delta_x = current_frame.translation[0] - prev_translation[0];
                    float delta_y = current_frame.translation[1] - prev_translation[1];
                    
                    std::vector<float> shift = calculateShift(delta_x, delta_y, yaw);
                    shift_dict[current_frame_id] = shift;
                }
                
                // 次のフレームの準備
                current_frame = FrameData();
                current_frame_id++;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "ROSバッグの読み込み中にエラーが発生: " << e.what() << std::endl;
        throw;
    }
    
    return std::make_tuple(can_bus_dict, shift_dict);
}

std::optional<int> extract_autoware_camera_id(
    const std::string& topic_name,
    const std::unordered_map<int, int>& vad_to_autoware_camera) {
    
    for (const auto& [vad_id, autoware_id] : vad_to_autoware_camera) {
        std::string camera_topic = "/sensing/camera/camera" + std::to_string(autoware_id) + "/camera_info";
        if (topic_name == camera_topic) {
            return autoware_id;
        }
    }
    return std::nullopt;
}

std::unordered_map<int, std::vector<float>> 
load_lidar2img_from_rosbag(const std::string& bag_path, int32_t n_frames, float scale_width, float scale_height) {
    std::unordered_map<int, std::vector<float>> result;
    
    // VADカメラIDからAutowareカメラIDへのマッピング
    std::unordered_map<int, int> vad_to_autoware_camera = {
        {0, 0},  // CAM_FRONT -> camera0
        {1, 4},  // CAM_FRONT_RIGHT -> camera4
        {2, 2},  // CAM_FRONT_LEFT -> camera2
        {3, 1},  // CAM_BACK -> camera1
        {4, 3},  // CAM_BACK_LEFT -> camera3
        {5, 5}   // CAM_BACK_RIGHT -> camera5
    };
    
    // Autowareカメラ名からVADカメラIDへの逆マッピング
    std::unordered_map<int, int> autoware_to_vad_camera;
    for (const auto& [vad_id, autoware_id] : vad_to_autoware_camera) {
        autoware_to_vad_camera[autoware_id] = vad_id;
    }

    try {
        // ROSバッグの設定
        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = bag_path;
        storage_options.storage_id = "sqlite3";

        rosbag2_cpp::ConverterOptions converter_options;
        converter_options.input_serialization_format = "cdr";
        converter_options.output_serialization_format = "cdr";

        // カメラの内部パラメータを格納する辞書
        std::unordered_map<int, Eigen::Matrix3f> cameras_intrinsics;

        // ROSバッグリーダーの初期化
        rosbag2_cpp::readers::SequentialReader reader;
        reader.open(storage_options, converter_options);

        // まずはすべてのカメラの内部パラメータを読み込む
        while (reader.has_next()) {
            auto bag_message = reader.read_next();
            
            if (bag_message->topic_name.find("/camera_info") != std::string::npos) {
                // AutowareカメラIDを抽出
                std::optional<int> autoware_camera_id = extract_autoware_camera_id(bag_message->topic_name, vad_to_autoware_camera);

                if (autoware_camera_id.has_value()) {
                    // CameraInfoメッセージをデシリアライズ
                    auto msg = std::make_shared<sensor_msgs::msg::CameraInfo>();
                    rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                    rclcpp::Serialization<sensor_msgs::msg::CameraInfo>().deserialize_message(
                        &serialized_msg, msg.get());

                    // カメラ行列Kを3x3行列として抽出
                    Eigen::Matrix3f k_matrix;
                    for (int i = 0; i < 3; ++i) {
                        for (int j = 0; j < 3; ++j) {
                            k_matrix(i, j) = msg->k[i * 3 + j];
                        }
                    }
                    cameras_intrinsics[autoware_camera_id.value()] = k_matrix;
                } else {
                    RCLCPP_ERROR(rclcpp::get_logger("load_lidar2img_from_rosbag"),
                        "Could not extract Autoware camera ID from topic name: %s", bag_message->topic_name.c_str());
                }
            }
        }

        // バッグファイルを再度開く
        rosbag2_cpp::readers::SequentialReader reader2;
        reader2.open(storage_options, converter_options);

        int32_t current_frame_id = 1;
        while (reader2.has_next() && current_frame_id <= n_frames) {
            auto bag_message = reader2.read_next();
            
            if (bag_message->topic_name == "/tf_static") {
                // 現在のフレームのlidar2imgデータを格納する一時配列
                std::vector<float> frame_lidar2img(16 * 6, 0.0f);  // 6カメラ分のスペースを確保
                bool frame_complete = true;

                // TFメッセージをデシリアライズ
                auto msg = std::make_shared<tf2_msgs::msg::TFMessage>();
                rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);
                rclcpp::Serialization<tf2_msgs::msg::TFMessage>().deserialize_message(
                    &serialized_msg, msg.get());

                // 各カメラのTF変換を処理
                for (const auto& transform : msg->transforms) {
                    std::string child_frame_id = transform.child_frame_id;
                    
                    if (child_frame_id.find("camera") != std::string::npos && 
                        child_frame_id.find("/camera_optical_link") != std::string::npos) {
                        // Autowareカメラ名からカメラIDを抽出
                        int32_t autoware_camera_id = std::stoi(child_frame_id.substr(
                            child_frame_id.find("camera") + 6, 1));

                        // VADカメラIDに変換
                        if (autoware_to_vad_camera.find(autoware_camera_id) == autoware_to_vad_camera.end()) {
                            continue;
                        }
                        int32_t vad_camera_id = autoware_to_vad_camera[autoware_camera_id];

                        // カメラの内部パラメータを確認
                        if (cameras_intrinsics.find(autoware_camera_id) == cameras_intrinsics.end()) {
                            std::cout << "Warning: No camera intrinsics for camera" << autoware_camera_id << std::endl;
                            continue;
                        }

                        // 変換行列の構築
                        Eigen::Vector3f aw_translation(
                            transform.transform.translation.x,
                            transform.transform.translation.y,
                            transform.transform.translation.z
                        );
                        
                        // Apply Autoware to nuScenes coordinate transformation to translation
                        auto [ns_x, ns_y] = aw2ns_xy(aw_translation[0], aw_translation[1]);
                        Eigen::Vector3f ns_translation(ns_x, ns_y, aw_translation[2]);

                        Eigen::Quaternionf q_aw(
                            transform.transform.rotation.w,
                            transform.transform.rotation.x,
                            transform.transform.rotation.y,
                            transform.transform.rotation.z
                        );
                        
                        // Apply Autoware to nuScenes coordinate transformation to quaternion
                        Eigen::Quaternionf q_ns = aw2ns_quaternion(q_aw);

                        // lidar2cam_rtを構築 (nuScenes座標系)
                        Eigen::Matrix4f lidar2cam_rt = Eigen::Matrix4f::Identity();
                        lidar2cam_rt.block<3,3>(0,0) = q_ns.toRotationMatrix();
                        lidar2cam_rt.block<3,1>(0,3) = ns_translation;

                        // lidar2cam_rt.Tを計算
                        Eigen::Matrix4f lidar2cam_rt_T = lidar2cam_rt.transpose();

                        // viewpadを作成
                        Eigen::Matrix4f viewpad = Eigen::Matrix4f::Zero();
                        viewpad.block<3,3>(0,0) = cameras_intrinsics[autoware_camera_id];
                        viewpad(3,3) = 1.0f;

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
                        if (vad_camera_id >= 0 && vad_camera_id < 6) {
                            std::copy(lidar2img_flat.begin(), 
                                    lidar2img_flat.end(), 
                                    frame_lidar2img.begin() + vad_camera_id * 16);
                        }
                    }
                }

                // すべてのカメラのデータが揃っているか確認
                for (int i = 0; i < 6; i++) {
                    bool camera_data_exists = false;
                    for (int j = 0; j < 16; j++) {
                        if (frame_lidar2img[i * 16 + j] != 0.0f) {
                            camera_data_exists = true;
                            break;
                        }
                    }
                    if (!camera_data_exists) {
                        frame_complete = false;
                        break;
                    }
                }

                if (frame_complete) {
                    result[current_frame_id] = frame_lidar2img;
                    current_frame_id++;
                } else {
                    std::cerr << "Frame " << current_frame_id << " is incomplete. Aborting." << std::endl;
                    throw std::runtime_error("Incomplete frame");
                }
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "ROSバッグの読み込み中にエラーが発生: " << e.what() << std::endl;
        throw;
    }

    return result;
}

void compare_with_reference_lidar2img(
    const std::vector<float>& lidar2img_data,
    const std::string& reference_file_path
) {
    std::ifstream file(reference_file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "参照ファイルを開けませんでした: " << reference_file_path << std::endl;
        throw std::runtime_error("参照ファイルを開けませんでした");
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size != lidar2img_data.size() * sizeof(float)) {
        std::cerr << "参照ファイルのサイズが一致しません: " << reference_file_path << std::endl;
        std::cerr << "期待されるサイズ: " << lidar2img_data.size() << " バイト, 実際のサイズ: " << size << " バイト" << std::endl;
        throw std::runtime_error("参照ファイルのサイズが一致しません");
    }

    std::vector<float> reference_data(lidar2img_data.size());
    if (!file.read(reinterpret_cast<char*>(reference_data.data()), size)) {
        std::cerr << "参照ファイルの読み込みに失敗しました: " << reference_file_path << std::endl;
        throw std::runtime_error("参照ファイルの読み込みに失敗しました");
    }

    for (size_t i = 0; i < lidar2img_data.size(); ++i) {
        if (std::abs(lidar2img_data[i] - reference_data[i]) > 1e-3) {
            std::cerr << "値が一致しません: " << reference_file_path << " のインデックス " << i << std::endl;
            std::cerr << "binファイルの値: " << reference_data[i] << ", lidar2imgの値: " << lidar2img_data[i] << std::endl;
            throw std::runtime_error("値が一致しません");
        }
    }

    std::cout << "lidar2imgデータは参照ファイルと一致しています: " << reference_file_path << std::endl;
}

int main(int argc, char** argv) {
  // ROSの初期化
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VADNode>();
  
  printf("nvinfer: %d.%d.%d\n", NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
  cudaSetDevice(0);

  auto runtime_deleter = [](nvinfer1::IRuntime *runtime) {};
	std::unique_ptr<nvinfer1::IRuntime, decltype(runtime_deleter)> runtime{
    nvinfer1::createInferRuntime(gLogger), runtime_deleter};

  const std::string config = argv[1];
  fs::path cfg_pth = config;
  fs::path cfg_dir = cfg_pth.parent_path();
  printf("[INFO] setting up from %s\n", config.c_str());
  printf("[INFO] assuming data dir is %s\n", cfg_dir.string().c_str());

  std::ifstream f(config);
  json cfg = json::parse(f);

  std::vector<void*> plugins;
  for( std::string plugin_name: cfg["plugins"]) {
    std::string plugin_dir = cfg_dir.string() + "/" + plugin_name;
    void* h_ = dlopen(plugin_dir.c_str(), RTLD_NOW);
    printf("[INFO] loading plugin from: %s\n", plugin_dir.c_str());
    if (!h_) {
      const char* error = dlerror();
      std::cerr << "Failed to load library: " << error << std::endl;
      return -1;
    }
    plugins.push_back(h_);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // init engines
  std::unordered_map<std::string, std::shared_ptr<nv::Net>> nets;
  for( auto engine: cfg["nets"]) {
    if (engine["name"] == "head") {
      continue;  // headは後で初期化
    }
    
    std::string eng_name = engine["name"];
    std::string eng_file = engine["file"];
    std::string eng_pth = cfg_dir.string() + "/" + eng_file;
    printf("-> engine: %s\n", eng_name.c_str());
    
    std::unordered_map<std::string, std::shared_ptr<nv::Tensor>> ext;
    // reuse memory
    auto inputs =  engine["inputs"];
    for (auto it = inputs.begin(); it != inputs.end(); ++it) {
      std::string k = it.key();
      auto ext_map = it.value();      
      std::string ext_net = ext_map["net"];
      std::string ext_name = ext_map["name"];
      printf("%s <- %s[%s]\n", k.c_str(), ext_net.c_str(), ext_name.c_str());
      ext[k] = nets[ext_net]->bindings[ext_name];
    }

    nets[eng_name] = std::make_shared<nv::Net>(eng_pth, runtime.get(), ext);

    bool use_graph = engine["use_graph"];
    if( use_graph ) {
      nets[eng_name]->EnableCudaGraph(stream);
    }
  }
  
  int32_t warm_up = cfg["warm_up"];
  printf("[INFO] warm_up=%d\n", warm_up);
  for( int32_t iw=0; iw < warm_up; iw++ ) {
    nets["backbone"]->Enqueue(stream);
    nets["head_no_prev"]->Enqueue(stream);
    cudaStreamSynchronize(stream);
  }

  EventTimer timer;
  std::string data_dir = cfg_dir.string() + "/data/";
  int32_t n_frames = cfg["n_frames"];
  printf("[INFO] n_frames=%d\n", n_frames);
  std::shared_ptr<nv::Tensor> saved_prev_bev;
  std::vector<float> lidar2img;
  bool is_first_frame = true;

  auto subscribed_image_dict = load_image_from_rosbag("/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/output_bag/", n_frames);
  auto [subscribed_can_bus_dict, subscribed_shift_dict] = load_can_bus_shift_from_rosbag("/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/output_bag/", n_frames);
  int32_t input_image_width = cfg["input_image_width"];
  int32_t input_image_height = cfg["input_image_hight"];
  auto subscribed_lidar2img_dict = load_lidar2img_from_rosbag("/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/output_bag/", n_frames, 640.0f / static_cast<float>(input_image_width), 384.0f / static_cast<float>(input_image_height));
  // img.binと値を比較
  for (int frame_id = 1; frame_id < n_frames; frame_id++) {
    std::string frame_dir = data_dir + std::to_string(frame_id) + "/";
    
    // 画像をconcatenate
    auto image_data = processImageForInference(subscribed_image_dict[frame_id]);    
    autoware::tensorrt_vad::VadInputData vad_input_data{
        image_data,    // camera_images_
        subscribed_shift_dict[frame_id],    // shift_
        subscribed_lidar2img_dict[frame_id], // lidar2img_
        subscribed_can_bus_dict[frame_id],  // can_bus_
        2                                   // command_
    };
    nets["backbone"]->bindings["img"]->load(vad_input_data.camera_images_, stream);
    nets["backbone"]->Enqueue(stream);

    if (is_first_frame) {
        nets["head_no_prev"]->bindings["img_metas.0[shift]"]->load(vad_input_data.shift_, stream);
        nets["head_no_prev"]->bindings["img_metas.0[lidar2img]"]->load(vad_input_data.lidar2img_, stream);
        nets["head_no_prev"]->bindings["img_metas.0[can_bus]"]->load(vad_input_data.can_bus_, stream);
        nets["head_no_prev"]->Enqueue(stream);
        
        // prev_bevを保存
        auto bev_embed = nets["head_no_prev"]->bindings["out.bev_embed"];
        saved_prev_bev = std::make_shared<nv::Tensor>("prev_bev", bev_embed->dim, bev_embed->dtype);
        cudaMemcpyAsync(saved_prev_bev->ptr, bev_embed->ptr, bev_embed->nbytes(), 
                      cudaMemcpyDeviceToDevice, stream);
        
        // head_no_prevを解放
        releaseNetwork(nets, "head_no_prev");
        cudaStreamSynchronize(stream);  // メモリ解放を確実に
        
        // headをロード
        loadHeadEngine(nets, cfg, cfg_dir.string(), runtime.get(), stream);
        
        is_first_frame = false;
    }
    else {
        nets["head"]->bindings["prev_bev"] = saved_prev_bev;
        nets["head"]->bindings["img_metas.0[shift]"]->load(vad_input_data.shift_, stream);
        nets["head"]->bindings["img_metas.0[lidar2img]"]->load(vad_input_data.lidar2img_, stream);
        nets["head"]->bindings["img_metas.0[can_bus]"]->load(vad_input_data.can_bus_, stream);
        nets["head"]->Enqueue(stream);
        // prev_bevを保存
        auto bev_embed = nets["head"]->bindings["out.bev_embed"];
        saved_prev_bev = std::make_shared<nv::Tensor>("prev_bev", bev_embed->dim, bev_embed->dtype);
        cudaMemcpyAsync(saved_prev_bev->ptr, bev_embed->ptr, bev_embed->nbytes(), 
                      cudaMemcpyDeviceToDevice, stream);
    }

    cudaStreamSynchronize(stream);

    std::string viz_dir = cfg["viz"];
    viz_dir = cfg_dir.string() + "/" + viz_dir;

    std::vector<unsigned char*> images;
    for( std::string image_name: cfg["images"]) {    
      std::string image_pth = data_dir + std::to_string(frame_id) + "/" + image_name;
      
      int32_t width, height, channels;
      images.push_back(stbi_load(image_pth.c_str(), &width, &height, &channels, 0));
    }
    std::string font_path = cfg_dir.string() + "/" + cfg["font_path"].get<std::string>();

    nv::VisualizeFrame frame;

    std::ifstream cmd_file(frame_dir + "cmd.bin", std::ios::binary);    
    cmd_file.read((char*)(&frame.cmd), sizeof(int));
    cmd_file.close();

    frame.img_metas_lidar2img = nets["head"]->bindings["img_metas.0[lidar2img]"]->cpu<float>();

    // pred -> frame.planning
    std::vector<float> ego_fut_preds = nets["head"]->bindings["out.ego_fut_preds"]->cpu<float>();
    std::vector<float> planning = std::vector<float>(
      ego_fut_preds.begin() + frame.cmd * 12, 
      ego_fut_preds.begin() + (frame.cmd + 1) * 12);
    // cumsum to build trajectory in 3d space
    for( int32_t i=1; i<6; i++) {
      planning[i * 2    ] += planning[(i-1) * 2    ];
      planning[i * 2 + 1] += planning[(i-1) * 2 + 1];
    }    
    frame.planning = planning;
    node->publishTrajectory(frame.planning);
    printf("publish trajectory");
    rclcpp::spin_some(node);

    std::vector<float> bbox_preds = nets["head"]->bindings["out.all_bbox_preds"]->cpu<float>();
    std::vector<float> cls_scores = nets["head"]->bindings["out.all_cls_scores"]->cpu<float>();

    // det to frame.det
    constexpr int32_t N_MAX_DET = 300;
    for( int32_t d=0; d<N_MAX_DET; d++ ) {
      // 3, 1, 100, 10
      std::vector<float> box_score(
        cls_scores.begin() + d * 10, 
        cls_scores.begin() + d * 10 + 10);
      float max_score = -1;
      int32_t max_label = -1;
      for( int32_t l=0; l<10; l++ ) {
        // sigmoid
        float this_score = 1.0f / (1.0f + std::exp(-box_score[l]));
        if( this_score > max_score ) {
          max_score = this_score;
          max_label = l;
        }
      }
      if( max_score > 0.35 ) {
        // from: cx, cy, w, l, cz, h, sin, cos, vx, vy
        //   to:  x,  y, z, w,  l, h, yaw, vx, vy, label, score
        std::vector<float> raw(
          bbox_preds.begin() + d * 10, 
          bbox_preds.begin() + d * 10 + 10);
        std::vector<float> ret(11);
        ret[0] = raw[0]; ret[1] = raw[1]; ret[2] = raw[4];
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
    nv::visualize(
      images, 
      frame,
      font_path,
      viz_dir + "/" + std::to_string(frame_id) + ".jpg",
      stream);

    printf("[INFO] %d, cmd=%d finished\n", frame_id, frame.cmd);
  }

  int32_t perf_loop = cfg.value("perf_loop", 0);
  if( perf_loop > 0 ) {
    printf("[INFO] running %d rounds of perf_loop\n", perf_loop);
  }
  for( int32_t i=0; i < perf_loop; i++ ) {
    timer.start(stream);
    nets["backbone"]->Enqueue(stream);
    nets["head"]->Enqueue(stream);
    timer.end(stream);
    cudaStreamSynchronize(stream);
    timer.report("vad-trt");
  }
  
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  // ROSのシャットダウン
  rclcpp::shutdown();
  // dlclose(so_handle);
  return 0;
}