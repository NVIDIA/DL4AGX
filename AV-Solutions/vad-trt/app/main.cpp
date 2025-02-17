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

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

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

using json = nlohmann::json;
namespace fs = std::filesystem;

#include "net.h"
#include "visualize.hpp"  

#include <rclcpp/rclcpp.hpp>
#include <autoware_planning_msgs/msg/trajectory.hpp>
#include <autoware_planning_msgs/msg/trajectory_point.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>

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
        for (int i = 0; i < 6; ++i) {
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

    void onImageReceived(const sensor_msgs::msg::CompressedImage::SharedPtr msg, int camera_index)
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
void processImageForInference(
    const std::string& img_path,
    std::shared_ptr<nv::Net>& net,
    const std::string& binding_name,
    cudaStream_t stream) {
    
    auto tensor = net->bindings[binding_name];
    if (!tensor) {
        throw std::runtime_error("Binding not found: " + binding_name);
    }

    // バイナリファイルを読み込む
    std::ifstream file(img_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open image file: " + img_path);
    }
    
    // ファイルサイズを取得
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // データを読み込む
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Could not read image file");
    }
    
    // ROSのCompressedImageメッセージを作成
    auto compressed_msg = std::make_shared<sensor_msgs::msg::CompressedImage>();
    compressed_msg->format = "jpeg";  // または適切なフォーマット
    compressed_msg->data = std::vector<uint8_t>(buffer.begin(), buffer.end());
    
    // TensorRTのバインディングにデータを渡す
    // データ型に応じて適切なサイズでコピー
    cudaMemcpyAsync(
        tensor->ptr,                               // destination (GPU memory)
        compressed_msg->data.data(),              // source (CPU memory)
        tensor->volume * nv::getElementSize(tensor->dtype),  // size
        cudaMemcpyHostToDevice,                   // direction
        stream                                    // CUDA stream
    );
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
  
  int warm_up = cfg["warm_up"];
  printf("[INFO] warm_up=%d\n", warm_up);
  for( int iw=0; iw < warm_up; iw++ ) {
    nets["backbone"]->Enqueue(stream);
    nets["head_no_prev"]->Enqueue(stream);
    cudaStreamSynchronize(stream);
  }

  EventTimer timer;
  std::string data_dir = cfg_dir.string() + "/data/";
  int n_frames = cfg["n_frames"];
  printf("[INFO] n_frames=%d\n", n_frames);
  std::shared_ptr<nv::Tensor> saved_prev_bev;
  std::vector<float> lidar2img;
  bool is_first_frame = true;

  for( int frame_id=1; frame_id < n_frames; frame_id++ ) {
    std::string frame_dir = data_dir + std::to_string(frame_id) + "/";
    // nets["backbone"]->bindings["img"]->load(frame_dir + "img.bin");
    // 画像処理と推論
    try {
        processImageForInference(
            frame_dir + "img.bin",
            nets["backbone"],
            "img",
            stream
        );
    } catch (const std::exception& e) {
        std::cerr << "Error processing image: " << e.what() << std::endl;
        return -1;
    }
    if (is_first_frame) {
        nets["head_no_prev"]->bindings["img_metas.0[shift]"]->load(frame_dir + "img_metas.0[shift].bin");
        nets["head_no_prev"]->bindings["img_metas.0[lidar2img]"]->load(frame_dir + "img_metas.0[lidar2img].bin");
        nets["head_no_prev"]->bindings["img_metas.0[can_bus]"]->load(frame_dir + "img_metas.0[can_bus].bin");
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
        nets["head"]->bindings["img_metas.0[shift]"]->load(frame_dir + "img_metas.0[shift].bin");
        nets["head"]->bindings["img_metas.0[lidar2img]"]->load(frame_dir + "img_metas.0[lidar2img].bin");
        nets["head"]->bindings["img_metas.0[can_bus]"]->load(frame_dir + "img_metas.0[can_bus].bin");
        nets["head"]->Enqueue(stream);
    }

    nets["backbone"]->Enqueue(stream);
    cudaStreamSynchronize(stream);

    std::string viz_dir = cfg["viz"];
    viz_dir = cfg_dir.string() + "/" + viz_dir;

    std::vector<unsigned char*> images;
    for( std::string image_name: cfg["images"]) {    
      std::string image_pth = data_dir + std::to_string(frame_id) + "/" + image_name;
      
      int width, height, channels;
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
    for( int i=1; i<6; i++) {
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
    constexpr int N_MAX_DET = 300;
    for( int d=0; d<N_MAX_DET; d++ ) {
      // 3, 1, 100, 10
      std::vector<float> box_score(
        cls_scores.begin() + d * 10, 
        cls_scores.begin() + d * 10 + 10);
      float max_score = -1;
      int max_label = -1;
      for( int l=0; l<10; l++ ) {
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

  int perf_loop = cfg.value("perf_loop", 0);
  if( perf_loop > 0 ) {
    printf("[INFO] running %d rounds of perf_loop\n", perf_loop);
  }
  for( int i=0; i < perf_loop; i++ ) {
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