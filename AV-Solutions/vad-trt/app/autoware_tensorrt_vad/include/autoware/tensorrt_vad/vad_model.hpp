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

#ifndef AUTOWARE_TENSORRT_VAD_VAD_MODEL_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_MODEL_HPP_

#include <optional>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <unordered_map>
#include <map>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <dlfcn.h>
#include "net.h"

namespace autoware::tensorrt_vad
{

// ロガーインターフェース
class VadLogger {
public:
  virtual ~VadLogger() = default;
  
  // 各ログレベルのメソッドを純粋仮想関数として定義
  virtual void debug(const std::string& message) = 0;
  virtual void info(const std::string& message) = 0;
  virtual void warn(const std::string& message) = 0;
  virtual void error(const std::string& message) = 0;
};

// Loggerクラス（VadModel内で使用）
class Logger : public nvinfer1::ILogger {
private:
    std::shared_ptr<VadLogger> custom_logger_;

public:
    Logger(std::shared_ptr<VadLogger> logger) : custom_logger_(logger) {}
    
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // Only print error messages
        if (severity == nvinfer1::ILogger::Severity::kERROR && custom_logger_) {
            custom_logger_->error(std::string(msg));
        }
    }
};

// VAD推論の入力データ構造
struct VadInputData
{
  // カメラ画像データ（複数カメラ対応）
  std::vector<float> camera_images_;

  // シフト情報（img_metas.0[shift]）
  std::vector<float> shift_;

  // LiDAR座標系からカメラ画像座標系への変換行列（img_metas.0[lidar2img]）
  std::vector<float> lidar2img_;

  // CAN-BUSデータ（車両状態情報：速度、角速度など）(img_metas.0[can_bus])
  std::vector<float> can_bus_;

  // コマンドインデックス（軌道選択用）
  int32_t command_{2};
};

// VAD推論の出力データ構造
struct VadOutputData
{
  // 予測された軌道（6つの2D座標点、累積座標として表現）
  // planning[0,1] = 1st point (x,y), planning[2,3] = 2nd point (x,y), ...
  std::vector<float> predicted_trajectory_{};  // size: 12 (6 points * 2 coordinates)

  // 推論時間
  double inference_time_ms_{0.0};

  // コマンドインデックス（選択された軌道のインデックス）
  int32_t selected_command_index_{2};
};

// config for Net class
struct NetConfig
{
  std::string name;
  std::string engine_file;
  bool use_graph;
  std::map<std::string, std::map<std::string, std::string>> inputs;
};

// config for VadModel class
struct VadConfig
{
  std::string plugins_path;
  int32_t warm_up_num;
  std::vector<NetConfig> nets_config;
};

class NetworkParam
{
public:
  NetworkParam(std::string onnx_path, std::string engine_path, std::string trt_precision)
  : onnx_path_(std::move(onnx_path)), engine_path_(std::move(engine_path)), trt_precision_(std::move(trt_precision))
  {
  }

  std::string onnx_path() const { return onnx_path_; }
  std::string engine_path() const { return engine_path_; }
  std::string trt_precision() const { return trt_precision_; }

private:
  std::string onnx_path_;
  std::string engine_path_;
  std::string trt_precision_;
};

// VADモデルクラス - CUDA/TensorRTを用いた推論を担当
template<typename LoggerType>
class VadModel
{
public:
  VadModel(const VadConfig& config, std::shared_ptr<LoggerType> logger)
    : runtime_(nullptr)
    , stream_(nullptr)
    , nets_()
    , initialized_(false)
    , saved_prev_bev_(nullptr)
    , is_first_frame_(true)
    , config_(config)
    , logger_(std::move(logger))
  {
    // loggerはVadLoggerを継承したclassのみ受け取る
    static_assert(std::is_base_of_v<VadLogger, LoggerType>, 
      "LoggerType must be VadLogger or derive from VadLogger.");    
    // 初期化を実行
    runtime_ = create_runtime();
    
    if (!load_plugin(config.plugins_path)) {
      logger_->error("Failed to load plugin");
      return;
    }
    
    cudaStreamCreate(&stream_);
    
    nets_ = init_engines(config.nets_config);
    
    logger_->info("warm_up=" + std::to_string(config.warm_up_num));
    warm_up(config.warm_up_num);
    
    initialized_ = true;
  }

  // デストラクタ
  ~VadModel()
  {
    if (stream_) {
      cudaStreamDestroy(stream_);
      stream_ = nullptr;
    }
    
    // netsのクリーンアップ
    nets_.clear();
    
    initialized_ = false;
  }

  // メイン推論API
  [[nodiscard]] std::optional<VadOutputData> infer(const VadInputData & vad_input) {
    // 最初のフレームかどうかでheadの名前を変更
    std::string head_name;
    if (is_first_frame_) {
      head_name = "head_no_prev";
    } else {
      head_name = "head";
    }

    // bindingsにload
    load_inputs(vad_input, head_name);

    // backboneとheadをenqueue
    enqueue(head_name);

    // prev_bevを保存
    saved_prev_bev_ = save_prev_bev(head_name);
    // VadOutputDataに出力を変換
    VadOutputData output = postprocess(head_name, vad_input.command_);

    // 最初のフレームなら"head_no_prev"をリリースして"head"をload
    if (is_first_frame_) {
      release_network("head_no_prev");
      load_head();
      is_first_frame_ = false;
    }
    
    return output;
  }

  // メンバ変数
  std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)>> runtime_;
  cudaStream_t stream_;
  std::unordered_map<std::string, std::shared_ptr<nv::Net>> nets_;
  bool initialized_;

  // 前回のBEV特徴量保存用
  std::shared_ptr<nv::Tensor> saved_prev_bev_;
  bool is_first_frame_;

  // 設定情報の保存
  VadConfig config_;

  // ロガーインスタンス
  std::shared_ptr<VadLogger> logger_;

private:
  // メンバ関数
  std::unique_ptr<nvinfer1::IRuntime, std::function<void(nvinfer1::IRuntime*)>> create_runtime() {
    static std::unique_ptr<Logger> logger_instance = std::make_unique<Logger>(logger_);
    auto runtime_deleter = [](nvinfer1::IRuntime *runtime) { (void)runtime; };
    std::unique_ptr<nvinfer1::IRuntime, decltype(runtime_deleter)> runtime{
        nvinfer1::createInferRuntime(*logger_instance), runtime_deleter};
    return runtime;
  }

  bool load_plugin(const std::string& plugin_dir) {
    void* h_ = dlopen(plugin_dir.c_str(), RTLD_NOW);
    logger_->info("loading plugin from: " + plugin_dir);
    if (!h_) {
      const char* error = dlerror();
      logger_->error("Failed to load library: " + std::string(error ? error : "unknown error"));
      return false;
    }
    return true;
  }

  std::unordered_map<std::string, std::shared_ptr<nv::Net>> init_engines(
      const std::vector<NetConfig>& nets_config) {
    
    std::unordered_map<std::string, std::shared_ptr<nv::Net>> nets;
    
    for (const auto& engine : nets_config) {
      if (engine.name == "head") {
        continue;  // headは後で初期化
      }
      
      std::string engine_name = engine.name;
      std::string engine_file_path = engine.engine_file;
      logger_->info("-> engine: " + engine_name);
      
      std::unordered_map<std::string, std::shared_ptr<nv::Tensor>> external_bindings;
      // reuse memory
      for (const auto& input_pair : engine.inputs) {
        const std::string& k = input_pair.first;
        const auto& ext_map = input_pair.second;      
        std::string ext_net = ext_map.at("net");
        std::string ext_name = ext_map.at("name");
        logger_->info(k + " <- " + ext_net + "[" + ext_name + "]");
        external_bindings[k] = nets[ext_net]->bindings[ext_name];
      }

      nets[engine_name] = std::make_shared<nv::Net>(engine_file_path, runtime_.get(), external_bindings);

      if (engine.use_graph) {
        nets[engine_name]->EnableCudaGraph(stream_);
      }
    }
    
    return nets;
  }

  void warm_up(int32_t warm_up_num) {
    for(int32_t iw=0; iw < warm_up_num; iw++) {
      nets_["backbone"]->Enqueue(stream_);
      nets_["head_no_prev"]->Enqueue(stream_);
      cudaStreamSynchronize(stream_);
    }
  }
  
  // infer関数で使用するヘルパー関数
  void load_inputs(const VadInputData& vad_input, const std::string& head_name) {
    nets_["backbone"]->bindings["img"]->load(vad_input.camera_images_, stream_);
    nets_[head_name]->bindings["img_metas.0[shift]"]->load(vad_input.shift_, stream_);
    nets_[head_name]->bindings["img_metas.0[lidar2img]"]->load(vad_input.lidar2img_, stream_);
    nets_[head_name]->bindings["img_metas.0[can_bus]"]->load(vad_input.can_bus_, stream_);

    if (head_name == "head") {
      nets_["head"]->bindings["prev_bev"] = saved_prev_bev_;
    }
  }

  void enqueue(const std::string& head_name) {
    nets_["backbone"]->Enqueue(stream_);
    nets_[head_name]->Enqueue(stream_);
  }

  std::shared_ptr<nv::Tensor> save_prev_bev(const std::string& head_name) {
    auto bev_embed = nets_[head_name]->bindings["out.bev_embed"];
    auto prev_bev = std::make_shared<nv::Tensor>("prev_bev", bev_embed->dim, bev_embed->dtype);
    cudaMemcpyAsync(prev_bev->ptr, bev_embed->ptr, bev_embed->nbytes(), 
                    cudaMemcpyDeviceToDevice, stream_);
    return prev_bev;
  }

  void release_network(const std::string& network_name) {
    if (nets_.find(network_name) != nets_.end()) {
      // まずbindingsをクリア
      nets_[network_name]->bindings.clear();
      cudaDeviceSynchronize();
      
      // 次にNetオブジェクトを解放
      nets_[network_name].reset();
      nets_.erase(network_name);
      cudaDeviceSynchronize();
    }
  }

  void load_head() {
    auto head_engine = std::find_if(config_.nets_config.begin(), config_.nets_config.end(),
        [](const NetConfig& engine) { return engine.name == "head"; });
    
    if (head_engine == config_.nets_config.end()) {
      logger_->error("Head engine configuration not found");
      return;
    }
    
    std::string engine_file_path = head_engine->engine_file;
    logger_->info("-> loading head engine: " + engine_file_path);
    
    std::unordered_map<std::string, std::shared_ptr<nv::Tensor>> external_bindings;
    for (const auto& input_pair : head_engine->inputs) {
      const std::string& k = input_pair.first;
      const auto& ext_map = input_pair.second;      
      std::string ext_net = ext_map.at("net");
      std::string ext_name = ext_map.at("name");
      logger_->info(k + " <- " + ext_net + "[" + ext_name + "]");
      external_bindings[k] = nets_[ext_net]->bindings[ext_name];
    }

    nets_["head"] = std::make_shared<nv::Net>(engine_file_path, runtime_.get(), external_bindings);

    if (head_engine->use_graph) {
      nets_["head"]->EnableCudaGraph(stream_);
    }
  }

  VadOutputData postprocess(const std::string& head_name, int32_t cmd) {
    std::vector<float> ego_fut_preds = nets_[head_name]->bindings["out.ego_fut_preds"]->cpu<float>();
    
    // Extract planning for the given command
    std::vector<float> planning(
        ego_fut_preds.begin() + cmd * 12,
        ego_fut_preds.begin() + (cmd + 1) * 12
    );
    
    // cumsum to build trajectory in 3d space
    for (int32_t i = 1; i < 6; i++) {
      planning[i * 2] += planning[(i-1) * 2];
      planning[i * 2 + 1] += planning[(i-1) * 2 + 1];
    }
    
    return VadOutputData{planning};
  }
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_MODEL_HPP_
