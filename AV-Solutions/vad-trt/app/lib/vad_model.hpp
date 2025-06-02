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

#ifndef AUTOWARE_TENSORRT_VAD_VAD_TRT_HPP_
#define AUTOWARE_TENSORRT_VAD_VAD_TRT_HPP_

#include <optional>
#include <string>
#include <vector>

namespace autoware::tensorrt_vad
{

// VAD推論の入力データ構造
struct VadInputData
{
  // カメラ画像データ（複数カメラ対応）
  std::vector<float> camera_images_;

  // 前回のBEV特徴量（時系列処理用）
  // nets["head_no_prev"]->bindings["out.bev_embed"]
  // std::vector<float> prev_bev_{};

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

  // // 検出されたオブジェクト
  // std::vector<std::vector<float>> detected_objects_{};

  // // コマンドインデックス（選択された軌道のインデックス）
  // int32_t selected_command_index_{2};
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
class VadModel
{
public:
  // コンストラクタ
  VadModel();

  // デストラクタ
  ~VadModel();

  // モデルの初期化
  [[nodiscard]] bool initialize(const std::string & model_path);

  // メイン推論API
  [[nodiscard]] std::optional<VadOutputData> infer(const VadInputData & input);

private:
  // TODO(Shin-kyoto): TensorRTエンジン関連のメンバ変数を追加
  // nvinfer1::IRuntime* runtime_{nullptr};
  // nvinfer1::ICudaEngine* engine_{nullptr};
  // nvinfer1::IExecutionContext* context_{nullptr};
  // cudaStream_t stream_{nullptr};
};

}  // namespace autoware::tensorrt_vad

#endif  // AUTOWARE_TENSORRT_VAD_VAD_TRT_HPP_
