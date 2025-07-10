#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>
#include "autoware/tensorrt_vad/vad_model.hpp"
#include "mock_vad_logger.hpp"

namespace autoware::tensorrt_vad
{

// テストフィクスチャ: 各テストのセットアップと後片付けを共通化
class VadModelTest : public ::testing::Test {
protected:
    void SetUp() override
    {
        // モックロガーを作成
        mock_logger_ = std::make_shared<MockVadLogger>();

        // ダミーのテスト設定
        config_.plugins_path = "/tmp/test_plugin.so";
        config_.warm_up_num = 1;
        
        NetConfig backbone_config;
        backbone_config.name = "backbone";
        backbone_config.engine_file = "/tmp/test_backbone.engine";
        backbone_config.use_graph = false; // グラフはインテグレーションテストで検証
        
        config_.nets_config.push_back(backbone_config);
    }

    void TearDown() override
    {
        // テストごとに状態をリセット
    }

    std::shared_ptr<MockVadLogger> mock_logger_;
    VadConfig config_;
};

// 1. VadInputData構造体の基本的な検証
TEST_F(VadModelTest, VadInputDataStructure)
{
    VadInputData input_data;

    // デフォルトコンストラクタでは各ベクターは空のはず
    EXPECT_EQ(input_data.camera_images_.size(), 0);
    EXPECT_EQ(input_data.shift_.size(), 0);
    EXPECT_EQ(input_data.lidar2img_.size(), 0);
    EXPECT_EQ(input_data.can_bus_.size(), 0);
    EXPECT_EQ(input_data.command_, 2); // commandのデフォルト値は2

    // 値を代入して確認
    const size_t correct_image_size = 6 * 3 * 256 * 704;
    input_data.camera_images_.resize(correct_image_size);
    input_data.shift_ = {1.0f, 2.0f, 3.0f};
    input_data.lidar2img_.resize(96);
    input_data.can_bus_.resize(18);
    input_data.command_ = 1;

    EXPECT_EQ(input_data.camera_images_.size(), correct_image_size);
    EXPECT_EQ(input_data.shift_.size(), 3);
    EXPECT_FLOAT_EQ(input_data.shift_[1], 2.0f);
    EXPECT_EQ(input_data.lidar2img_.size(), 96);
    EXPECT_EQ(input_data.can_bus_.size(), 18);
    EXPECT_EQ(input_data.command_, 1);
}

// 2. VadOutputData構造体の基本的な検証
TEST_F(VadModelTest, VadOutputDataStructure)
{
    VadOutputData output_data;

    // デフォルトのサイズを確認
    EXPECT_EQ(output_data.predicted_trajectory_.size(), 0);

    // 値を代入して確認
    output_data.predicted_trajectory_ = {
        1.0, 0.1, 2.0, 0.2, 3.0, 0.3, 4.0, 0.4, 5.0, 0.5, 6.0, 0.6
    };

    EXPECT_EQ(output_data.predicted_trajectory_.size(), 12);
}

// 3. VadConfig構造体の検証
TEST_F(VadModelTest, VadConfigStructure)
{
    EXPECT_EQ(config_.plugins_path, "/tmp/test_plugin.so");
    EXPECT_EQ(config_.warm_up_num, 1);
    EXPECT_EQ(config_.nets_config.size(), 1);
    
    // ネットワーク設定の検証
    const auto& net_conf = config_.nets_config[0];
    EXPECT_EQ(net_conf.name, "backbone");
    EXPECT_EQ(net_conf.engine_file, "/tmp/test_backbone.engine");
    EXPECT_FALSE(net_conf.use_graph);
}

// 4. NetworkParamクラスの基本的な振る舞いをテスト
TEST_F(VadModelTest, NetworkParamClass)
{
    std::string onnx_path = "model.onnx";
    std::string engine_path = "model.engine";
    std::string trt_precision = "fp16";

    NetworkParam param(onnx_path, engine_path, trt_precision);

    EXPECT_EQ(param.onnx_path(), onnx_path);
    EXPECT_EQ(param.engine_path(), engine_path);
    EXPECT_EQ(param.trt_precision(), trt_precision);
}

}  // namespace autoware::tensorrt_vad
