#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <memory>
#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <dlfcn.h>
#include <yaml-cpp/yaml.h>
#include "mock_vad_logger.hpp"
#include "autoware/tensorrt_vad/vad_model.hpp"

namespace autoware::tensorrt_vad {
namespace test {

// テスト用の設定構造体
struct TestConfig {
    struct {
        struct {
            std::string src_path;
            std::string dst_path;
        } bev_embed;
        struct {
            std::string path;
        } camera_images;
        std::vector<float> shift;
        std::vector<float> lidar2img;
        std::vector<float> can_bus;
        int command;
    } input_data;
    struct {
        std::vector<float> trajectory;
    } expected_output;
};

std::pair<VadConfig, TestConfig> load_config_from_yaml(const std::string& config_path) {
    try {
        YAML::Node yaml_config = YAML::LoadFile(config_path);
        const auto& test_config_node = yaml_config["test_config"];
        
        VadConfig vad_config;
        vad_config.plugins_path = test_config_node["plugins_path"].as<std::string>();
        vad_config.warm_up_num = test_config_node["warm_up_num"].as<int>();
        
        const auto& nets = test_config_node["nets"];
        for (const auto& net : nets) {
            NetConfig net_config;
            net_config.name = net.second["name"].as<std::string>();
            net_config.engine_file = net.second["engine_file"].as<std::string>();
            net_config.use_graph = net.second["use_graph"].as<bool>();
            
            if (net.second["inputs"]) {
                const auto& inputs = net.second["inputs"];
                for (const auto& input : inputs) {
                    std::map<std::string, std::string> input_map;
                    for (const auto& param : input.second) {
                        input_map[param.first.as<std::string>()] = param.second.as<std::string>();
                    }
                    net_config.inputs[input.first.as<std::string>()] = input_map;
                }
            }
            vad_config.nets_config.push_back(net_config);
        }

        TestConfig test_config;
        const auto& test_data = test_config_node["test_data"];
        const auto& input_data = test_data["input_data"];
        test_config.input_data.bev_embed.src_path = input_data["bev_embed"]["src_path"].as<std::string>();
        test_config.input_data.bev_embed.dst_path = input_data["bev_embed"]["dst_path"].as<std::string>();
        test_config.input_data.camera_images.path = input_data["camera_images"]["path"].as<std::string>();
        test_config.input_data.shift = input_data["shift"].as<std::vector<float>>();
        test_config.input_data.lidar2img = input_data["lidar2img"].as<std::vector<float>>();
        test_config.input_data.can_bus = input_data["can_bus"].as<std::vector<float>>();
        test_config.input_data.command = input_data["command"].as<int>();

        const auto& expected_output = test_data["expected_output"];
        test_config.expected_output.trajectory = expected_output["trajectory"].as<std::vector<float>>();

        return {vad_config, test_config};
    } catch (const YAML::Exception& e) {
        throw std::runtime_error("Failed to load config from YAML: " + std::string(e.what()));
    }
}

}  // namespace test

// 統合テスト用のフィクスチャ
class VadIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        mock_logger_ = std::make_shared<MockVadLogger>();
        auto [vad_config, test_config] = test::load_config_from_yaml("../../install/autoware_tensorrt_vad/share/autoware_tensorrt_vad/test/test_config.yaml");
        config_ = vad_config;
        test_config_ = test_config;
    }

    std::shared_ptr<MockVadLogger> mock_logger_;
    VadConfig config_;
    test::TestConfig test_config_;
};

// VadModelの初期化が、実際のエンジンファイルを用いて成功することを確認するテスト
TEST_F(VadIntegrationTest, ModelInitializationWithRealEngines)
{
    // エラーログが呼ばれないことを期待
    EXPECT_CALL(*mock_logger_, error(testing::_)).Times(0);
    // infoログは何回か呼ばれるはず
    EXPECT_CALL(*mock_logger_, info(testing::_)).Times(testing::AtLeast(1));

    // VadModelのコンストラクタが例外を投げずに完了すればテスト成功
    // これにより、プラグインのロード、エンジンのデシリアライズ、CUDAコンテキストの初期化が
    // 正常に行われることを検証する
    std::unique_ptr<VadModel<MockVadLogger>> model;
    ASSERT_NO_THROW({
        model = std::make_unique<VadModel<MockVadLogger>>(config_, mock_logger_);
    }) << "VadModel initialization failed with real engine files. "
       << "Check if paths are correct and files are not corrupted.";
    
    ASSERT_TRUE(model->initialized_) << "Model should be marked as initialized.";
}

class VadInferIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        logger_ = std::make_shared<MockVadLogger>();
        auto [vad_config, test_config] = test::load_config_from_yaml("../../install/autoware_tensorrt_vad/share/autoware_tensorrt_vad/test/test_config.yaml");
        config_ = vad_config;
        test_config_ = test_config;
        
        // 前提条件のチェック
        bool engines_exist = 
            std::filesystem::exists(config_.nets_config[0].engine_file) &&
            std::filesystem::exists(config_.nets_config[1].engine_file) &&
            std::filesystem::exists(config_.nets_config[2].engine_file);
        bool plugin_exists = std::filesystem::exists(config_.plugins_path);
        
        int device_count = 0;
        cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
        bool cuda_available = (cuda_status == cudaSuccess && device_count > 0);
        
        integration_test_enabled_ = engines_exist && plugin_exists && cuda_available;

        if (!integration_test_enabled_) {
            GTEST_SKIP() << "Integration test requirements not met.";
        }

        // bev_embedはビルドディレクトリからコピーする
        const std::string src_bev_path = test_config_.input_data.bev_embed.src_path;
        const std::string dst_bev_path = test_config_.input_data.bev_embed.dst_path;
        if (std::filesystem::exists(src_bev_path)) {
            std::filesystem::copy(src_bev_path, dst_bev_path, std::filesystem::copy_options::overwrite_existing);
        } else {
            GTEST_SKIP() << "bev_embed_frame1.bin not found. Run vad_app first.";
        }
    }

    VadConfig createRealConfig() {
        VadConfig config;
        config.plugins_path = config_.plugins_path;
        config.warm_up_num = config_.warm_up_num;
        
        NetConfig backbone_config;
        backbone_config.name = config_.nets_config[0].name;
        backbone_config.engine_file = config_.nets_config[0].engine_file;
        backbone_config.use_graph = config_.nets_config[0].use_graph;
        
        NetConfig head_no_prev_config;
        head_no_prev_config.name = config_.nets_config[1].name;
        head_no_prev_config.engine_file = config_.nets_config[1].engine_file;
        head_no_prev_config.use_graph = config_.nets_config[1].use_graph;
        head_no_prev_config.inputs["img_feats"] = config_.nets_config[1].inputs["img_feats"];
        
        NetConfig head_config;
        head_config.name = config_.nets_config[2].name;
        head_config.engine_file = config_.nets_config[2].engine_file;
        head_config.use_graph = config_.nets_config[2].use_graph;
        head_config.inputs["img_feats"] = config_.nets_config[2].inputs["img_feats"];
        
        config.nets_config = {backbone_config, head_no_prev_config, head_config};
        return config;
    }

    std::vector<float> loadBevEmbedFromFile(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open bev_embed file: " + path);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<float> data(size / sizeof(float));
        if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
            throw std::runtime_error("Failed to read bev_embed data from file: " + path);
        }
        return data;
    }

    VadInputData createFrame2InputData() {
        VadInputData input_data;
        
        std::ifstream file(test_config_.input_data.camera_images.path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open image data file: " + test_config_.input_data.camera_images.path);
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        size_t expected_elements = 6 * 3 * 384 * 640;
        size_t expected_size_bytes = expected_elements * sizeof(float);
        
        if (static_cast<size_t>(size) != expected_size_bytes) {
            throw std::runtime_error("Image data file has incorrect size. Expected: " + 
                                     std::to_string(expected_size_bytes) + ", Got: " + std::to_string(size));
        }
        
        input_data.camera_images_.resize(expected_elements);
        if (!file.read(reinterpret_cast<char*>(input_data.camera_images_.data()), size)) {
            throw std::runtime_error("Failed to read image data from file: " + test_config_.input_data.camera_images.path);
        }
        
        input_data.shift_ = test_config_.input_data.shift;
        input_data.lidar2img_ = test_config_.input_data.lidar2img;
        input_data.can_bus_ = test_config_.input_data.can_bus;
        input_data.command_ = test_config_.input_data.command;
        
        return input_data;
    }

    std::shared_ptr<MockVadLogger> logger_;
    VadConfig config_;
    test::TestConfig test_config_;
    bool integration_test_enabled_ = false;
};

// 1. モデルが例外を投げずに初期化できることを確認
TEST_F(VadInferIntegrationTest, ModelInitialization) {
    VadConfig config = createRealConfig();
    std::unique_ptr<VadModel<MockVadLogger>> model;
    
    ASSERT_NO_THROW({
        model = std::make_unique<VadModel<MockVadLogger>>(config, logger_);
    }) << "Model initialization failed. Check paths and permissions.";
}

// 2. 実際のinfer実行テスト
TEST_F(VadInferIntegrationTest, RealInferExecution) {
    auto model = std::make_unique<VadModel<MockVadLogger>>(createRealConfig(), logger_);
    
    auto prev_bev_data = loadBevEmbedFromFile("bev_embed_frame1.bin");
    
    auto dummy_input = createFrame2InputData();
    auto result1 = model->infer(dummy_input); 
    (void)result1; // 戻り値を明示的に無視
    
    model->is_first_frame_ = false;

    VadInputData input_data_frame2 = createFrame2InputData();

    auto result = model->infer(input_data_frame2);
    ASSERT_TRUE(result.has_value()) << "Inference failed to return a result.";
    EXPECT_EQ(result->predicted_trajectory_.size(), test_config_.expected_output.trajectory.size());
    
    for (size_t i = 0; i < test_config_.expected_output.trajectory.size(); ++i) {
        EXPECT_NEAR(result->predicted_trajectory_[i], test_config_.expected_output.trajectory[i], 1e-5)
            << "Mismatch at index " << i;
    }
}

}  // namespace autoware::tensorrt_vad
