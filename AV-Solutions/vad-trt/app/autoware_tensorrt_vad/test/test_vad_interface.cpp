#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <tf2_msgs/msg/tf_message.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include "vad_interface.hpp"

namespace autoware::tensorrt_vad {

// dummy tf_static, camera_info, scale, lidar2imgの値を用いたテスト
TEST(VadLidar2ImgTest, DummyInputOutput)
{
    // tf_staticのダミー作成
    auto tf_static = std::make_shared<tf2_msgs::msg::TFMessage>();
    // 6カメラ分のtransformを追加
    std::vector<std::tuple<std::string, std::string, std::array<double,3>, std::array<double,4>>> tf_params = {
        {"base_link", "camera0/camera_optical_link", {0.761597, 0.0020668, 0.741433}, {-0.487, 0.486, -0.507, 0.519}},
        {"base_link", "camera4/camera_optical_link", {0.381621, -0.496336, -0.308103}, {0.176601, -0.672482, 0.684519, -0.21912}},
        {"base_link", "camera2/camera_optical_link", {0.241922, 0.505025, -0.330718}, {-0.665072, 0.193319, -0.225921, 0.68503}},
        {"base_link", "camera1/camera_optical_link", {-1.00382, 0.0101697, -0.32055}, {0.509639, 0.509785, -0.485972, -0.494185}},
        {"base_link", "camera3/camera_optical_link", {0.0698264, 0.482037, -0.276687}, {-0.694254, -0.128327, 0.100488, 0.701032}},
        {"base_link", "camera5/camera_optical_link", {-0.0754992, -0.465983, -0.29509}, {0.153793, 0.691312, -0.69544, -0.121654}}
    };
    for (const auto& [parent, child, trans, rot] : tf_params) {
        geometry_msgs::msg::TransformStamped t;
        t.header.frame_id = parent;
        t.child_frame_id = child;
        t.transform.translation.x = trans[0];
        t.transform.translation.y = trans[1];
        t.transform.translation.z = trans[2];
        t.transform.rotation.x = rot[0];
        t.transform.rotation.y = rot[1];
        t.transform.rotation.z = rot[2];
        t.transform.rotation.w = rot[3];
        tf_static->transforms.push_back(t);
    }
    // camera_infoのダミー作成
    std::vector<std::array<double, 9>> k_values = {
        {960, 0, 960.5, 0, 959.391, 540.5, 0, 0, 1},
        {796.891, 0, 857.777, 0, 796.891, 476.885, 0, 0, 1},
        {1257.86, 0, 827.241, 0, 1257.86, 450.915, 0, 0, 1},
        {1254.99, 0, 829.577, 0, 1254.99, 467.168, 0, 0, 1},
        {1256.75, 0, 817.789, 0, 1256.75, 451.954, 0, 0, 1},
        {1249.96, 0, 825.377, 0, 1249.96, 462.548, 0, 0, 1}
    };
    std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> camera_infos;
    for (const auto& k : k_values) {
        auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
        for (int i = 0; i < 9; ++i) info->k[i] = k[i];
        camera_infos.push_back(info);
    }
    // 期待されるlidar2img
    std::vector<float> expected = {
        315.886, 323.942, 13.5783, -256.128, -6.31035, 209.941, -330.421, 85.0808, -0.0127048, 0.998511, 0.053057, -0.799828, 0, 0, 0, 1,
        455.039, -206.316, -13.1366, -151.165, 135.287, 114.121, -440.643, -246.462, 0.84326, 0.536482, 0.0331592, -0.613056, 0, 0, 0, 1,
        10.425, 501.045, 26.146, -107.302, -138.021, 114.07, -440.075, -242.841, -0.82384, 0.565366, 0.0406133, -0.539403, 0, 0, 0, 1,
        -267.966, -283.597, -8.95374, -290.275, -3.71822, -158.29, -289.762, -251.815, -0.00823006, -0.999196, -0.0392251, -1.01567, 0, 0, 0, 1,
        -395.513, 307.766, 17.7519, -207.23, -164.462, -36.4216, -445.338, -199.953, -0.947597, -0.319451, 0.00308716, -0.433616, 0, 0, 0, 1,
        95.15, -489.74, -19.9046, -87.1869, 158.46, -43.5394, -444.479, -208.288, 0.924112, -0.382109, -0.00312805, -0.460392, 0, 0, 0, 1
    };
    // tf_bufferにtf_static->transformsを追加
    auto tf_buffer = std::make_shared<tf2_ros::Buffer>(rclcpp::Clock::make_shared());
    for (const auto& t : tf_static->transforms) {
        tf_buffer->setTransform(t, "default_authority", true);
    }
    int32_t input_image_width = 1920;
    int32_t input_image_height = 1080;
    int32_t target_image_width = 640;
    int32_t target_image_height = 384;
    std::vector<double> point_cloud_range = {-15.0, -30.0, -2.0, 15.0, 30.0, 2.0};
    int32_t bev_h = 100;
    int32_t bev_w = 100;
    double default_patch_angle = -1.0353195667266846;
    int32_t default_command = 0;
    std::vector<double> default_shift = {0.0, 0.0};
    std::vector<double> image_normalization_param_mean = {103.530, 116.280, 123.675};
    std::vector<double> image_normalization_param_std = {1.0, 1.0, 1.0};
    std::vector<double> vad2base = {
        0.0, 1.0, 0.0, 0.0,
       -1.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    };
    std::vector<int64_t> autoware_to_vad_camera_mapping = {0, 0, // FRONT
                                                           4, 1, // FRONT_RIGHT
                                                           2, 2, // FRONT_LEFT
                                                           1, 3, // BACK
                                                           3, 4, // BACK_LEFT
                                                           5, 5  // BACK_RIGHT
                                                           };
    autoware::tensorrt_vad::VadInterfaceConfig vad_interface_config(
        input_image_width, input_image_height,
        target_image_width, target_image_height,
        point_cloud_range,
        bev_h, bev_w,
        default_patch_angle,
        default_command,
        default_shift,
        image_normalization_param_mean,
        image_normalization_param_std,
        vad2base,
        autoware_to_vad_camera_mapping);
    VadInterface vad_interface(vad_interface_config, tf_buffer);
    float scale_width = static_cast<float>(target_image_width) / static_cast<float>(input_image_width);
    float scale_height = static_cast<float>(target_image_height) / static_cast<float>(input_image_height);
    auto result = vad_interface.process_lidar2img(tf_static, camera_infos, scale_width, scale_height);
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result[i], expected[i], 1e-2);
    }
}

} // namespace autoware::tensorrt_vad
