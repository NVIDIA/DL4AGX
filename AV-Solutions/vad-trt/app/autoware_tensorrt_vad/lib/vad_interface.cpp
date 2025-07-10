#include "vad_interface.hpp"
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <cmath>

namespace autoware::tensorrt_vad
{

VadInterface::VadInterface(const VadInterfaceConfig& config, std::shared_ptr<tf2_ros::Buffer> tf_buffer)
  : tf_buffer_(tf_buffer),
    target_image_width_(config.target_image_width),
    target_image_height_(config.target_image_height),
    input_image_width_(config.input_image_width),
    input_image_height_(config.input_image_height),
    point_cloud_range_(config.point_cloud_range),
    bev_h_(config.bev_h),
    bev_w_(config.bev_w),
    default_patch_angle_(config.default_patch_angle),
    default_command_(config.default_command),
    default_shift_(config.default_shift),
    image_normalization_param_mean_(config.image_normalization_param_mean),
    image_normalization_param_std_(config.image_normalization_param_std),
    vad2base_(config.vad2base),
    base2vad_(config.base2vad)
{
  // AutowareカメラインデックスからVADカメラインデックスへのマッピング
  autoware_to_vad_camera_mapping_ = config.autoware_to_vad_camera_mapping;
}

VadInputData VadInterface::convert_input(const VadInputTopicData & vad_input_topic_data, const std::vector<float> & prev_can_bus)
{
  VadInputData vad_input_data;

  float scale_width = target_image_width_ / static_cast<float>(input_image_width_);
  float scale_height = target_image_height_ / static_cast<float>(input_image_height_);

  // Process lidar2img transformation
  vad_input_data.lidar2img_ = process_lidar2img(
    vad_input_topic_data.tf_static,
    vad_input_topic_data.camera_infos,
    scale_width, scale_height
  );
  
  // Process can_bus and shift data
  vad_input_data.can_bus_ = process_can_bus(
    vad_input_topic_data.kinematic_state,
    vad_input_topic_data.imu_raw,
    prev_can_bus
  );
  vad_input_data.shift_ = process_shift(vad_input_data.can_bus_, prev_can_bus);
  
  // Process image data
  vad_input_data.camera_images_ = process_image(vad_input_topic_data.images);
  
  // Set default command
  vad_input_data.command_ = default_command_;
  
  return vad_input_data;
}

std::optional<Eigen::Matrix4f> VadInterface::lookup_base_to_camera_rt(tf2_ros::Buffer & buffer, int32_t autoware_camera_id) const
{
  std::string target_frame = "camera" + std::to_string(autoware_camera_id) + "/camera_optical_link";
  std::string source_frame = "base_link";

  try {
    geometry_msgs::msg::TransformStamped lookup_result =
        buffer.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
    
    // geometry_msgs::msg::TransformからEigen::Matrix4fへの手動変換
    Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity();
    
    // 並進部分
    transform_matrix(0, 3) = lookup_result.transform.translation.x;
    transform_matrix(1, 3) = lookup_result.transform.translation.y;
    transform_matrix(2, 3) = lookup_result.transform.translation.z;
    
    // 回転部分（クォータニオンから回転行列への変換）
    Eigen::Quaternionf q(
        lookup_result.transform.rotation.w,
        lookup_result.transform.rotation.x,
        lookup_result.transform.rotation.y,
        lookup_result.transform.rotation.z);
    transform_matrix.block<3, 3>(0, 0) = q.toRotationMatrix();
    
    return transform_matrix;

  } catch (const tf2::TransformException &ex) {
    RCLCPP_ERROR(rclcpp::get_logger("VadInterface"), "TF変換の取得に失敗: %s -> %s. Reason: %s",
                 source_frame.c_str(), target_frame.c_str(), ex.what());
    return std::nullopt;
  }
}

Eigen::Matrix4f VadInterface::create_viewpad(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info) const
{
  Eigen::Matrix3f k_matrix;
  for (int32_t i = 0; i < 3; ++i) {
    for (int32_t j = 0; j < 3; ++j) {
      k_matrix(i, j) = camera_info->k[i * 3 + j];
    }
  }
  
  // viewpadを作成
  Eigen::Matrix4f viewpad = Eigen::Matrix4f::Zero();
  viewpad.block<3, 3>(0, 0) = k_matrix;
  viewpad(3, 3) = 1.0f;
  
  return viewpad;
}

Eigen::Matrix4f VadInterface::apply_scaling(const Eigen::Matrix4f & lidar2img, float scale_width, float scale_height) const
{
  Eigen::Matrix4f scale_matrix = Eigen::Matrix4f::Identity();
  scale_matrix(0, 0) = scale_width;
  scale_matrix(1, 1) = scale_height;
  return scale_matrix * lidar2img;
}

std::vector<float> VadInterface::matrix_to_flat(const Eigen::Matrix4f & matrix) const
{
  std::vector<float> flat(16);
  int32_t k = 0;
  for (int32_t i = 0; i < 4; ++i) {
    for (int32_t j = 0; j < 4; ++j) {
      flat[k++] = matrix(i, j);
    }
  }
  return flat;
}

Lidar2ImgData VadInterface::process_lidar2img(
    const tf2_msgs::msg::TFMessage::ConstSharedPtr & tf_static,
    const std::vector<sensor_msgs::msg::CameraInfo::ConstSharedPtr> & camera_infos,
    float scale_width, float scale_height) const
{
  std::vector<float> frame_lidar2img(16 * 6, 0.0f); // 6カメラ分のスペースを確保



  // vad_base_link -> base_link の変換をバッファに登録
  rclcpp::Time stamp(0, 0, RCL_ROS_TIME);
  if (!tf_static->transforms.empty()) {
    stamp = tf_static->transforms[0].header.stamp;
  }

  // 各カメラの処理
  for (int32_t autoware_camera_id = 0; autoware_camera_id < 6; ++autoware_camera_id) {
    if (!camera_infos[autoware_camera_id]) {
      continue;
    }

    auto base_to_camera_rt_opt = lookup_base_to_camera_rt(*tf_buffer_, autoware_camera_id);
    if (!base_to_camera_rt_opt) continue;
    Eigen::Matrix4f base_to_camera_rt = *base_to_camera_rt_opt;

    Eigen::Matrix4f viewpad = create_viewpad(camera_infos[autoware_camera_id]);
    Eigen::Matrix4f lidar2cam_rt = base_to_camera_rt * vad2base_;
    Eigen::Matrix4f lidar2img = viewpad * lidar2cam_rt;

    // スケーリングを適用
    lidar2img = apply_scaling(lidar2img, scale_width, scale_height);

    // 結果を格納
    std::vector<float> lidar2img_flat = matrix_to_flat(lidar2img);

    // lidar2imgの計算後、VADカメラIDの位置に格納
    int32_t vad_camera_id = autoware_to_vad_camera_mapping_.at(autoware_camera_id);
    if (vad_camera_id >= 0 && vad_camera_id < 6) {
      std::copy(lidar2img_flat.begin(), lidar2img_flat.end(),
                frame_lidar2img.begin() + vad_camera_id * 16);
    }
  }

  return frame_lidar2img;
}

std::vector<float> VadInterface::normalize_image(unsigned char *image_data, int32_t width, int32_t height) const
{
  std::vector<float> normalized_image_data(width * height * 3);
  
  // BGRの順で処理
  for (int32_t c = 0; c < 3; ++c) {
    for (int32_t h = 0; h < height; ++h) {
      for (int32_t w = 0; w < width; ++w) {
        int32_t src_idx = (h * width + w) * 3 + (2 - c); // BGR -> RGB
        int32_t dst_idx = c * height * width + h * width + w; // CHW形式
        float pixel_value = static_cast<float>(image_data[src_idx]);
        normalized_image_data[dst_idx] = (pixel_value - image_normalization_param_mean_[c]) / image_normalization_param_std_[c];
      }
    }
  }
  
  return normalized_image_data;
}

CameraImagesData VadInterface::process_image(
  const std::vector<sensor_msgs::msg::Image::ConstSharedPtr> & images) const
{
  std::vector<std::vector<float>> frame_images;
  frame_images.resize(6); // VADカメラ順序で初期化

  // 各カメラの画像を処理
  for (int32_t autoware_idx = 0; autoware_idx < 6; ++autoware_idx) {
    const auto &image_msg = images[autoware_idx];

    // OpenCVで画像データをデコード
    cv::Mat bgr_img = cv::imdecode(cv::Mat(image_msg->data), cv::IMREAD_COLOR); // BGRでデコード
    if (bgr_img.empty()) {
      throw std::runtime_error("画像データのデコードに失敗しました: " + std::to_string(autoware_idx));
    }

    // RGBに変換
    cv::Mat rgb_img;
    cv::cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);

    // サイズが目標と違う場合はリサイズする
    if (rgb_img.cols != target_image_width_ || rgb_img.rows != target_image_height_) {
      cv::resize(rgb_img, rgb_img, cv::Size(target_image_width_, target_image_height_));
    }

    // 画像を正規化
    std::vector<float> normalized_image_data = normalize_image(rgb_img.data, rgb_img.cols, rgb_img.rows);

    // VADカメラ順序で格納
    int32_t vad_idx = autoware_to_vad_camera_mapping_.at(autoware_idx);
    frame_images[vad_idx] = normalized_image_data;
  }

  // 画像データを連結
  std::vector<float> concatenated_data;
  size_t single_camera_size = 3 * target_image_height_ * target_image_width_;
  concatenated_data.reserve(single_camera_size * 6);

  // カメラの順序: {0, 1, 2, 3, 4, 5}
  for (int32_t camera_idx = 0; camera_idx < 6; ++camera_idx) {
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

CanBusData VadInterface::process_can_bus(
  const nav_msgs::msg::Odometry::ConstSharedPtr & kinematic_state,
  const sensor_msgs::msg::Imu::ConstSharedPtr & imu_raw,
  const std::vector<float> & prev_can_bus) const
{
  CanBusData can_bus(18, 0.0f);

  // Apply Autoware to VAD base_link coordinate transformation to position
  auto [vad_x, vad_y, vad_z] =
      aw2vad_xyz(kinematic_state->pose.pose.position.x,
                kinematic_state->pose.pose.position.y,
                kinematic_state->pose.pose.position.z);

  // translation (0:3)
  can_bus[0] = vad_x;
  can_bus[1] = vad_y;
  can_bus[2] = vad_z;

  // Apply Autoware to VAD base_link coordinate transformation to orientation
  Eigen::Quaternionf q_aw(
      kinematic_state->pose.pose.orientation.w,
      kinematic_state->pose.pose.orientation.x,
      kinematic_state->pose.pose.orientation.y,
      kinematic_state->pose.pose.orientation.z);

  Eigen::Quaternionf q_vad = aw2vad_quaternion(q_aw);

  // rotation (3:7)
  can_bus[3] = q_vad.x();
  can_bus[4] = q_vad.y();
  can_bus[5] = q_vad.z();
  can_bus[6] = q_vad.w();

  // Apply Autoware to VAD base_link coordinate transformation to acceleration
  auto [vad_ax, vad_ay, vad_az] =
      aw2vad_xyz(imu_raw->linear_acceleration.x,
                imu_raw->linear_acceleration.y,
                imu_raw->linear_acceleration.z);

  // acceleration (7:10)
  can_bus[7] = vad_ax;
  can_bus[8] = vad_ay;
  can_bus[9] = vad_az;

  // Apply Autoware to VAD base_link coordinate transformation to angular velocity
  auto [vad_wx, vad_wy, vad_wz] =
      aw2vad_xyz(kinematic_state->twist.twist.angular.x,
                kinematic_state->twist.twist.angular.y,
                kinematic_state->twist.twist.angular.z);

  // angular velocity (10:13)
  can_bus[10] = vad_wx;
  can_bus[11] = vad_wy;
  can_bus[12] = vad_wz;

  // Apply Autoware to VAD base_link coordinate transformation to velocity
  auto [vad_vx, vad_vy, vad_vz] =
      aw2vad_xyz(kinematic_state->twist.twist.linear.x,
                kinematic_state->twist.twist.linear.y,
                0.0f); // z方向の速度は0とする

  // velocity (13:16)
  can_bus[13] = vad_vx;
  can_bus[14] = vad_vy;
  can_bus[15] = vad_vz;

  // patch_angle[rad]の計算 (16)
  double yaw = std::atan2(
      2.0 * (can_bus[6] * can_bus[5] + can_bus[3] * can_bus[4]),
      1.0 - 2.0 * (can_bus[4] * can_bus[4] + can_bus[5] * can_bus[5]));
  if (yaw < 0)
    yaw += 2 * M_PI;
  can_bus[16] = static_cast<float>(yaw);

  // patch_angle[deg]の計算 (17)
  if (!prev_can_bus.empty()) {
    float prev_angle = prev_can_bus[16];
    can_bus[17] = (yaw - prev_angle) * 180.0f / M_PI;
  } else {
    can_bus[17] = default_patch_angle_; // 最初のフレームのデフォルト値
  }

  return can_bus;
}

ShiftData VadInterface::process_shift(
  const CanBusData & can_bus,
  const CanBusData & prev_can_bus) const
{
  if (prev_can_bus.empty()) {
    return default_shift_;
  }

  float delta_x = can_bus[0] - prev_can_bus[0];  // translation difference
  float delta_y = can_bus[1] - prev_can_bus[1];  // translation difference
  float patch_angle_rad = can_bus[16];  // current patch_angle[rad]

  float real_w = point_cloud_range_[3] - point_cloud_range_[0];
  float real_h = point_cloud_range_[4] - point_cloud_range_[1];
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

std::tuple<float, float, float> VadInterface::aw2vad_xyz(float aw_x, float aw_y, float aw_z) const
{
  // Autoware(base_link)座標[x, y, z]をVAD base_link座標に変換
  Eigen::Vector4f aw_xyz(aw_x, aw_y, aw_z, 1.0f);
  Eigen::Vector4f vad_xyz = base2vad_ * aw_xyz;
  return {vad_xyz[0], vad_xyz[1], vad_xyz[2]};
}

Eigen::Quaternionf VadInterface::aw2vad_quaternion(const Eigen::Quaternionf & q_aw) const
{
  // base2vad_の回転部分をクォータニオンに変換
  Eigen::Matrix3f rot = base2vad_.block<3,3>(0,0);
  Eigen::Quaternionf q_v2a(rot); // base_link→vadの回転
  Eigen::Quaternionf q_v2a_inv = q_v2a.conjugate(); // 単位クオータニオンなので逆=共役
  // q_vad = q_v2a * q_aw * q_v2a_inv
  return q_v2a * q_aw * q_v2a_inv;
}

}  // namespace autoware::tensorrt_vad
