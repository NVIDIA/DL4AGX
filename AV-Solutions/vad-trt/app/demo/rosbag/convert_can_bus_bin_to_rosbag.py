#!/usr/bin/env python3
import argparse
import os
import yaml
import numpy as np
from pyquaternion import Quaternion as pyquaternion_Quaternion
from nuscenes.eval.common.utils import quaternion_yaw

# ROS2 関連のインポート
from sensor_msgs.msg import Imu, CameraInfo
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message, deserialize_message
import rosbag2_py
import rclpy
from nav_msgs.msg import Odometry

point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
bev_h_ = 100
bev_w_ = 100

real_w = point_cloud_range[3] - point_cloud_range[0]
real_h = point_cloud_range[4] - point_cloud_range[1]
grid_length = [real_h / bev_h_, real_w / bev_w_]

def calculate_shift(delta_x, delta_y, patch_angle_rad, grid_length=grid_length, bev_h=bev_h_, bev_w=bev_w_):
    ego_angle = np.array(patch_angle_rad / np.pi * 180)

    grid_length_y = grid_length[0]
    grid_length_x = grid_length[1]

    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w

    return [shift_x, shift_y]

def calculate_patch_angle_rad(can_bus_rotation_quaternion):
    patch_angle_deg = quaternion_yaw(pyquaternion_Quaternion(can_bus_rotation_quaternion))/np.pi*180
    if patch_angle_deg < 0:
        patch_angle_deg += 360

    patch_angle_rad = patch_angle_deg / 180 * np.pi
    return patch_angle_rad

def convert_bin_to_imu(can_bus_data: np.ndarray, timestamp: Time, frame_id: str = "imu_link") -> Imu:
    """
    CAN busデータから/sensing/imu/tamagawa/imu_rawトピックへの変換
    
    Args:
        can_bus_data: CANバスデータ
        timestamp: メッセージのタイムスタンプ
        frame_id: フレームID
        
    Returns:
        Imu: 変換されたIMUメッセージ
    """
    # can_bus[7:9] = 加速度(x,y)
    # can_bus[12] = ego_w（yaw角速度）
    
    # 加速度の取得（float型に明示的に変換）
    accel_x = float(can_bus_data[7])
    accel_y = float(can_bus_data[8])
    accel_z = float(can_bus_data[9])
    
    # 角速度の取得（float型に明示的に変換）
    # TODO(Shin-kyoto): これで良いのかを確認する
    angular_velocity_z = float(can_bus_data[12])  # Yaw角速度
    angular_velocity_x = float(can_bus_data[10])
    angular_velocity_y = float(can_bus_data[11])
    
    # IMUメッセージの作成
    imu_msg = Imu()
    imu_msg.header.stamp = timestamp
    imu_msg.header.frame_id = frame_id
    
    # 線形加速度の設定
    imu_msg.linear_acceleration.x = accel_x
    imu_msg.linear_acceleration.y = accel_y
    imu_msg.linear_acceleration.z = accel_z
    
    # 角速度の設定
    imu_msg.angular_velocity.x = angular_velocity_x
    imu_msg.angular_velocity.y = angular_velocity_y
    imu_msg.angular_velocity.z = angular_velocity_z
    
    # 共分散行列の設定（不明な場合は-1で埋める）
    imu_msg.linear_acceleration_covariance = [-1.0] * 9
    imu_msg.angular_velocity_covariance = [-1.0] * 9
    imu_msg.orientation_covariance = [-1.0] * 9
    
    return imu_msg

def convert_bin_to_kinematic_state(can_bus_data: np.ndarray, timestamp: Time, frame_id: str = "base_link") -> Odometry:
    """
    CAN busデータから/localization/kinematic_stateトピックへの変換
    
    Args:
        can_bus_data: CANバスデータ
        timestamp: メッセージのタイムスタンプ
        frame_id: フレームID
        
    Returns:
        Odometry: 変換された運動学状態メッセージ
    """
    # can_bus[0:3] = ego2global_translation
    # can_bus[3:7] = ego2global_rotation
    # can_bus[12] = ego_w（yaw角速度）
    # can_bus[13:15] = ego_vx, ego_vy（x,y方向速度）
    
    # Odometryメッセージの作成
    odom_msg = Odometry()
    odom_msg.header.stamp = timestamp
    odom_msg.header.frame_id = "map"
    odom_msg.child_frame_id = frame_id
    
    # 位置情報の設定
    odom_msg.pose.pose.position.x = float(can_bus_data[0])
    odom_msg.pose.pose.position.y = float(can_bus_data[1])
    odom_msg.pose.pose.position.z = float(can_bus_data[2])
    
    # 姿勢情報の設定（クォータニオン）
    odom_msg.pose.pose.orientation.x = float(can_bus_data[3])
    odom_msg.pose.pose.orientation.y = float(can_bus_data[4])
    odom_msg.pose.pose.orientation.z = float(can_bus_data[5])
    odom_msg.pose.pose.orientation.w = float(can_bus_data[6])

    assert np.isclose(calculate_patch_angle_rad(can_bus_data[3:7]), can_bus_data[-2], atol=0.1), f"calculate_patch_angle_rad(can_bus_data[3:7]: {calculate_patch_angle_rad(can_bus_data[3:7])} is not equal to can_bus_data[-2]: {can_bus_data[-2]}"

    # 速度情報の設定
    odom_msg.twist.twist.linear.x = float(can_bus_data[13])  # x方向速度
    odom_msg.twist.twist.linear.y = float(can_bus_data[14])  # y方向速度
    odom_msg.twist.twist.linear.z = float(can_bus_data[15])  # z方向速度
    
    odom_msg.twist.twist.angular.x = float(can_bus_data[10])
    odom_msg.twist.twist.angular.y = float(can_bus_data[11])
    odom_msg.twist.twist.angular.z = float(can_bus_data[12])  # yaw角速度
    
    # 共分散行列の設定（不明な場合は大きな値を設定）
    odom_msg.pose.covariance = [0.01] * 36  # 6x6行列
    odom_msg.twist.covariance = [0.01] * 36  # 6x6行列
    
    return odom_msg

def convert_bin_to_tf_static(lidar2img_data: dict[int, np.ndarray], timestamp: Time) -> TFMessage:
    """
    lidar2imgデータからTFメッセージへの変換
    
    Args:
        lidar2img_data: カメラIDをキーとするlidar2imgデータの辞書
        timestamp: メッセージのタイムスタンプ
        
    Returns:
        TFMessage: 変換されたTFメッセージ
    """
    # カメラの内部パラメータの設定
    cameras_intrinsics = {
        # VADのカメラID: intrinsic
        0: np.array([  # CAM_FRONT
            [1.25281310e+03, 0.00000000e+00, 8.26588115e+02],
            [0.00000000e+00, 1.25281310e+03, 4.69984663e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        1: np.array([  # CAM_FRONT_RIGHT
            [1.25674851e+03, 0.00000000e+00, 8.17788757e+02],
            [0.00000000e+00, 1.25674851e+03, 4.51954178e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        2: np.array([  # CAM_FRONT_LEFT
            [1.25786253e+03, 0.00000000e+00, 8.27241063e+02],
            [0.00000000e+00, 1.25786253e+03, 4.50915498e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        3: np.array([  # CAM_BACK
            [796.89106345, 0.0, 857.77743269],
            [0.0, 796.89106345, 476.88489884],
            [0.0, 0.0, 1.0]
        ]),
        4: np.array([  # CAM_BACK_LEFT
            [1.25498606e+03, 0.00000000e+00, 8.29576933e+02],
            [0.00000000e+00, 1.25498606e+03, 4.67168056e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        5: np.array([  # CAM_BACK_RIGHT
            [1.24996293e+03, 0.00000000e+00, 8.25376805e+02],
            [0.00000000e+00, 1.24996293e+03, 4.62548164e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
    }
    
    # VADカメラIDからAutowareカメラIDへのマッピング
    vad_to_autoware_camera_map = {
        0: 0,  # CAM_FRONT -> camera0
        1: 4,  # CAM_FRONT_RIGHT -> camera4
        2: 2,  # CAM_FRONT_LEFT -> camera2
        3: 1,  # CAM_BACK -> camera1
        4: 3,  # CAM_BACK_LEFT -> camera3
        5: 5   # CAM_BACK_RIGHT -> camera5
    }
    
    # TFメッセージの作成
    tf_msg = TFMessage()
    
    # 各カメラの変換を処理
    for vad_camera_id, lidar2img in lidar2img_data.items():
        # 4x4の変換行列に変形
        lidar2img_matrix = lidar2img.reshape(4, 4)
        
        # カメラの内部パラメータを取得
        intrinsic = cameras_intrinsics[vad_camera_id]
        
        # intrinsicからviewpadを作成
        viewpad = np.zeros((4, 4), dtype=np.float32)
        viewpad[:3, :3] = intrinsic
        viewpad[3, 3] = 1.0
        
        # lidar2img = viewpad @ lidar2cam_rt.T から lidar2cam_rt を復元
        # lidar2cam_rt.T = viewpad^(-1) @ lidar2img
        
        # viewpadの上3行を使用して擬似逆行列を計算
        viewpad_inv = np.linalg.inv(viewpad)
        
        # lidar2cam_rt.T の計算
        lidar2cam_rt_T = np.dot(viewpad_inv, lidar2img_matrix)
        
        # lidar2cam_rt = (lidar2cam_rt.T).T
        lidar2cam_rt = lidar2cam_rt_T.T
        
        # 回転行列と平行移動ベクトルを抽出
        rotation_matrix = lidar2cam_rt[:3, :3]
        translation = lidar2cam_rt[:3, 3]
        assert np.allclose(viewpad @ lidar2cam_rt.T, lidar2img_matrix, atol=1e-5)
        
        # 回転行列からクォータニオンへの変換
        # TODO: ValueError: Matrix must be orthogonal, i.e. its transpose should be its inverse
        import pdb;pdb.set_trace()
        quaternion = pyquaternion_Quaternion(matrix=rotation_matrix)
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        qw = quaternion.w
        
        # AutowareのカメラIDを取得
        autoware_camera_id = vad_to_autoware_camera_map[vad_camera_id]
        
        # TransformStampedメッセージの作成
        transform_stamped = TransformStamped()
        transform_stamped.header.stamp = timestamp
        transform_stamped.header.frame_id = "base_link"  # LiDARのフレーム
        transform_stamped.child_frame_id = f"camera{autoware_camera_id}/optical_link"  # カメラのフレーム
        
        # 平行移動の設定
        transform_stamped.transform.translation.x = float(translation[0])
        transform_stamped.transform.translation.y = float(translation[1])
        transform_stamped.transform.translation.z = float(translation[2])
        
        # 回転の設定（クォータニオン）
        transform_stamped.transform.rotation.x = float(qx)
        transform_stamped.transform.rotation.y = float(qy)
        transform_stamped.transform.rotation.z = float(qz)
        transform_stamped.transform.rotation.w = float(qw)
        
        # TFMessageに追加
        tf_msg.transforms.append(transform_stamped)
    
    return tf_msg

def create_camera_info_messages(timestamp: Time) -> dict[int, CameraInfo]:
    """
    カメラの内部パラメータからCameraInfoメッセージを作成

    Args:
        timestamp: メッセージのタイムスタンプ

    Returns:
        dict[int, CameraInfo]: AutowareカメラIDをキーとするCameraInfoメッセージの辞書
    """
    # カメラの内部パラメータの設定
    cameras_intrinsics = {
        # VADのカメラID: intrinsic
        0: np.array([  # CAM_FRONT
            [1.25281310e+03, 0.00000000e+00, 8.26588115e+02],
            [0.00000000e+00, 1.25281310e+03, 4.69984663e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        1: np.array([  # CAM_FRONT_RIGHT
            [1.25674851e+03, 0.00000000e+00, 8.17788757e+02],
            [0.00000000e+00, 1.25674851e+03, 4.51954178e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        2: np.array([  # CAM_FRONT_LEFT
            [1.25786253e+03, 0.00000000e+00, 8.27241063e+02],
            [0.00000000e+00, 1.25786253e+03, 4.50915498e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        3: np.array([  # CAM_BACK
            [796.89106345, 0.0, 857.77743269],
            [0.0, 796.89106345, 476.88489884],
            [0.0, 0.0, 1.0]
        ]),
        4: np.array([  # CAM_BACK_LEFT
            [1.25498606e+03, 0.00000000e+00, 8.29576933e+02],
            [0.00000000e+00, 1.25498606e+03, 4.67168056e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]),
        5: np.array([  # CAM_BACK_RIGHT
            [1.24996293e+03, 0.00000000e+00, 8.25376805e+02],
            [0.00000000e+00, 1.24996293e+03, 4.62548164e+02],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
    }
    
    # VADカメラIDからAutowareカメラIDへのマッピング
    vad_to_autoware_camera_map = {
        0: 0,  # CAM_FRONT -> camera0
        1: 4,  # CAM_FRONT_RIGHT -> camera4
        2: 2,  # CAM_FRONT_LEFT -> camera2
        3: 1,  # CAM_BACK -> camera1
        4: 3,  # CAM_BACK_LEFT -> camera3
        5: 5   # CAM_BACK_RIGHT -> camera5
    }
    
    # 結果を格納する辞書
    camera_info_msgs = {}
    
    # 各カメラのCameraInfoメッセージを作成
    for vad_camera_id, intrinsic in cameras_intrinsics.items():
        autoware_camera_id = vad_to_autoware_camera_map[vad_camera_id]
        
        # intrinsicからviewpadを作成
        viewpad = np.zeros((4, 4), dtype=np.float32)
        viewpad[:3, :3] = intrinsic
        viewpad[3, 3] = 1.0
        
        # CameraInfoメッセージの作成
        camera_info = CameraInfo()
        camera_info.header.stamp = timestamp
        camera_info.header.frame_id = f"camera{autoware_camera_id}/optical_link"
        
        # 画像サイズの設定（一般的な値を仮定）
        camera_info.width = 1920
        camera_info.height = 1080
        
        # 歪み係数（ここでは歪みなしと仮定）
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # カメラ行列K（内部パラメータ）
        camera_info.k = [
            intrinsic[0, 0], intrinsic[0, 1], intrinsic[0, 2],
            intrinsic[1, 0], intrinsic[1, 1], intrinsic[1, 2],
            intrinsic[2, 0], intrinsic[2, 1], intrinsic[2, 2]
        ]
        
        # 射影行列P（プロジェクション行列の上3行）
        # intrinsicから直接作成
        camera_info.p = [
            intrinsic[0, 0], intrinsic[0, 1], intrinsic[0, 2], 0.0,
            intrinsic[1, 0], intrinsic[1, 1], intrinsic[1, 2], 0.0,
            intrinsic[2, 0], intrinsic[2, 1], intrinsic[2, 2], 0.0
        ]
        
        # 回転行列R（ここでは単位行列と仮定）
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        
        # ディストーションモデルの設定
        camera_info.distortion_model = "plumb_bob"
        
        # 結果に追加
        camera_info_msgs[autoware_camera_id] = camera_info
    
    return camera_info_msgs

def unix_to_ros_time(unix_time_sec: float) -> Time:
    """
    Unixタイムスタンプ(秒)をROS2のTimeメッセージに変換

    Args:
        unix_time_sec: Unixタイムスタンプ(秒)

    Returns:
        Time: ROS2のTimeメッセージ
    """
    sec = int(unix_time_sec)
    nanosec = int((unix_time_sec - sec) * 1e9)
    return Time(sec=sec, nanosec=nanosec)

def write_to_rosbag(writer, topic: str, msg, timestamp: Time):
    """
    メッセージをROSバッグに書き込む
    
    Args:
        writer: ROSバッグライター
        topic: トピック名
        msg: 書き込むメッセージ
        timestamp: メッセージのタイムスタンプ
    """
    # タイムスタンプをナノ秒に変換（小数点以下の精度を保持）
    ros_timestamp = int(timestamp.sec * 1e9) + timestamp.nanosec
    writer.write(topic, serialize_message(msg), ros_timestamp)

def main():
    parser = argparse.ArgumentParser(
        description="Convert can_bus.bin data to an MCAP rosbag using rosbag2_py with VAD->Autoware mapping."
    )
    parser.add_argument("--config", default="/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/private/rosbag_conversion_config.yaml", help="Path to rosbag_conversion_config.yaml")

    args = parser.parse_args()

    # 設定ファイルの読み込み
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 設定値の取得
    input_dir = config.get("input_dir", "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data")
    n_frames = config.get("n_frames", 30)
    output_file = config.get("output_file", "output_can_bus.mcap")
    init_time = config.get("init_time", 1672531200)  # Unix time (秒)
    cycle_time_ms = config.get("cycle_time_ms", 100)  # 動作周期（ミリ秒）
    
    # ROSバッグの初期化
    writer = rosbag2_py.SequentialWriter()
    
    # ストレージオプションの設定
    storage_options = rosbag2_py._storage.StorageOptions(
        uri=output_file,
        storage_id="mcap"
    )
    
    # 変換オプションの設定
    converter_options = rosbag2_py._storage.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    
    # バッグファイルのオープン
    writer.open(storage_options, converter_options)
    
    # メタデータの作成と登録
    camera_topic_types = []
    for autoware_camera_id in range(6):
        camera_topic_types.append((f"/sensing/camera/camera{autoware_camera_id}/camera_info", "sensor_msgs/msg/CameraInfo"))
    topic_types = [
        ("/sensing/imu/tamagawa/imu_raw", "sensor_msgs/msg/Imu"),
        ("/localization/kinematic_state", "nav_msgs/msg/Odometry"),
        ("/tf_static", "tf2_msgs/msg/TFMessage"),
    ] + camera_topic_types
    
    # トピックの情報を登録
    for topic_name, topic_type in topic_types:
        topic_info = rosbag2_py._storage.TopicMetadata(
            name=topic_name,
            type=topic_type,
            serialization_format="cdr"
        )
        writer.create_topic(topic_info)
    
    # 各フレームのデータ処理
    can_bus_data_dict: dict[int, np.ndarray] = {}
    shift_data_dict: dict[int, np.ndarray] = {}
    lidar2img_data_dict: dict[int, dict[int, np.ndarray]] = {}
    for frame in range(1, n_frames + 1):
        lidar2img_data_dict[frame] = {}
        frame_dir = os.path.join(input_dir, str(frame))
        can_bus_bin_path = os.path.join(frame_dir, "img_metas.0[can_bus].bin")

        shift_bin_path = os.path.join(frame_dir, "img_metas.0[shift].bin")

        lidar2img_bin_path = os.path.join(frame_dir, "img_metas.0[lidar2img].bin")
        
        if not os.path.exists(can_bus_bin_path):
            print(f"Warning: {can_bus_bin_path} does not exist, skipping frame {frame}")
            continue

        if not os.path.exists(shift_bin_path):
            print(f"Warning: {shift_bin_path} does not exist, skipping frame {frame}")
            continue

        if not os.path.exists(lidar2img_bin_path):
            print(f"Warning: {lidar2img_bin_path} does not exist, skipping frame {frame}")
            continue
        
        # タイムスタンプの計算
        timestamp_sec = init_time + (frame - 1) * (cycle_time_ms / 1000.0)
        ros_timestamp = unix_to_ros_time(timestamp_sec)
        
        # バイナリファイルをバイト単位で読み込む
        with open(can_bus_bin_path, "rb") as f_can_bus:
            can_bus_binary = f_can_bus.read()

        with open(shift_bin_path, "rb") as f_shift:
            shift_binary = f_shift.read()

        with open(lidar2img_bin_path, "rb") as f_lidar2img:
            lidar2img_binary = f_lidar2img.read()

        # バイナリデータをnumpy配列に変換
        can_bus_data = np.frombuffer(can_bus_binary, dtype=np.float32)
        can_bus_data_dict[frame] = can_bus_data

        shift_data = np.frombuffer(shift_binary, dtype=np.float32)
        shift_data_dict[frame] = shift_data

        lidar2img_data = np.frombuffer(lidar2img_binary, dtype=np.float32)
        for vad_camera_id in range(6):
            lidar2img_data_dict[frame][vad_camera_id] = lidar2img_data[16*vad_camera_id:16*(vad_camera_id+1)]

        # 各トピックへの変換と書き込み        
        # 1. IMUの変換と書き込み
        imu_msg = convert_bin_to_imu(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/sensing/imu/tamagawa/imu_raw", imu_msg, ros_timestamp)
        
        # 2. 運動学状態の変換と書き込み
        kinematic_msg = convert_bin_to_kinematic_state(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/localization/kinematic_state", kinematic_msg, ros_timestamp)

        # # 3. lidar2imgのtf_staticへの変換と書き込み
        # # base_link(lidar) to camera{vad_camera_id}/optical_link
        # tf_static_msg = convert_bin_to_tf_static(lidar2img_data_dict[frame], ros_timestamp)
        # write_to_rosbag(writer, "/tf_static", tf_static_msg, ros_timestamp)

        # # 4. intrinsicsのcamera_infoへの変換と書き込み
        # camera_infos = create_camera_info_messages(ros_timestamp)
        # for autoware_camera_id in range(6):
        #     write_to_rosbag(writer, f"/sensing/camera/camera{autoware_camera_id}/camera_info", camera_infos[autoware_camera_id], ros_timestamp)
        
        print(f"Processed frame {frame}")
    
    # バッグファイルのクローズ
    del writer
    print(f"Successfully created rosbag: {output_file}")

    # reconstructed can_busのバイナリを作成
    reconstructed_can_bus: dict[int, np.ndarray] = reconstruct_can_bus_from_rosbag(
        output_file,
        init_time,
        cycle_time_ms
    )

    reconstructed_lidar2img: dict[int, np.ndarray] = reconstruct_lidar2img_from_rosbag(
        output_file,
        init_time,
        cycle_time_ms
    )

    # reconstructed can_busのバイナリとcan_bus.binを比較
    for frame_id in reconstructed_can_bus.keys():
        if not np.array_equal(reconstructed_can_bus[frame_id][0:9], can_bus_data_dict[frame_id][0:9]):
            if not np.array_equal(reconstructed_can_bus[frame_id][12:17], can_bus_data_dict[frame_id][12:17]):
                print(f"Error: reconstructed_can_bus[{frame_id}] != can_bus_data[{frame_id}]")
                break

        delta_x, delta_y = reconstructed_can_bus[frame_id][0:2]
        patch_angle_rad = reconstructed_can_bus[frame_id][-2]

        if not np.allclose(calculate_shift(delta_x, delta_y, patch_angle_rad), shift_data_dict[frame_id], atol=0.0000001):
            print(f"Error: calculated_shift on {frame_id} != shift_data_dict[{frame_id}]")
            break

        # import pdb;pdb.set_trace()
        # for vad_camera_id in range(6):
        #     if not np.array_equal(reconstructed_lidar2img[frame_id][vad_camera_id], lidar2img_data_dict[frame_id][vad_camera_id]):
        #         import pdb;pdb.set_trace()

    print("Success: reconstructed can_bus matches can_bus.bin")

def reconstruct_can_bus_from_rosbag(bag_file: str, init_time: float, cycle_time_ms: float) -> dict[int, np.ndarray]:
    """
    ROSバッグから can_bus データを再構築する

    Args:
        bag_file: ROSバッグファイルのパス
        init_time: 初期時刻（Unix時間、秒）
        cycle_time_ms: フレーム間の時間間隔（ミリ秒）

    Returns:
        dict[int, np.ndarray]: フレームIDをキーとするcan_busデータの辞書
    """ 
    # ROSバッグリーダーの初期化
    reader = rosbag2_py.SequentialReader()
    
    storage_options = rosbag2_py._storage.StorageOptions(
        uri=bag_file,
        storage_id="mcap"
    )
    
    converter_options = rosbag2_py._storage.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    
    # バッグファイルを開く
    reader.open(storage_options, converter_options)
    
    # フレームごとのデータを格納する辞書
    reconstructed_data: dict[int, dict[str, any]] = {}
    result: dict[int, np.ndarray] = {}
    
    # メッセージの読み込み
    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()
        # 整数の除算を避けるため、浮動小数点数で計算
        timestamp_sec = float(timestamp_ns) / 1e9
        frame_id = round((timestamp_sec - init_time) / (cycle_time_ms / 1000.0)) + 1
                
        if frame_id not in reconstructed_data:
            reconstructed_data[frame_id] = {}
        
        if topic_name == "/localization/kinematic_state":
            msg = deserialize_message(data, Odometry)
            # 位置情報を取得
            reconstructed_data[frame_id]["translation"] = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            # 姿勢情報を取得
            reconstructed_data[frame_id]["rotation"] = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            # 速度情報を取得
            reconstructed_data[frame_id]["velocity"] = [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y
            ]
            
            # 角速度情報を取得
            reconstructed_data[frame_id]["roll_rate"] = msg.twist.twist.angular.x
            reconstructed_data[frame_id]["pitch_rate"] = msg.twist.twist.angular.y
            reconstructed_data[frame_id]["yaw_rate"] = msg.twist.twist.angular.z
        
        elif topic_name == "/sensing/imu/tamagawa/imu_raw":
            msg = deserialize_message(data, Imu)
            reconstructed_data[frame_id]["acceleration"] = [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ]
    
    # 各フレームのデータを can_bus 形式に変換
    for frame_id, data in reconstructed_data.items():
        can_bus = np.zeros(18, dtype=np.float32)  # can_busの長さは18と仮定
        
        # translation (0:3)
        can_bus[0:3] = data["translation"]
        
        # rotation (3:7)
        can_bus[3:7] = data["rotation"]
        
        # acceleration (7:10)
        can_bus[7:10] = data["acceleration"]
        
        # angular velocity (10:12)
        can_bus[10] = data["roll_rate"]
        can_bus[11] = data["pitch_rate"]
        can_bus[12] = data["yaw_rate"]
        
        # velocity (13:15)
        can_bus[13:15] = data["velocity"]
        
        # patch_angle
        can_bus[16] = calculate_patch_angle_rad(data["rotation"])  # rad
        if frame_id > 1:
            patch_angle_deg_last_frame = result[frame_id - 1][-2] / np.pi * 180
            patch_angle_deg = can_bus[16] / np.pi * 180
            can_bus[17] =  patch_angle_deg - patch_angle_deg_last_frame  # deg
        else:
            # TODO(Shin-kyoto): use magic number for first frame
            can_bus[17] = -1.0353195667266846
        
        result[frame_id] = can_bus
    
    return result

# if "/camera_info" in topic_name:
#     # AutowareカメラIDを抽出
#     autoware_camera_id = int(topic_name.split("/camera")[-2])

# if "camera" in child_frame_id and "/optical_link" in child_frame_id:
#     # AutowareカメラIDを抽出
#     autoware_camera_id = int(child_frame_id.split("/")[0].split("camera")[-1])

def reconstruct_lidar2img_from_rosbag(bag_file: str, init_time: float, cycle_time_ms: float) -> dict[int, dict[int, np.ndarray]]:
    """
    ROSバッグからlidar2imgデータを再構築する

    Args:
        bag_file: ROSバッグファイルのパス
        init_time: 初期時刻（Unix時間、秒）
        cycle_time_ms: フレーム間の時間間隔（ミリ秒）

    Returns:
        dict[int, dict[int, np.ndarray]]: フレームIDとVADカメラIDをキーとするlidar2imgデータの二重辞書
    """
    # VADカメラIDからAutowareカメラIDへのマッピング
    vad_to_autoware_camera_map = {
        0: 0,  # CAM_FRONT -> camera0
        1: 4,  # CAM_FRONT_RIGHT -> camera4
        2: 2,  # CAM_FRONT_LEFT -> camera2
        3: 1,  # CAM_BACK -> camera1
        4: 3,  # CAM_BACK_LEFT -> camera3
        5: 5   # CAM_BACK_RIGHT -> camera5
    }
    
    # Autowareカメラ名からVADカメラIDへの逆マッピング
    autoware_to_vad_camera_map = {v: k for k, v in vad_to_autoware_camera_map.items()}
    
    # ROSバッグリーダーの初期化
    reader = rosbag2_py.SequentialReader()
    
    storage_options = rosbag2_py._storage.StorageOptions(
        uri=bag_file,
        storage_id="mcap"
    )
    
    converter_options = rosbag2_py._storage.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    
    # バッグファイルを開く
    reader.open(storage_options, converter_options)
    
    # カメラ内部パラメータを格納する辞書
    cameras_intrinsics = {}  # AutowareカメラIDをキーとする辞書
    
    # まずはすべてのカメラの内部パラメータを読み込む
    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()
        
        if "/camera_info" in topic_name:
            # AutowareカメラIDを抽出
            autoware_camera_id = int(topic_name.split("/camera")[-2])

            
            # CameraInfoメッセージをデシリアライズ
            camera_info_msg = deserialize_message(data, CameraInfo)
            
            # カメラ行列Kを3x3行列として抽出
            k_matrix = np.array(camera_info_msg.k).reshape(3, 3)
            
            # カメラIDを登録
            cameras_intrinsics[autoware_camera_id] = k_matrix
    
    # バッグファイルを再度開く
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    
    # 結果を格納する辞書
    result: dict[int, dict[int, np.ndarray]] = {}
    
    # tf_staticメッセージを読み込む
    while reader.has_next():
        topic_name, data, timestamp_ns = reader.read_next()
        
        if topic_name == "/tf_static":
            # タイムスタンプからフレームIDを計算
            timestamp_sec = float(timestamp_ns) / 1e9
            frame_id = round((timestamp_sec - init_time) / (cycle_time_ms / 1000.0)) + 1
            
            # このフレームの辞書を初期化
            if frame_id not in result:
                result[frame_id] = {}
            
            # TFメッセージをデシリアライズ
            tf_msg = deserialize_message(data, TFMessage)
            
            # 各カメラのTF変換を処理
            for transform in tf_msg.transforms:
                child_frame_id = transform.child_frame_id
                
                # カメラのTF変換を識別
                if "camera" in child_frame_id and "/optical_link" in child_frame_id:
                    # Autowareカメラ名からカメラIDを抽出
                    autoware_camera_id = int(child_frame_id.split("/")[0].split("camera")[-1])

                    
                    # VADカメラIDに変換
                    if autoware_camera_id not in autoware_to_vad_camera_map:
                        continue
                    vad_camera_id = autoware_to_vad_camera_map[autoware_camera_id]
                    
                    # カメラの内部パラメータを確認
                    if autoware_camera_id not in cameras_intrinsics:
                        print(f"Warning: No camera intrinsics for camera{autoware_camera_id}")
                        continue
                    
                    # 平行移動を取得
                    translation = np.array([
                        transform.transform.translation.x,
                        transform.transform.translation.y,
                        transform.transform.translation.z
                    ])
                    
                    # クォータニオンを取得
                    quaternion = np.array([
                        transform.transform.rotation.x,
                        transform.transform.rotation.y,
                        transform.transform.rotation.z,
                        transform.transform.rotation.w
                    ])
                    
                    # クォータニオンから回転行列への変換
                    rotation_matrix = quaternion_to_rotation_matrix(quaternion)
                    
                    # lidar2cam_rtを構築
                    lidar2cam_rt = np.eye(4, dtype=np.float32)
                    lidar2cam_rt[:3, :3] = rotation_matrix
                    lidar2cam_rt[:3, 3] = translation
                    
                    # lidar2cam_rt.Tを計算
                    lidar2cam_rt_T = lidar2cam_rt.T
                    
                    # カメラの内部パラメータを取得
                    intrinsic = cameras_intrinsics[autoware_camera_id]
                    
                    # intrinsicからviewpadを作成
                    viewpad = np.zeros((4, 4), dtype=np.float32)
                    viewpad[:3, :3] = intrinsic
                    viewpad[3, 3] = 1.0
                    
                    # lidar2img = viewpad @ lidar2cam_rt.T を計算
                    lidar2img = np.dot(viewpad, lidar2cam_rt_T)
                    
                    # 平坦化して保存
                    result[frame_id][vad_camera_id] = lidar2img.flatten()
    
    return result

def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    クォータニオンから回転行列への変換

    Args:
        q: クォータニオン [x, y, z, w]

    Returns:
        np.ndarray: 3x3回転行列
    """
    # クォータニオンを正規化
    q = q / np.linalg.norm(q)
    
    # クォータニオンの要素を展開
    x, y, z, w = q
    
    # 回転行列を計算
    rotation_matrix = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ], dtype=np.float32)
    
    return rotation_matrix

if __name__ == "__main__":
    main()
