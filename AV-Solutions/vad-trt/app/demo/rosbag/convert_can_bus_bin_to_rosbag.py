#!/usr/bin/env python3
import argparse
import os
import yaml
import numpy as np

# ROS2 関連のインポート
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from autoware_vehicle_msgs.msg import VelocityReport
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message, deserialize_message
import rosbag2_py
import rclpy

def convert_bin_to_tf(can_bus_data: np.ndarray, timestamp: Time, frame_id: str = "base_link", child_frame_id: str = "map") -> TFMessage:
    """
    CAN busデータから/tfトピックへの変換
    
    Args:
        can_bus_data: CANバスデータ
        timestamp: メッセージのタイムスタンプ
        frame_id: 親フレームID
        child_frame_id: 子フレームID
        
    Returns:
        TFMessage: 変換されたtfメッセージ
    """
    # can_bus[0:3] = ego2global_translation(の差分)
    # can_bus[3:7] = ego2global_rotation(の差分?)
    
    # 変換ベクトルの取得（x, y, zは0と仮定）
    delta_x = float(can_bus_data[0])
    delta_y = float(can_bus_data[1])
    z = 0.0  # z方向は0と仮定
    
    # 回転クォータニオンの取得
    qx = float(can_bus_data[3])
    qy = float(can_bus_data[4])
    qz = float(can_bus_data[5])
    qw = float(can_bus_data[6])
    
    # TransformStampedメッセージの作成
    transform_stamped = TransformStamped()
    transform_stamped.header.stamp = timestamp
    transform_stamped.header.frame_id = frame_id
    transform_stamped.child_frame_id = child_frame_id
    
    # 変換情報の設定
    transform_stamped.transform.translation = Vector3(x=delta_x, y=delta_y, z=z)
    transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    
    # TFMessageの作成
    tf_msg = TFMessage()
    tf_msg.transforms = [transform_stamped]
    
    return tf_msg

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
    accel_z = 0.0
    
    # 角速度の取得（float型に明示的に変換）
    angular_velocity_z = float(can_bus_data[12])  # Yaw角速度
    angular_velocity_x = 0.0
    angular_velocity_y = 0.0
    
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

def convert_bin_to_kinematic_state(can_bus_data: np.ndarray, timestamp: Time, frame_id: str = "base_link") -> VelocityReport:
    """
    CAN busデータから/localization/kinematic_stateトピックへの変換
    
    Args:
        can_bus_data: CANバスデータ
        timestamp: メッセージのタイムスタンプ
        frame_id: フレームID
        
    Returns:
        VelocityReport: 変換された運動学状態メッセージ
    """
    # can_bus[12] = ego_w（yaw角速度）
    # can_bus[13:15] = ego_vx, ego_vy（x,y方向速度）
    
    # 速度の取得（float型に明示的に変換）
    vx = float(can_bus_data[13])
    vy = float(can_bus_data[14])
    
    # 速度の大きさを計算
    speed = float(np.sqrt(vx**2 + vy**2))
    
    # VelocityReportメッセージの作成
    velocity_msg = VelocityReport()
    velocity_msg.header.stamp = timestamp
    velocity_msg.header.frame_id = frame_id
    
    # 速度情報の設定
    velocity_msg.longitudinal_velocity = vx  # x方向速度
    velocity_msg.lateral_velocity = vy      # y方向速度
    velocity_msg.heading_rate = float(can_bus_data[12])  # yaw角速度
    
    return velocity_msg

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
    topic_types = [
        ("tf", "tf2_msgs/msg/TFMessage"),
        ("/sensing/imu/tamagawa/imu_raw", "sensor_msgs/msg/Imu"),
        ("/localization/kinematic_state", "autoware_vehicle_msgs/msg/VelocityReport")
    ]
    
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
    for frame in range(1, n_frames + 1):
        frame_dir = os.path.join(input_dir, str(frame))
        can_bus_bin_path = os.path.join(frame_dir, "img_metas.0[can_bus].bin")
        
        if not os.path.exists(can_bus_bin_path):
            print(f"Warning: {can_bus_bin_path} does not exist, skipping frame {frame}")
            continue
        
        # タイムスタンプの計算
        timestamp_sec = init_time + (frame - 1) * (cycle_time_ms / 1000.0)
        ros_timestamp = unix_to_ros_time(timestamp_sec)
        
        # バイナリファイルをバイト単位で読み込む
        with open(can_bus_bin_path, "rb") as f_can_bus:
            data = f_can_bus.read()
        
        # バイナリデータをnumpy配列に変換
        can_bus_data = np.frombuffer(data, dtype=np.float32)
        can_bus_data_dict[frame] = can_bus_data
        # 各トピックへの変換と書き込み
        # 1. TFの変換と書き込み
        tf_msg = convert_bin_to_tf(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "tf", tf_msg, ros_timestamp)
        
        # 2. IMUの変換と書き込み
        imu_msg = convert_bin_to_imu(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/sensing/imu/tamagawa/imu_raw", imu_msg, ros_timestamp)
        
        # 3. 運動学状態の変換と書き込み
        kinematic_msg = convert_bin_to_kinematic_state(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/localization/kinematic_state", kinematic_msg, ros_timestamp)
        
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

    # reconstructed can_busのバイナリとcan_bus.binを比較
    for frame_id in reconstructed_can_bus.keys():
        if not np.array_equal(reconstructed_can_bus[frame_id][0:9], can_bus_data_dict[frame_id][0:9]):
            if not np.array_equal(reconstructed_can_bus[frame_id][12:15], can_bus_data_dict[frame_id][12:15]):
                import pdb;pdb.set_trace()
                print(f"Error: reconstructed_can_bus[{frame_id}] != can_bus_data[{frame_id}]")
                break
    else:
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
        
        if topic_name == "tf":
            msg = deserialize_message(data, TFMessage)
            transform = msg.transforms[0].transform
            reconstructed_data[frame_id]["translation"] = [
                transform.translation.x,
                transform.translation.y,
                0.0  # z方向は0と仮定
            ]
            reconstructed_data[frame_id]["rotation"] = [
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
                transform.rotation.w
            ]
        
        elif topic_name == "/sensing/imu/tamagawa/imu_raw":
            msg = deserialize_message(data, Imu)
            reconstructed_data[frame_id]["acceleration"] = [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y
            ]
        
        elif topic_name == "/localization/kinematic_state":
            msg = deserialize_message(data, VelocityReport)
            reconstructed_data[frame_id]["velocity"] = [
                msg.longitudinal_velocity,
                msg.lateral_velocity
            ]
            reconstructed_data[frame_id]["yaw_rate"] = msg.heading_rate
    
    # 各フレームのデータを can_bus 形式に変換
    for frame_id, data in reconstructed_data.items():
        can_bus = np.zeros(18, dtype=np.float32)  # can_busの長さは18と仮定
        
        # translation (0:3)
        can_bus[0:3] = data["translation"]
        
        # rotation (3:7)
        can_bus[3:7] = data["rotation"]
        
        # acceleration (7:9)
        can_bus[7:9] = data["acceleration"]
        
        # yaw_rate (12)
        can_bus[12] = data["yaw_rate"]
        
        # velocity (13:15)
        can_bus[13:15] = data["velocity"]
        
        # patch_angle（16:18）は0とする
        can_bus[16] = 0.0  # rad
        can_bus[17] = 0.0  # deg
        
        result[frame_id] = can_bus
    
    return result

if __name__ == "__main__":
    main()