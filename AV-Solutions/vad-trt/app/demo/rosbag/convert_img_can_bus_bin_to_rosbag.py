#!/usr/bin/env python3
import argparse
import os
import yaml
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage, PointCloud2, PointField, Imu
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from tf2_msgs.msg import TFMessage
from builtin_interfaces.msg import Time
from rclpy.serialization import serialize_message, deserialize_message
import rosbag2_py
from nav_msgs.msg import Odometry

def convert_bin_to_tf(can_bus_data: np.ndarray, timestamp: Time, frame_id: str = "base_link", child_frame_id: str = "map") -> TFMessage:
    """
    CAN busデータから/tfトピックへの変換
    """
    delta_x = float(can_bus_data[0])
    delta_y = float(can_bus_data[1])
    z = 0.0
    
    qx = float(can_bus_data[3])
    qy = float(can_bus_data[4])
    qz = float(can_bus_data[5])
    qw = float(can_bus_data[6])
    
    transform_stamped = TransformStamped()
    transform_stamped.header.stamp = timestamp
    transform_stamped.header.frame_id = frame_id
    transform_stamped.child_frame_id = child_frame_id
    
    transform_stamped.transform.translation = Vector3(x=delta_x, y=delta_y, z=z)
    transform_stamped.transform.rotation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    
    tf_msg = TFMessage()
    tf_msg.transforms = [transform_stamped]
    
    return tf_msg

def convert_bin_to_imu(can_bus_data: np.ndarray, timestamp: Time, frame_id: str = "imu_link") -> Imu:
    """
    CAN busデータから/sensing/imu/tamagawa/imu_rawトピックへの変換
    """
    accel_x = float(can_bus_data[7])
    accel_y = float(can_bus_data[8])
    accel_z = 0.0
    
    angular_velocity_z = float(can_bus_data[12])
    angular_velocity_x = 0.0
    angular_velocity_y = 0.0
    
    imu_msg = Imu()
    imu_msg.header.stamp = timestamp
    imu_msg.header.frame_id = frame_id
    
    imu_msg.linear_acceleration.x = accel_x
    imu_msg.linear_acceleration.y = accel_y
    imu_msg.linear_acceleration.z = accel_z
    
    imu_msg.angular_velocity.x = angular_velocity_x
    imu_msg.angular_velocity.y = angular_velocity_y
    imu_msg.angular_velocity.z = angular_velocity_z
    
    imu_msg.linear_acceleration_covariance = [-1.0] * 9
    imu_msg.angular_velocity_covariance = [-1.0] * 9
    imu_msg.orientation_covariance = [-1.0] * 9
    
    return imu_msg

def convert_bin_to_kinematic_state(can_bus_data: np.ndarray, timestamp: Time, frame_id: str = "base_link") -> Odometry:
    """
    CAN busデータから/localization/kinematic_stateトピックへの変換
    """
    odom_msg = Odometry()
    odom_msg.header.stamp = timestamp
    odom_msg.header.frame_id = "map"
    odom_msg.child_frame_id = frame_id
    
    odom_msg.pose.pose.position.x = float(can_bus_data[0])
    odom_msg.pose.pose.position.y = float(can_bus_data[1])
    odom_msg.pose.pose.position.z = 0.0
    
    odom_msg.pose.pose.orientation.x = float(can_bus_data[3])
    odom_msg.pose.pose.orientation.y = float(can_bus_data[4])
    odom_msg.pose.pose.orientation.z = float(can_bus_data[5])
    odom_msg.pose.pose.orientation.w = float(can_bus_data[6])
    
    odom_msg.twist.twist.linear.x = float(can_bus_data[13])
    odom_msg.twist.twist.linear.y = float(can_bus_data[14])
    odom_msg.twist.twist.linear.z = 0.0
    
    odom_msg.twist.twist.angular.x = 0.0
    odom_msg.twist.twist.angular.y = 0.0
    odom_msg.twist.twist.angular.z = float(can_bus_data[12])
    
    odom_msg.pose.covariance = [0.01] * 36
    odom_msg.twist.covariance = [0.01] * 36
    
    return odom_msg

def write_to_rosbag(writer, topic: str, msg, timestamp: Time):
    """
    メッセージをROSバッグに書き込む
    """
    ros_timestamp = int(timestamp.sec * 1e9) + timestamp.nanosec
    writer.write(topic, serialize_message(msg), ros_timestamp)

def main():
    parser = argparse.ArgumentParser(
        description="Convert img.bin and can_bus.bin data to an MCAP rosbag using rosbag2_py."
    )
    parser.add_argument("--config", default="/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/private/rosbag_conversion_config.yaml", help="Path to rosbag_conversion_config.yaml")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    input_dir = config.get("input_dir", "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data")
    n_frames = config.get("n_frames", 30)
    output_file = config.get("output_file", "output.mcap")
    topic_template = config.get("topic", "/sensing/camera/camera{i}/image_rect_color/compressed")
    image_format = config.get("image_format", "jpeg")
    init_time = config.get("init_time", 1672531200)
    cycle_time_ms = config.get("cycle_time_ms", 100)

    cameras = config.get("cameras", [])
    time_offsets_list = config.get("time_offsets", [])
    time_offsets = {entry["camera_index"]: entry["offset_ms"] for entry in time_offsets_list}

    target_h, target_w = 384, 640
    num_cams = 6
    mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
    std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=output_file, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    writer.open(storage_options, converter_options)

    # カメラトピックの作成
    topics = {}
    for cam in cameras:
        autoware_index = cam["autoware"]
        topic_name = topic_template.format(i=autoware_index)
        topics[cam["vad"]] = topic_name
        topic_metadata = rosbag2_py.TopicMetadata(
            name=topic_name,
            type="sensor_msgs/msg/CompressedImage",
            serialization_format="cdr"
        )
        writer.create_topic(topic_metadata)

    # CANバス関連トピックの作成
    can_bus_topics = [
        ("/tf", "tf2_msgs/msg/TFMessage"),
        ("/sensing/imu/tamagawa/imu_raw", "sensor_msgs/msg/Imu"),
        ("/localization/kinematic_state", "nav_msgs/msg/Odometry")
    ]
    
    for topic_name, topic_type in can_bus_topics:
        topic_info = rosbag2_py.TopicMetadata(
            name=topic_name,
            type=topic_type,
            serialization_format="cdr"
        )
        writer.create_topic(topic_info)

    # LiDARトピックの作成
    lidar_topic_metadata = rosbag2_py.TopicMetadata(
        name="/sensing/lidar/concatenated/pointcloud",
        type="sensor_msgs/msg/PointCloud2",
        serialization_format="cdr"
    )
    writer.create_topic(lidar_topic_metadata)

    for frame in range(1, n_frames + 1):
        frame_dir = os.path.join(input_dir, str(frame))
        
        # 画像データの処理
        img_bin_path = os.path.join(frame_dir, "img.bin")
        if not os.path.exists(img_bin_path):
            print(f"Warning: {img_bin_path} does not exist, skipping frame {frame}")
            continue

        with open(img_bin_path, "rb") as f_img:
            raw_data = f_img.read()
        try:
            arr = np.frombuffer(raw_data, dtype=np.float32).reshape((num_cams, 3, target_h, target_w))
        except Exception as e:
            print(f"Frame {frame}: Error reshaping image data: {e}")
            continue

        # CANバスデータの処理
        can_bus_bin_path = os.path.join(frame_dir, "img_metas.0[can_bus].bin")
        if not os.path.exists(can_bus_bin_path):
            print(f"Warning: {can_bus_bin_path} does not exist, skipping frame {frame}")
            continue

        with open(can_bus_bin_path, "rb") as f_can_bus:
            can_bus_data = np.frombuffer(f_can_bus.read(), dtype=np.float32)

        base_timestamp = init_time + (frame - 1) * (cycle_time_ms / 1000.0)
        ros_timestamp = Time(
            sec=int(base_timestamp),
            nanosec=int((base_timestamp - int(base_timestamp)) * 1e9)
        )

        # CANバスデータの処理
        tf_msg = convert_bin_to_tf(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/tf", tf_msg, ros_timestamp)
        
        imu_msg = convert_bin_to_imu(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/sensing/imu/tamagawa/imu_raw", imu_msg, ros_timestamp)
        
        kinematic_msg = convert_bin_to_kinematic_state(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/localization/kinematic_state", kinematic_msg, ros_timestamp)

        # LiDARデータの処理（ダミー）
        lidar_msg = PointCloud2()
        lidar_msg.header.stamp = ros_timestamp
        lidar_msg.header.frame_id = "lidar"
        lidar_msg.height = 1
        lidar_msg.width = 0
        lidar_msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        write_to_rosbag(writer, "/sensing/lidar/concatenated/pointcloud", lidar_msg, ros_timestamp)

        # 画像データの処理
        for cam in cameras:
            vad_index = cam["vad"]
            autoware_index = cam["autoware"]
            topic_name = topics[vad_index]
            offset_ms = time_offsets[autoware_index]
            cam_timestamp = base_timestamp + (offset_ms / 1000.0)
            cam_ros_timestamp = Time(
                sec=int(cam_timestamp),
                nanosec=int((cam_timestamp - int(cam_timestamp)) * 1e9)
            )

            cam_img = arr[vad_index]
            cam_img = cam_img * std[:, None, None] + mean[:, None, None]
            cam_img = np.clip(cam_img, 0, 255).astype(np.uint8)
            cam_img = cam_img.transpose(1, 2, 0)

            ret, jpeg_encoded = cv2.imencode(".jpg", cam_img)
            if not ret:
                print(f"Frame {frame}, camera {cam['name']}: JPEG encoding failed.")
                continue
            jpeg_bytes = jpeg_encoded.tobytes()

            msg = CompressedImage()
            msg.header = Header()
            msg.header.stamp = cam_ros_timestamp
            msg.header.frame_id = f"camera{autoware_index}"
            msg.format = image_format
            msg.data = jpeg_bytes

            write_to_rosbag(writer, topic_name, msg, cam_ros_timestamp)
            print(f"Frame {frame}, camera {cam['name']} processed")

        print(f"Processed frame {frame}")

    del writer
    print(f"Successfully created rosbag: {output_file}")

if __name__ == "__main__":
    main()
