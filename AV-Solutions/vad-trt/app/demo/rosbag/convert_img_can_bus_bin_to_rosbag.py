#!/usr/bin/env python3
import argparse
import os
import yaml
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage, PointCloud2, PointField
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
import rosbag2_py

from convert_can_bus_bin_to_rosbag import convert_bin_to_imu, convert_bin_to_kinematic_state, write_to_rosbag, convert_bin_to_tf_static, create_camera_info_messages

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

    # tf_staticトピックの作成
    tf_static_topic_metadata = rosbag2_py.TopicMetadata(
        name="/tf_static",
        type="tf2_msgs/msg/TFMessage",
        serialization_format="cdr"
    )
    writer.create_topic(tf_static_topic_metadata)

    # camera_infoトピックの作成
    for autoware_camera_id in range(6):
        camera_info_topic_metadata = rosbag2_py.TopicMetadata(
            name=f"/sensing/camera/camera{autoware_camera_id}/camera_info",
            type="sensor_msgs/msg/CameraInfo",
            serialization_format="cdr"
        )
        writer.create_topic(camera_info_topic_metadata)

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

        # IMUメッセージの処理
        imu_msg = convert_bin_to_imu(can_bus_data, ros_timestamp)
        write_to_rosbag(writer, "/sensing/imu/tamagawa/imu_raw", imu_msg, ros_timestamp)
        
        # 運動学状態メッセージの処理
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

        # lidar2imgのtf_staticへの変換と書き込み
        lidar2img_bin_path = os.path.join(frame_dir, "img_metas.0[lidar2img].bin")
        if os.path.exists(lidar2img_bin_path):
            with open(lidar2img_bin_path, "rb") as f_lidar2img:
                lidar2img_data = np.frombuffer(f_lidar2img.read(), dtype=np.float32)
            
            lidar2img_dict = {}
            for vad_camera_id in range(6):
                lidar2img_dict[vad_camera_id] = lidar2img_data[16*vad_camera_id:16*(vad_camera_id+1)]
            
            tf_static_msg = convert_bin_to_tf_static(lidar2img_dict, ros_timestamp)
            write_to_rosbag(writer, "/tf_static", tf_static_msg, ros_timestamp)

        # camera_infoメッセージの作成と書き込み
        camera_infos = create_camera_info_messages(ros_timestamp)
        for autoware_camera_id in range(6):
            write_to_rosbag(writer, 
                          f"/sensing/camera/camera{autoware_camera_id}/camera_info", 
                          camera_infos[autoware_camera_id], 
                          ros_timestamp)

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
