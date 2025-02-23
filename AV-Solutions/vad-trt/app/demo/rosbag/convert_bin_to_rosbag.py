#!/usr/bin/env python3
import argparse
import os
import yaml
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Header
from rclpy.serialization import serialize_message
import rosbag2_py

def main():
    parser = argparse.ArgumentParser(
        description="Convert img.bin images to an MCAP rosbag using rosbag2_py with VAD->Autoware mapping."
    )
    parser.add_argument("--config", default="/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/private/rosbag_conversion_config.yaml", help="Path to rosbag_conversion_config.yaml")

    args = parser.parse_args()

    # 設定ファイルの読み込み
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 設定値の取得
    input_dir = config.get("input_dir", "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data")
    n_frames = config.get("n_frames", 30)
    output_file = config.get("output_file", "output.mcap")
    topic_template = config.get("topic", "/sensing/camera/camera{i}/image_rect_color/compressed")
    image_format = config.get("image_format", "jpeg")
    init_time = config.get("init_time", 1672531200)  # Unix time (秒)
    cycle_time_ms = config.get("cycle_time_ms", 100)  # 動作周期（ミリ秒）

    # カメラの mapping 設定（config の cameras リスト）
    cameras = config.get("cameras", [])
    # time_offsets: リストから辞書に変換（キー: VAD の camera_index）
    time_offsets_list = config.get("time_offsets", [])
    time_offsets = {entry["camera_index"]: entry["offset_ms"] for entry in time_offsets_list}

    # ここでは、img.bin に保存されている画像データは (6, 3, 768, 1280) と仮定
    target_h, target_w = 768, 1280
    num_cams = 6

    # rosbag2_py の SequentialWriter のセットアップ
    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=output_file, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    writer.open(storage_options, converter_options)

    # 各カメラごとに topic を作成する
    # config の cameras リストには、各項目に "name", "vad", "autoware" がある
    topics = {}  # キー: vad index, 値: topic name
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

    # 各フレームごとに処理
    for frame in range(1, n_frames + 1):
        frame_dir = os.path.join(input_dir, str(frame))
        img_bin_path = os.path.join(frame_dir, "img.bin")
        if not os.path.exists(img_bin_path):
            print(f"Warning: {img_bin_path} does not exist, skipping frame {frame}")
            continue

        # img.bin を読み込み、(6, 3, 768, 1280) の uint8 配列に変換
        with open(img_bin_path, "rb") as f_img:
            raw_data = f_img.read()
        try:
            arr = np.frombuffer(raw_data, dtype=np.uint8).reshape((num_cams, 3, target_h, target_w))
        except Exception as e:
            print(f"Frame {frame}: Error reshaping data: {e}")
            continue

        # 基本タイムスタンプ（フレームごと）
        base_timestamp = init_time + (frame - 1) * (cycle_time_ms / 1000.0)

        # 各カメラについてメッセージ作成
        for cam in cameras:
            vad_index = cam["vad"]
            autoware_index = cam["autoware"]
            topic_name = topics[vad_index]
            # 時刻オフセットを加味（offset_ms は VAD index に対応）
            offset_ms = time_offsets.get(vad_index, 0)
            cam_timestamp = base_timestamp + (offset_ms / 1000.0)

            # arr から対象カメラの画像を抽出 → shape: (3, 768, 1280)
            cam_img = arr[vad_index]
            # (3, H, W) → (H, W, 3)
            cam_img = cam_img.transpose(1, 2, 0)
            # ここでは、学習パイプラインで to_rgb=True が適用されていると仮定し、保存された img.bin は RGB 順
            # JPEG 圧縮
            ret, jpeg_encoded = cv2.imencode(".jpg", cam_img)
            if not ret:
                print(f"Frame {frame}, camera {cam['name']}: JPEG encoding failed.")
                continue
            jpeg_bytes = jpeg_encoded.tobytes()

            # CompressedImage メッセージの生成
            msg = CompressedImage()
            msg.header = Header()
            sec = int(cam_timestamp)
            nsec = int((cam_timestamp - sec) * 1e9)
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nsec
            msg.header.frame_id = f"camera{autoware_index}"
            msg.format = image_format  # "jpeg"
            msg.data = jpeg_bytes

            serialized_msg = serialize_message(msg)
            mcap_timestamp = int(cam_timestamp * 1e9)
            writer.write(topic_name, serialized_msg, mcap_timestamp)
            print(f"Frame {frame}, camera {cam['name']} (VAD {vad_index} -> autoware {autoware_index}) processed with timestamp: {mcap_timestamp}")

    # クローズして終了
    del writer

if __name__ == "__main__":
    main()
