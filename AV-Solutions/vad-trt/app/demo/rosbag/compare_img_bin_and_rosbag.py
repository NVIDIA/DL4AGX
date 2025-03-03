# img.binを読み込み，rosbagの画像データと比較する

import cv2
import numpy as np
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

autoware_to_vad_camera_idx = {
    0: 0,
    1: 3,
    2: 2,
    3: 4,
    4: 1,
    5: 5
}


def main():

    # rosbagを読み込み
    bag = rosbag2_py.SequentialReader()
    
    # StorageOptionsとConverterOptionsを設定
    storage_options = StorageOptions(
        uri="/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/output_bag/",
        storage_id="sqlite3"
    )
    
    converter_options = ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )
    
    # bagファイルを開く
    bag.open(storage_options, converter_options)
    
    bag_images: dict[int, dict[int, np.ndarray]] = {} # bag_images[frame_id][autoware_camera_idx]

    
    # 各フレームIDに空の辞書を初期化
    for frame_id in range(1, 40):
        bag_images[frame_id] = {}
    current_frame_id = 1    
    while bag.has_next():
        topic, msg_data, t = bag.read_next()
        for autoware_camera_idx in range(6):
            if f"/sensing/camera/camera{autoware_camera_idx}/image_rect_color/compressed" in topic:
                # バイナリデータをCompressedImageメッセージにデシリアライズ
                msg = deserialize_message(msg_data, CompressedImage)
                # JPEGデータをデコード
                img = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
                # uint8 -> float32に変換
                img = img.astype(np.float32)
                # 正規化（元のimg.binの形式に合わせる）
                mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
                std = np.array([1.0, 1.0, 1.0], dtype=np.float32)
                img = (img - mean) / std
                # (H, W, 3) -> (3, H, W)に変換
                img = img.transpose(2, 0, 1)
                vad_camera_idx = autoware_to_vad_camera_idx[autoware_camera_idx]
                bag_images[current_frame_id][vad_camera_idx] = img
        if bag_images[current_frame_id].keys() == {0, 1, 2, 3, 4, 5}:
            current_frame_id += 1
            # all cameras are loaded, so frame_id is incremented

    # compare img.bin and rosbag images
    for frame_id in bag_images.keys():
        # img.binを読み込み
        img_bin = np.fromfile(f"/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data/{frame_id}/img.bin", dtype=np.float32)
        img_bin = img_bin.reshape(6, 3, 384, 640)
        print(img_bin.shape)
        for i in range(6):
            # comare shape
            if img_bin[i].shape != bag_images[frame_id][i].shape:
                print(f"shape mismatch: {img_bin[i].shape} != {bag_images[frame_id][i].shape}")
                import pdb; pdb.set_trace()
            # compare pixel value with tolerance
            diff = np.abs(img_bin[i] - bag_images[frame_id][i])
            if (diff > 4.0).any():
                print(f"pixel value mismatch greater than 4.0 found")
                print(f"Max difference: {np.max(diff)}")
                print(f"Mean difference: {np.mean(diff)}")
                # 差分が4.0を超える位置を特定
                large_diff_indices = np.where(diff > 4.0)
                print(f"Indices with large difference:")
                print(f"Channel: {large_diff_indices[0]}")
                print(f"Height: {large_diff_indices[1]}")
                print(f"Width: {large_diff_indices[2]}")
                print(f"Values at these positions:")
                print(f"img_bin values: {img_bin[i][large_diff_indices]}")
                print(f"bag_images values: {bag_images[frame_id][i][large_diff_indices]}")
                # max differenceのindexを探す
                max_diff_index = np.argmax(diff)
                # max differenceの位置を探す
                max_diff_position = np.unravel_index(max_diff_index, diff.shape)
                print(f"max difference position: {max_diff_position}")
                print(f"img_bin value at max difference position: {img_bin[i][max_diff_position]}")
                print(f"bag_images value at max difference position: {bag_images[frame_id][i][max_diff_position]}")
                import pdb; pdb.set_trace()
main()
