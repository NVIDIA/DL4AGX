import numpy as np
import os
import cv2

# img.bin のパス
bin_path = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data/1/img.bin"
# 出力先のディレクトリ
output_dir = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/private/"

# 画像の正規化パラメータ（NormalizeMultiviewImage の mean/std に基づく）
mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

# バイナリデータを float32 として読み込む
data = np.fromfile(bin_path, dtype=np.float32)

# 画像データの形状 (N, C, H, W) = (6, 3, 384, 640) を想定
images = data.reshape((6, 3, 384, 640))

# データの範囲確認
print("Data range before denormalization: min={}, max={}".format(data.min(), data.max()))

# **正規化の逆変換（denormalization）**
images = images * std[:, None, None] + mean[:, None, None]

# データの範囲を 0-255 にクリップし、uint8 に変換
images = np.clip(images, 0, 255).astype(np.uint8)

# 6 枚の画像を個別に JPEG 形式で保存
for i, img in enumerate(images):
    img_bgr = img.transpose(1, 2, 0)  # (C, H, W) → (H, W, C)

    # OpenCV は BGR 形式なので、そのまま保存
    output_path = os.path.join(output_dir, f"image_{i+1}_denorm.jpg")
    cv2.imwrite(output_path, img_bgr)
    
    print(f"Saved corrected image {i+1} to {output_path}")
