import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
# img.bin のパス
bin_path = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data/1/img.bin"

# 出力先のディレクトリ
output_dir = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/private/"

# バイナリデータを uint8 として読み込む
data = np.fromfile(bin_path, dtype=np.uint8)
print("Data min:", data.min(), "max:", data.max(), "size:", data.size)

# 画像データは (N, C, H, W) = (6, 3, 768, 1280) の形状と仮定
images = data.reshape((6, 3, 768, 1280))
print("Images shape:", images.shape)

# 各画像ごと、各チャネルの統計量を計算して表示
for i, img in enumerate(images):
    print(f"Image {i+1}:")
    for c in range(3):
        channel = img[c]
        mean = channel.mean()
        std = channel.std()
        mi = channel.min()
        ma = channel.max()
        print(f"  Channel {c}: mean={mean:.2f}, std={std:.2f}, min={mi}, max={ma}")

# 画像データは (N, C, H, W) = (6, 3, 768, 1280) の形状で保存されていると仮定
images = data.reshape((6, 3, 768, 1280))
print("Images shape:", images.shape)  # (6, 3, 768, 1280)

# 6 枚の画像を個別に JPEG 形式で保存
for i, img in enumerate(images):
    # (C, H, W) → (H, W, C) に変換
    img_rgb = img.transpose(1, 2, 0)
    # OpenCV は BGR 形式なので、RGB → BGR に変換
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, f"image_{i+1}.jpg")
    cv2.imwrite(output_path, img_bgr)
    print(f"Saved image {i+1} to {output_path}")
     
