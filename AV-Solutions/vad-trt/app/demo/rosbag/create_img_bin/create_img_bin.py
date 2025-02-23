import os
import numpy as np
import mmcv
import cv2

# 画像の順序は、VAD の設定に基づくと仮定します
camera_names = [
    "CAM_FRONT", 
    "CAM_FRONT_RIGHT", 
    "CAM_FRONT_LEFT", 
    "CAM_BACK", 
    "CAM_BACK_LEFT", 
    "CAM_BACK_RIGHT"
]

# jpg ファイルが格納されているディレクトリ
img_dir = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data/1"

# 最終的なターゲットサイズ（H, W）
target_h, target_w = 768, 1280

# 各画像を読み込み、リサイズして RGB → (3, H, W) に変換
imgs_list = []
for cam in camera_names:
    filename = os.path.join(img_dir, f"{cam}.jpg")
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        continue
    # mmcv.imread は BGR で読み込むので注意
    img_bgr = mmcv.imread(filename)
    # リサイズ（mmcv.imresize の引数は (width, height)）
    img_bgr_resized = mmcv.imresize(img_bgr, (target_w, target_h))
    # BGR から RGB に変換
    img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
    imgs_list.append(img_rgb)

if len(imgs_list) != 6:
    print("Error: 6 枚の画像が読み込めていません！")
    exit(1)

# (N, H, W, C) の形状にする
imgs_array = np.stack(imgs_list, axis=0)
print("Reconstructed images shape (HWC):", imgs_array.shape)

# (N, H, W, C) → (N, C, H, W)
imgs_array = imgs_array.transpose(0, 3, 1, 2)
print("Reconstructed images shape (CHW):", imgs_array.shape)

# ここで、もし学習時と同じ前処理（正規化→逆正規化など）が行われていれば、
# その処理をここで再現する必要がありますが、今回は単にリサイズした jpg を用います。

# 保存用にバイナリに変換（uint8 のまま）
reconstructed_bin = "reconstructed_img.bin"
imgs_array.tofile(reconstructed_bin)
print(f"Reconstructed binary saved to {reconstructed_bin}")

# 既存の img.bin を読み込む
original_bin_path = os.path.join(img_dir, "img.bin")
orig_data = np.fromfile(original_bin_path, dtype=np.uint8)
# 既知の形状は (6, 3, 768, 1280)
orig_data = orig_data.reshape((6, 3, target_h, target_w))
print("Original img.bin shape:", orig_data.shape)

# 差分を計算
diff = imgs_array.astype(np.int32) - orig_data.astype(np.int32)
print("Difference stats:")
print("  min:", diff.min())
print("  max:", diff.max())
print("  mean:", diff.mean())
print("  std:", diff.std())

# 各画像ごと、各チャネルの統計も比較してみる
for i in range(6):
    for c in range(3):
        mean_recon = imgs_array[i, c].mean()
        mean_orig = orig_data[i, c].mean()
        print(f"Image {i+1} Channel {c}: reconstructed mean={mean_recon:.2f}, original mean={mean_orig:.2f}")
