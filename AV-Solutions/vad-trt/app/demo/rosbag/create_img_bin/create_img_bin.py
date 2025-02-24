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
target_h, target_w = 384, 640

# 画像の正規化パラメータ（NormalizeMultiviewImage の mean/std に基づく）
mean = np.array([103.530, 116.280, 123.675], dtype=np.float32)
std = np.array([1.0, 1.0, 1.0], dtype=np.float32)

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

# 保存用にバイナリに変換（float32 のまま）
reconstructed_bin = "reconstructed_img.bin"

# 既存の img.bin を読み込む
original_bin_path = os.path.join(img_dir, "img.bin")
orig_data = np.fromfile(original_bin_path, dtype=np.float32)
# 既知の形状は (6, 3, 384, 640)
orig_data = orig_data.reshape((6, 3, target_h, target_w))
print("Original img.bin shape:", orig_data.shape)

# 正規化の逆変換（denormalization）
orig_data = orig_data * std[:, None, None] + mean[:, None, None]

# データの範囲を 0-255 にクリップし、uint8 に変換
orig_data = np.clip(orig_data, 0, 255).astype(np.uint8)

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


base_img_dir = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/data/demo_data"
base_out_dir = "/home/autoware/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag/private/reconstructed_img"
# frame 1 から 36 までループ
for frame_id in range(1, 37):
    frame_str = str(frame_id)
    img_dir = os.path.join(base_img_dir, frame_str)
    
    # 出力ディレクトリを作成
    out_dir = os.path.join(base_out_dir, frame_str)
    os.makedirs(out_dir, exist_ok=True)
    
    imgs_list = []
    missing = False
    for cam in camera_names:
        filename = os.path.join(img_dir, f"{cam}.jpg")
        if not os.path.exists(filename):
            print(f"Frame {frame_id}: File not found: {filename}")
            missing = True
            break
        # mmcv.imread は BGR で読み込むので注意
        img_bgr = mmcv.imread(filename)
        # リサイズ（mmcv.imresize の引数は (width, height)）
        img_bgr_resized = mmcv.imresize(img_bgr, (target_w, target_h))
        # BGRのまま．つまり以下のような処理はかけない．
        # img_rgb = cv2.cvtColor(img_bgr_resized, cv2.COLOR_BGR2RGB)
        imgs_list.append(img_bgr_resized)

    # (N, H, W, C) の形状にする
    imgs_array = np.stack(imgs_list, axis=0)
    print(f"Frame {frame_id}: Reconstructed images shape (HWC): {imgs_array.shape}")

    # (N, H, W, C) → (N, C, H, W)
    imgs_array = imgs_array.transpose(0, 3, 1, 2)
    print(f"Frame {frame_id}: Reconstructed images shape (CHW): {imgs_array.shape}")

    # normalizationをかける(meanをひいてstdで割る)
    imgs_array = (imgs_array - mean[:, None, None]) / std[:, None, None]
    imgs_array = imgs_array.astype(np.float32)

    # 保存用にバイナリに変換（float32 のまま）BGRで保存
    reconstructed_bin_path = os.path.join(out_dir, "reconstructed_img.bin")
    print(f"Frame {frame_id}: Reconstructed binary saved to {reconstructed_bin_path}")
    imgs_array.tofile(reconstructed_bin_path)