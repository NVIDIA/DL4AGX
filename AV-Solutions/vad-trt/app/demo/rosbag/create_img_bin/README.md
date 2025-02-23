# create_img_bin

## 環境構築

### 最初だけ実施

```bash
uv init create_img_bin
```

```bash
uv venv --python=python3.11 .mmcv
```

### 毎回実施

```bash
source .mmcv/bin/activate
```

```bash
uv pip install --editable .
```

## 実行

```bash
(.mmcv) autoware@dpc2308007:~/ghq/github.com/Shin-kyoto/DL4AGX/AV-Solutions/vad-trt/app/demo/rosbag$ .mmcv/bin/python create_img_bin.py
```