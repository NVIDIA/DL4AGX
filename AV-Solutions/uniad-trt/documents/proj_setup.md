## Project setup for deployment
Step 1: clone `uniad-trt`
```
git clone https://github.com/NVIDIA/DL4AGX.git
cd DL4AGX/AV-Solutions/uniad-trt
git submodule update --init --recursive
```

Step 2: apply a patch to make `UniAD` compatible with `torch1.12` and corresponding `mmcv/mmdet/mmdet3d` version
```
cd UniAD && git apply --reject --whitespace=fix --exclude='*.DS_Store' ../patch/uniad-torch1.12.patch
```

Step 3: apply a patch related to modification of original `UniAD` code for onnx export
```
git apply --reject --whitespace=fix --exclude='*.DS_Store' ../patch/uniad-onnx-export.patch && cd ..
```

Step 4: copy `bev_mmdet3d` to `UniAD`
```
cp -r ./dependencies/BEVFormer_tensorrt/third_party ./UniAD/
```

Step 5: rename `bev_mmdet3d` as `uniad_mmdet3d`
```
mv ./UniAD/third_party/bev_mmdet3d ./UniAD/third_party/uniad_mmdet3d
```

Step 6: apply a patch to borrow more modules and functions from `mmdet3d`
```
cd UniAD
git apply --exclude='*.DS_Store' ../patch/mmdet3d.patch
```

Step 7: copy part of `BEVFormer_tensorrt` util functions to `UniAD` & apply a small patch for onnx export support
```
cd ..
chmod +x ./tools/add_bevformer_tensorrt_support.sh
./tools/add_bevformer_tensorrt_support.sh
cd UniAD
git apply --exclude='*.DS_Store' ../patch/bevformer_tensorrt.patch
```

Step 8: copy `tool/config/helper` files for `UniAD` onnx export
```
cd ..
chmod +x ./tools/add_onnx_export_support.sh
./tools/add_onnx_export_support.sh
chmod +x ./UniAD/tools/*.sh
```

-> Next Page: [Environment Preparation](env_prep.md)
