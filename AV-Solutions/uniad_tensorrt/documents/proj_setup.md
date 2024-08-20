## Re-create UniAD deployment project
Step 1: clone `uniad_tensorrt`
```
git clone uniad_tensorrt
cd uniad_tensorrt
git submodule update --init --recursive
```

Step 2: apply a patch to make `UniAD` compatible with `torch1.12` and corresponding `mmcv/mmdet/mmdet3d` version
```
cd UniAD && git apply --exclude='*.DS_Store' ../patch/uniad-torch1.12.patch
```

Step 3: apply a patch related to modification of original `UniAD` code for onnx export
```
git apply --exclude='*.DS_Store' ../patch/uniad-onnx-export.patch && cd ..
```

Step 4: copy `bev_mmdet3d` to `UniAD`
```
cp -r ./dependencies/BEVFormer_tensorrt/third_party ./UniAD/
```

Step 5: rename `bev_mmdet3d` as `uniad_mmdet3d`
```
mv ./UniAD/third_party/bev_mmdet3d ./UniAD/third_party/uniad_mmdet3d
```

Step 6: apply a patch to borrow more modules and functions from `mmdet3d` official source code
```
cd UniAD
git apply --exclude='*.DS_Store' ../patch/mmdet3d.patch
```

Step 7: copy part of `BEVFormer_tensorrt` util functions to `UniAD` & apply a small patch for onnx export support
```
cd ..
chmod +x ./tools/step7.sh
./tools/step7.sh
cd UniAD
git apply --exclude='*.DS_Store' ../patch/bevformer_tensorrt.patch
```

Step 8: copy `BEVFormer_tensorrt` plugin & rename & replace the `CMakeLists.txt` with ours
```
cd ..
cp -r ./dependencies/BEVFormer_tensorrt/TensorRT ./UniAD/tools/
mv ./UniAD/tools/TensorRT ./UniAD/tools/tensorrt_plugin
cp ./tools/CMakeLists.txt ./UniAD/tools/tensorrt_plugin/
```

Step 9: copy our prepared tool/config/helper files for `UniAD` onnx export
```
chmod +x ./tools/step9.sh
./tools/step9.sh
chmod +x ./UniAD/tools/*.sh
```

-> Next Page: [Environment Preparation](env_prep.md)
