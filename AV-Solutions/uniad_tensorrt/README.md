# Planning-Oriented End-to-End DNN:​ Adaptation, Acceleration, and Deployment​ on DRIVE Orin

## Re-create UniAD deployment project
### Steps for Clones, Copies, Downloads, and Modifications
Step 1: clone `uniad_tensorrt`
```
git clone uniad_tensorrt
cd uniad_tensorrt
git submodule update --init --recursive
```

Step 2: apply a patch to make `UniAD` compatible with `torch1.12` and corresponding `mmcv/mmdet/mmdet3d` version
```
cd UniAD && git apply --exclude='*.DS_Store' ../patch/0001-step2-make-UniAD-compatible-with-torch1.12.patch
```

Step 3: apply a patch related to modification of original `UniAD` code for onnx export
```
git apply --exclude='*.DS_Store' ../patch/0001-step3-modification-of-UniAD-code-for-onnx-export.patch && cd ..
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
git apply --exclude='*.DS_Store' ../patch/0001-step6-borrowed-more-code-from-mmdet3d-official-sourc.patch
```

Step 7: copy part of `BEVFormer_tensorrt` util functions to `UniAD` & apply a small patch for onnx export support
```
cd ..
chmod +x ./tools/step7.sh
./tools/step7.sh
cd UniAD
git apply --exclude='*.DS_Store' ../patch/0001-step7-modifications-on-derrhub-code-for-onnx-export.patch
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

Step 10: copy C++ inference and visualization App
```
cp -r ./inference_app ./UniAD/
```

Step 11: 

1. download prepare nuscenes dataset as UniAD [instructed](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md) to `./UniAD/data`

2. download [pretrained weights](https://nvidia-my.sharepoint.com/:u:/r/personal/joshp_nvidia_com/Documents/Internal/onnx/UniAD_weights/tiny_imgx0.25_e2e_ep20.pth?csf=1&web=1&e=C8khhs) of UniAD_tiny to `./UniAD/ckpts`

3. download `TensorRT-8.6.15.17`(CUDA 11.4) to `./UniAD/TensorRT-8.6.15.17`


### Steps for Preparing Environments
Step 1: Apply a patch to `nuscenes-devkit` for current env support
```
cd dependencies/nuscenes-devkit
git apply --exclude='*.DS_Store' ../../patch/0001-update-nuscenes_python-sdk-for-torch1.12.patch
cp -r ./python-sdk/nuscenes ../../docker
```
Step 2: build docker image
```
cd ../../docker
docker build -t uniad_torch1.12 -f uniad_torch1.12.dockerfile .
```

Step 3: create docker containder (add `-v /host/system/path/to/UniAD/xxx:/workspace/UniAD/FOLDER` if any host system `./UniAD/FOLDER` is symbolic link)
```
docker run -it --gpus all --shm-size=8g -v /host/system/path/to/UniAD:/workspace/UniAD -d uniad_torch1.12 /bin/bash
```
Step 4: show container and run 
```
docker ps
docker exec -it CONTAINER_NAME /bin/bash
```
Step 5: inside docker container, build `uniad_mmdet3d`
```
cd /workspace/UniAD/third_party/uniad_mmdet3d/
python3 setup.py build develop --user
```

### Generate Preprocessed Data
Inside docker container, generate `5` preprocessed sample input to `./UniAD/nuscenes_np/uniad_onnx_input` for onnx exporter use, and `NUM_FRAME` preprocessed trt input to `./UniAD/nuscenes_np/uniad_onnx_input` for C++ Inference App use. `5 < NUM_FRAME < 6019`, by default we set `NUM_FRAME = 69` for the first `2` scenes.
```
cd /workspace/UniAD
PYTHONPATH=$(pwd) python3  ./tools/process_metadata.py --num_frame NUM_FRAME
```

### The Overall Structure

Please make sure the structure of `UniAD` is as follows:
```
UniAD
├── inference_app/
├── third_party/
│   ├── uniad_mmdet3d/
├── nuscenes_np/
│   ├── uniad_onnx_input/
│   ├── uniad_trt_input/
├── projects/
├── tools/
├── TensorRT-8.6.15.17/
├── ckpts/
│   ├── tiny_imgx0.25_e2e_ep20.pth
├── data/
│   ├── nuscenes/
│   │   ├── can_bus/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-trainval/
│   ├── infos/
│   │   ├── nuscenes_infos_temporal_train.pkl
│   │   ├── nuscenes_infos_temporal_val.pkl
│   ├── others/
│   │   ├── motion_anchor_infos_mode6.pkl
```

## UniAD_tiny Deployment
### Pytorch to ONNX
```
cd /workspace/UniAD
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py ./ckpts/tiny_imgx0.25_e2e_ep20.pth 1
```


### ONNX to TensorRT

Engine has been verified on `TensorRT-8.6.15.17`

#### TensorRT Plugin Compilation:

Change line 16&17 of `/workspace/UniAD/tools/tensorrt_plugin/CMakeList.txt`

from
```
set(TENSORRT_INCLUDE_DIRS /usr/include/x86_64-linux-gnu/)
set(TENSORRT_LIBRARY_DIRS /usr/lib/x86_64-linux-gnu/)
```
to
```
set(TENSORRT_INCLUDE_DIRS /workspace/UniAD/TensorRT-8.6.15.17/include/)
set(TENSORRT_LIBRARY_DIRS /workspace/UniAD/TensorRT-8.6.15.17/lib/)
```


Then complie by

```
cd /workspace/UniAD/tools/tensorrt_plugin
mkdir build
cd build
cmake .. -DCMAKE_TENSORRT_PATH=/workspace/UniAD/TensorRT-8.6.15.17
make -j$(nproc)
make install
```


#### Engine Build
TensorRT FP32 engine build and latency measurement (modify TensorRT version inside `run_trtexec.sh` if needed)
```
cd /workspace/UniAD
./run_trtexec.sh
```


## C++ Inference App and Visualization

See [C++ Inference App Instructions](cpp/)


## Appendix: UniAD_tiny Training

| model | bev size | img size | with bevslicer? | pretrained weights |
| :---: | :---: | :---: | :---:|:---:| 
| UniAD Base  | 200x200  | 1600x928 | Y | [link](https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth) |
| UniAD_tiny | 50x50 | 400x256 | N  | [link](https://nvidia-my.sharepoint.com/:u:/r/personal/joshp_nvidia_com/Documents/Internal/onnx/UniAD_weights/tiny_imgx0.25_e2e_ep20.pth?csf=1&web=1&e=C8khhs) |

Download [pretrained weights](https://nvidia-my.sharepoint.com/:u:/r/personal/joshp_nvidia_com/Documents/Internal/onnx/UniAD_weights/tiny_imgx0.25_e2e_ep20.pth?csf=1&web=1&e=C8khhs) of `UniAD_tiny` to `./UniAD/ckpts` to skip re-train.

### Steps to re-train
Follow [training instructions](https://github.com/OpenDriveLab/UniAD/blob/main/docs/TRAIN_EVAL.md) from official UniAD

Necessary files: 

1. Configs: [stage1](projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py) and [stage2](projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py) for `UniAD_tiny` training.

2. [BEVFormer_tiny weights](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth) for stage1 initialization.