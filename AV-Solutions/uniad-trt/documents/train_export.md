
## Model Training and Exportation
### Training
We trained a UniAD-tiny model for deployment, the differences compared to the original model are summarized in the following table:
| model | img backbone | bev size | img size | with bevslicer? | with bev upsample? |
| :---: | :---: | :---: | :---: | :---:|:---:| 
| UniAD  | ResNet-101| 200x200  | 1600x928 | Y | Y |
| UniAD-tiny | ResNet-50 | 50x50 | 400x256 | N  | N |


Please follow [training instructions](https://github.com/OpenDriveLab/UniAD/blob/main/docs/TRAIN_EVAL.md) from official UniAD for details on UniAD model training.

To train this variant, the following files are needed:

1. Configs: [stage1](../projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py) and [stage2](../projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py) for `UniAD-tiny` training.

2. Download BEVFormer-tiny weights from [BEVFormer Model Zoo](https://github.com/fundamentalvision/BEVFormer?tab=readme-ov-file#model-zoo) for stage1 initialization.

Launch UniAD-tiny training and evaluation in a separate training docker container and separate UniAD project\
Step 1: Create a separate training project from scratch: clone a UniAD Repo to a separate project and checkout, apply patch to support torch-1.12 training, borrow `third_party.uniad_mmdet3d`, copy tools and configs, and download weights
```
cd uniad-trt
mkdir UniAD_train && cd UniAD_train
git clone https://github.com/OpenDriveLab/UniAD.git
cd UniAD
git checkout 02fa68c5
git apply <path_to_uniad-trt/patch/uniad-tiny-training-support.patch>
cp -r <path_to_uniad-trt/UniAD/third_party> .
cp <path_to_uniad-trt/tools/postprocess_bevformer_tiny_epoch_24_pth.py> ./tools/
cp <path_to_uniad-trt/projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py> ./projects/configs/stage1_track_map/
cp <path_to_uniad-trt/projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py> ./projects/configs/stage2_e2e/
mkdir ckpts
wget -O ./ckpts/bevformer_tiny_epoch_24.pth <link_to_BEVFormer-tiny_weights>
```
Step 2: prepare nuscenes dataset to `<uniad-trt/UniAD_train/UniAD/data>`\
Step 3: launch a training docker container for UniAD-tiny training, compile `third_party.uniad_mmdet3d` and modify BEVformer-tiny pretrained weights
```
docker run -it --gpus all --shm-size=8g -v </host/system/path/to/UniAD_train/UniAD>:/workspace/UniAD_train/UniAD uniad_torch1.12 /bin/bash
cd /workspace/UniAD_train/UniAD/third_party/uniad_mmdet3d/
python3 setup.py build develop --user
cd /workspace/UniAD_train/UniAD
python3 ./tools/postprocess_bevformer_tiny_epoch_24_pth.py
```
Step 4: train and evaluate
```
./tools/uniad_dist_train.sh ./projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py NUM_GPUs
cp ./projects/work_dirs/stage1_track_map/tiny_imgx0.25_track_map/epoch_6.pth ./ckpts/
mv ./ckpts/epoch_6.pth ./ckpts/tiny_imgx0.25_track_map.pth
./tools/uniad_dist_train.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py NUM_GPUs
cp ./projects/work_dirs/stage2_e2e/tiny_imgx0.25_e2e/epoch_20.pth ./ckpts/
mv ./ckpts/epoch_20.pth ./ckpts/tiny_imgx0.25_e2e.pth
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_dist_eval.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py ./ckpts/tiny_imgx0.25_e2e.pth 1
```

#### File Structure

After training, please put `tiny_imgx0.25_e2e_ep20.pth` into deployment project `<uniad-trt/UniAD/ckpts>` and make sure the structure of `<uniad-trt/UniAD>` is as follows:
```
UniAD
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
├── nuscenes_np/
│   ├── uniad_onnx_input/
│   ├── uniad_trt_input/
├── projects/
├── third_party/
│   ├── uniad_mmdet3d/
├── tools/
```

### Pytorch to ONNX
To export an ONNX model, inside the deployment docker container, please run the following commands
```
cd /workspace/UniAD
CUDA_VISIBLE_DEVICES=0 ./tools/uniad_export_onnx.sh ./projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py ./ckpts/tiny_imgx0.25_e2e_ep20.pth 1
```

Due to legal reasons, we can only provid an [ONNX](../onnx/uniad_tiny_dummy.onnx) model of UniAD-tiny with random weights. Please follow instructions on training to obtain a model with real weights.

<- Last Page: [Data Preparation](data_prep.md)

-> Next Page: [Engine Build, C++ Inference and Visualization](../inference_app_enqueueV3/README.md)

