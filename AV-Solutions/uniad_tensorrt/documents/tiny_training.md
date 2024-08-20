
## UniAD Tiny Training
For efficiency when deploying a UniAD model on DRIVE platform, we train a tiny version of UniAD, with a smaller ResNet backbone and reduced image size & bev size.

| model | img backbone | bev size | img size | with bevslicer? | with bev upsample? |
| :---: | :---: | :---: | :---: | :---:|:---:| 
| UniAD Base  | ResNet-101| 200x200  | 1600x928 | Y | Y |
| UniAD Tiny | ResNet-50 | 50x50 | 400x256 | N  | N |


Please follow [training instructions](https://github.com/OpenDriveLab/UniAD/blob/main/docs/TRAIN_EVAL.md) from official UniAD for details on UniAD model training.

To train this variant, the following files are needed:

1. Configs: [stage1](projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py) and [stage2](projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py) for `UniAD_tiny` training.

2. [BEVFormer_tiny weights](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth) for stage1 initialization.

### File Structure

After training, please put `tiny_imgx0.25_e2e_ep20.pth` into `./ckpts` and make sure the structure of `UniAD` is as follows:
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
├── dumped_inputs/
│   ├── uniad_trtexec_fp64/
├── nuscenes_np/
│   ├── uniad_onnx_input/
│   ├── uniad_trt_input/
├── projects/
├── third_party/
│   ├── uniad_mmdet3d/
├── tools/
```

<- Last Page: [Data Preparation](data_prep.md)

-> Next Page: [ONNX and Engine Build](onnx_engine_build.md)

