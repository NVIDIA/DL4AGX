
## Data Preparation
### Download Nuscenes
Download and prepare nuscenes dataset following UniAD's [instruction](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md) to `./UniAD/data`

### Create a docker container
Step 1: create a docker container and run
```
docker run -it --gpus all --shm-size=8g -v </host/system/path/to/UniAD>:/workspace/UniAD uniad_torch1.12 /bin/bash
```
Step 2: inside the docker container, build `uniad_mmdet3d`
```
cd /workspace/UniAD/third_party/uniad_mmdet3d/
python3 setup.py build develop --user
```


### Generate Preprocessed Data

Inside docker container, generate six inputs to `./UniAD/nuscenes_np/uniad_onnx_input` for ONNX exportation, and `NUM_FRAME` preprocessed inputs to `./UniAD/nuscenes_np/uniad_trt_input` for inference application. By default we set `NUM_FRAME` to `69` which covers the first two scenes, user can choose any number in the range of `[6, 6018]`.
```
cd /workspace/UniAD
PYTHONPATH=$(pwd) python3  ./tools/process_metadata.py --num_frame NUM_FRAME
```


<- Last Page: [Environment Preparation](env_prep.md)

-> Next Page: [UniAD-tiny Traning and Exportation](train_export.md)

