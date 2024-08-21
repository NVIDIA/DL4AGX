
## Data Preparation
### Download Nuscenes
Download and prepare nuscenes dataset following UniAD's [instruction](https://github.com/OpenDriveLab/UniAD/blob/main/docs/DATA_PREP.md) to `./UniAD/data`

### Create a docker container
Step 1: create docker containder (add `-v /host/system/path/to/UniAD/FOLDER:/workspace/UniAD/FOLDER` if any host system `./UniAD/FOLDER` is symbolic link)
```
docker run -it --gpus all --shm-size=8g -v /host/system/path/to/UniAD:/workspace/UniAD -d uniad_torch1.12 /bin/bash
```
Step 2: show container and run 
```
docker ps
docker exec -it CONTAINER_NAME /bin/bash
```
Step 3: inside docker container, build `uniad_mmdet3d`
```
cd /workspace/UniAD/third_party/uniad_mmdet3d/
python3 setup.py build develop --user
```


### Generate Preprocessed Data

Inside docker container, generate `6` preprocessed sample input to `./UniAD/nuscenes_np/uniad_onnx_input` for onnx exporter use, and `NUM_FRAME` preprocessed trt input to `./UniAD/nuscenes_np/uniad_onnx_input` for C++ Inference App use. `5 < NUM_FRAME < 6019`, by default we set `NUM_FRAME = 69` for the first `2` scenes.
```
cd /workspace/UniAD
PYTHONPATH=$(pwd) python3  ./tools/process_metadata.py --num_frame NUM_FRAME
```


<- Last Page: [Environment Preparation](env_prep.md)

-> Next Page: [UniAD-tiny Traning and Exportation](tiny_train_export.md)

