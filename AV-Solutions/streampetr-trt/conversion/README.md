# StreamPETR Model Export
This page showcases the process of exporting a StreamPETR model (.pth) to ONNX models. We've selected the configuration file [stream_petr_r50_704_bs2_seq_428q_nui_speed_test.py](https://github.com/exiawsh/StreamPETR/blob/main/projects/configs/test_speed/stream_petr_r50_704_bs2_seq_428q_nui_speed_test.py) as our target model configuration. In this configuration, a ResNet-50 serves as the image backbone, while a 6-layer transformer-based decoder is utilized to predict boxes within the scene. By following the instructions provided below, you'll obtain two ONNX model files: one for the image backbone and another for the perception decoding head.

## StreamPETR Environment Setup
1. Setup StreamPETR source code
```bash
git clone https://github.com/exiawsh/StreamPETR
cd ./StreamPETR
git clone https://github.com/open-mmlab/mmdetection3d.git  # also clone and install mmdetection3d
cd mmdetection3d
git checkout v1.0.0rc6 
pip install -e .
```
2. Follow official instructions at [streampetr/docs/setup.md](https://github.com/exiawsh/StreamPETR/blob/main/docs/setup.md). Please beware of flash-attn version if you are using it.
If you don't have a conda environment, you may follow below command lines.
```bash
# prepare conda environment
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
# after installation
~/miniconda3/bin/conda init bash  # if you are using bash
~/miniconda3/bin/conda init zsh   # if you are using zsh
```
Then we can start working on setting up environment for StreamPETR
```bash
# start with a conda virtual environment
conda create -n streampetr python=3.8 -y
conda activate streampetr
# prepare for pytorch environment
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install flash-attn==0.2.2
# prepare for mmcv related packagess
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
```
3. Other than StreamPETR's dependencies, you may install onnx related dependencies inside the docker or virtual environment with 
```bash 
pip install onnx onnxruntime onnxsim
```
4. Prepare dataset following the official instructions at [streampetr/docs/data_preparation.md](https://github.com/exiawsh/StreamPETR/blob/main/docs/data_preparation.md). Assume nuscenes dataset is stored at **./data/nuscenes**. Don't forget to run 'create-infos-file' as mentioned in the [link](https://github.com/exiawsh/StreamPETR/blob/main/docs/data_preparation.md#2-creating-infos-file). 

## Conversion
In this conversion demo, we start with following the 'Estimate the inference speed of StreamPETR' section in [streampetr/docs/training_inference.md](https://github.com/exiawsh/StreamPETR/blob/main/docs/training_inference.md#estimate-the-inference-speed-of-streampetr).

1. Download the pretrained pth file at [pth file link](https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth) to **StreamPETRRepoDir**/work_dirs/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth
```bash
mkdir -p work_dirs/  # assume we are already at the root path of StreamPETR repo
wget https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth -O ./work_dirs/stream_petr_r50_flash_704_bs2_seq_428q_nui_60e.pth
```
2. Verify all environmental setup with 
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python tools/benchmark.py projects/configs/test_speed/stream_petr_r50_704_bs2_seq_428q_nui_speed_test.py
```
This script is expected to load model pth file and report the running speed for this settings. If it can finish without any issue, it means that the environment has been setup properly.

3. **Copy** the python file pth2onnx.py to **StreamPETRRepoDir/tools** and run the following commands at **StreamPETRRepoDir** to build backbone and head onnx files
```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python tools/pth2onnx.py projects/configs/test_speed/stream_petr_r50_704_bs2_seq_428q_nui_speed_test.py --section extract_img_feat
CUBLAS_WORKSPACE_CONFIG=:4096:8 python tools/pth2onnx.py projects/configs/test_speed/stream_petr_r50_704_bs2_seq_428q_nui_speed_test.py --section pts_head_memory
```

4. Now you should get two onnx files for image backbone and head. These onnx files contains no special op and are supported by TensorRT natively.

## Notes
1. In current conversion script, timestamp input handle is different compared with other tensors in memory. It can't be directly processed with TensorRT along with other memory tensors.
2. The script consider some extra tensors ( reference_points, tgt, temp_memory, temp_pos, query_pos, query_pos_in, outs_dec ) as output. Although they are not required if we only need detection results, they are useful for numerical debug purpose.
