FROM nvcr.io/nvidia/pytorch:22.06-py3
RUN chmod -R 777 /tmp
RUN apt update
RUN apt install libgl1-mesa-glx libsm6 libxext6  -y
RUN pip install "opencv-python-headless<4.3"
RUN pip install "opencv-python<=4.5" 
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
# need at least mmcv-full 1.7.0 https://github.com/Sense-X/Co-DETR/issues/6
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.0 mmdet==2.28.2 mmsegmentation==0.30.0 mmdet3d==1.0.0rc6
RUN FLASH_ATTENTION_FORCE_BUILD=TRUE MAX_JOBS=12 pip install flash-attn==0.2.8
RUN pip install debugpy
RUN pip install "opencv-python-headless<4.3" "opencv-python<=4.5" --force-reinstall
RUN pip install "motmetrics<1.2"
RUN pip install onnx onnxruntime onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install onnxsim
RUN pip install colored
RUN pip install av2 kornia refile
# verify installation
RUN python -c "import cv2"
RUN python -m torch.utils.collect_env

RUN apt-get install -y libyaml-cpp-dev libeigen3-dev pybind11-dev

ENTRYPOINT /bin/bash
WORKDIR /workspace/far3d-trt