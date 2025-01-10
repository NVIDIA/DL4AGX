FROM nvcr.io/nvidia/pytorch:22.04-py3

RUN chmod 1777 /tmp
RUN apt update && apt install -y libturbojpeg libsm6 libxext6 -y
RUN pip install setuptools==69.5.1
RUN apt install libgl1-mesa-glx libsm6 libxext6  -y
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv_full==1.7.0 mmdet==2.28.2 mmdet3d==1.0.0rc6 mmsegmentation==0.30.0
RUN pip install numpy==1.23.5 nuscenes-devkit==1.1.10 yapf==0.33.0 tensorboard==2.14.0 motmetrics==1.1.3 pandas==1.1.5
RUN pip install "opencv-python-headless<4.3" "opencv-python<=4.5" --force-reinstall
RUN pip install numba==0.53.0 numpy==1.22.3
RUN pip install onnx onnxsim onnxruntime
RUN pip install onnx onnxruntime onnx_graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN pip install similaritymeasures
RUN apt install libeigen3-dev
RUN pip install pybind11 pycuda numpy==1.23