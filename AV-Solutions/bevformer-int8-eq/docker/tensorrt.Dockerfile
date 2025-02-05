FROM nvcr.io/nvidia/tensorrt:24.08-py3

ARG CMAKE_VERSION=3.29.3
ARG NUM_JOBS=8

ENV DEBIAN_FRONTEND noninteractive
ENV PROJECT_DIR=/workspace

# Install package dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        software-properties-common \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        locales \
        locales-all \
        wget \
        git \
        libcurl4-openssl-dev \
        sudo \
        ssh \
        libssl-dev \
        pbzip2 \
        pv \
        bzip2 \
        unzip \
        devscripts \
        lintian \
        fakeroot \
        dh-make \
        vim && \
    apt-get clean

# System locale
# Important for UTF-8
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8

# Install CMake
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license
RUN rm -rf /tmp/*

RUN pip install --upgrade pip setuptools wheel
RUN pip install cuda-python==12.3.0 \
                numpy==1.26.3 \
                onnx==1.16.1 \
                onnxsim
RUN pip install --extra-index-url https://pypi.ngc.nvidia.com onnx_graphsurgeon==0.3.27
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121

# ======== Install ModelOpt Toolkit for quantization and ORT for CUDA 12 =========
RUN pip install nvidia-modelopt[onnx]==0.15.0 --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# ======== Prepare repo to convert BEVFormer model from PyTorch to ONNX ========
ARG TORCH_CUDA_ARCH_LIST="7.5;6.1;8.0;8.6"
ENV FORCE_CUDA="1"
ENV TRT_LIBPATH="/usr/lib/x86_64-linux-gnu"

ENV DERRYHUB_PROJECT_DIR=${PROJECT_DIR}/BEVFormer_tensorrt
RUN cd ${PROJECT_DIR} && git clone https://github.com/DerryHub/BEVFormer_tensorrt.git && \
    cd BEVFormer_tensorrt && git checkout 303d3140

# Install requirements except pytorch_quantization (requirements.txt file except the last 3 lines)
RUN head -n $(($(wc -l < ${DERRYHUB_PROJECT_DIR}/requirements.txt) - 3)) ${DERRYHUB_PROJECT_DIR}/requirements.txt > temp.txt && \
    sed -i 's/==/>=/g' temp.txt && \
    mv temp.txt ${DERRYHUB_PROJECT_DIR}/requirements.txt && \
    pip install -r ${DERRYHUB_PROJECT_DIR}/requirements.txt

# Install pytorch-quantization from source (needed for PyTorch 2 support)
# WAR for issue with package installed via "pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com pytorch-quantization"
RUN cd ${PROJECT_DIR} && git clone https://github.com/NVIDIA/TensorRT.git -b release/10.0 && \
    cd TensorRT/tools/pytorch-quantization && \
    MAX_JOBS=4 python setup.py install

# Install MMCV. First replace c++14 by c++17 to be able to compile PyTorch 2.x then do pip install with MAX_JOBS=4 to prevent workstation from freezing.
RUN cd ${DERRYHUB_PROJECT_DIR} && \
    git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && git checkout v1.5.0 && \
    pip install -r requirements/optional.txt && \
    sed -i "s/c++14/c++17/g" setup.py && \
    MAX_JOBS=4 MMCV_WITH_OPS=1 pip install -v -e .

RUN cd ${DERRYHUB_PROJECT_DIR} && \
    git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && git checkout v2.25.1 && \
    pip install -v -e .

RUN cd ${DERRYHUB_PROJECT_DIR}/third_party/bev_mmdet3d && \
    MAX_JOBS=4 python setup.py build develop --user

RUN sudo apt-get install -y libgl1-mesa-glx

WORKDIR /mnt
