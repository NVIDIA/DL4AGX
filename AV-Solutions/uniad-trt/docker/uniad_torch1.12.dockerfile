# SPDX-FileCopyrightText: Copyright (c) 2022-2023 DerryHub. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/DerryHub/BEVFormer_tensorrt/blob/main/docker/Dockerfile
# Removed TensorRT, NGC client, and mmdeploy installation related code
# Added UniAD environment support and updated TORCH_CUDA_ARCH_LIST


ARG CUDA_VERSION=11.8.0
ARG OS_VERSION=20.04

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu${OS_VERSION}
LABEL maintainer="NVIDIA CORPORATION"

SHELL ["/bin/bash", "-c"]

# Setup user account
ARG uid=1000
ARG gid=1000
RUN groupadd -r -f -g ${gid} trtuser && useradd -o -r -l -u ${uid} -g ${gid} -ms /bin/bash trtuser
RUN usermod -aG sudo trtuser
RUN echo 'trtuser:nvidia' | chpasswd
RUN mkdir -p /workspace && chown trtuser /workspace

# Required to build Ubuntu 20.04 without user prompts with DLFW container
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntu-toolchain-r/test
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev \
    wget \
    git \
    pkg-config \
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
    build-essential

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0

# Install Cmake
RUN cd /tmp && \
    wget https://cmake.org/files/v3.14/cmake-3.14.4-Linux-x86_64.sh && \
    chmod +x cmake-3.14.4-Linux-x86_64.sh && \
    ./cmake-3.14.4-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.14.4-Linux-x86_64.sh

# Install PyTorch
RUN pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install BEVFormer_tensorrt required
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt

# Install MMLab
ARG TORCH_CUDA_ARCH_LIST="7.0;7.5;6.1;8.0;8.6"
ENV FORCE_CUDA="1"

RUN cd / && \
    git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && git checkout v1.5.0 && \
    pip3 install -r requirements/optional.txt && \
    MMCV_WITH_OPS=1 pip3 install -e .

RUN cd / && \
    git clone https://github.com/open-mmlab/mmdetection.git && \
    cd mmdetection && git checkout v2.25.1 && \
    pip3 install -v -e .

RUN pip install mmsegmentation>=0.20.0

# Install UniAD required
RUN python3 -m pip install --upgrade pip
RUN pip3 install google-cloud-bigquery==3.25.0 motmetrics==1.1.3 einops==0.4.1 casadi==3.5.6rc2 pytorch-lightning==1.2.5

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install misc
RUN pip3 install ipython==8.12.3
RUN pip3 install scikit-image==0.21.0
RUN pip3 install yapf==0.40.1

WORKDIR /workspace
COPY ./nuscenes /usr/local/lib/python3.8/dist-packages/nuscenes
RUN pip install torchmetrics==0.11.4
RUN pip install pandas==1.4.4
RUN pip install onnx==1.16.2 onnx_graphsurgeon==0.5.2