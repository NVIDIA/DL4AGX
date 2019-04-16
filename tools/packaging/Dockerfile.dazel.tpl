##########################################################################
# Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: Dockerfile.dazel
# Description: Dockerfile for the bazel layer of the development 
#              enviorment  
##########################################################################

FROM nvidia/drive_os_pdk 

# Creating the man pages directory to deal with the slim variants not having it.
RUN mkdir -p /usr/share/man/man1 
RUN apt-get update && apt-get install -y --no-install-recommends openjdk-8-jdk ca-certificates curl gnupg

RUN apt-get install -y --no-install-recommends \
    bash-completion \
    g++ \
    python \
    unzip \
    zlib1g-dev \
    && curl -LO "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/bazel_${BAZEL_VERSION}-linux-x86_64.deb" \
    && dpkg -i bazel_*.deb \
    && rm -rf /etc/apt/sources.list.d/bazel.list \
    && rm -rf /var/lib/apt/lists/*
