# DL4AGX

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

This repository contains applications and tools to help understand and develop Deep Learning Applications for NVIDIA AGX Platforms (DRIVE, Jetson and CLARA). The AGX Family is based around the Xavier SoC, a high performance Aarch64 based processor that is automotive safety grade. On board are a number of accelerators to help accelerate Deep Learning workloads. These include a Volta Based Integrated GPU, multiple Deep Learning Accelerators (DLA), multiple Programmable Vision Accelerators (PVA) as well as other ISPs and Video processors. For more information on Xavier check [https://developer.nvidia.com/drive/drive-agx](https://developer.nvidia.com/drive/drive-agx).

## Getting Started

This repo uses bazel via a tool called dazel ([https://github.com/nadirizr/dazel](https://github.com/nadirizr/dazel)) to manage builds and cross-compilation inside a docker container.

### Installing Dependencies

1. Install Docker

   - https://docs.docker.com/install/

2. Install NVIDIA-Docker

   - https://github.com/NVIDIA/nvidia-docker

3. Install Dazel

   - ```pip3 install dazel```

4. Build the relevant docker container using one of the Dockerfiles provided in `//docker`

   - More precise instructions can be found in that directory's ([README.md](https://github.com/NVIDIA/DL4AGX/blob/master/docker/README.md))

5. Modify Dockerfile.dazel to be based on the image you just built

   - e.g. `FROM nvidia/drive_pdk:5.1.3.0`

### Compiling Applications

Dazel behaves like bazel but runs the compilation in a specified docker container. Therefore traditional bazel commands work like:

```sh
dazel build //plugins/dali/TensorRTInferOp:libtensorrtinferop.so
```

You will find the associated binaries in `//bazel-out/k8-fastbuild/plugins/dali/TensorRTInferOp/libtensorrtinferop.so`

### Cross-Compiling Applications

The AGX platforms are aarch64 based, so we need to cross compile the applications:

There are two supported toolchains:

#### aarch64-linux

##### Applicable to DRIVE AGX Platforms flashed with the Linux PDK and Jetson AGX Platforms

> In order to use this toolchain you must and have built a container that supports aarch64-linux (Dockerfiles will have names that contain `aarch64-linux` or `both`)

To cross-compile targets for aarch64-linux append the following flag to your build command: `--config=D5L-toolchain`

​- e.g. `dazel build //plugins/dali/TensorRTInferOp:libtensorrtinferop.so --config=D5L-toolchain`

You will find the associated binaries in `//bazel-out/aarch64-fastbuild/plugins/dali/TensorRTInferOp/libtensorrtinferop.so`

> Note: D5L-toolchain is aliased to L4T-toolchain for Jetson users' convenience

#### aarch64-qnx

##### Applicable to DRIVE AGX Platforms flashed with the QNX PDK

> In order to use this toolchain you must obtain the QNX Toolchain (found here: ) and have built a container that supports QNX (Dockerfiles will have names that contain `aarch64-qnx` or `both`)

To cross-compile targets for aarch64-qnx append the following flag to your build command: `--config=D5Q-toolchain`

- e.g. `dazel build //plugins/dali/TensorRTInferOp:libtensorrtinferop.so --config=D5Q-toolchain`

You will find the associated binaries in `//bazel-out/aarch64-fastbuild/plugins/dali/TensorRTInferOp/libtensorrtinferop.so`

### Running Compiled Targets in a Container

If you want to run a target in a container, the following command should work:

```sh
docker run --runtime=nvidia -v $(realpath bazel-bin):/DL4AGX -it <NAME OF ENV DOCKER IMAGE> /DL4AGX/<PATH TO YOUR SAMPLE IN bazel-bin>
```

## Applications

### Multi-Device Inference Pipelines 

This application demonstrates how to use DALI ([https://github.com/NVIDIA/DALI](https://github.com/NVIDIA/DALI)) and TensorRT ([https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)) in order to create accelerated inference pipelines that leverage more than one accelerator on the Xavier SoC.

## Troubleshooting Steps

### Refreshing the Build container

If you rebuild a container but have not changed the name of it, dazel may not pick up that the environment has changed. To trigger a manual rebuild of the environment do:

``` sh
touch Dockerfile.dazel
```