# NVIDIA DRIVE OS PDK Containers 

> **WARNING: This container is going to install some or all of the DRIVE OS PDK and SDK components. By building this contianer, you are accepting the NVIDIA Software License Agreements contained in the DRIVE OS PDK and SDK installers. You may wish to review the license agreements in the runfiles before building the containers.**

The Dockerfile(s) in this directory are setup to install the DRIVE OS PDK in a container to be used to build apps 


## Expected Files

### For DRIVE OS versions > 5.0.13.0 or Jetson

Using the SDK manager, download the host componets of the PDK version or Jetpack specified in the name of the Dockerfile. To do this:

1. [**SDK Manager Step 01**] Log into the SDK manager
2. [**SDK Manager Step 01**] Select the correct platform and PDK version (should be corresponding to the name of the Dockerfile you are building (e.g. `Linux - DRIVE OS 5.1.0.0 PDK`), then click `Continue`
3. [**SDK Manager Step 02**] Under `Download & Install Options` make note of or change the download folder **and Select Download now. Install later.** then agree to the license terms and click `Continue`

You should now have all expected files to build the container. Move these into the same directory as the Dockerfile in a directory called `**pdk_files**` with the names unmodified.

### For DRIVE OS versions <= 5.0.13.0
Take a look at the included Dockerfiles for a complete list of necessary files but in general, these containers will depend on all of the PDK run files for a platform and any additional tarballs or debian files provided to you. These files should be located in a directory called `pdk_files` with the names unmodified. Note that if you want to use specific versions of TensorRT or cudnn, those tar files shall also be placed in the `pdk_files` folder.

### QNX Toolchain

If the container is targeted at QNX, the QNX Toolchain is expected in a directory named `qnx_toolchain` with a layout similar to:

```
qnx_toolchain
|__ host
|   |__ ...
|
|__lib64
|  |__ ...
|
|__target
   |__ ...
```

## Building the Container 

With the dependency files in the correct places, you can build the docker image with 

``` sh
docker build -t nvidia/drive_os_pdk -f <NAME OF DOCKERFILE> .
```
> Note: the period at the end of the command is important and the **name of the container needs to be nvidia/drive_os_pdk** unless you rename the base image in `Dockerfile.dazel`

> For the Jetson container, you need to access NGC's container registry [https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt) to download the TensorRT base image. 

### Recipes 
#### DRIVE PDK 5.1.3.0 aarch64-linux 
``` sh
docker build -t nvidia/drive_os_pdk:5.1.3.0-linux -f DRIVE/Dockerfile.aarch64-linux.5.1.3.0 DRIVE
```

#### DRIVE PDK 5.1.3.0 aarch64-qnx 
``` sh
docker build -t nvidia/drive_os_pdk:5.1.3.0-qnx -f DRIVE/Dockerfile.aarch64-qnx.5.1.3.0 DRIVE
```

#### DRIVE PDK 5.1.3.0 Both (aarch64-linux and aarch64-qnx) 
``` sh
docker build -t nvidia/drive_os_pdk:5.1.3.0-both -f DRIVE/Dockerfile.both.5.1.3.0 DRIVE
```
_For the container that supports both, copy both the QNX and Linux pdk files into the same directory_

#### JetPack 4.1 aarch64-linux
``` sh
docker build -t nvidia/jetpack:4.1 -f Jetson/Dockerfile.aarch64-linux.4.1 Jetson
```

## Using the Container to Build DL4AGX Targets
You can now build apps using `dazel`. Change the `FROM` line in //Dockerfile.dazel to point at the image you just created.
Then, follow the instructions in the DL4AGX README to install dazel and build.

You may want to change assorted settings in `.dazelrc` to your liking, especially `DAZEL_IMAGE_NAME`

**You may also want to move the pdk files out of the workspace after you are done building the container to avoid long build times due to increased the size of the docker context**

> Note: depending in some containers only one toolchain will work. The Dockerfiles have steps to ensure you 
> will not encounter errors for missing directories when trying to build but for a QNX container only the `D5Q-toolchain` will 
> actually compile targets, similarly only the `D5L-toolchain` wil only compile targets when using a 
> container with the aarch64 Linux PDK
