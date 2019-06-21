# Contributing to DL4AGX

Forking and modifying the applications and tools in this repo to suite your own use cases is welcomed. If you think you have something that the community may benefit from, upstreaming those changes to this repo would be appreciated but will be accepted at the discretion of the maintainers (currently NVIDIA's Automotive Deep Learning Solution Architects Group). The goal of this repo is to help people using NVIDIA AGX platforms, so make sure PRs serve that purpose.  

## Getting Started

There are a couple key components to the development enviorment: (NVIDIA) Docker, and Dazel/Bazel

The actual development enviorment is defined by the Dockerfiles found in `//docker` and reflect the development enviorment provided by DRVIE PDKs and Jetpack. We use containers because it allows us to version the development enviorment and not require a ton of pollution of the host system.

The next layer is defined but `//Dockerfile.dazel` which installs bazel in the enviorment + an assortment of development tools. The combination of the base Docker Image and Dockerfile.dazel defines the development enviorment we work in. This enviorment after the base container is built automatically when invoking a `dazel` command (though manual refreshes may be needed using `touch Dockerfile.dazel`). It might make sense to tag this image with a memorable name with a command like `docker build -t nvidia/dl4agx -f Dockerfile.dazel .` for use in debugging.

After that things work like a normal Bazel project, just invoking commands with `dazel` instead of `bazel` and that the external dependencies will be located in the container and not the host machine.

### Adding a new project

To add a project, create a new directory at the root of the repo and create a `BUILD` file, place your sources in that directory.

When writting C++ code, include using quotations instead of angle brackets
​    - e.g. `"cuda_runtime_api.h"` vs. `<cuda_runtime_api.h>` unless its a stdlib library

### Bazel Crash Course

#### Writing a `BUILD` file

1. Most likely you will be writing an executable, so start out by declaring a `cc_binary`

2. Name and enumerate the source files, if you are compiling a library there and there are header files you would like make available to other projects in the build 
system, list them in the headers section

3. List the dependencies for the project. The format is to reference the location of the "project" (directory on the system) through a nickname, then the specific library needed. ex. `@cuda10` refers to `/usr/local/cuda-10.0`. These nicknames are defined in `WORKSPACE`. If you are loading from a project in the workspace then you don't need this prefix. Then the actual libraries can be loaded using the name of the `cc_library` object.

- e.g.

```py
cc_binary(
    name="ApplicationName",
    srcs=["main.cpp"],
    deps=["//common:common"]
)
```

#### Building a project

To build you can just run:

```sh
dazel build //<project name>
```

#### Cross Compilation

The toolchains are setup based on which environment container you are using and are accessible via the `--config=[D5L/L4T]-toolchain` (aarch64-linux) and `--config=D5Q-toolchain` (aarch64-qnx) flags 

#### Setting up the `BUILD` file

Since external libraries (CUDA, cuDNN, TensorRT) do not auto resolve based on toolchain on you will have to declare explictly dependencies for each platform. Within the `BUILD` file you can set toolchain specific rules to govern compilation. `config_setting` lets you inject a variable based on command line values. Then using those variables you can optionally include dependencies

- ex

  ```py
  package(default_visibility = ["//visibility:public"])

  config_setting(
      name = "aarch64_linux",
      values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
  )
  
  config_setting(
      name = "aarch64_qnx",
      values = { "crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx" }
  )
  
  cc_binary(
      name="libflattenconcatplugin.so",
      linkshared=True,
      srcs=["FlattenConcat.h",
            "FlattenConcat.cpp"],
      deps=["//common:common"]
      + select({":aarch64_linux":["@tensorrt_aarch64_linux//:nvinferplugin",
                                  "@cuda_aarch64_linux//:cublas"],
                ":aarch64_qnx":["@tensorrt_aarch64_qnx//:nvinferplugin",
                                "@cuda_aarch64_qnx//:cublas"],
                "//conditions:default":["@tensorrt_x86_64_linux//:nvinferplugin",
                                       "@cuda_x86_64_linux//:cublas"]}),
  )
  ```

## Debugging Recommendations

### Refreshing the Build container

First thing to try if you code is not compiling because of some missing file but you know you've changed the development enviorment is to manually trigger a rebuild of the dazel container with:

```sh
touch Dockerfile.dazel 
```

### Working inside the Development Container 
Sometimes its useful to work inside of the container so you can move outside the bazel sandbox and look around. This command run from the repo root allows you to do that:

```sh
docker run --runtime=nvidia --rm -it -v $(realpath .):/DL4AGX nvidia/dl4agx #The name of the container built by Dockerfile.dazel (see above)
```

This will mount the repo as a volume (so file changes propogate in and out of the container). The only difference is inside the container use the command `bazel` instead of `dazel`

## Installing New System Libraries

1. Its recommended you add a Dockerfile to create a layer between the base PDK/Jetpack image and the dazel image, with your new system dependency then base the Dockerfile.dazel image on your custom image. For example a Dockerfile could install Tensorflow:

```Dockerfile
FROM nvidia/drive_os_pdk

RUN pip3 install tensorflow-gpu
```

Then Dockerfile.dazel would look like:

```Dockerfile
FROM my_custom_tf_drive_os_pdk_container
...
```

2. Add the directory where the libraries will be accessible in `WORKSPACE` and name it something reasonable. We use the convention that versions of the library for different platforms have different workspace entries but ideally share the same BUILD file

   - e.g.

   ```py
   new_local_repository(
       name="tensorrt_x86_64_linux",
       path="/usr/local/cuda-10.1/dl/targets/x86_64-linux/",
       build_file="libs/tensorrt.BUILD"
   )
   
   new_local_repository(
       name="tensorrt_aarch64_linux",
       path="/usr/local/cuda-10.1/dl/targets/aarch64-linux/",
       build_file="libs/tensorrt.BUILD"
   )
   
   new_local_repository(
       name="tensorrt_aarch64_qnx",
       path="/usr/local/cuda-10.1/dl/targets/aarch64-qnx/",
       build_file="libs/tensorrt.BUILD"
   )
   ```

3. Enumerate the libraries in a file called `libs/<reasonable_name>.BUILD` and add the relative path to that file in the WORKSPACE entry

   - e.g.

   ```py
   package(default_visibility = ["//visibility:public"])
   
   config_setting(
       name = "aarch64_linux",
       values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
   )
   
   config_setting(
       name = "aarch64_qnx",
       values = { "crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx" }
   )
   
   cc_library(
       name="nvinfer_headers",
       hdrs = ["include/NvInfer.h",
               "include/NvUtils.h"],
       includes = ["include/"],
       visibility=["//visibility:private"],
   )
   
   cc_import(
       name="nvinfer_lib",
       shared_library="lib/libnvinfer.so",
       visibility=["//visibility:private"],
   )
   
   cc_library(
       name="nvinfer",
       deps=["nvinfer_headers",
             "nvinfer_lib"]
       + select({":aarch64_linux":["@cuda_aarch64_linux//:cudart",
                                   "@cuda_aarch64_linux//:cublas",
                                   "@cudnn_aarch64_linux//:cudnn"],
                 ":aarch64_qnx":["@cuda_aarch64_qnx//:cudart",
                                 "@cuda_aarch64_qnx//:cublas",
                                 "@cudnn_aarch64_qnx//:cudnn"],
                 "//conditions:default":["@cuda_x86_64_linux//:cudart",
                                        "@cuda_x86_64_linux//:cublas",
                                        "@cudnn_x86_64_linux//:cudnn"]}),
       visibility=["//visibility:public"],
   )
   ```

### Library Location Conventions 
We try our best to stay as close to what DRIVE Software does:
Hence:

- CUDA: Should be installed at `/usr/local`, CUDA for various platforms should be in the target directory of `/usr/local/cuda-X`
  - e.g. aarch64-linux CUDA 10.1 should be located at `/usr/local/cuda-10.1/targets/aarch64-linux`	 
- CUDA-X DL Libs (i.e. TensorRT and cuDNN): Should be located at `/usr/local/cuda-X/dl/targets/<PLATFORM>/{include, lib}`
- Other system dependencies: Dependencies should be located in `/usr/local/{include, lib}` for x86_64, `/usr/aarch64-linux-gnu/` for aarch64-linux
  and `/usr/aarch64-unknown-nto-qnx/aarch64le` for aarch64-qnx	

## Commit Messages

We would like commit messages to stay as close to the Conventional Commits Standard as we can, please refer to [https://www.conventionalcommits.org/en/v1.0.0-beta.4/](https://www.conventionalcommits.org/en/v1.0.0-beta.4/) for the full specification.

## CI/CD

For the time being CI/CD will be done manually, code must compile for at least x86_64-linux and one of the aarch64-linux platforms (DRIVE or Jetson) using the latest enviorment container before we will consider merging the code. We will test the code internally on aarch64-qnx platforms and provide feedback / patches to support. 

## License Header for New Files

New source files must include the following license header:

```txt
Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

File: [FILE PATH]
Description: [WHAT THE FILE DOES]

```

## Sign Your Work

We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.

To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:

    $ git commit -s -m "Add cool feature."

This will append the following to your commit message:

    Signed-off-by: Your Name <your@email.com>

By doing this you certify the below:

    Developer Certificate of Origin
    Version 1.1
    
    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
    1 Letterman Drive
    Suite D4700
    San Francisco, CA, 94129
    
    Everyone is permitted to copy and distribute verbatim copies of this license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1
    
    By making a contribution to this project, I certify that:
    
    (a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license indicated in the file; or
    
    (b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source license and I have the right under that license to submit that work with modifications, whether created in whole or in part by me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file; or
    
    (c) The contribution was provided directly to me by some other person who certified (a), (b) or (c) and I have not modified it.
    
    (d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent with this project or the open source license(s) involved.

## Conforming to Coding Guidelines

There are tools built into dazel to help you make sure your code conforms to coding guidelines.

### C++ Linting 

To see where your code deviates from the coding guidelines run:

```sh
dazel run //tools/linter:cpplint_diff -- <dazel target>
```

Ex.

```sh
dazel run //tools/linter:cpplint_diff -- //plugins/tensorrt/...
```

To run linting on all projects you can run:

```sh
dazel run //tools/linter:cpplint_diff -- //...
```

To change your files to conform with coding guidelines you can run:

```sh
dazel run //tools/linter:cpplint -- <dazel target>
```

Ex.

```sh
dazel run //tools/linter:cpplint -- //plugins/tensorrt/...
```

To change all projects you can run:

```sh
dazel run //tools/linter:cpplint -- //...
```

### Python Linting

To see where your code deviates from the coding guidelines run:

```sh
dazel run //tools/linter:pylint_diff -- <dazel target>
```

Ex.

```sh
dazel run //tools/linter:pylint_diff -- //examplePythonProject
```

To run linting on all projects you can run:

```sh
dazel run //tools/linter:pylint_diff -- //...
```

To change your files to conform with coding guidelines you can run:

```sh
dazel run //tools/linter:pylint -- <dazel target>
```

Ex.

```sh
dazel run //tools/linter:pylint -- //examplePythonProject
```

To change all projects you can run:

```sh
dazel run //tools/linter:pylint -- //...
```

## CUDA

There is experimental support for CUDA in bazel included in this repository. Only the `cu_library` rule is considered usable, but bugs may still pop up. We make no guarantees about anything with this rule, so **USE AT YOUR RISK**. However internally a couple of applications have successfully been compiled using the rule and it does contain a number of useful features. You can include the rule using the following line in a BUILD file:

```py
load("//tools/nvcc:cuda.bzl", "cu_library")
```

`cu_library` can be considered the CUDA analog to `cc_library`. It behaves very similar but with a couple differences:

Source files passed into this rule are assumed as CUDA code only (i.e. will be compiled with NVCC with the `-x cu` flag). This may dictate code structuring. The output of `cu_library` will be a shared library and static library, these can be consumed by `cc_*` rules as deps to create full apps. cu_libraries can consume cc_libraries as dependencies themselves as well. In addition to the standard options for `cc_library` are two additional ones: `gpu_arch` and `gen_code`. These options map directly to NVCC's `--gpu-architecutre` and `-gencode` flags respectively. Cross-compilation is also supported via CROSSTOOL/cc_toolchain_config configuration. `cc_library`'s `include_prefix` and `strip_include_prefix` are not currently implemented and, `copts` and `linkopts` refer to host compiler options (i.e. will be passed to NVCC perpended with `--Xcompiler` and `--Xlinker` respectively). For options to be passed to NVCC itself use `nvcc_copts` and `nvcc_linkopts`

An example BUILD file is the following:

```python
package(default_visibility = ["//visibility:public"])
load("//tools/nvcc:cuda.bzl", "cu_library")

config_setting(
    name = "aarch64_linux",
    values = { "crosstool_top": "//toolchains/D5L:aarch64-unknown-linux-gnu" }
)

config_setting(
    name = "aarch64_qnx",
    values = { "crosstool_top": "//toolchains/D5Q:aarch64-unknown-nto-qnx" }
)

cc_binary (
    name="VecAdd",
    srcs=["main.cpp"],
    deps=["//VecAdd:vector_addition_kernel",
          "//VecAdd:myprintf"]
        + select({":aarch64_linux":["@cuda_aarch64_linux//:cuda",
                                    "@cuda_aarch64_linux//:cudart"],
                  ":aarch64_qnx":["@cuda_aarch64_qnx//:cuda",
                                  "@cuda_aarch64_qnx//:cudart"],
                  "//conditions:default":["@cuda_x86_64_linux//:cuda",
                                          "@cuda_x86_64_linux//:cudart"]}),
)

cu_library(
    name="vector_addition_kernel",
    srcs=["vectorAdd.cu"],
    hdrs=["vectorAdd.h"],
    gpu_arch="sm_70",
    gen_code=["arch=compute_75,code=sm_75",  
              "arch=compute_70,code=sm_70",  
              "arch=compute_61,code=sm_61",], 
    deps=["//VecAdd:myprintf"] 
         + select({":aarch64_linux":["@cuda_aarch64_linux//:cuda",
                                     "@cuda_aarch64_linux//:cudart"],
                   ":aarch64_qnx":["@cuda_aarch64_qnx//:cuda",
                                   "@cuda_aarch64_qnx//:cudart"],
                   "//conditions:default":["@cuda_x86_64_linux//:cuda",
                                           "@cuda_x86_64_linux//:cudart"]}),
)

cc_library(
    name="myprintf",
    srcs=["myprintf.cpp"],
    hdrs=["myprintf.h"]
)


```

## Creating a Tarball with a Subset of the Applications and a Docker Container

There is a custom bazel command created called `src_package` which describes the 
tarball to be created, usually these packages should be defined in the root level `BUILD` file. 

- The `name` argument will end up being the name of the tarball file
- `components` is a list of the targets that should be included, the bazel command will analyze the targets and include all local dependencies (e.g. third party libraries)
- `platform` is the intended platform that should be used to build the apps, this must correspond to a dockerfile in `//docker` 
- `pdk_version` is the intended version of the pdk that should be used to build the apps, this must correspond to a dockerfile in `//docker` 
- `pdk_platform` is the intended OS that the apps should be compiled for, this should also correspond to a dockerfile in `//docker` (e.g. if the plaform is `qnx` and the pdk version is `5.0.13.0.patched` then there must be a dockerfile called `Dockerfile.qnx.5.0.13.0.patched` in `//docker`
  - Available platforms are `qnx`, `aarch64-linux`, and `both` which will include a dockerfile for each platform targeting the same pdk version
- `documentation` allows you to insert custom top level documentation to the end of the root README file in the tarball which by default has setup instructions, this will be in addition to any sample documentation in the sample directory. The format is markdown (see below for an example). 

```py
load("//tools/packaging:src_package.bzl", "src_package")

src_package(
    name = "DRIVE_OS-reference-app-5.1.3.0-aarch64-linux-OPENSOURCE",
    components = [
        "//MultiDeviceInferencePipeline:MultiDeviceInferencePipeline",
        "//plugins/tensorrt/FlattenConcatPlugin:libflattenconcatplugin.so",
        "//plugins/dali/TensorRTInferOp:libtensorrtinferop.so",
    ],
    documentation = """
## DRIVE OS Reference Application

This sample takes you through developing a inference pipeline utilizing the compute capabilities of the NVIDIA DRIVE AGX Developer Kit. The pipeline will do Object Detection and Lane Segementation on an image concurrently using both the integrated GPU of the Xavier Chip and one of the Deep Learning Accelerators onboard.

### Getting Started with the DRIVE OS Reference Application
Follow the instructions above to setup the development enviorment. Then look at the README in the project directory for details on the components of the reference application and how to compile them. 
""",
    min_bazel_version = "0.21.0",
    platform = "DRIVE",
    pdk_platform = "aarch64-linux",
    pdk_version = "5.1.3.0",
)
```

The tarball can be constructed with a command as follows:

```sh
dazel build //:DRIVE_OS-reference-app-5.1.3.0-aarch64-linux-OPENSOURCE
```

and can be found in `bazel-bin`
