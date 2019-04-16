# ${title}

${toc}
    
## Getting Started Guide

We use a build system called [Bazel](https://bazel.build/) for its reproducible builds and ability to support multiple toolchains and a wrapper around Bazel called [Dazel](https://github.com/nadirizr/dazel), which allows use to do the compilation within a docker container, thus giving us a reproducible developement enviorment as well. 
    
### Prerequisites

#### Installing Docker 

Follow the instructions provided here for installing Docker onto your host machine: https://docs.docker.com/install/

#### Installing NVIDIA Docker 

Follow these instructions for installing the NVIDIA Docker runtime onto your host machine: https://github.com/NVIDIA/nvidia-docker

#### Installing Dazel 
Dazel is a wrapper around bazel which lets developers use bazel in a docker container seamlessly. 

You can install dazel with:

```
pip3 install dazel
```

### Building the Base Container
Within the `docker` directory, there will be a dockerfile labeled with the particular platform and pdk version it targets. Follow the instructions in the `README.md` in that directory to build the base container.

### Building Apps

You can now build components with dazel. Make sure to be in the bazel workspace (the directory containing this file) then invoke `dazel` with a command similar to the following

```
dazel build //exampleProject
```

The binaries from the build will be found in the `bazel-out` directory 

#### Cross Compiling Apps

We use this docker build system so that cross compilation is really simple. Provided you built the base container properly, i.e. all the libraries and toolchains necessary are present inside the container, you should be able to compile the components for ${platform} by invoking a command similar to the following:

```sh
${toolchain_command}
```

${component_specific_instructions}
