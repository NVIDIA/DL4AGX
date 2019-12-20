# How to Create a TensorRT Plugin (S3Pooling in LeNet)
## Target platform
- Ubuntu 18.04 Host machine(tested)

## Training
### Installing Dependencies

1. Create a virtual enviromment

```sh
mkdir -p $HOME/dl4agx/venv
virtualenv $HOME/dl4agx/venv/pytorch -p /usr/bin/python3
source $HOME/dl4agx/venv/pytorch/bin/activate
```
2. Install python dependencies

```sh
cd <PATH TO DL4AGX>/LeNetWithS3Pooling/training
pip3 install -r requirements.txt
```

### Train model

1.  Train the pytorch model with S3Pool layerh. The model is saved as `mnist.onnx`

```sh
python3 main.py --save-model --s3pool
```

2. Train the pytorch mnist model with Average poooling in inference path. The model is saved as `mnist_with_avgpool.onnx`

```sh
python3 main.py --save-model
```

### Modify ONNX model
 
As we know that ONNX is an IR, so custom layer can be decomposed into simpler ops. In this case S3Pool includes MaxPooling as the 2nd stage so  we need to update the onnx file to S3Pool operation in order to prevent the AveragePooling mapping in ONNX parser.

```sh
cd <PATH TO DL4AGX>/LeNetWithS3Pooling/utils
pip3 install -r requirements.txt
python3 modify_onnx.py --onnx=<PATH TO TRAINED ONNX FILE> --output=mnist_plugin.onnx
deactivate       #decativate virtual environment
```

## Inference
### Build the Build Enviornment

Follow the instructions `//docker` to build a base container for the system you develop with.

Then build the container in this directory based on the container you just built by doing the following:

1. Download the TensorRT 6.0.1.8 tarball for CUDA 10.2 Ubuntu 18.04 x86_64 and place it in the `//LeNetWithS3Pooling/docker` directory

2. Replace the first line of ``//LeNetWithS3Pooling/docker/Dockerfile.x86_64-linux.LeNetWithS3Pooling` with the name of the base container you built. For example `FROM nvidia/drive_os_pdk:5.1.6.0-linux`

3. Build the container using `docker build -t nvidia/s3pool -f Dockerfile.x86_64-linux.LeNetWithS3Pooling .` This will replace the ONNX parser with a custom one that supports the S3Pooling plugin 

4. Replace the first line of `//Dockerfile.dazel` with the new container you built. For example `FROM nvidia/s3pool`
   
### Compile and build S3Pool plugin

Run the following command to build TensorRT plugin and inference application. You can find the binaries here: `<PATH_TO_DL4AGX>/bazel-out/k8-fastbuild/bin/`

```sh
cd <PATH_TO_DL4AGX>
dazel build //plugins/tensorrt/… //LeNetWithS3Pooling/inference/...         # Builds TensorRT plugin and application for Host
```
### Running the App

To avoid recompiling dependencies, copy the custom onnx parser from the build container, to do this start an instance of the docker container:

``` sh
docker create --name s3pooling s3pool
docker cp s3pooling:/usr/local/cuda-10.2/dl/targets/x86_64-linux/lib/ .
```

Export this lib directory in your `LD_LIBRARY_PATH`

``` sh
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
```

It's easiest to copy your trained model in ONNX format and the example image `//LeNetWithS3Pooling/inference/2.pgm` into `//bazel-out/k8-fastbuild/bin/LeNetS3Pooling/inference` 

Then run the app as follows:
```sh
./lenet_s3pooling_trt mnist_with_avgpool.onnx input.1 30     # To test sample without using plugin
LD_PRELOAD=<PATH_TO_DL4AGX>/bazel-bin/plugins/tensorrt/S3PoolPlugin/libs3poolplugin.so ./lenet_s3ooling_trt mnist_plugin.onnx input.1 30     # To test sample with using plugin
```
