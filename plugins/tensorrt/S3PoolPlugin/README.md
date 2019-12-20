# S3Pooling Plugin 

Implements a S3Pooling Op for TensorRT

In inference S3Pooling is implemented with Average Pooling. So for TensorRT we implement an Average Pooling Op mapped to S3Pooling. 

## Usage 

### Shared Object File 
For applications that use a shared object, not linked but loaded in during the applications runtime, you can generate a `.so` file to use with the following command:

- For x86_64-linux

``` sh
dazel build //plugins/tensorrt/S3Pooling:libs3poolingplugin.so 
```

Outputs will be found in: 

- For x86_64

``` sh
//bazel-out/k8-fastbuild/bin/plugins/tensorrt/S3Pooling:libs3poolingplugin.so
```
