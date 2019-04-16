# FlattenConcat Plugin 

Implements a FlattenConcat Op for TensorRT
 
FlattenConcat is used to flatten each input to the op and then concatenate the results. This plugin is usually used in conjunction with the NMS op where it is applied to location and confidence data before it is fed to the NMS plugin since it requires the data to be in this format

## Usage 

### Shared Object File 
For applications that use a shared object, not linked but loaded in during the applications runtime, you can generate a `.so` file to use with the following command:

- For x86_64-linux

``` sh
dazel build //plugins/FlattenConcat:libflattenconcatplugin.so 
```

- For aarch64-linux

``` sh
dazel build //plugins/FlattenConcat:libflattenconcatplugin.so --config=D5L-toolchain
```

- For aarch64-qnx

``` sh
dazel build //plugins/FlattenConcat:libflattenconcatplugin.so --config=D5Q-toolchain
```

Outputs will be found in: 

- For x86_64

``` sh
//bazel-out/k8-fastbuild/bin/plugins/FlattenConcat:libflattenconcatplugin.so
```

- For aarch64

``` sh
//bazel-out/aarch64-fastbuild/bin/plugins/FlattenConcat:libflattenconcatplugin.so
```

### Included In Source

By listing the `flattenconcat` target as a dependency of another bazel target you can then include the plugin and use it in an application. 
