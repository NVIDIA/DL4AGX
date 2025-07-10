# TODO

## engine build

- 現状
  - engineをtrtexecでbuildし，pathをyamlに記載して読み込んでいる
- 理想状態
  - engine pathではなく，onnx pathをyamlに記載して読み込む
- 参考: BEVFusion nodeの例
    - [onnx path読み込み](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_bevfusion/src/bevfusion_node.cpp#L43)
    - [TrtCommonConfigにonnx_pathを渡す](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_bevfusion/src/bevfusion_node.cpp#L143)
    - [TrtCommonConfig](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_tensorrt_common/include/autoware/tensorrt_common/utils.hpp#L51)
    - [TrtCommon::setupの中でengineがbuildされる](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_tensorrt_common/src/tensorrt_common.cpp#L147)
        - [最終的にはbuildEngineFromOnxxが呼ばれている](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_tensorrt_common/src/tensorrt_common.cpp#L114)
- やること
  - VadModelの中でTrtCommonを呼ぶようにする
  - onnxをTrtCommonに渡して，engineをbuildする

## plugins build

- 現状
  - pluigns.soをbuildし，pathをyamlに記載して読み込んでいる
- 理想状態
    - [autoware_tensorrt_plugins](https://github.com/autowarefoundation/autoware_universe/tree/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_tensorrt_plugins)に[vad-trtのplugins](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/vad-trt/plugins)を追加
    - pluginsをcolcon build時にbuild
- 参考: BEVFusion nodeの例
  - [autoware_tensorrt_pluginsをpackage.xmlに追加](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_bevfusion/package.xml#L34)し，colcon build時に一緒にbuild
  - [configにてplugins_pathを指定](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_bevfusion/config/bevfusion_camera_lidar.param.yaml#L8)
  - [TrtrCommonにplugins_pathを渡す](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_bevfusion/lib/bevfusion_trt.cpp#L220)
  - [TrtCommonの中でdlopen()](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_tensorrt_common/src/tensorrt_common.cpp#L62)
- やること
  - [autoware_tensorrt_plugins](https://github.com/autowarefoundation/autoware_universe/tree/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_tensorrt_plugins)に[vad-trtのplugins](https://github.com/NVIDIA/DL4AGX/tree/master/AV-Solutions/vad-trt/plugins)を追加する
- 課題: IPluginのversion
  - [vad-trtではIPluginV2DynamicExtを使用](https://github.com/NVIDIA/DL4AGX/blob/482b3f53e7b73b431aa1c4c146a74735aaf558ef/AV-Solutions/vad-trt/plugins/multi_scale_deform_attn/ms_deform_attn.cpp#L152C1-L152C20)
  - [autoware_tensorrt_pluginsではIPluginV3を使用](https://github.com/autowarefoundation/autoware_universe/blob/3dc2a8529bc73aacc524e65d7059f05defe5d264/perception/autoware_tensorrt_plugins/src/argsort_plugin.cpp#L169)
