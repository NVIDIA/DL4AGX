######################################################################################################
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
# File: DL4AGX/MultiDeviceInferencePipeline/training/objectDetection/ssdConvertUFF/BUILD
#####################################################################################################
package(default_visibility = ["//visibility:public"])

py_binary(
    name = "convert_to_trt",
    srcs = ["convert_to_trt.py"],
    python_version = "PY3",
    deps = ["//MultiDeviceInferencePipeline/training/objectDetection/ssdConvertUFF/utils"],
)

filegroup(
    name = "ssd_convert_uff",
    srcs = ["README.md"] + glob(["data/*"]),
    data = ["//MultiDeviceInferencePipeline/training/objectDetection/ssdConvertUFF:convert_to_trt"],
)
