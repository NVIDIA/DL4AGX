#############################################################################
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
# File: DL4AGX/MultiDeviceInferencePipeline/inference/conf/BUILD
############################################################################
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "engine_settings",
    hdrs = ["EngineSettings.h"],
)

cc_library(
    name = "execution_settings",
    hdrs = ["ExecutionSettings.h"],
)

cc_library(
    name = "conf",
    hdrs = ["conf.h"],
    deps = [
        "//MultiDeviceInferencePipeline/inference/conf:engine_settings",
        "//MultiDeviceInferencePipeline/inference/conf:execution_settings",
        "//MultiDeviceInferencePipeline/inference/conf/PipelineSpec:pipeline_spec",
        "//third_party/cpptoml",
    ],
)
