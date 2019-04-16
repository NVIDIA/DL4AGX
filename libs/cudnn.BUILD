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
# File: DL4AGX/libs/cudnn.BUILD
############################################################################
package(default_visibility = ["//visibility:public"])

cc_library(
    name = "cudnn_headers",
    hdrs = ["include/cudnn.h"] + glob(["include/cudnn+.h"]),
    includes = ["include/"],
    visibility = ["//visibility:private"],
)
       
cc_import(
    name = "cudnn_lib",
    shared_library = "lib/libcudnn.so",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "cudnn",
    deps = [
        "cudnn_headers", 
        "cudnn_lib"
    ],
    visibility = ["//visibility:public"],
)


