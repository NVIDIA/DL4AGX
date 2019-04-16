##########################################################################
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
# File: DL4AGX/tools/nvcc/private/constants.bzl
# Description: Constants for use in creating cuda rules 
##########################################################################

CUDA_ACCEPTABLE_SRC_EXTENSIONS = [".cu", ".c", ".cc", ".cxx", ".cpp"]
CUDA_ACCEPTABLE_HDR_EXTENSIONS = [".h", ".cuh", ".hpp", ".inl"]
CUDA_ACCEPTABLE_BIN_EXTENSIONS = [".ptx", ".cubin", ".fatbin", 
                                   ".o", ".obj", ".a", ".lib",
                                   ".res", ".so"]

CUDA_ACCEPTABLE_EXTENSIONS = CUDA_ACCEPTABLE_SRC_EXTENSIONS + CUDA_ACCEPTABLE_BIN_EXTENSIONS + CUDA_ACCEPTABLE_HDR_EXTENSIONS