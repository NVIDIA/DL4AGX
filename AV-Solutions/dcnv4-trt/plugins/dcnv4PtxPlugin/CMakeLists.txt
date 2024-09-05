# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(make_includable input_file output_file)
    file(READ ${input_file} content)
    set(DELIM "PTX")
    set(content "R\"${DELIM}(\n${content})${DELIM}\"")
    file(WRITE ${output_file} "${content}")
endfunction(make_includable)

foreach(PTX dcnv4_stage0_b128.ptx dcnv4_stage0_b1.ptx dcnv4_stage1_b1.ptx dcnv4_stage2_b1.ptx dcnv4_stage3_b1.ptx)
    make_includable(${CMAKE_CURRENT_SOURCE_DIR}/${PTX} ${CMAKE_CURRENT_SOURCE_DIR}/${PTX}.h)
endforeach()

set(DCNV4_plugin_cu_SRCS ${DCNV4_plugin_cu_SRCS}
                         ${CMAKE_CURRENT_SOURCE_DIR}/dcnv4_ptx.cu PARENT_SCOPE)
