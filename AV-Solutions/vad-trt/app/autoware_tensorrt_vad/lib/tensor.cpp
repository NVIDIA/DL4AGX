/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensor.h"

std::ostream& operator<<(std::ostream& os, nv::Tensor& t) {
  // os << "[" << (int)(t.iomode) << "] ";
  os << t.name << ", [";
  
  for( int nd=0; nd<t.dim.nbDims; nd++ ) {
    if( nd == 0 ) {
      os << t.dim.d[nd];
    } else {
      os << ", " << t.dim.d[nd];
    }
  }
  std::cout << "]";
  std::cout << ", type = " << int(t.dtype);
  return os;
}
