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

import onnx
from onnxsim import simplify, model_info

model = onnx.load("onnx_files/mtmi.onnx")
model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"
simplified_path = "onnx_files/mtmi_slim.onnx"
onnx.save(model_simp, simplified_path)
model_info.print_simplifying_info(model, model_simp)
print(f"Simplified onnx model saved to {simplified_path}")
