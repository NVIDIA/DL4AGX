## SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch

input_file = "./ckpts/bevformer_tiny_epoch_24.pth"
output_file = "./ckpts/bevformer_tiny_epoch_24_updated2.pth"

ckpt = torch.load(input_file, map_location="cpu")

key_mapping = {
    "pts_bbox_head.query_embedding.weight": "query_embedding.weight",
    "pts_bbox_head.transformer.reference_points.weight": "reference_points.weight",
    "pts_bbox_head.transformer.reference_points.bias": "reference_points.bias"
}

state_dict = ckpt['state_dict']
for key in list(state_dict.keys()):
    if key in key_mapping:
        new_key = key_mapping[key]
        if key == "pts_bbox_head.query_embedding.weight":
            # Adjust size by appending 899th value to the 900th position
            old_tensor = state_dict[key]
            if old_tensor.size(0) == 900:  
                appended_tensor = torch.cat([old_tensor, old_tensor[-1:]], dim=0)  
                state_dict[new_key] = appended_tensor
                print(f"Appended last value to {key}, new size: {appended_tensor.size()}")
        else:
            new_key = key_mapping[key]
            state_dict[new_key] = state_dict.pop(key)
            print(f"Renamed {key} to {new_key}")

torch.save(ckpt, output_file)
print(f"Updated checkpoint saved to {output_file}")
