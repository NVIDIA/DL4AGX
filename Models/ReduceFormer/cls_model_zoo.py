# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from efficientvit.models.reduceformer import (
    ReduceFormerCls,
    ######################
    reduceformer_cls_b1,
    reduceformer_cls_b2,
    reduceformer_cls_b3,
)
from efficientvit.models.nn.norm import set_norm_eps
from efficientvit.models.utils import load_state_dict_from_file

__all__ = ["create_cls_model"]

def create_cls_model(name: str, pretrained=True, weight_url: str or None = None, **kwargs) -> ReduceFormerCls:
    model_dict = {
        "b1_rf": reduceformer_cls_b1,
        "b2_rf": reduceformer_cls_b2,
        "b3_rf": reduceformer_cls_b3,
    }

    model_id = name.split("-")[0]
    if model_id not in model_dict:
        raise ValueError(f"Do not find {name} in the model zoo. List of models: {list(model_dict.keys())}")
    else:
        model = model_dict[model_id](**kwargs)
        
    return model
