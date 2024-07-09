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

import torch
import torch.nn as nn

from efficientvit.models.reduceformer.backbone import ReduceFormerBackbone
from efficientvit.models.nn import ConvLayer, LinearLayer, OpSequential
from efficientvit.models.utils import build_kwargs_from_config

__all__ = [
    "ReduceFormerCls",
    ######################
    "reduceformer_cls_b1",
    "reduceformer_cls_b2",
    "reduceformer_cls_b3",
]


class ClsHead(OpSequential):
    def __init__(
        self,
        in_channels: int,
        width_list: list[int],
        n_classes=1000,
        dropout=0.0,
        norm="bn2d",
        act_func="hswish",
        fid="stage_final",
    ):
        ops = [
            ConvLayer(in_channels, width_list[0], 1, norm=norm, act_func=act_func),
            nn.AdaptiveAvgPool2d(output_size=1),
            LinearLayer(width_list[0], width_list[1], False, norm="ln", act_func=act_func),
            LinearLayer(width_list[1], n_classes, True, dropout, None, None),
        ]
        super().__init__(ops)

        self.fid = fid

    def forward(self, feed_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        x = feed_dict[self.fid]
        return OpSequential.forward(self, x)


class ReduceFormerCls(nn.Module):
    def __init__(self, backbone: ReduceFormerBackbone, head: ClsHead) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feed_dict = self.backbone(x)
        output = self.head(feed_dict)
        return output


def reduceformer_cls_b1(**kwargs) -> ReduceFormerCls:
    from efficientvit.models.reduceformer.backbone import reduceformer_backbone_b1

    backbone = reduceformer_backbone_b1(**kwargs)

    head = ClsHead(
        in_channels=256,
        width_list=[1536, 1600],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ReduceFormerCls(backbone, head)
    return model


def reduceformer_cls_b2(**kwargs) -> ReduceFormerCls:
    from efficientvit.models.reduceformer.backbone import reduceformer_backbone_b2

    backbone = reduceformer_backbone_b2(**kwargs)

    head = ClsHead(
        in_channels=384,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ReduceFormerCls(backbone, head)
    return model


def reduceformer_cls_b3(**kwargs) -> ReduceFormerCls:
    from efficientvit.models.reduceformer.backbone import reduceformer_backbone_b3

    backbone = reduceformer_backbone_b3(**kwargs)

    head = ClsHead(
        in_channels=512,
        width_list=[2304, 2560],
        **build_kwargs_from_config(kwargs, ClsHead),
    )
    model = ReduceFormerCls(backbone, head)
    return model

