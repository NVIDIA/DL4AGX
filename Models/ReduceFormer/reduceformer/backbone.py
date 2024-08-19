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
from torch.cuda.amp import autocast

from efficientvit.models.utils import build_kwargs_from_config, get_same_padding, val2tuple
from efficientvit.models.nn.act import build_act
from efficientvit.models.nn import (
    ConvLayer,
    DSConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
)


__all__ = [
    "ReduceFormerBackbone",
    "reduceformer_backbone_b1",
    "reduceformer_backbone_b2",
    "reduceformer_backbone_b3",
]


class RF_Attn(nn.Module):
    r"""ReduceFormer Attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
    ):
        super(RF_Attn, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim
        self.total_dim = total_dim * (1 + len(scales))
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)
        self.dim = dim
        self.num_head = heads
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    )
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @autocast(enabled=False)
    def rf_att(self, qkv: torch.Tensor) -> torch.Tensor:
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        
        if len(qkv.shape) == 4: 
            B, _, H, W = list(qkv.size())     
            qkv = torch.reshape(qkv, (B, 3, -1, H, W))
            
        q, k, v = (
            qkv[:, 0, :, :, :], 
            qkv[:, 1, :, :, :],
            qkv[:, 2, :, :, :]
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        sum_k = torch.sum(k, (-1, -2), keepdim=True) 
        sum_v = torch.sum(v * sum_k, (-1, -2), keepdim=True) 
        sum_kv = torch.sum(k * sum_v, (-1, -2), keepdim=True)          
        sum_q = torch.sum(q, 1, keepdim=True)

        out = (q * sum_kv) / (sum_q * sum_k + 1.0e-15)
        out = self.proj(out)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        B, _, H, W = list(qkv.size())
        multi_scale_qkv = [qkv.reshape(B, 3, -1, H, W)]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv).reshape(B, 3, -1, H, W))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=2)

        out = self.rf_att(multi_scale_qkv)
        return out

    @staticmethod
    def configure_rfattn(model: nn.Module, **kwargs) -> None:
        eps = kwargs.get("eps", None)
        for m in model.modules():
            if isinstance(m, RF_Attn):
                if eps is not None:
                    m.eps = eps


class ReduceFormerBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
    ):
        super(ReduceFormerBlock, self).__init__()
        self.context_module = ResidualBlock(
            RF_Attn(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ReduceFormerBackbone(nn.Module):
    def __init__(
        self,
        width_list: list[int],
        depth_list: list[int],
        in_channels=3,
        dim=32,
        expand_ratio=4,
        norm="bn2d",
        act_func="hswish",
    ) -> None:
        super().__init__()

        self.width_list = []
        # input stem
        self.input_stem = [
            ConvLayer(
                in_channels=3,
                out_channels=width_list[0],
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth_list[0]):
            block = self.build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(self.input_stem)
        self.width_list.append(in_channels)

        # stages
        self.stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self.build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
                block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        for w, d in zip(width_list[3:], depth_list[3:]):
            stage = []
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
                fewer_norm=True,
            )
            stage.append(ResidualBlock(block, None))
            in_channels = w

            for _ in range(d):
                stage.append(
                    ReduceFormerBlock(
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=expand_ratio,
                        norm=norm,
                        act_func=act_func,
                    )
                )
            self.stages.append(OpSequential(stage))
            self.width_list.append(in_channels)
        self.stages = nn.ModuleList(self.stages)

    @staticmethod
    def build_local_block(
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: float,
        norm: str,
        act_func: str,
        fewer_norm: bool = False,
    ) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None),
            )
        return block

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_dict = {"input": x}
        output_dict["stage0"] = x = self.input_stem(x)  
        for stage_id, stage in enumerate(self.stages, 1):
            output_dict["stage%d" % stage_id] = x = stage(x)
        output_dict["stage_final"] = x
        return output_dict

def reduceformer_backbone_b1(**kwargs) -> ReduceFormerBackbone:
    backbone = ReduceFormerBackbone(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[1, 2, 3, 3, 4] ,
        dim=16,
        **build_kwargs_from_config(kwargs, ReduceFormerBackbone),
    )
    return backbone

def reduceformer_backbone_b2(**kwargs) -> ReduceFormerBackbone:
    backbone = ReduceFormerBackbone(
        width_list=[24, 48, 96, 192, 384],
        depth_list=[1, 3, 4, 4, 6], 
        dim=32,
        **build_kwargs_from_config(kwargs, ReduceFormerBackbone),
    )
    return backbone

def reduceformer_backbone_b3(**kwargs) -> ReduceFormerBackbone:
    backbone = ReduceFormerBackbone(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 4, 6, 6, 9],
        dim=32,
        **build_kwargs_from_config(kwargs, ReduceFormerBackbone),
    )
    return backbone
