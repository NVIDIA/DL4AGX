# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model


class Block_Conv_SelfAttn(nn.Module):
    """
    Convolutional Self-Attention module

    Parameters
    ----------
    dim : int
        Number of input channels.
    drop_path : float
        Stochastic depth rate. Default: 0.0.
    layer_scale_init_value : float
        Init value for Layer Scale. Default: 1e-6.
    sr_to : int
        Target spatial reduction size. Default: 14.
    num_heads : int
        Number of heads. Defulat: 4.
    mlp_ratio : int
        Number to multiply input dimension for the last mlp layer. Default: 3.
    neighbors : int
        Kernel window size for depth-wise convolution. Default: 5
    resize_mode : string
        Algorithm used for resizing: ['nearest' | 'bilinear']. Default: 'bilinear'.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=0., sr_to=14, num_heads=4, mlp_ratio=3,
                 neighbors=7, resize_mode='bilinear', **kwargs):
        super().__init__()
        self.dim = mlp_ratio * dim
        self.num_heads = num_heads
        self.resize_mode = resize_mode
        self.sr_to = sr_to
        self.HW = sr_to ** 2

        self.v = nn.Conv2d(dim, dim, kernel_size=neighbors, padding=neighbors//2, groups=dim)
        self.act_v = nn.Sequential(nn.BatchNorm2d(dim), Swish(dim, trainable=False))

        self.q = nn.Conv2d(dim, self.num_heads * self.HW, 1)
        self.norm_q = nn.BatchNorm2d(self.num_heads * self.HW)
        self.qk = nn.Conv2d(self.num_heads * self.HW, dim, 1)
        self.act_qk = nn.Sequential(nn.BatchNorm2d(dim), nn.Sigmoid())

        self.qkv = nn.Conv2d(dim, self.dim, 1)

        self.act_qkv = nn.Sequential(nn.BatchNorm2d(self.dim), Swish(self.dim, trainable=False))
        self.mlp = nn.Conv2d(self.dim, dim, 1)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((1, dim, 1, 1)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        _, _, H_og, W_og = input.shape

        # TensorRT-8.6.11.4 - restricted mode does NOT support resize with size parameter
        if type(H_og) != int:
            H_og = H_og.item()
        if type(W_og) != int:
            W_og = W_og.item()
        sr_h, sr_w = self.sr_to / H_og, self.sr_to / W_og

        v = self.act_v(self.v(input))

        # TensorRT-8.6.11.4 - restricted mode does NOT support resize with size parameter
        v_ = torch.nn.functional.interpolate(v, scale_factor=(sr_h, sr_w), mode='bilinear', align_corners=False)
        # v_ = torch.nn.functional.interpolate(v, size=(self.sr_to, self.sr_to), mode='bilinear', align_corners=False)

        q = self.norm_q(self.q(v_))
        B_, C, H, W = q.shape
        k = q.view(B_, self.num_heads, self.HW, self.HW).transpose(3, 2).contiguous().view(B_, self.num_heads * self.HW, H, W)

        qk = torch.nn.functional.interpolate(q * k, scale_factor=(1/sr_h, 1/sr_w), mode='bilinear', align_corners=False)
        # qk = torch.nn.functional.interpolate(q * k, size=(H_og, W_og), mode='bilinear', align_corners=False)

        qk = self.act_qk(self.qk(qk))
        x = self.act_qkv(self.qkv(qk * v))
        x = self.mlp(x)

        if self.gamma is not None:
            x = self.gamma * x

        return input + self.drop_path(x)


class Swish(nn.Module):
    """
    Swish activation [b * x * sigmoid(x)] : https://arxiv.org/abs/1710.05941v2

    Parameters
    ----------
    dim : int
        Number of input channels.
    trainable : bool
        Whether to include a trainable parameter b or not. Default: False.
    """
    def __init__(self, dim, trainable=False):
        super().__init__()
        if trainable:
            self.beta = nn.Parameter(torch.ones((1, dim, 1, 1)), requires_grad=True)
        else:
            self.beta = 1.
        self.trainable = trainable

    def forward(self, x):
        if self.trainable:
            x = self.beta * x
        return x * self.sigm(x)


class CSA_backbone(nn.Module):
    """
    Backbone Network that incorporates CSA modules.

    Parameters
    ----------
    in_chans : int
        Number of input channels. Default: 3.
    num_classes : int
        Number of output classes for prediction. Default: 1000.
    depths : list
        Numbers of blocks per phase. Default: [3, 3, 9, 3]
    dims : list
        Numbers of channels for each block per phase. Default: [96, 192, 384, 768]
    drop_path_rate : float
        Stochastic depth rate. Default: 0.
    layer_scale_init_value : float
        Init value for Layer Scale. Default: 0.
    head_init_scale : float
        Init scaling value for classifier weights and biases. Default: 1.
    ds_patch : list
        Kernel window sizes for downsampling layers per phase. Default: [7, 3, 3, 3]
    strides : list
        Stride sizes for downsampling layers per phase. Default: [4, 2, 2, 2]
    num_heads : list
        Numbers of heads per phase. Default: [1, 2, 4, 8]
    mlp_dim : list
        Numbers to multiply input dimension for the last mlp layer per phase. Default: [2, 2, 2, 2].
    sr_to : list
        Sizes to reduce feature maps to. Default: [14, 14, 14, 7].
    neighbors : 5
        Kernel window size for depth-wise convolution for all CSA blocks. Default: 5.
    resize_mode : string
        Algorithm used for resizing: ['nearest' | 'bilinear']. Default: 'bilinear'.
    """
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                 layer_scale_init_value=0, head_init_scale=1., ds_patch=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                 num_heads=[1, 2, 4, 8], mlp_dim=[2, 2, 2, 2], sr_to=[14, 14, 14, 7], neighbors=5, resize_mode='bilinear'):
        super().__init__()
        self.num_phases = len(depths)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=ds_patch[0], stride=strides[0], padding=ds_patch[0]//2),
            nn.BatchNorm2d(dims[0])
            )
        self.downsample_layers.append(stem)

        for i in range(self.num_phases - 1):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=ds_patch[i + 1], stride=strides[i + 1]),
                nn.BatchNorm2d(dims[i + 1])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(self.num_phases):
            stage = nn.Sequential(
                *[Block_Conv_SelfAttn(dim=dims[i],
                                      drop_path=dp_rates[cur + j],
                                      layer_scale_init_value=layer_scale_init_value,
                                      num_heads=num_heads[i],
                                      mlp_ratio=mlp_dim[i],
                                      sr_to=sr_to[i],
                                      neighbors=neighbors,
                                      resize_mode=resize_mode) for j in
                  range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.BatchNorm2d(dims[-1])
        self.head = nn.Conv2d(dims[-1], num_classes, 1)

        self.apply(self._init_weights)
        self.avgpool = nn.AvgPool2d(6)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(self.num_phases):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        # TensorRT-8.6.11.4 - restricted mode does not support ReduceMean
        # x = x.mean([-2, -1]).view(x.size(0), x.size(1), 1, 1)
        x = self.avgpool(x)
        return self.norm(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x.squeeze()


@register_model
def convselfattn(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=False, in_22k=False, **kwargs):
    model = CSA_backbone(depths=[3, 4, 6, 3], dims=[96, 192, 384, 768], num_heads=[1, 2, 4, 8], mlp_dim=[3, 3, 3, 3],
                         sr_to=[14, 14, 14, 7], neighbors=5, **kwargs)
    return model

