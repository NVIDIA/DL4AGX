# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

import math

from packnet_sfm.networks.DEST.simplified_attention import OverlapPatchEmbed, Mlp, Attention_MaxPool


class Attention_Joint_MaxPool(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim

        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Conv1d(dim, dim, 1, bias=qkv_bias)
        self.k = nn.Conv1d(dim, dim, 1, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm2 = nn.BatchNorm1d(self.dim)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.BatchNorm1d(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y, H, W):
        B, C, N = x.shape

        q = self.q(x)
        q = q.reshape(B, self.num_heads, C // self.num_heads, N)
        q = q.permute(0, 1, 3, 2)

        if self.sr_ratio > 1:
            x_ = x.reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1)
            x_ = self.norm(x_)
            k = self.k(x_).reshape(B, self.num_heads, C // self.num_heads, -1)
        else:
            k = self.k(x).reshape(B, self.num_heads, C // self.num_heads, -1)

        v = torch.mean(x, 2, True).repeat(1, 1, self.num_heads).transpose(-2, -1)


        attn = (q @ k) * self.scale

        attn, _ = torch.max(attn, -1)
        
        out = (attn.transpose(-2, -1) @ v) 
        out = out.transpose(-2, -1)

        out = self.proj(out) 

        return out


class JointBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm0 = nn.BatchNorm1d(dim)
        self.norm1_ref = nn.BatchNorm1d(dim)
        self.norm1_src = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)

        self.attn_joint = Attention_Joint_MaxPool(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, ref_feat, src_feat, H, W):
        src_feat = src_feat + self.drop_path(self.attn_joint(self.norm1_ref(ref_feat), self.norm1_src(src_feat), H, W))
        src_feat = src_feat + self.drop_path(self.mlp(self.norm2(src_feat), H, W))
        return src_feat



class SimplifiedJointTransformer(nn.Module):
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=(img_size[0] // 4, img_size[1] // 4), patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=(img_size[0] // 8, img_size[1] // 8), patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=(img_size[0] // 16, img_size[1] // 16), patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([JointBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = nn.BatchNorm1d(self.patch_embed1.N)


        cur += depths[0]
        self.block2 = nn.ModuleList([JointBlock(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = nn.BatchNorm1d(self.patch_embed2.N)

        cur += depths[1]
        self.block3 = nn.ModuleList([JointBlock(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = nn.BatchNorm1d(self.patch_embed3.N)

        cur += depths[2]
        self.block4 = nn.ModuleList([JointBlock(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = nn.BatchNorm1d(self.patch_embed4.N)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, ref_feat, x):
        B = x.shape[0]

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            if i > len(ref_feat['1']) - 1 : i = -1
            x = blk(ref_feat['1'][i], x, H, W)
        x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, -1, H, W).contiguous()

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            if i > len(ref_feat['2']) -1: i=-1
            x = blk(ref_feat['2'][i], x, H, W)
        x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, -1, H, W).contiguous()

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            if i > len(ref_feat['3']) -1: i = -1
            x = blk(ref_feat['3'][i], x, H, W)
        x = self.norm3(x.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, -1, H, W).contiguous()

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            if i > len(ref_feat['4'])-1: i = -1
            x = blk(ref_feat['4'][i], x, H, W)
        x = self.norm4(x.transpose(-2, -1)).transpose(-2, -1)
        x = x.reshape(B, -1, H, W).contiguous()
        return x

    def forward(self, ref_feat, x):
        return self.forward_features(ref_feat, x)

