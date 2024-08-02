# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from functools import partial
import torch
from torch import nn

from packnet_sfm.networks.DEST.simplified_attention import SimplifiedTransformer as SimpTR
from packnet_sfm.networks.DEST.simplified_joint_attention import SimplifiedJointTransformer as SimpTR_Joint


def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


class DEST_Encoder_Decoder(nn.Module):
    def __init__(
            self,
            *,
            img_size=(192, 640),
            dims=(32, 64, 160, 256),
            heads=(1, 2, 4, 8),
            ff_expansion=(8, 8, 4, 4),
            reduction_ratio=(8, 4, 2, 1),
            num_layers=(2, 2, 2, 2),
            channels=3,
            decoder_dim=128,
            num_classes=64,
            semseg=False
    ):
        super().__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'
        
        self.dest_encoder = SimpTR(
            img_size=img_size, in_chans=channels, num_classes=num_classes,
            embed_dims=dims, num_heads=heads, mlp_ratios=ff_expansion, qkv_bias=True, qk_scale=None, drop_rate=0,
            drop_path_rate=0.1, attn_drop_rate=0., norm_layer=nn.LayerNorm, depths=num_layers, sr_ratios=reduction_ratio)

        self.dims = dims
        self.fuse_conv1 = nn.Sequential(nn.Conv2d(dims[-1], dims[-1], 1), nn.ReLU(inplace=True))
        self.fuse_conv2 = nn.Sequential(nn.Conv2d(dims[-2], dims[-2], 1), nn.ReLU(inplace=True))
        self.fuse_conv3 = nn.Sequential(nn.Conv2d(dims[-3], dims[-3], 1), nn.ReLU(inplace=True))
        self.fuse_conv4 = nn.Sequential(nn.Conv2d(dims[-4], dims[-4], 1), nn.ReLU(inplace=True))

        self.upsample = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'))]*len(dims))

        self.fused_1 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dims[-1], dims[-1], 3), nn.ReLU(inplace=True))
        self.fused_2 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dims[-2] + dims[-1], dims[-2], 3), nn.ReLU(inplace=True))
        self.fused_3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dims[-3] + dims[-2], dims[-3], 3), nn.ReLU(inplace=True))
        self.fused_4 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dims[-4] + dims[-3], dims[-4], 3), nn.ReLU(inplace=True))
        self.fused_5 = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                         nn.Conv2d(dims[-4], dims[-4], 1),
                                         nn.ReLU(True))
        self.semseg = semseg

    def dest_decoder(self, lay_out):
        fused_1 = self.fuse_conv1(lay_out[-1])
        fused_1 = self.upsample[-1](fused_1)
        fused_1 = self.fused_1(fused_1)
        fused_2 = torch.cat([fused_1, self.fuse_conv2(lay_out[-2])], 1)

        fused_2 = self.upsample[-2](fused_2)
        fused_2 = self.fused_2(fused_2)
        fused_3 = torch.cat([fused_2, self.fuse_conv3(lay_out[-3])], 1)

        fused_3 = self.upsample[-3](fused_3)
        fused_3 = self.fused_3(fused_3)
        fused_4 = torch.cat([fused_3, self.fuse_conv4(lay_out[-4])], 1)

        fused_4 = self.upsample[-4](fused_4)
        fused_4 = self.fused_4(fused_4)
        
        if self.semseg: 
            return fused_4
            
        fused_5 = self.fused_5(fused_4)
        return fused_5, fused_4, fused_3, fused_2

    def forward(self, x):
        layer_outputs, ref_feat = self.dest_encoder(x)
        
        out = self.dest_decoder(layer_outputs)
        
        return out, layer_outputs, ref_feat

def DEST_Pose(
        img_size=(192, 640),
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 8, 8),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = (2, 2, 2, 2),
        channels=3,
        num_classes=512,
        connectivity=True):

        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'


        if connectivity :
            model = SimpTR_Joint(
                img_size=img_size, in_chans=channels, num_classes=num_classes,
                embed_dims=dims, num_heads=heads, mlp_ratios=ff_expansion, qkv_bias=True, qk_scale=None, drop_rate=0.,
                drop_path_rate=0.1, attn_drop_rate= 0., norm_layer=nn.LayerNorm, depths=num_layers, sr_ratios=reduction_ratio)
        else:
            model = SimpTR(
                img_size=img_size, in_chans=channels, num_classes=num_classes,
                embed_dims=dims, num_heads=heads, mlp_ratios=ff_expansion, qkv_bias=True, qk_scale=None, drop_rate=0.,
                drop_path_rate=0.1, attn_drop_rate= 0., norm_layer=nn.LayerNorm, depths=num_layers, sr_ratios=reduction_ratio)

        return num_classes, model



def SimpleTR_B0(img_size=(192, 640), num_out_ch=64, semseg=False):
    model = DEST_Encoder_Decoder(
        img_size=img_size,
        dims=(32, 64, 160, 256),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=(2, 2, 2, 2),
        channels=3,
        decoder_dim=256,
        num_classes=num_out_ch, 
        semseg=semseg)
    return num_out_ch, model

def SimpleTR_B1(img_size=(192, 640), num_out_ch=256, semseg=False):
    model = DEST_Encoder_Decoder(
        img_size=img_size,
        dims=(64, 128, 250, 320),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=(2, 2, 2, 2),
        channels=3,
        decoder_dim=num_out_ch,
        num_classes=num_out_ch, 
        semseg=semseg)
    return num_out_ch, model

def SimpleTR_B2(img_size=(192, 640), num_out_ch=256, semseg=False):
    model = DEST_Encoder_Decoder(
        img_size=img_size,
        dims=(64, 128, 250, 320),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=(3, 3, 6, 3),
        channels=3,
        decoder_dim=num_out_ch,
        num_classes=num_out_ch, 
        semseg=semseg)
    return num_out_ch, model


def SimpleTR_B3(img_size=(192, 640), num_out_ch=256, semseg=False):
    model = DEST_Encoder_Decoder(
        img_size=img_size,
        dims=(64, 128, 250, 320),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=(3, 6, 8, 3), 
        channels=3,
        decoder_dim=512,
        num_classes=256, 
        semseg=semseg)
    return num_out_ch, model

def SimpleTR_B4(img_size=(192, 640), num_out_ch=512, semseg=False):
    model = DEST_Encoder_Decoder(
        img_size=img_size,
        dims=(64, 128, 250, 320),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=(3, 8, 12, 5),
        channels=3,
        decoder_dim=num_out_ch,
        num_classes=num_out_ch, 
        semseg=semseg)
    return num_out_ch, model

def SimpleTR_B5(img_size=(192, 640), num_out_ch=512, semseg=False):
    model = DEST_Encoder_Decoder(
        img_size=img_size,
        dims=(64, 128, 250, 320),
        heads=(1, 2, 5, 8),
        ff_expansion=(8, 8, 4, 4),
        reduction_ratio=(8, 4, 2, 1),
        num_layers=(3, 10, 16, 5),
        channels=3,
        decoder_dim=num_out_ch,
        num_classes=num_out_ch, 
        semseg=semseg)
    return num_out_ch, model

