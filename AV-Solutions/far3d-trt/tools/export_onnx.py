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
# modified from https://github.com/megvii-research/Far3D/blob/main/tools/test.py 

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------


import os
import math
import pickle as pkl
from argparse import ArgumentParser
import inspect
import numpy as np

import torch
from mmcv import Config

from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from projects.mmdet3d_plugin.models.detectors.far3d import Far3D
import projects.mmdet3d_plugin.models.utils.detr3d_transformer as detr3d_transformer

import onnx_graphsurgeon as gs
import onnx
import onnxsim


onnx_opset_version = 15
precision = 'fp32'

msda_plugin_name = "MultiscaleDeformableAttnPlugin_TRT"

def recursive_apply(obj, func):
    if isinstance(obj, (list, tuple)):
        out = []
        for v in obj:
            out.append(recursive_apply(v, func))
        if isinstance(obj, tuple):
            return tuple(out)
        return out
    if isinstance(obj, dict):
        out_dict = dict()
        for k,v in obj.items():
            out_dict[k] = recursive_apply(v, func)
        return out_dict
    if isinstance(obj, torch.Tensor):
        return func(obj)
    return obj


def recursive_to_gpu(obj):
    return recursive_apply(obj, lambda x: x.cuda())

base_msda = detr3d_transformer.MultiScaleDeformableAttnFunction
class PatchMSDA(torch.autograd.Function):

    # this is necessary to remove im2col_step
    # but we don't really want to pass in reference_points..

    @staticmethod
    def symbolic(
        g,
        value,
        value_spatial_shapes,
        level_start_index, 
        sampling_offsets,
        attention_weights,
        im2col_step
    ):
        value_spatial_shapes = g.op("Cast", value_spatial_shapes, to_i=torch._C._onnx.TensorProtoDataType.INT32)
        level_start_index = g.op("Cast", level_start_index, to_i=torch._C._onnx.TensorProtoDataType.INT32)
        attention_op = g.op(
            f"trt::{msda_plugin_name}",
            value,
            value_spatial_shapes,
            level_start_index,
            sampling_offsets,
            attention_weights,
        )
        
        return attention_op

    @staticmethod
    def forward(ctx, feat_flatten, spatial_flatten, level_start_index, points_2d,
                weights, im2col_step):
        return base_msda.forward(ctx, feat_flatten, spatial_flatten, level_start_index, points_2d, weights, im2col_step)
        

def patch_inverse(g, matrix):
    output_op = g.op(f"trt::{inverse_plugin_name}", matrix)
    output_op.setType(matrix.type().with_sizes(matrix.type().sizes()))
    return output_op

# from https://github.com/pytorch/pytorch/pull/100040/files
def patch_atan2(g, input, other):
    # self is y, and other is x on coordinate
    slope = g.op("Div", input, other)
    atan = g.op("Atan", slope)
    const_zero = g.op("Constant", value_t=torch.tensor(0))
    const_pi = g.op("Constant", value_t=torch.tensor(math.pi))

    condition_second_or_third_quadrant = g.op("Greater", input, const_zero)
    second_third_quadrant = g.op(
        "Where",
        condition_second_or_third_quadrant,
        g.op("Add", atan, const_pi),
        g.op("Sub", atan, const_pi),
    )

    condition_14_or_23_quadrant = g.op("Less", other, const_zero)
    result = g.op("Where", condition_14_or_23_quadrant, second_third_quadrant, atan)

    return result


torch.onnx.register_custom_op_symbolic(f"trt::{msda_plugin_name}", PatchMSDA.symbolic, 9)
torch.onnx.register_custom_op_symbolic("::inverse", patch_inverse, 9)
torch.onnx.register_custom_op_symbolic("aten::linalg_inv", patch_inverse, 9)
torch.onnx.register_custom_op_symbolic("aten::atan2", patch_atan2, 9)

def noop_complex_operators(g, input):
    # ONNX does not have operators to *directly* manipulate real/imaginary components
    # However, a few torch APIs (e.g. .tolist()) use complex operations when input is real,
    # which results in failures due to missing operators for complex numbers

    # `aten::resolve_conj` and `aten::resolve_neg` can safely be implemented as no-op
    return input

torch.onnx.register_custom_op_symbolic("aten::resolve_conj", noop_complex_operators, 9)
torch.onnx.register_custom_op_symbolic("aten::resolve_neg", noop_complex_operators, 9)

class ImageEncoderExportWrapper(torch.nn.Module):
    def __init__(self, module: Far3D):
        super().__init__()
        self.module = module

    def get_input_names(self):
        return list(inspect.signature(self.forward).parameters)
    
    def get_output_names(self):
        return ['enc_cls_scores_0', 'enc_cls_scores_1', 'enc_cls_scores_2', 'enc_cls_scores_3', 
                'enc_bbox_preds_0', 'enc_bbox_preds_1', 'enc_bbox_preds_2', 'enc_bbox_preds_3', 
                'pred_centers2d_offset_0', 'pred_centers2d_offset_1', 'pred_centers2d_offset_2', 'pred_centers2d_offset_3', 
                'objectnesses_0', 'objectnesses_1', 'objectnesses_2', 'objectnesses_3', 
                'depth_logit', 'pred_depth', 
                'bbox_list_0', 'bbox_list_1', 'bbox_list_2', 'bbox_list_3', 'bbox_list_4', 'bbox_list_5', 'bbox_list_6', 
                'bbox2d_scores', 'valid_indices', 
                'img_feats_0', 'img_feats_1', 'img_feats_2', 'img_feats_3']

    """
        img: [batch, camera, height, width, channels]
    """
    def forward(self, img, intrinsics, extrinsics, 
                      lidar2img, img2lidar, ego_pose_inv, 
                      ego_pose, timestamp, 
                      prev_exists, memory_embedding, 
                      memory_reference_point, memory_timestamp, memory_egopose, memory_velo):
        img = img.permute(0,1,4,2,3)
        
        # memory is not used when doing just image encoding, but it is kept here to keep the interface the same
        # for the full model
        img_feats = self.module.extract_img_feat(img=img, return_depth=False)
        img_metas = [dict(pad_shape=[torch.Tensor([640, 960, 3]).to(device=img.device)])]
        
        
        dets, img_feats = self.module.get_bboxes(img_metas=img_metas, img_feats=img_feats, intrinsics=intrinsics, extrinsics=extrinsics, lidar2img=lidar2img, img2lidar=img2lidar, ego_pose_inv=ego_pose_inv,
                               ego_pose=ego_pose, timestamp=timestamp, prev_exists=prev_exists)
        # unpack output into a flat list
        output = []
        for value in dets.values():
            if value is None:
                continue
            if isinstance(value, list):
                for v in value:
                    output.append(v)
            else:
                output.append(value)
        # img feats is a list over scales
        for v in img_feats:
            output.append(v)
        return output

far3d_state_names = ["memory_embedding", "memory_reference_point", "memory_egopose", "memory_velo", "memory_timestamp"]

class TransformerDecoderExportWrapper(torch.nn.Module):
    def __init__(self, module: Far3D):
        super().__init__()
        self.module = module

    def get_input_names(self):
        return list(inspect.signature(self.forward).parameters)
    
    def get_output_names(self):
        output = ['bboxes', 'scores', 'labels']
        output += [x + "_out" for x in far3d_state_names]
        return output
    
    def forward(self, 
                img_feats_0, img_feats_1, img_feats_2, img_feats_3,
                bbox_list_0, bbox_list_1, bbox_list_2, bbox_list_3, bbox_list_4, bbox_list_5, bbox_list_6,
                prev_exists, 
                timestamp, 
                ego_pose_inv, 
                intrinsics, 
                extrinsics, 
                pred_depth, 
                bbox2d_scores, 
                valid_indices,
                lidar2img,
                img2lidar,
                ego_pose,
                memory_embedding, memory_reference_point, memory_egopose, memory_velo, memory_timestamp):
        img_metas = [dict(pad_shape=[torch.Tensor([640, 960, 3]).to(device=img_feats_0.device)])]
        data = dict(img_feats=[img_feats_0, img_feats_1, img_feats_2, img_feats_3], 
                    prev_exists=prev_exists, 
                    timestamp=timestamp, 
                    ego_pose_inv=ego_pose_inv,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    pred_depth=pred_depth,
                    bbox2d_scores=bbox2d_scores,
                    valid_indices=valid_indices,
                    lidar2img=lidar2img,
                    img2lidar=img2lidar,
                    ego_pose=ego_pose
                    )
        state = dict(memory_embedding=memory_embedding, 
                     memory_reference_point=memory_reference_point, 
                     memory_egopose=memory_egopose,
                      memory_velo=memory_velo, 
                     memory_timestamp=memory_timestamp)
        outs_roi=dict(pred_depth=pred_depth, 
                                            bbox2d_scores=bbox2d_scores, 
                                            valid_indices=valid_indices,
                                            bbox_list=[bbox_list_0, bbox_list_1, bbox_list_2, bbox_list_3, bbox_list_4, bbox_list_5, bbox_list_6],
                                            )
        bbox_list, state = self.module.simple_test_bboxes(img_metas=img_metas, outs_roi=outs_roi, state=state, **data)
        output = [bbox_list[0][0], bbox_list[0][1], bbox_list[0][2].to(torch.int32)]
        for name in far3d_state_names:
            output.append(state[name])
        return output

class Far3DExportWrapper(torch.nn.Module):
    def __init__(self, module: Far3D):
        super().__init__()
        self.module = module

    def get_input_names(self):
        return list(inspect.signature(self.forward).parameters)
    
    def get_output_names(self):
        output = ['bboxes', 'scores', 'labels']
        output += [x + "_out" for x in far3d_state_names]
        return output
    
    def forward(self, 
                img,
                intrinsics,
                extrinsics,
                lidar2img,
                img2lidar,
                ego_pose_inv,
                ego_pose,
                timestamp,
                prev_exists, 
                memory_embedding, memory_reference_point, memory_timestamp, memory_egopose, memory_velo):
        img_metas = [dict(pad_shape=[torch.Tensor([640, 960, 3]).to(img.device)])]
        data = dict(img=img, 
                    prev_exists=prev_exists, 
                    timestamp=timestamp, 
                    ego_pose_inv=ego_pose_inv,
                    intrinsics=intrinsics,
                    extrinsics=extrinsics,
                    lidar2img=lidar2img,
                    img2lidar=img2lidar,
                    ego_pose=ego_pose
                    )
        state = dict(memory_embedding=memory_embedding, 
                     memory_reference_point=memory_reference_point, 
                     memory_egopose=memory_egopose,
                      memory_velo=memory_velo, 
                     memory_timestamp=memory_timestamp)
        
        bbox_list, state = self.module.simple_test(img_metas=img_metas, state=state, **data)
        bbox_list = bbox_list[0]['pts_bbox']
        output = [bbox_list['boxes_3d'], bbox_list['scores_3d'], bbox_list['labels_3d'].to(torch.int32)]
        for name in far3d_state_names:
            output.append(state[name])
        return output   

def patch_msda_onnx(onnx_file_path):
    base = os.path.splitext(onnx_file_path)[0]
    model = onnx.load(onnx_file_path)
    inferred = onnx.shape_inference.infer_shapes(model)
    sim_inferred, _ = onnxsim.simplify(inferred)
    # Have to do a while loop here to re-run simplify and infer_shapes
    g = gs.import_onnx(sim_inferred)
    count = 0
    while(True):
        found = False
        # unfortunately since we re-sort the model, it is necessary to just re-iterate over the whole node list
        for node in g.nodes:
            if node.op == 'MultiscaleDeformableAttnPlugin_TRT':
                if(len(node.inputs[-1].shape)) == 4:
                    print('Updating {}'.format(node.name))
                    if count == 0:
                        # all of the attention layers are of the same shape in Far3D,
                        # but shape inference breaks after the first one, so we store the shapes
                        # from the first attention layer here
                        N, _, M, _ = node.inputs[0].shape
                        _, Lq, _, L, P, _  = node.inputs[-2].shape
                    count += 1
                    new_shape = np.array([N, Lq, M, L, P], dtype=np.int32)
                    
                    name = node.name
                    name = '/'.join(name.split('/')[:-1] + ['reshape_constant'])
                    new_shape = gs.Constant(name=name, values=new_shape)
                    #casted = g.layer(op="Cast", inputs=[new_shape], outputs=["Cast_{}_out".format(node.name)], attrs=dict(to=onnx.TensorProto.INT32))[0]
                    reshape = g.layer(op="Reshape", inputs=[node.inputs[-1], new_shape], outputs=["Reshape_{}_out".format(node.name)])[0]
                    reshape.shape = [N, Lq, M, L, P]
                    reshape.dtype = node.inputs[0].dtype
                    node.inputs[-1] = reshape
                    g.toposort()
                    new_onnx = gs.export_onnx(g)
                    inferred = onnx.shape_inference.infer_shapes(new_onnx)
                    sim_inferred, _ = onnxsim.simplify(inferred)
                    g = gs.import_onnx(sim_inferred)
                    found = True
                    break
        if not found:
            break
    onnx.save_model(sim_inferred, f"{base}.patched.onnx")
    
def load_input(path):
    input_dict = pkl.load(open(path, 'rb'))
    input_names = input_dict["input_names"]
    output = []
    for name in input_names:
        value = input_dict["data"][name]
        if not isinstance(value, torch.Tensor):
            value = torch.rand(size=value)
        output.append(value)
    output = recursive_to_gpu(output)
    return tuple(output)

    

def main():
    parser = ArgumentParser("far3d onnx exporter")
    parser.add_argument("config", help="Path to configuration file")
    parser.add_argument("checkpoint", help="Path to checkpoint")
    args = parser.parse_args()
    
    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    
    import importlib
    if hasattr(cfg, 'plugin_dir'):
        plugin_dir = cfg.plugin_dir
        _module_dir = os.path.dirname(plugin_dir)
        _module_dir = _module_dir.split('/')
        _module_path = _module_dir[0]

        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)

    
    cfg.data.test.test_mode = True
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model: Far3D = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    
    # need to use cuda for export due to tracing attention modules that only have  GPU implementation
    model.cuda()
    model.eval()

    model.img_roi_head.topk_proposal=50
    model.img_roi_head.sample_with_score = False

    encoder_input_data = load_input('data/encoder_input.pkl')
    decoder_input_data = load_input('data/decoder_input.pkl')
    model_input_data = load_input('data/model_input.pkl')
    
    
    with torch.no_grad():
        detr3d_transformer.MultiScaleDeformableAttnFunction = PatchMSDA
        encoder = ImageEncoderExportWrapper(model)
        encoder_input_names = encoder.get_input_names()
        encoder_output_names = encoder.get_output_names()
        encoder_input_data_ = list(encoder_input_data)
        encoder_input_data_[0] = encoder_input_data_[0].permute(0,1,3,4,2)
        encoder_input_data = tuple(encoder_input_data_)
        # this changes it from bgr input to rgb input
        encoder.module.img_backbone.stem[0].weight = torch.nn.Parameter(encoder.module.img_backbone.stem[0].weight.flip(1))
        encoder.module.mean = encoder.module.mean.flip(0)
        encoder.module.std = encoder.module.std.flip(0)
        torch.onnx.export(encoder, args=encoder_input_data, 
                        f="far3d.encoder.onnx",
                        do_constant_folding=False,
                        input_names=encoder_input_names,
                        output_names=encoder_output_names,
                        opset_version=15)
        model.eval()
        decoder = TransformerDecoderExportWrapper(model)
        decoder_input_names = decoder.get_input_names()
        decoder_output_names = decoder.get_output_names()
        decoder_file_name= "far3d.decoder.onnx"
        torch.onnx.export(decoder, args=decoder_input_data,
                          f=decoder_file_name,
                          do_constant_folding=False,
                            input_names=decoder_input_names,
                            output_names=decoder_output_names,
                            opset_version=15)
        patch_msda_onnx(decoder_file_name)

        
        model.eval()
        full_model = Far3DExportWrapper(model)
        input_names = full_model.get_input_names()
        output_names = full_model.get_output_names()
        full_file_name = "far3d.onnx"
        torch.onnx.export(full_model, args=model_input_data,
                          f=full_file_name,
                          do_constant_folding=False,
                            input_names=input_names,
                            output_names=output_names,
                            opset_version=15)
        patch_msda_onnx(full_file_name)

        
        # MMCV expected attention_weights to be of the form [N, Lq, M, L * P] while tensorrt open source expects
        # the form [N, Lq, M, L, P]
        # the below work around is due to bugs in shape inference during the pytorch -> onnx conversion process
        # thus we operate directly on the produced onnx file to add a reshape layer prior to each MSDA




if __name__ == '__main__':
    main()
