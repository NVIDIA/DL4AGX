# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os
import numpy as np
import torch
import copy
from third_party.uniad_mmdet3d.datasets.builder import build_dataloader, build_dataset
from third_party.uniad_mmdet3d.models.builder import build_model
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from tqdm import tqdm
import random

warnings.filterwarnings("ignore")

# Set environment variables
os.environ['PYTHONHASHSEED'] = str(42)

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)  # if you are using multi-GPU

# Ensure deterministic algorithms in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
The metadata that needed to be prepared before pytorch/onnx inference:
timestamp, l2g_r_mat, l2g_t, img_metas_scene_token
command, img_metas_can_bus, img_metas_lidar2img
"""

def scene_token_preprocess(scene_token):
    scene_token_list = []
    for ch in scene_token:
        scene_token_list.append(ord(ch))
    return torch.tensor(scene_token_list)

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
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
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    # torch.use_deterministic_algorithms(True)
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # ########## Prepare Inputs for Pytorch Inference ##########
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=0, #cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']

    bevh=50
    img_h = 256
    img_w = 416

    torch.random.manual_seed(0)
    model=model.cuda()
    model=model.eval()
    
    input_shapes = dict(
        prev_track_intances0=[-1, 512],  #901-1149
        prev_track_intances1=[-1, 3],
        prev_track_intances2=[-1, 256],
        prev_track_intances3=[-1],
        prev_track_intances4=[-1],
        prev_track_intances5=[-1],
        prev_track_intances6=[-1],
        prev_track_intances7=[-1],
        prev_track_intances8=[-1],
        prev_track_intances9=[-1, 10],
        prev_track_intances10=[-1, 10],
        prev_track_intances11=[-1, 4, 256],
        prev_track_intances12=[-1, 4],
        prev_track_intances13=[-1],
        prev_timestamp=[1],
        prev_l2g_r_mat=[1, 3, 3],
        prev_l2g_t=[1, 3],
        prev_bev=[bevh**2, 1, 256],
        gt_lane_labels=[1, -1], 
        gt_lane_masks=[1, -1, bevh, bevh],
        gt_segmentation=[1, 7, bevh, bevh],
        img_metas_scene_token=[32],
        timestamp=[1],
        l2g_r_mat=[1, 3, 3], 
        l2g_t=[1, 3], 
        img=[1, 6, 3, img_h, img_w],
        img_metas_can_bus=[18],
        img_metas_lidar2img=[1, 6, 4, 4],
        image_shape=[2],
        command=[1],
        use_prev_bev=[1],
        max_obj_id=[1],
    )

    img_metas_scene_token = torch.zeros(32).cuda()
    def _generate_empty_zeros_tracks_trt():
        num_queries = 901
        dim = 512
        query = torch.zeros((901, 512), dtype=torch.float)
        ref_pts = torch.zeros((901, 3), dtype=torch.float)

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros(
            (len(ref_pts), 10), dtype=torch.float
        )

        output_embedding = torch.zeros(
            (num_queries, dim >> 1)
        )

        obj_idxes = torch.full(
            (len(ref_pts),), -1, dtype=torch.int
        )
        matched_gt_idxes = torch.full(
            (len(ref_pts),), -1, dtype=torch.int
        )
        disappear_time = torch.zeros(
            (len(ref_pts),), dtype=torch.int
        )

        iou = torch.zeros(
            (len(ref_pts),), dtype=torch.float
        )
        scores = torch.zeros(
            (len(ref_pts),), dtype=torch.float
        )
        track_scores = torch.zeros(
            (len(ref_pts),), dtype=torch.float
        )
        # xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes = pred_boxes_init

        pred_logits = torch.zeros(
            (len(ref_pts), 10), dtype=torch.float
        )

        mem_bank = torch.zeros(
            (len(ref_pts), 4, dim // 2),
            dtype=torch.float32,
        )
        mem_padding_mask = torch.ones(
            (len(ref_pts), 4), dtype=torch.int
        )
        save_period = torch.zeros(
            (len(ref_pts),), dtype=torch.float32
        )

        return [
            query,
            ref_pts,
            output_embedding,
            obj_idxes,
            matched_gt_idxes,
            disappear_time,
            iou,
            scores,
            track_scores,
            pred_boxes,
            pred_logits,
            mem_bank,
            mem_padding_mask,
            save_period,
        ]
    test_track_instances =[item*0 for item in _generate_empty_zeros_tracks_trt()] 
    l2g_r_mat0 = torch.zeros(1, 3, 3)
    l2g_t0 = torch.zeros(1,3)
    prev_pos = 0
    prev_angle = 0
    shape0 = None
    shape0_dict = {}
    for sample_id, data in tqdm(enumerate(data_loader)):
        img_metas = data["img_metas"][0].data
        timestamp = data["timestamp"][0] if data["timestamp"] is not None else None
        if sample_id == 0:
            timestamp0 = timestamp
        
        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        img_metas[0][0]['can_bus'][:3] -= prev_pos
        img_metas[0][0]['can_bus'][-1] -= prev_angle
        prev_pos = tmp_pos
        prev_angle = tmp_angle

        onnx_inputs = dict()
        onnx_inputs["img_metas_lidar2img"] = torch.from_numpy(np.float32(np.stack(img_metas[0][0]["lidar2img"])[None,...])).cuda()
        onnx_inputs["img_metas_scene_token"] = scene_token_preprocess(img_metas[0][0]["scene_token"]).float().cuda()
        onnx_inputs["l2g_t"] = data["l2g_t"].float().cuda()
        onnx_inputs["l2g_r_mat"] = data["l2g_r_mat"].float().cuda()
        onnx_inputs["timestamp"] = (timestamp-timestamp0).float().cuda()
        onnx_inputs["command"] = data["command"][0].float().cuda()
        onnx_inputs["img_metas_can_bus"] = torch.from_numpy(np.float32(img_metas[0][0]["can_bus"])).cuda()
        onnx_inputs['img'] = data["img"][0].data[0].float().cuda()
        onnx_inputs['gt_segmentation'] = data["gt_segmentation"][0].float().cuda()
        onnx_inputs['gt_lane_masks'] = data["gt_lane_masks"][0].float().cuda()
        onnx_inputs['gt_lane_labels'] = data["gt_lane_labels"][0].float().cuda()
        for key in input_shapes.keys():
            if not key in onnx_inputs:
                if key=='image_shape':
                    onnx_inputs[key] = np.array([img_h,img_w]).astype(np.float32)
                    onnx_inputs[key] = torch.from_numpy(onnx_inputs[key]).cuda()
                elif key=='prev_bev':
                    if sample_id==0:
                        onnx_inputs[key] = torch.from_numpy(np.zeros([bevh**2, 1, 256]).astype(np.float32)).cuda()
                    else:
                        onnx_inputs[key]= bev_embed
                elif key=='max_obj_id':
                    onnx_inputs[key] = torch.Tensor([0]).int().cuda() if sample_id==0 else max_obj_id
                elif 'prev_track_intances' in key:
                    if sample_id==0:
                        onnx_inputs[key] = test_track_instances[int(key[19:])].cuda()
                    else:
                        if int(key[19:]) in (2,7,10): # 2,7,10 will not be used in ONNX graph
                            # update shape0
                            shape = copy.deepcopy(input_shapes[key])
                            shape[0] = onnx_inputs['prev_track_intances0'].shape[0]
                            # put dummy inputs for pytorch
                            onnx_inputs[key] = torch.zeros(shape).float().cuda()
                        else:
                            onnx_inputs[key]= prev_track_intances_out[key+'_out']
                            if onnx_inputs[key].dtype == torch.int64:
                                onnx_inputs[key] = onnx_inputs[key].int()
                elif key=='prev_l2g_r_mat':
                    onnx_inputs[key] = l2g_r_mat0.float().cuda() if sample_id==0 else prev_l2g_r_mat_out
                elif key=='prev_l2g_t':
                    onnx_inputs[key] = l2g_t0.float().cuda() if sample_id==0 else prev_l2g_t_out
                elif key=='prev_timestamp':
                    onnx_inputs[key] = torch.zeros([1]).float().cuda() if sample_id==0 else prev_timestamp_out
                elif key=='use_prev_bev':
                    cur_img_metas_scene_token = onnx_inputs["img_metas_scene_token"]
                    if not torch.equal(cur_img_metas_scene_token,img_metas_scene_token):
                        onnx_inputs[key] = np.array([0]).astype(np.int32)
                    else:
                        onnx_inputs[key] = np.array([1]).astype(np.int32)
                    onnx_inputs[key] = torch.from_numpy(onnx_inputs[key]).cuda()
                    img_metas_scene_token = cur_img_metas_scene_token

        # import pdb; pdb.set_trace()
        # inputs = tuple(onnx_inputs.values())
        with torch.no_grad():
            dummy_outputs = model.forward_uniad_trt(**onnx_inputs)
        max_obj_id = dummy_outputs[-3]
        bev_embed = dummy_outputs[-9]
        prev_l2g_r_mat_out = dummy_outputs[-10]
        prev_l2g_t_out = dummy_outputs[-11]
        prev_timestamp_out = dummy_outputs[-12]
        prev_track_intances_out = {}
        prev_track_intances_out['prev_track_intances0_out'] = dummy_outputs[0]
        prev_track_intances_out['prev_track_intances1_out'] = dummy_outputs[1]
        prev_track_intances_out['prev_track_intances3_out'] = dummy_outputs[2]
        prev_track_intances_out['prev_track_intances4_out'] = dummy_outputs[3]
        prev_track_intances_out['prev_track_intances5_out'] = dummy_outputs[4]
        prev_track_intances_out['prev_track_intances6_out'] = dummy_outputs[5]
        prev_track_intances_out['prev_track_intances8_out'] = dummy_outputs[6]
        prev_track_intances_out['prev_track_intances9_out'] = dummy_outputs[7]
        prev_track_intances_out['prev_track_intances11_out'] = dummy_outputs[8]
        prev_track_intances_out['prev_track_intances12_out'] = dummy_outputs[9]
        prev_track_intances_out['prev_track_intances13_out'] = dummy_outputs[10]

        
        onnx_input_shapes = dict(
            prev_track_intances0=[-1, 512],  #901~1149
            prev_track_intances1=[-1, 3],
            prev_track_intances3=[-1],
            prev_track_intances4=[-1],
            prev_track_intances5=[-1],
            prev_track_intances6=[-1],
            prev_track_intances8=[-1],
            prev_track_intances9=[-1, 10],
            prev_track_intances11=[-1, 4, 256],
            prev_track_intances12=[-1, 4],
            prev_track_intances13=[-1],
            prev_timestamp=[1],
            prev_l2g_r_mat=[1, 3, 3],
            prev_l2g_t=[1, 3],
            prev_bev=[bevh**2, 1, 256],
            timestamp=[1],
            l2g_r_mat=[1, 3, 3], 
            l2g_t=[1, 3], 
            img=[1, 6, 3, img_h, img_w],
            img_metas_can_bus=[18],
            img_metas_lidar2img=[1, 6, 4, 4],
            command=[1],
            use_prev_bev=[1],
            max_obj_id=[1],
            g2l_r=[1, 3, 3],
        )
        
        if shape0 is None:
            # collect which shape0 contains the most number of samples
            shape_0 = str(onnx_inputs['prev_track_intances0'].shape[0])
            if shape_0 not in shape0_dict:
                shape0_dict[shape_0] = 1
            else:
                shape0_dict[shape_0] += 1
            # print the shape0_dict with the sorted by the max value
            print(" ")
            print(sorted(shape0_dict.items(), key=lambda item: item[1], reverse=True))
            
        else:
            # save calibration data
            num_calib_data = 0
            if shape0 == onnx_inputs['prev_track_intances0'].shape[0]:
                for key in onnx_input_shapes.keys():
                    if key not in onnx_inputs:
                        npz_data = {}
                        npz_data[key] = onnx_inputs[key].detach().cpu().numpy()
                    else:
                        npz_data[key] = np.concatenate([npz_data[key], onnx_inputs[key].detach().cpu().numpy()], axis=0)
                np.savez('/workspace/UniAD/calib_data_shape0_'+str(shape0)+'.npz', **npz_data)
                num_calib_data += 1
                print('num_calib_data: ', num_calib_data)

if __name__ == '__main__':
    main()
