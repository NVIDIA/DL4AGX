# SPDX-FileCopyrightText: Copyright (c) 2023-2024 OpenMMLab. All rights reserved.
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

# Modified from https://github.com/OpenDriveLab/UniAD/blob/main/tools/test.py
# Added support for launching ONNX exportation for UniAD-tiny model

import argparse
import torch
import numpy as np
import mmcv
import os
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (init_dist, load_checkpoint,
                         wrap_fp16_model)
from third_party.uniad_mmdet3d.models.builder import build_model
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from torch.onnx import OperatorExportTypes
from tqdm import tqdm
import random
import onnx
import onnx_graphsurgeon as gs

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
    if args.launcher != 'none':
        init_dist(args.launcher, **cfg.dist_params)

    # set random seeds
    # torch.use_deterministic_algorithms(True)
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

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

    # ########## ONNX Export Starts###############
    onnx_folder = "./onnx/"
    folder_dat = './dumped_inputs/'
    onnx_file_name = onnx_folder+"uniad_tiny_imgx0.25_cp.onnx"
    onnx_export_input = './nuscenes_np/uniad_onnx_input/'
    onnx_export_output = './nuscenes_np/uniad_pth_trtp_out/'
    if 'tiny' in onnx_file_name:
        bevh=50
        img_h = 480
        img_w = 800
        print('bevh=', bevh)
        if 'x0.25' in onnx_file_name:
            img_h = 256
            img_w = 416
    else:
        bevh=200
        img_h = 928
        img_w = 1600

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
    output_shapes = dict(
        prev_track_intances0_out=[-1, 512],
        prev_track_intances1_out=[-1, 3],
        # prev_track_intances2_out=[-1, 256], will not be used in ONNX graph
        prev_track_intances3_out=[-1],
        prev_track_intances4_out=[-1],
        prev_track_intances5_out=[-1],
        prev_track_intances6_out=[-1],
        # prev_track_intances7_out=[-1], will not be used in ONNX graph
        prev_track_intances8_out=[-1],
        prev_track_intances9_out=[-1, 10],
        # prev_track_intances10_out=[-1, 10], will not be used in ONNX graph
        prev_track_intances11_out=[-1, 4, 256],
        prev_track_intances12_out=[-1, 4],
        prev_track_intances13_out=[-1],
        prev_timestamp_out=[1],
        prev_l2g_t_out=[1, 3],
        prev_l2g_r_mat_out=[1, 3, 3],
        bev_embed=[bevh**2, 1, 256], 
        bboxes_dict_bboxes=[-1, 9],
        scores=[-1],
        labels=[-1],
        bbox_index=[-1],
        obj_idxes=[-1],
        max_obj_id_out=[1],
        outs_planning=[1,6,2],
    )

    dynamic_axes = {
    'prev_track_intances0':[0],
    'prev_track_intances1':[0],
    'prev_track_intances2':[0],
    'prev_track_intances3':[0],
    'prev_track_intances4':[0],
    'prev_track_intances5':[0],
    'prev_track_intances6':[0],
    'prev_track_intances7':[0],
    'prev_track_intances8':[0],
    'prev_track_intances9':[0],
    'prev_track_intances10':[0],
    'prev_track_intances11':[0],
    'prev_track_intances12':[0],
    'prev_track_intances13':[0],
    'gt_lane_labels':[1],
    'gt_lane_masks':[1],

    'prev_track_intances0_out':[0],
    'prev_track_intances1_out':[0],
    # 'prev_track_intances2_out':[0], will not be used in ONNX graph
    'prev_track_intances3_out':[0],
    'prev_track_intances4_out':[0],
    'prev_track_intances5_out':[0],
    'prev_track_intances6_out':[0],
    # 'prev_track_intances7_out':[0], will not be used in ONNX graph
    'prev_track_intances8_out':[0],
    'prev_track_intances9_out':[0],
    # 'prev_track_intances10_out':[0], will not be used in ONNX graph
    'prev_track_intances11_out':[0],
    'prev_track_intances12_out':[0],
    'prev_track_intances13_out':[0],
    'bboxes_dict_bboxes':[0],
    'scores':[0],
    'labels':[0],
    'bbox_index':[0],
    'obj_idxes':[0],
    }

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
    for iid in tqdm(range(6)):
        inputs = {}
        for key in input_shapes.keys():
            if os.path.isfile(onnx_export_input+key+'/'+str(iid)+'.npy'):
                inputs[key]= np.load(onnx_export_input+key+'/'+str(iid)+'.npy')
                inputs[key] = torch.from_numpy(inputs[key]).cuda()
                if key=='timestamp':
                    if iid==0:
                        import copy
                        timestamp0=copy.deepcopy(inputs[key])
                    inputs[key] = (inputs[key] - timestamp0).float()
            else:
                if key=='image_shape':
                    inputs[key] = np.array([img_h,img_w]).astype(np.float32)
                    inputs[key] = torch.from_numpy(inputs[key]).cuda()
                elif key=='prev_bev':
                    if iid==0:
                        inputs[key] = np.zeros([bevh**2, 1, 256]).astype(np.float32)
                    else:
                        inputs[key]= np.load(onnx_export_output+'bev_embed'+'/'+str(iid-1)+'.npy').astype(np.float32)
                    inputs[key] = torch.from_numpy(inputs[key]).cuda()
                elif key=='max_obj_id':
                    inputs[key] = torch.Tensor([0]).int().cuda() if iid==0 else max_obj_id
                elif 'prev_track_intances' in key:
                    if iid==0:
                        inputs[key] = test_track_instances[int(key[19:])].cuda()
                    else:
                        if int(key[19:]) in (2,7,10): # 2,7,10 will not be used in ONNX graph
                            # update shape0
                            shape = copy.deepcopy(input_shapes[key])
                            shape[0] = inputs['prev_track_intances0'].shape[0]
                            # put dummy inputs for pytorch
                            inputs[key] = torch.zeros(shape).float().cuda()
                        else:
                            inputs[key]= np.load(onnx_export_output+key+'_out/'+str(iid-1)+'.npy')
                            inputs[key] = torch.from_numpy(inputs[key]).cuda()
                            if inputs[key].dtype == torch.int64:
                                inputs[key] = inputs[key].int()
                elif key=='prev_l2g_r_mat':
                    inputs[key] = l2g_r_mat0.float().cuda() if iid==0 else prev_l2g_r_mat_out
                elif key=='prev_l2g_t':
                    inputs[key] = l2g_t0.float().cuda() if iid==0 else prev_l2g_t_out
                elif key=='prev_timestamp':
                    inputs[key] = torch.zeros([1]).float().cuda() if iid==0 else prev_timestamp_out
                elif key=='use_prev_bev':
                    cur_img_metas_scene_token = torch.from_numpy(np.load(onnx_export_input+'img_metas_scene_token'+'/'+str(iid)+'.npy')).cuda()
                    if not torch.equal(cur_img_metas_scene_token,img_metas_scene_token):
                        inputs[key] = np.array([0]).astype(np.int32)
                    else:
                        inputs[key] = np.array([1]).astype(np.int32)
                    inputs[key] = torch.from_numpy(inputs[key]).cuda()
                    img_metas_scene_token = cur_img_metas_scene_token
        inputs = tuple(inputs.values())
        with torch.no_grad():
            dummy_outputs = model.forward_uniad_trt(*inputs)
        max_obj_id = dummy_outputs[-2]
        prev_l2g_r_mat_out = dummy_outputs[-9]
        prev_l2g_t_out = dummy_outputs[-10]
        prev_timestamp_out = dummy_outputs[-11]
        output_name = list(output_shapes.keys())
        if not os.path.exists(onnx_export_output):
            os.mkdir(onnx_export_output)
        for i, out in enumerate(dummy_outputs):
            if not os.path.exists(onnx_export_output+output_name[i]):
                os.mkdir(onnx_export_output+output_name[i])
            np.save(onnx_export_output+output_name[i]+'/'+str(iid)+'.npy', out.detach().cpu().numpy())

        if iid==5:
            print('start deploying iid: ', iid)
            model.forward = model.forward_uniad_trt
            input_name = list(input_shapes.keys())
            output_name = list(output_shapes.keys())
            if not os.path.exists(onnx_folder):
                os.mkdir(onnx_folder)
            if not os.path.exists(folder_dat):
                os.mkdir(folder_dat)
            for i in range(len(inputs)):
                inputs[i].cpu().numpy().tofile(folder_dat+input_name[i]+'.dat')
            torch.onnx.export(model, 
                    inputs, 
                    onnx_file_name,
                    verbose=True, 
                    export_params=True, 
                    keep_initializers_as_inputs=True, 
                    do_constant_folding=False, 
                    input_names = input_name, 
                    output_names=output_name, 
                    opset_version=16,
                    operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
                    dynamic_axes=dynamic_axes)
            
            graph = gs.import_onnx(onnx.load(onnx_file_name))
            for node in graph.nodes:
                if node.op == "Reshape":
                    node.attrs["allowzero"] = 1
            onnx.save(gs.export_onnx(graph), onnx_file_name[:-4]+'repaired.onnx')
    # ########## ONNX Export Ends###############

if __name__ == '__main__':
    main()
