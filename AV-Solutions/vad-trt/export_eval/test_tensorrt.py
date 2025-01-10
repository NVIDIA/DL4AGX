# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Modified from https://github.com/hustvl/VAD/blob/main/tools/test.py
# Licensed by https://github.com/hustvl/VAD/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import sys
sys.path.append('')
import numpy as np
import argparse
import mmcv
import os
import copy
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
# from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.VAD.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
import json
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--json_dir', help='json parent dir name file') # NOTE: json file parent folder name
    parser.add_argument('--out', help='output result file in pickle format')
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
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
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
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
        model.eval()

        from pathlib import Path
        me = Path(__file__)
        
        import sys
        import ctypes
        ctypes.CDLL(me.parent / "../plugins/build/libplugins.so")

        import numpy as np

        from bev_deploy.hook import HookHelper, Hook
        from bev_deploy.trt.inference import InferTrt
        from bev_deploy.patch.bevformer.patch import patch_spatial_cross_attn_forward, patch_point_sampling, \
            patch_bevformer_encoder_forward
        from bev_deploy.patch.bevformer.patch import fn_lidar2img, fn_canbus
        
        ch = HookHelper()
        ch.attach_hook(model.module, "vadv1")
        
        for i in range(3):
            ch.hooks[f"vadv1.pts_bbox_head.transformer.encoder.layers.{i}.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
        ch.hooks["vadv1.pts_bbox_head.transformer.encoder.point_sampling"]._patch(patch_point_sampling)
        ch.hooks["vadv1.pts_bbox_head.transformer.encoder.forward"]._patch(patch_bevformer_encoder_forward)
        from patch.patch_head import patch_VADHead_select_and_pad_query, patch_VADPerceptionTransformer_get_bev_features, patch_VADHead_select_and_pad_pred_map
        
        ch.hooks["vadv1.pts_bbox_head.select_and_pad_query"]._patch(patch_VADHead_select_and_pad_query)
        ch.hooks["vadv1.pts_bbox_head.select_and_pad_pred_map"]._patch(patch_VADHead_select_and_pad_pred_map)
        ch.hooks["vadv1.pts_bbox_head.transformer.get_bev_features"]._patch(patch_VADPerceptionTransformer_get_bev_features)

        class TrtExtractImgFeatHelper(object):
            def __init__(self) -> None:
                self.infer = InferTrt()
                self.infer.read("scratch/vadv1.extract_img_feat/sim_vadv1.extract_img_feat_fp16.engine")
                print(self.infer)

            def get_func(self):
                me = self
                def trt_extract_img_feat(self, img, img_metas, len_queue=None):
                    torch.cuda.synchronize() # TODO: maybe we can send stream to trt?
                    dev = img.device
                    # vadv1.extract_img_feat
                    feat_shapes = [[1, 6, 256, 12, 20]]
                    # ["torch.Tensor(shape=[1, 6, 256, 12, 20], dtype=torch.float32)"]        
                    img_feats = [torch.zeros(size=s, dtype=torch.float32, device=dev) for s in feat_shapes]
                    args = [img.data_ptr()] + [f.data_ptr() for f in img_feats]
                    me.infer.forward(args)
                    torch.cuda.synchronize()
                    return img_feats
                return trt_extract_img_feat

        h1 = TrtExtractImgFeatHelper()  
        ch.hooks["vadv1.extract_img_feat"]._patch(h1.get_func())

        from patch.patch_rotate import rotate

        mod = sys.modules["patch.patch_head"]
        ch.hooks["vadv1.func.rotate"] = Hook(mod, "rotate", "vadv1.func")._patch(rotate)
        Hook.cache._capture["vadv1.func.rotate"] = []

        class TrtPtsBboxHeadHelper(object):
            def __init__(self) -> None:
                self.infer = InferTrt()
                self.infer.read("scratch/vadv1.pts_bbox_head.forward/sim_vadv1.pts_bbox_head.forward.engine")
                print(self.infer)
                self.infer_prev = InferTrt()
                self.infer_prev.read("scratch/vadv1_prev.pts_bbox_head.forward/sim_vadv1_prev.pts_bbox_head.forward.engine")
                print(self.infer_prev)
                self.func = None

            def get_func(self):
                me = self
                def trt_pts_bbox_head_fwd(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False, ego_his_trajs=None, ego_lcf_feat=None):
                    dev = mlvl_feats[0].device
                    m = fn_lidar2img(img_metas)
                    m = fn_canbus(mlvl_feats[0], m, 100, 100, [0.6, 0.3])
                    lidar2img = m[0]["lidar2img"].to(dev)
                    torch.cuda.synchronize()

                    lut = [
                        ("bev_embed", [10000, 1, 256]),
                        ("all_cls_scores", [3, 1, 300, 10]),
                        ("all_bbox_preds", [3, 1, 300, 10]),
                        ("all_traj_preds", [3, 1, 300, 6, 12]),
                        ("all_traj_cls_scores", [3, 1, 300, 6]),
                        ("map_all_cls_scores", [3, 1, 100, 3]),
                        ("map_all_bbox_preds", [3, 1, 100, 4]),
                        ("map_all_pts_preds", [3, 1, 100, 20, 2]),
                        ("enc_cls_scores", None),
                        ("enc_bbox_preds", None),
                        ("map_enc_cls_scores", None),
                        ("map_enc_bbox_preds", None),
                        ("map_enc_pts_preds", None),
                        ("ego_fut_preds", [1, 3, 6, 2])]
                    ret = dict()
                    # please beware of the order of input tensors
                    # make sure they match with the engine
                    if prev_bev is not None:
                        args = [
                            mlvl_feats[0].data_ptr(),
                            m[0]["can_bus"].data_ptr(),
                            m[0]["shift"].data_ptr(),
                            lidar2img.data_ptr(),]
                        args.append(prev_bev.data_ptr())
                    else:
                        args = [
                            mlvl_feats[0].data_ptr(),
                            m[0]["shift"].data_ptr(),
                            lidar2img.data_ptr(),
                            m[0]["can_bus"].data_ptr(),]

                    for k, v in lut:
                        if v is None:
                            ret[k] = None
                        else:
                            ret[k] = torch.zeros(size=v, dtype=torch.float32, device=dev)
                            args.append(ret[k].data_ptr())

                    if prev_bev is not None:
                        me.infer_prev.forward(args)
                    else:
                        me.infer.forward(args)

                    torch.cuda.synchronize()
                    return ret
                return trt_pts_bbox_head_fwd
            
        h2 = TrtPtsBboxHeadHelper()
        h2.func = ch.hooks["vadv1.pts_bbox_head.forward"].func

        ch.hooks["vadv1.pts_bbox_head.forward"]._patch(h2.get_func())

        outputs = []
        for i, data in enumerate(data_loader):
            with torch.no_grad():           
                result = model(return_loss=False, rescale=True, **data)
                outputs.extend(result)
    else:           
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                        args.gpu_collect)

    tmp = {}
    tmp['bbox_results'] = outputs
    outputs = tmp
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            # assert False
            if isinstance(outputs, list):
                mmcv.dump(outputs, args.out)
            else:
                mmcv.dump(outputs['bbox_results'], args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
            '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
        if args.format_only:
            dataset.format_results(outputs['bbox_results'], **kwargs)

        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))

            print(dataset.evaluate(outputs['bbox_results'], **eval_kwargs))

if __name__ == '__main__':
    main()
