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
        # assert False
        from bev_deploy.hook import HookHelper, Hook
        from bev_deploy.patch.inspect import AutoInspectHelper
        from bev_deploy.patch.bevformer.patch import patch_spatial_cross_attn_forward, patch_point_sampling, \
            patch_bevformer_encoder_forward
        from bev_deploy.patch.bevformer.patch import fn_lidar2img, fn_canbus

        ch = HookHelper()
        ch.attach_hook(model, "vadv1")

        model = MMDataParallel(model, device_ids=[0])
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)        

        with torch.no_grad():
            model = model.eval().cuda().float()
            for i, data in enumerate(data_loader):
                if i == 5:
                    Hook.cache._capture["vadv1.extract_img_feat"] = []
                    Hook.cache._capture["vadv1.pts_bbox_head.forward"] = []
                    
                    with Hook.capture() as _:
                        result = model(return_loss=False, rescale=True, **data)

                    def pth_mapping(src):
                        return src.replace("/ws_oss", "/home/yuchaoj/ws_oss")
                    
                    for r in Hook.history:
                        meta = r.meta
                        pth, lineno, fname, key, sig, body = meta
                        # stack[], f.filename, f.lineno, f.function, context
                        print(f"{pth_mapping(pth)}:{lineno}, {fname}\n{sig}\n{key}")
                        print(r.meta_args)
                        print(r.meta_kwargs)
                        print(r.meta_ret)
                        for s in r.stack:
                            pname, lineno, fn, ctx = s
                            if pname == "/ws_oss/VAD/projects/mmdet3d_plugin/VAD/VAD.py" and \
                               fn == "forward_test":
                                break
                            print(f"{pth_mapping(pname)}:{lineno}, {fn}, {ctx}, {body}")
                        print("-")
                    print("\n".join(
                        [k + ", " + v for k, v in ch.seen_classes(pth_mapping)]
                    ))
                    Hook.call_stack.view(pth_mapping)
                    print("*" * 80)

                    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.layers.0.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
                    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.layers.1.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
                    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.layers.2.attentions.1.forward"]._patch(patch_spatial_cross_attn_forward)
                    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.point_sampling"]._patch(patch_point_sampling)
                    ch.hooks["vadv1.pts_bbox_head.transformer.encoder.forward"]._patch(patch_bevformer_encoder_forward)
                     
                    def fn(args, kwargs):
                        m = kwargs["img_metas"]
                        for i in range(len(m)):
                            m[i]["lidar2img"] = torch.from_numpy(np.asarray(m[i]["lidar2img"])).to(torch.float32)
                        kwargs["img_metas"] = fn_canbus(args[1], m, 100, 100, [0.6, 0.3])
                        return args, kwargs

                    from patch.patch_head import patch_VADHead_select_and_pad_query, patch_VADPerceptionTransformer_get_bev_features, \
                        patch_VADHead_select_and_pad_pred_map, patch_VADHead_forward
                    
                    ch.hooks["vadv1.pts_bbox_head.select_and_pad_query"]._patch(patch_VADHead_select_and_pad_query)
                    ch.hooks["vadv1.pts_bbox_head.transformer.get_bev_features"]._patch(patch_VADPerceptionTransformer_get_bev_features)
                    ch.hooks["vadv1.pts_bbox_head.select_and_pad_pred_map"]._patch(patch_VADHead_select_and_pad_pred_map)

                    def fn_fwd(args, kwargs):
                        m = args[1] # kwargs["img_metas"]
                        m = fn_lidar2img(m)
                        m = fn_canbus(args[0][0], m, 100, 100, [0.6, 0.3])
                        return (args[0], m), kwargs
                    
                    ch.hooks["vadv1.pts_bbox_head.forward"]._patch(patch_VADHead_forward)

                    ah = AutoInspectHelper(ch.hooks["vadv1.pts_bbox_head.forward"], [fn_fwd]); ah.export()
                    ah = AutoInspectHelper(ch.hooks["vadv1.extract_img_feat"], []); ah.export()
                    exit(0)
                else:
                    pass
                    # _ = model(return_loss=False, rescale=True, **data)
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
    
        # # # NOTE: record to json
        # json_path = args.json_dir
        # if not os.path.exists(json_path):
        #     os.makedirs(json_path)
        
        # metric_all = []
        # for res in outputs['bbox_results']:
        #     for k in res['metric_results'].keys():
        #         if type(res['metric_results'][k]) is np.ndarray:
        #             res['metric_results'][k] = res['metric_results'][k].tolist()
        #     metric_all.append(res['metric_results'])
        
        # print('start saving to json done')
        # with open(json_path+'/metric_record.json', "w", encoding="utf-8") as f2:
        #     json.dump(metric_all, f2, indent=4)
        # print('save to json done')

if __name__ == '__main__':
    main()
