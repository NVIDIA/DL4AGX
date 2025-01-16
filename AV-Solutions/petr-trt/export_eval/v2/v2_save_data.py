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

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import mmcv
import os
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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
    
    # when saving data, we change to [5] so the 'previous' frame is exactly the frame in the previous round
    # default value will cause mismatch here
    cfg.data.test.pipeline[1]["sweeps_id"] = [5]

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

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
        model.eval()
        # outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)

        from bev_deploy.hook import HookHelper, Hook
        from bev_deploy.patch.inspect import AutoInspectHelper
        from bev_deploy.trt.inference import InferTrt

        ch = HookHelper()
        ch.attach_hook(model.module, "PETRv2")
        outputs = []

        class TrtExtractFeatHelper(torch.nn.Module):
            def __init__(self, mod) -> None:
                super().__init__()
                self.infer = InferTrt()
                self.infer.read("engines/PETRv2.extract_feat.onnx.fp16.engine")
                print(self.infer)
                self.mod = mod
            
            def get_func(self):
                me = self
                def trt_extract_feat(self, img, img_metas):
                    dev = img.device
                    a = img[:, 0: 6, ...].contiguous()
                    b = img[:, 6:12, ...].contiguous()
                    prev = me.mod.extract_img_feat(b, img_metas)
                    outs = [
                        torch.zeros(size=[1, 12, 256, 20, 50], dtype=torch.float32, device=dev),
                        torch.zeros(size=[1, 12, 256, 10, 25], dtype=torch.float32, device=dev)]
                    args = [a.data_ptr(), prev[0].data_ptr(), prev[1].data_ptr()] + \
                        [outs[0].data_ptr(), outs[1].data_ptr()]
                    me.infer.forward(args, stream=torch.cuda.current_stream())
                    return outs
                return trt_extract_feat
        
        import torch.nn.functional as F
        import numpy as np

        class TrtPtsBboxHeadHelper(torch.nn.Module):
            def __init__(self, mod) -> None:
                super().__init__()
                self.infer = InferTrt()
                self.infer.read("engines/PETRv2.pts_bbox_head.forward.onnx.fp16.engine")
                print(self.infer)
                self.mod = mod
                self.coords_position_embeding = None
                self.mean_time_stamp = None

            def get_func(self):
                me = self
                def trt_pts_bbox_head(self, mlvl_feats, img_metas):
                    dev = mlvl_feats[0].device
                    batch_size, num_cams = 1, 12
                    input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
                    masks = mlvl_feats[0].new_ones(
                        (batch_size, num_cams, input_img_h, input_img_w))
                    for img_id in range(batch_size):
                        for cam_id in range(num_cams):
                            img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                            masks[img_id, cam_id, :img_h, :img_w] = 0
                    
                    time_stamps = []
                    for img_meta in img_metas:    
                        time_stamps.append(np.asarray(img_meta['timestamp']))

                    time_stamp = mlvl_feats[0].new_tensor(np.array(time_stamps))
                    time_stamp = time_stamp.view(batch_size, -1, 6)
                    mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

                    # interpolate masks to have the same spatial shape with x
                    masks = F.interpolate(masks, size=mlvl_feats[0].shape[-2:]).to(torch.bool)
                    coords_position_embeding, _ = model.module.pts_bbox_head.position_embeding(mlvl_feats, img_metas, masks)
                    
                    if me.coords_position_embeding is None:
                        me.mean_time_stamp = mean_time_stamp
                        me.coords_position_embeding = coords_position_embeding

                    outs = {
                        "all_cls_scores": torch.zeros(size=[6, 1, 900, 10], dtype=torch.float32, device=dev),
                        "all_bbox_preds": torch.zeros(size=[6, 1, 900, 10], dtype=torch.float32, device=dev),
                        "enc_cls_scores": None,
                        "enc_bbox_preds": None
                    }

                    lut = {
                        "mlvl_feats.0": mlvl_feats[0].data_ptr(), 
                        "img_metas.0[coords_position_embeding]": coords_position_embeding.data_ptr(),
                        "img_metas.0[mean_time_stamp]": mean_time_stamp.data_ptr(),
                        "out.all_cls_scores": outs["all_cls_scores"].data_ptr(), 
                        "out.all_bbox_preds": outs["all_bbox_preds"].data_ptr()
                    }
                    args = [lut[n] for n in me.infer.names]
                    me.infer.forward(args, stream=torch.cuda.current_stream())
                    return outs
                return trt_pts_bbox_head

        from tqdm import tqdm
        with torch.no_grad():
            head = TrtPtsBboxHeadHelper(model.module)
            ch.hooks["PETRv2.extract_feat"]._patch(TrtExtractFeatHelper(model.module).get_func())
            ch.hooks["PETRv2.pts_bbox_head.forward"]._patch(head.get_func())
            
            for i, data in tqdm(enumerate(data_loader), total=len(dataset.data_infos)):
                if i >= 30:
                    head.coords_position_embeding.detach().cpu().numpy().tofile(f"demo/data/v2_coords_pe.bin")
                    head.mean_time_stamp.detach().cpu().numpy().tofile(f"demo/data/v2_mean_time_stamp.bin")
                    exit(0)

                result = model(return_loss=False, rescale=True, **data)
                outputs.extend(result)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    main()
