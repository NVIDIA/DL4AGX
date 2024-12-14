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
import argparse
import os
import torch
import warnings
from tqdm import tqdm

import torch

from mmcv import Config, DictAction
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes

from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from mmdet.apis import set_random_seed

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config',help='test config file path')
    parser.add_argument("encoder_engine")
    parser.add_argument("decoder_engine")
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    
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
    
    args = parser.parse_args()

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


import tensorrt as trt
TRT_VERSION_MAJOR, TRT_VERSION_MINOR = trt.__version__.split('.')[0:2]
TRT_VERSION_MAJOR = int(TRT_VERSION_MAJOR)
TRT_VERSION_MINOR = int(TRT_VERSION_MINOR)

trt_dtype_to_torch = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT8: torch.int8,
    trt.DataType.INT32: torch.int32,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.UINT8: torch.uint8,
}
dtype_size = {
    trt.DataType.FLOAT: 4,
    trt.DataType.HALF: 2,
    trt.DataType.INT8: 1,
    trt.DataType.INT32: 4,
    trt.DataType.BOOL: 1,
    trt.DataType.UINT8: 1,
    
    torch.float32: 4,
    torch.int64: 8,
    torch.half: 2,
    torch.int32: 4
}

if TRT_VERSION_MAJOR >= 10:
    trt_dtype_to_torch[trt.DataType.INT64] = torch.int64
    dtype_size[trt.DataType.INT64] =  8



def align_address(data, alignment, device):
    tensor_address = data.data_ptr()
    # unfortunately just cloning is not sufficient since torch can still give you a missaligned tensor
    if(tensor_address % alignment != 0):
        total_size = data.shape.numel()
        dtype_size_bytes = dtype_size[data.dtype]
        total_size += (alignment // dtype_size_bytes)
        buffer = torch.zeros(total_size, dtype=data.dtype, device=device)
        buffer_addr = buffer.data_ptr()
        offset = buffer_addr % alignment
        offset_elements = offset // dtype_size_bytes
        total_size = data.shape.numel()
        buffer[offset_elements:offset_elements+total_size] = data.flatten()
        tensor_address = buffer_addr + offset
        return buffer[offset_elements:offset_elements+total_size].reshape(data.shape), tensor_address
    return data, tensor_address

trt_logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(trt_logger, "")
runtime = trt.Runtime(trt_logger)

class TRTInferenceBase:
    def __init__(self, engine_file, internal_state = [], list_output=False):
        self.runtime = runtime
        if os.path.isfile(engine_file):
            with open(engine_file, "rb") as f:
                engine_file = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_file)
        assert self.engine is not None, f"Failed to load {engine_file}"
        num_io_tensors = self.engine.num_io_tensors
        tensor_names = [self.engine.get_tensor_name(i) for i in range(num_io_tensors)]
        self.input_shapes = {}
        self.output_shapes = {}
        self.tensor_dtype = {}
        for name in tensor_names:
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            assert dtype in trt_dtype_to_torch
            self.tensor_dtype[name] = trt_dtype_to_torch[dtype]
            
            if mode == trt.TensorIOMode.INPUT:
                self.input_shapes[name] = shape
            else:
                self.output_shapes[name] = shape
        self.tensor_names = tensor_names
            
        assert self.engine is not None
        self.context = self.engine.create_execution_context()
        assert self.context is not None
        self.context.debug_sync = True
        self.internal_state = {}
        for key in internal_state:
            if key in self.input_shapes or f"{key}.1" in self.input_shapes:
                key = f"{key}.1" if f"{key}.1" in self.input_shapes else key
                shape = self.input_shapes[key]
                tensor = torch.zeros((*shape), dtype=self.tensor_dtype[key]).cuda()
                self.internal_state[key] = tensor
                self.context.set_tensor_address(key, tensor.data_ptr())
        self.list_output = list_output
        

    def reset_internal_state(self):
        for key in self.internal_state.keys():
            self.internal_state[key].fill_(0.0)
            
    def __call__(self, stream: torch.cuda.Stream, device="cuda:0", **kwargs):
        alignment = 256
        for key, value in self.input_shapes.items():
            name = key
            if '.1' in key: # for some reason onnx is appending a .1 to my inputs
                key = key.split('.')[0]
            if key in kwargs:
                if len(value) > 0:
                    if len(value) == (len(kwargs[key].shape) + 1):
                        assert kwargs[key].shape == value[1:]
                    elif len(kwargs[key].shape) == (len(value) + 1):
                        assert value == kwargs[key].shape[1:]
                    else:
                        assert value == kwargs[key].shape
                if kwargs[key].dtype != self.tensor_dtype[key]:
                    kwargs[key] = kwargs[key].to(self.tensor_dtype[key])
                kwargs[key] = kwargs[key].contiguous()
                kwargs[key], tensor_address = align_address(kwargs[key], alignment, device)
                    
                assert tensor_address % alignment == 0, "Input address ({}) of {} not aligned to {} byte boundary dtype:{} alignment error: {} shape:{}".format(address, key, alignment, dtype, address % 256, kwargs[key].shape)
                self.context.set_tensor_address(name, tensor_address)
                if device is None:
                    device = kwargs[key].device
        if self.list_output:
            outputs = [torch.zeros(*shape, device=device, dtype=self.tensor_dtype[key]) for key, shape in self.output_shapes.items()]
        else:
            outputs = { key: torch.zeros(*shape, device=device, dtype=self.tensor_dtype[key]) for key, shape in self.output_shapes.items()}
        for key, value in outputs.items():
            outputs[key], tensor_address = align_address(value, alignment, device)
            assert tensor_address % alignment == 0, "Output tensor address ({}) of {} not aligned to {} byte boundary shape:{}".format(value.data_ptr(), key, alignment, value.shape)
            self.context.set_tensor_address(key, tensor_address)
        
        for key in self.tensor_names:
            address = self.context.get_tensor_address(key)
            dtype = self.engine.get_tensor_dtype(key)
            assert address != 0, f"Didn't set tensor address for {key}"
            
        stream.synchronize()        
        success = self.context.execute_async_v3(stream.cuda_stream)
        assert success
        stream.synchronize()

        return outputs


class Far3DTRT(TRTInferenceBase):

    def __init__(self, engine_file, internal_state=["memory_embedding", "memory_reference_point", "memory_egopose", "memory_velo", "memory_timestamp"]):
        super().__init__(engine_file, internal_state=internal_state)
        self.sequence_id = None
        self.timestamp_offset = None
        if 'memory_embedding' in self.input_shapes:
            self.state_len = self.input_shapes['memory_embedding'][1]
        else:
            # If we're doing a partial conversion, we may not have a memory embedding
            self.state_len = 0

    def reset_internal_state(self):
        for key in self.internal_state.keys():
            self.internal_state[key].fill_(0.0)
            
    @staticmethod
    def unpack(data, device=True):
        # data.keys() -> 'img_metas', 'img', 'lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv'
        # img_metas.keys() -> ['filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'scene_token']
        assert data['img_metas'][0].data[0][0]['box_type_3d'] == LiDARInstance3DBoxes
        def to(x):
            if device:
                return x.cuda()
            return x
        lidar2img=to(data['lidar2img'][0].data[0][0].unsqueeze(0))
        img2lidar = lidar2img.inverse()
        unpacked = dict(img=to(data['img'][0].data[0]).flip(2), # this converts bgr to rgb
            intrinsics=to(data['intrinsics'][0].data[0][0].unsqueeze(0)),
            extrinsics=to(data['extrinsics'][0].data[0][0].unsqueeze(0)),
            lidar2img=lidar2img,    
            img2lidar=img2lidar,
            ego_pose=to(data['ego_pose'][0].data[0][0].unsqueeze(0)),
            ego_pose_inv=to(data['ego_pose_inv'][0].data[0][0].unsqueeze(0)),
            pad_shape=to(torch.tensor(data['img_metas'][0].data[0][0]['pad_shape'][0])),
            timestamp=to(torch.tensor(data['timestamp'][0].data[0][0])),
        )
        
        return unpacked

    def __call__(self, stream: torch.cuda.Stream, data):
        kwargs = self.unpack(data)
        
        if self.sequence_id != data['img_metas'][0].data[0][0]['scene_token']:
            self.reset_internal_state()
            self.sequence_id = data['img_metas'][0].data[0][0]['scene_token']
            self.timestamp_offset = kwargs['timestamp']
            kwargs['timestamp'] -= self.timestamp_offset
            kwargs['timestamp'] = kwargs['timestamp'].float()
            
            if 'prev_exists' in self.tensor_dtype:
                kwargs['prev_exists'] = torch.zeros(1, device=kwargs['img'].device, dtype=self.tensor_dtype['prev_exists'])
        else:
            if 'prev_exists' in self.tensor_dtype:
                kwargs['prev_exists'] = torch.ones(1, device=kwargs['img'].device, dtype=self.tensor_dtype['prev_exists'])
        
        outputs = super().__call__(stream, **kwargs)
        if "embedding_out" in outputs:
            self.internal_state["memory_embedding"].copy_(outputs['embedding_out'][:,:self.state_len,:])
            self.internal_state['memory_reference_point'].copy_(outputs['reference_point_out'][:,:self.state_len,:])
            self.internal_state['memory_timestamp'].copy_(outputs['memory_timestamp_out'][:,:self.state_len,:])
            self.internal_state['memory_egopose'].copy_(outputs['egopose_out'][:,:self.state_len,:])
            self.internal_state['memory_velo'].copy_(outputs['velocity_out'][:,:self.state_len,:])
        
        return outputs

class Far3DDecoderTRT(TRTInferenceBase):

    def __init__(self, engine_file, internal_state=["memory_embedding", "memory_reference_point", "memory_egopose", "memory_velo", "memory_timestamp"]):
        super().__init__(engine_file, internal_state=internal_state)
        self.sequence_id = None
        self.timestamp_offset = None
        if 'memory_embedding' in self.input_shapes:
            self.state_len = self.input_shapes['memory_embedding'][1]
        else:
            # If we're doing a partial conversion, we may not have a memory embedding
            self.state_len = 0

    def reset_internal_state(self):
        for key in self.internal_state.keys():
            self.internal_state[key].fill_(0.0)
            

    def __call__(self, stream: torch.cuda.Stream, img_metas, timestamp, **kwargs):
        
        if self.sequence_id != img_metas[0].data[0][0]['scene_token']:
            self.reset_internal_state()
            self.sequence_id = img_metas[0].data[0][0]['scene_token']
            self.timestamp_offset = timestamp
        timestamp -= self.timestamp_offset
        timestamp = timestamp.float()
            
        """if 'prev_exists' in self.tensor_dtype:
                prev_exists = torch.zeros(1, device=kwargs['img_feats_0'].device, dtype=self.tensor_dtype['prev_exists'])
        else:
            if 'prev_exists' in self.tensor_dtype:
                prev_exists = torch.ones(1, device=kwargs['img'].device, dtype=self.tensor_dtype['prev_exists'])
        """
        
        outputs = super().__call__(stream, timestamp=timestamp, **kwargs)
        if "memory_embedding_out" in outputs:
            self.internal_state["memory_embedding"].copy_(outputs['memory_embedding_out'][:,:self.state_len,:])
            self.internal_state['memory_reference_point'].copy_(outputs['memory_reference_point_out'][:,:self.state_len,:])
            self.internal_state['memory_timestamp'].copy_(outputs['memory_timestamp_out'][:,:self.state_len,:])
            self.internal_state['memory_egopose'].copy_(outputs['memory_egopose_out'][:,:self.state_len,:])
            self.internal_state['memory_velo'].copy_(outputs['memory_velo_out'][:,:self.state_len,:])
        
        return outputs


class Far3DSplitNetTRTInference(object):
    def __init__(self, image_encoder_engine_path, transformer_decoder_engine_path):
        self.encoder = TRTInferenceBase(image_encoder_engine_path)
        self.decoder = Far3DDecoderTRT(transformer_decoder_engine_path)

    def __call__(self, stream: torch.cuda.Stream, data):
        kwargs = Far3DTRT.unpack(data)
        kwargs['img'] = kwargs['img'].permute(0,1,3,4,2).contiguous()
        image_features = self.encoder(stream, **kwargs)
        data.update(kwargs)
        data.update(image_features)
        output = self.decoder(stream, **data)
        return output

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
    import importlib

    plugin_dir = cfg.plugin_dir
    _module_dir = os.path.dirname(plugin_dir)
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
    cfg.data.test.test_mode = True

    # set random seeds
    set_random_seed(0, deterministic=False)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    # build the model and load checkpoint
    trt_model = Far3DSplitNetTRTInference(args.encoder_engine, args.decoder_engine)

    trt_outputs = []
    
    stream = torch.cuda.Stream()
    scene_token = None
    
    for i, data in tqdm(enumerate(data_loader)): 
        data["prev_exists"] = torch.cuda.FloatTensor([1]).unsqueeze(0)
        if scene_token != data['img_metas'][0].data[0][0]['scene_token']:
            scene_token = data['img_metas'][0].data[0][0]['scene_token']
            data["prev_exists"] = torch.cuda.FloatTensor([0]).unsqueeze(0)
        trt_results = trt_model(stream=stream, data=data)
        for key in trt_results.keys():
            trt_results[key] = trt_results[key].cpu()
        
        trt_results = dict(pts_bbox=dict(boxes_3d=trt_results['bboxes'], scores_3d=trt_results['scores'], labels_3d=trt_results['labels']))
        trt_results['pts_bbox']['boxes_3d'] = LiDARInstance3DBoxes(trt_results['pts_bbox']['boxes_3d'])
        trt_outputs.append(trt_results)
        if i == 500:
            break
        
    eval_kwargs = cfg.get('evaluation', {}).copy()    
    remove_keys = ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best','rule']
    for key in remove_keys:
        eval_kwargs.pop(key, None)
    print('Tensorrt performance')
    print(dataset.evaluate(trt_outputs, **eval_kwargs))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
