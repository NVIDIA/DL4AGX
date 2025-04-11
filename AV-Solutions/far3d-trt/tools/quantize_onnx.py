# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import argparse
import torch

import onnx
import onnx_graphsurgeon as gs
import torch

from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from onnxruntime.quantization.calibrate import CalibrationDataReader


import onnx

import modelopt.onnx.quantization.int8
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxconverter_common.float16 import convert_float_to_float16


def parse_args():
    parser = argparse.ArgumentParser(description='Perform PTQ on the image encoder')
    parser.add_argument('config',help='test config file path')
    parser.add_argument('onnx_path', help='path to far3d.encoder.onnx', default='far3d.encoder.onnx')
    parser.add_argument('--exclude-key', type=str, help='exclude key')
    args = parser.parse_args()

    return args


def create_node_IO_mapping(graph):
    input_map = {}
    output_map = {}
    for node in graph.nodes:
        for input_variable in node.inputs:
            if input_variable.name not in input_map:
                input_map[input_variable.name] = []
            input_map[input_variable.name].append(node)
        for output_variable in node.outputs:
            output_map[output_variable.name] = node
    return input_map, output_map


class CalibrationReader(CalibrationDataReader):
    def __init__(self, data_loader, num_samples):
        self.iter = iter(data_loader)
        self.count = 0
        self.num_samples = num_samples

    def get_next(self) -> dict:
        # skip 20 frames
        for _ in range(20):
            data = next(self.iter)
        
        self.count += 1
        if self.count > self.num_samples:
            return None
        return dict(img=data['img'][0].data[0].cpu().permute(0,1,3,4,2).numpy())


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

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
            
    onnx_path = args.onnx_path
    quant_pre_process(onnx_path, onnx_path)
    graph = gs.import_onnx(onnx.load(onnx_path))
    input_map, _ = create_node_IO_mapping(graph)
    
    exclude_keys = ['OSA4_5']
    
    nodes = []
    def recursiveGlobChildren(graph, node):
        for output_tensor in node.outputs:
            if output_tensor.name in input_map:
                input_nodes = input_map[output_tensor.name]
                for input_node in input_nodes:
                    nodes.append(input_node.name)
                    recursiveGlobChildren(graph, input_node)
    
    for node in graph.nodes:
        if 'lateral_convs' in node.name:
            recursiveGlobChildren(graph, node)
    
    for node in graph.nodes:
        for exclude_key in exclude_keys:
            if exclude_key in node.name:
                nodes.append(node.name)
            
    data_reader = CalibrationReader(data_loader, num_samples=500)
    quantized_onnx_path = onnx_path.replace('.onnx', '.int8.onnx')
    quantized_model = modelopt.onnx.quantization.int8.quantize(onnx_path, output_path = quantized_onnx_path, 
                                                                calibration_data_reader=data_reader, 
                                                                nodes_to_exclude=nodes)
    quantized_model = convert_float_to_float16(quantized_model)    
    onnx.save_model(quantized_model, quantized_onnx_path)
    

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
