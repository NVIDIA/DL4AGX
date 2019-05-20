######################################################################################################
# Copyright (c) 2018-2019 NVIDIA Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: DL4AGX/MultiDeviceInferencePipeline/training/objectDetection/ssdConvertUFF/utils/paths.py
# Description: Model download and UFF conversion utils
#####################################################################################################
import os
import sys
import tarfile

import requests
import tensorflow as tf
import tensorrt as trt
import graphsurgeon as gs
import uff

# from utils.paths import PATHS
from utils.paths import PATHS


# UFF conversion functionality
def build_nms_node(
        name="NMS",
        op="NMS_TRT",
        backgroundLabelId=0,
        confSigmoid=True,
        confidenceThreshold=1e-8,
        isNormalized=True,
        topK=100,
        keepTopK=100,
        nmsThreshold=0.6,
        numClasses=91,  # +1 for background
        scoreConverter='SIGMOID',
        shareLocation=True,
        varianceEncodedInTarget=False,
        **kw_args):
    return gs.create_plugin_node(name,
                                 op=op,
                                 backgroundLabelId=backgroundLabelId,
                                 confSigmoid=confSigmoid,
                                 confidenceThreshold=confidenceThreshold,
                                 isNormalized=isNormalized,
                                 topK=topK,
                                 keepTopK=keepTopK,
                                 nmsThreshold=nmsThreshold,
                                 numClasses=numClasses,
                                 scoreConverter=scoreConverter,
                                 shareLocation=shareLocation,
                                 varianceEncodedInTarget=varianceEncodedInTarget,
                                 **kw_args)


def build_grid_anchor_node(name="GridAnchor",
                           op='GridAnchor_TRT',
                           aspectRatios=[1.0, 2.0, 0.5, 3.0, 0.33],
                           featureMapShapes=[19, 10, 5, 3, 2, 1],
                           maxSize=0.95,
                           minSize=0.2,
                           numLayers=6,
                           variance=[0.1, 0.1, 0.2, 0.2],
                           **kw_args):
    return gs.create_plugin_node(name,
                                 op=op,
                                 aspectRatios=aspectRatios,
                                 featureMapShapes=featureMapShapes,
                                 maxSize=maxSize,
                                 minSize=minSize,
                                 numLayers=numLayers,
                                 variance=variance,
                                 **kw_args)


# This class contains converted (UFF) model metadata
class ModelData(object):
    # Name of input node
    INPUT_NAME = "Input"
    # Name of output node
    OUTPUT_NAME = "NMS"


def ssd_unsupported_nodes_to_plugin_nodes(ssd_graph, n_classes, input_dims, feature_dims):
    """Makes ssd_graph TensorRT comparible using graphsurgeon.

    This function takes ssd_graph, which contains graphsurgeon
    DynamicGraph data structure. This structure describes frozen Tensorflow
    graph, that can be modified using graphsurgeon (by deleting, adding,
    replacing certain nodes). The graph is modified by removing
    Tensorflow operations that are not supported by TensorRT's UffParser
    and replacing them with custom layer plugin nodes.

    Note: This specific implementation works only for
    ssd_inception_v2_coco_2017_11_17 network.

    Args:
        ssd_graph (gs.DynamicGraph): graph to convert
    Returns:
        gs.DynamicGraph: UffParser compatible SSD graph
    """

    # Remove assert nodes
    all_assert_nodes = ssd_graph.find_nodes_by_op("Assert")
    # Remove those nodes from the graph.
    ssd_graph.remove(all_assert_nodes, remove_exclusive_dependencies=True)
    # Find all identity nodes.
    all_identity_nodes = ssd_graph.find_nodes_by_op("Identity")
    # Forward inputs those in the graph i.e. forward their inputs.
    ssd_graph.forward_inputs(all_identity_nodes)

    # Create TRT plugin nodes to replace unsupported ops in Tensorflow graph
    channels = input_dims[0]
    height = input_dims[1]
    width = input_dims[2]

    nodes = ssd_graph.node_map
    node_names = ssd_graph.node_map.keys()

    break_now = False
    class_predictor_label = 'concat_1'
    box_loc_label = 'concat'
    for inode in nodes['concat'].input:
        if break_now:
            break
        for jnode in nodes[inode].input:
            if break_now:
                break
            if 'ClassPredictor' in jnode:
                class_predictor_label = 'concat'
                box_loc_label = 'concat_1'
                break_now = True

    concat_namespace = "Concatenate"
    include_anchors = False
    for k in node_names:
        if "MultipleGridAnchorGenerator" in k:
            include_anchors = True
        if "MultipleGridAnchorGenerator/Concatenate" in k:
            concat_namespace = "MultipleGridAnchorGenerator/Concatenate"

    # Now we need to collapse a few namespaces.
    if include_anchors:
        Concat = gs.create_node("concat_priorbox", op="ConcatV2", dtype=tf.float32, axis=2)

    Input = gs.create_plugin_node(ModelData.INPUT_NAME,
                                  op="Placeholder",
                                  dtype=tf.float32,
                                  shape=[1, channels, height, width])
    FlattenConcat_box_conf = gs.create_plugin_node(
        "concat_box_conf",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
    )
    FlattenConcat_box_loc = gs.create_plugin_node(
        "concat_box_loc",
        op="FlattenConcat_TRT",
        dtype=tf.float32,
    )
    NMS = build_nms_node(numClasses=n_classes)

    # Create a mapping of namespace names -> plugin nodes.
    namespace_plugin_map = {
        "Postprocessor": NMS,
        "Preprocessor": Input,
        "ToFloat": Input,
        "image_tensor": Input,
        box_loc_label: FlattenConcat_box_loc,
        class_predictor_label: FlattenConcat_box_conf
    }

    # # Now create a new graph by collapsing namespaces
    # ssd_graph.collapse_namespaces(namespace_plugin_map)

    # # Determine the parameter for the GridAnchors
    # # print(ssd_graph.as_graph_def().node)
    # # print(tf.import_graph_def(ssd_graph.as_graph_def()))
    # for node in ssd_graph.as_graph_def().node:
    #     if node.name == "BoxPredictor_2/BoxEncodingPredictor/BiasAdd":
    #         print(type(node))
    #         print(node)
    # # print([dict(nodes[x].attr) for x in ssd_graph.node_map.keys() if 'BoxEncodingPredictor/BiasAdd' in x])
    # # print(ssd_graph.find_nodes_by_name("BoxPredictor_2/BoxEncodingPredictor/BiasAdd"))
    # exit()

    if include_anchors:
        GridAnchor = build_grid_anchor_node(featureMapShapes=feature_dims)
        namespace_plugin_map[concat_namespace] = Concat
        namespace_plugin_map["MultipleGridAnchorGenerator"] = GridAnchor

    # Now create a new graph by collapsing namespaces
    ssd_graph.collapse_namespaces(namespace_plugin_map)
    # Remove the outputs, so we just have a single output node (NMS).
    # If remove_exclusive_dependencies is True, the whole graph will be removed!
    ssd_graph.remove(ssd_graph.graph_outputs, remove_exclusive_dependencies=False)

    # add in grid anchors for SSDLite
    if not include_anchors:
        Const = gs.create_node("Const", op="Const", dtype=tf.float32, value=[128, 128])
        GridAnchor = build_grid_anchor_node(inputs=[Const], featureMapShapes=feature_dims)
        Concat = gs.create_node("concat_priorbox", inputs=[GridAnchor], op="ConcatV2", dtype=tf.float32, axis=2)
        ssd_graph.append(Const)
        ssd_graph.append(GridAnchor)
        ssd_graph.append(Concat)
        NMS = build_nms_node(inputs=list(NMS.input) + ["concat_priorbox"], numClasses=n_classes)
        namespace_plugin_map = {"NMS": NMS}
        ssd_graph.collapse_namespaces(namespace_plugin_map)

    # For exported graphs, we need to remove the squeeze node between concat plugin and the NMS plugin.
    # Downloaded graphs don't need this step. Refer to convert_ssd_v1.py
    all_squeeze_nodes = ssd_graph.find_nodes_by_name("Squeeze")
    # Forward inputs those in the graph i.e. forward their inputs.
    ssd_graph.forward_inputs(all_squeeze_nodes)

    # clean up NMS and Input nodes
    actualInputOrder = []
    for node in ssd_graph._internal_graphdef.node:
        if node.name == "NMS":
            if ModelData.INPUT_NAME in node.input:
                node.input.remove(ModelData.INPUT_NAME)
            for input_name in node.input:
                if "loc" in input_name:
                    actualInputOrder.append(0)
                elif "conf" in input_name:
                    actualInputOrder.append(1)
                elif "priorbox" in input_name:
                    actualInputOrder.append(2)
        elif node.name == ModelData.INPUT_NAME:
            if "image_tensor:0" in node.input:
                node.input.remove("image_tensor:0")

    # NOTE: since the actual order of the NMS nodes inputs differ between versions, I'll reinsert the NMS trt op
    NMS = build_nms_node(inputOrder=actualInputOrder, numClasses=n_classes)
    namespace_plugin_map = {"NMS": NMS}
    ssd_graph.collapse_namespaces(namespace_plugin_map)
    return ssd_graph


def model_to_uff(model_path, output_uff_path, n_classes, input_dims, feature_dims, silent=False):
    """Takes frozen .pb graph, converts it to .uff and saves it to file.

    Args:
        model_path (str): .pb model path
        output_uff_path (str): .uff path where the UFF file will be saved
        silent (bool): if True, writes progress messages to stdout

    """
    dynamic_graph = gs.DynamicGraph(model_path)
    dynamic_graph = ssd_unsupported_nodes_to_plugin_nodes(dynamic_graph, n_classes, input_dims, feature_dims)

    dynamic_graph.write_tensorboard(os.path.join(os.path.dirname(output_uff_path), 'trt_tensorboard'))

    uff.from_tensorflow(dynamic_graph.as_graph_def(), [ModelData.OUTPUT_NAME],
                        output_filename=output_uff_path,
                        text=True)


# Model download functionality


def maybe_print(should_print, print_arg):
    """Prints message if supplied boolean flag is true.

    Args:
        should_print (bool): if True, will print print_arg to stdout
        print_arg (str): message to print to stdout
    """
    if should_print:
        print(print_arg)


def maybe_mkdir(dir_path):
    """Makes directory if it doesn't exist.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
