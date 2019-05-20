#############################################################################
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
# File: DL4AGX/MultiDeviceInferencePipeline/enginecreator/utils/coco_eval.py
# Description: Evaluate a inference on COCO
############################################################################
import os
import sys
sys.path.append(os.path.join(os.environ['HOME'], 'cocoapi/PythonAPI'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse


def parse_command_line():
    """
    Parse command-line.

    Returns:
        Namespace with members for all parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Run evaluation on COCO dataset.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-a',
                        '--annotation_file',
                        required=True,
                        type=str,
                        default='instances_val2017.json',
                        help='Full path to the annotation.json file.')
    parser.add_argument('-r',
                        '--result_file',
                        required=True,
                        type=str,
                        default='COCO_val2017_float_eval_MV2-tmp2.json',
                        help='Full path to result.json file.')
    return parser.parse_args()


def main():
    args = parse_command_line()
    annType = "bbox"
    annFile = args.annotation_file
    resFile = args.result_file
    cocoGt = COCO(annFile)
    cocoDt = cocoGt.loadRes(resFile)

    # running evaluation
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, annType)
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
