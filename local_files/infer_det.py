import sys
sys.path.insert(0, "../")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os

from mmdet.apis import init_detector, inference_detector
from functools import partial
from types import MethodType
from mmdet.apis import inference_detector, show_result_pyplot


def example_det_infer():
    # download config and checkpoints
    # mim download mmdet - -config centernet_resnet18_dcnv2_140e_coco - -dest.
    # faster rcnn
    # config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # centernet
    config_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco.py'
    checkpoint_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'

    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    # inference the demo image
    results = inference_detector(model, '../demo/demo.jpg')

    # Let's plot the result
    show_result_pyplot(model, '../demo/demo.jpg', results, score_thr=0.3)


if __name__ == "__main__":
    print("Start Local Files")
    example_det_infer()
    print("End Local Files")