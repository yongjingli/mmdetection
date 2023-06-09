import sys
sys.path.insert(0, "../")

import cv2
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os

from mmdet.apis import init_detector, inference_detector
from functools import partial
from types import MethodType
from mmdet.apis import inference_detector, show_result_pyplot, inference_detector_pose
import matplotlib.pyplot as plt


def draw_orient_bins_in_cv_img(img, c_x, c_y, c_r, bins=72, color=(0, 0, 255), thickness=1):
    margin = c_r * 0.1
    bin_angle = 360.0 / bins

    # draw circle
    cv2.circle(img, (c_x, c_y), int(c_r), color, thickness)

    for i in range(bins):
        x1 = c_x + (c_r - margin) * math.cos(i * bin_angle * np.pi / 180.0)
        y1 = c_y + (c_r - margin) * math.sin(i * bin_angle * np.pi / 180.0)

        # 同心小圆，计算B点
        x2 = c_x + c_r * math.cos(i * bin_angle * np.pi/180.0)
        y2 = c_y + c_r * math.sin(i * bin_angle * np.pi/180.0)
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=2)
    return img


def example_det_infer():
    # download config and checkpoints
    # mim download mmdet - -config centernet_resnet18_dcnv2_140e_coco - -dest.
    # faster rcnn
    # config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # centernet
    # config_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco.py'
    # checkpoint_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco_20210702_155131-c8cd631f.pth'

    # centernet
    # config_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco_pose.py'
    # checkpoint_file = '../work_dirs/centernet_resnet18_dcnv2_140e_coco_pose_1013/epoch_23.pth'

    # centernet
    config_file = '../checkpoints/centernet_resnet18_dcnv2_140e_coco_pose_orient.py'
    checkpoint_file = '../work_dirs/centernet_resnet18_dcnv2_140e_coco_pose_orient_orient_weight_20230608/latest.pth'

    # local train centernet
    # config_file = "/userdata/liyj/programs/mmdetection/work_dirs/centernet_resnet18_dcnv2_140e_coco/centernet_resnet18_dcnv2_140e_coco.py"
    # checkpoint_file = "/userdata/liyj/programs/mmdetection/work_dirs/centernet_resnet18_dcnv2_140e_coco/epoch_13.pth"

    device = 'cuda:0'
    # init a detector
    model = init_detector(config_file, checkpoint_file, device=device)
    img_path = '/mnt/data10/liyj/data/open_dataset/coco/val2017/000000190007.jpg'
    # img_path = '/mnt/data10/liyj/data/open_dataset/coco/val2017/000000181303.jpg'
    # img_path = '../demo/demo_person.png'

    # inference the demo image
    # results = inference_detector(model, '../demo/demo_person.png')
    # 主要区别在模型推理时采用 pose_result=True
    # results = inference_detector_pose(model, '../demo/demo_person.png')
    results = inference_detector_pose(model, img_path)

    det_bboxes, det_labels, keypoints_wh, orients_score, orients = results
    if isinstance(det_bboxes, torch.Tensor):
        det_bboxes = det_bboxes.detach().cpu().numpy()
        det_labels = det_labels.detach().cpu().numpy()
        keypoints_wh = keypoints_wh.detach().cpu().numpy()
        orients_score = orients_score.detach().cpu().numpy()
        orients = orients.detach().cpu().numpy()

    # img = cv2.imread('../demo/demo_person.png')
    img = cv2.imread(img_path)
    for det_bbox, det_label, keypoint_wh, orient_score,\
            orient in zip(det_bboxes, det_labels, keypoints_wh, orients_score, orients):
        if det_bbox[4] > 0.3:
            x1, y1, x2, y2 = det_bbox[:4]
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
            for i in range(17):
                kp_x = keypoint_wh[2*i + 0]
                kp_y = keypoint_wh[2*i + 1]
                cv2.circle(img, (int(kp_x), int(kp_y)), 5, (0, 255, 0), -1)

            # 转回到MEBOW的表示
            if 180 >= orient >= 0:
                orient = 90 + (180 - orient)
            elif orient >= -90:
                orient = 270 + abs(orient)
            elif orient >= -180:
                orient = abs(orient) - 90
            else:
                orient = None

            print("orient:", orient)
            # draw orients
            if orient_score > 0.3:
                orient = 90 + orient
                if orient > 360:
                    orient = orient - 360

                w = x2 - x1
                h = y2 - y1
                c_x = int(x1 + w * 1 / 2)
                c_y = int(y1 + h * 1 / 2)
                c_r = int(min(w, h) * 0.5)

                x1 = c_x + c_r * math.cos(orient * np.pi / 180.0)
                y1 = c_y - c_r * math.sin(orient * np.pi / 180.0)
                cv2.line(img, (int(x1), int(y1)), (int(c_x), int(c_y)), (0, 0, 255), thickness=1)
                cv2.arrowedLine(img, (int(c_x), int(c_y)), (int(x1), int(y1)), (0, 0, 255), 3, 2, 0, 0.3)

                img = draw_orient_bins_in_cv_img(img, c_x, c_y, c_r, bins=36,
                                                         color=tuple(reversed((0, 0, 255))),
                                                         thickness=1)

    plt.imshow(img[:, :, ::-1])
    plt.show()
    # Let's plot the result
    # show_result_pyplot(model, '../demo/demo_person.png', results, score_thr=0.25)


if __name__ == "__main__":
    print("Start Local Files")
    example_det_infer()
    print("End Local Files")