import sys
sys.path.insert(0, "../")

import os
import copy
import cv2
import mmcv
import torch
import numpy as np
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import choice
from mmcv.runner import load_checkpoint
from matplotlib.patches import Polygon

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmdet.core.visualization.image import imshow_det_bboxes
from mmdet.core.mask.structures import bitmap_to_polygon


def gen_person_mask(result, mask_img, score_thr=0.3, class_names=None):
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)

    # self define draw
    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

        # img_show = cv2.imread(img)
        # boxes
        for box in bboxes:
            bbox_int = box.astype(np.int32)
            # cv2.rectangle(img, ([bbox_int[0], bbox_int[1]]), ([bbox_int[2], bbox_int[3]]),
            #               (255, 0, 255), 2)
        polygons = []
        for i, (label, box, segm) in enumerate(zip(labels, bboxes, segms)):
            cls_name = class_names[label]
            if cls_name in ['person', 'backpack']:
                bbox_int = box.astype(np.int32)
                cv2.rectangle(mask_img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), 0, -1)

        for i, (label, box, segm) in enumerate(zip(labels, bboxes, segms)):
            cls_name = class_names[label]
            if cls_name in ['person', 'backpack']:
                contours, _ = bitmap_to_polygon(segm)
                polygons += [Polygon(c) for c in contours]
                for contour in contours:
                    # cv2.polylines(img=img, pts=[contour], isClosed=True, color=(0, 0, 255), thickness=3)
                    cv2.fillPoly(mask_img, [contour], 255)


    return mask_img


def expand_mask(mask_img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_img = cv2.dilate(mask_img, kernel, iterations=1)
    return mask_img


def gan_remove_bg_2_reid(segm_model):
    root = "/home/dev/data_disk/liyj/data/collect_data/reid/label_data/20220322"
    dst_root = root + "_gan"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    class_name = segm_model.CLASSES
    img_paths = []
    for root, dirs, files in os.walk(root, topdown=False):
        for img_file in files:
            if img_file.endswith('.png'):
                img_path = os.path.join(root, img_file)
                img_paths.append(img_path)

    for img_path in tqdm(img_paths[:-1]):
        # img_path = "/home/dev/data_disk/liyj/data/collect_data/reid/label_data/20220322/camera_body_right_up_rgb/p00037/0699793d-28b2-11ec-92ea-000c293913c8/png/13149520-28b2-11ec-92ea-000c293913c8.png"
        # print(img_path)
        result = inference_detector(segm_model, img_path)
        img = cv2.imread(img_path)
        mask_img = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
        mask_img = gen_person_mask(result, mask_img, class_names=class_name)

        mask_img = expand_mask(mask_img)

        bg_img_path = choice(img_paths)
        bg_img = cv2.imread(bg_img_path)

        if img.shape != bg_img.shape:
            bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))

        assert img.shape == bg_img.shape, "img.shape != bg_img.shape, {}," \
                                          " {}".format(img_path, bg_img_path)
        gan_img = copy.deepcopy(bg_img)
        gan_img[mask_img > 0] = img[mask_img > 0]

        img_root, img_name = os.path.split(img_path)
        dst_img_root = img_root.replace(root, dst_root)
        dst_img_path = os.path.join(dst_img_root, img_name)
        if not os.path.exists(dst_img_root):
            os.makedirs(dst_img_root)
        cv2.imwrite(dst_img_path, gan_img)
        # print(dst_img_path)
        # exit(1)

        if 0:
            plt.subplot(4, 1, 1)
            plt.imshow(img[:, :, ::-1])
            plt.subplot(4, 1, 2)
            plt.imshow(mask_img)

            plt.subplot(4, 1, 3)
            plt.imshow(bg_img[:, :, ::-1])

            plt.subplot(4, 1, 4)
            plt.imshow(gan_img[:, :, ::-1])
            plt.show()
        # exit(1)



if __name__ == "__main__":
    # Choose to use a config and initialize the detector
    config = '../configs/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
    checkpoint = '../checkpoints/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth'

    # Set the device to be used for evaluation
    device='cuda:6'

    config = mmcv.Config.fromfile(config)
    config.model.pretrained = None
    model = build_detector(config.model)
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.cfg = config
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    gan_remove_bg_2_reid(model)