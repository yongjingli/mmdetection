import os
# set current dir
os.chdir("../")
cwd_dir = os.getcwd()
print("current dir:", cwd_dir)

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import torch
from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models.builder import DETECTORS, build_backbone, build_head
from mmdet.models.utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)


def topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds.to(dtype=torch.float32) / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def decode_heatmap_pose(center_heatmap_pred,
                        wh_pred,
                        offset_pred,
                        keypoint_heatmap_pred,
                        keypoint_wh_pred,
                        keypoint_offset_pred,
                        orients_heatmap_pred,
                        orients_pred,
                        img_shape,
                        k=100,
                        kernel=3):
    """Transform outputs into detections raw bbox prediction.

    Args:
        center_heatmap_pred (Tensor): center predict heatmap,
           shape (B, num_classes, H, W).
        wh_pred (Tensor): wh predict, shape (B, 2, H, W).
        offset_pred (Tensor): offset predict, shape (B, 2, H, W).
        img_shape (list[int]): image shape in [h, w] format.
        k (int): Get top k center keypoints from heatmap. Default 100.
        kernel (int): Max pooling kernel for extract local maximum pixels.
           Default 3.

    Returns:
        tuple[torch.Tensor]: Decoded output of CenterNetHead, containing
           the following Tensors:

          - batch_bboxes (Tensor): Coords of each box with shape (B, k, 5)
          - batch_topk_labels (Tensor): Categories of each box with \
              shape (B, k)
    """
    # print(center_heatmap_pred.shape)
    # print(center_heatmap_pred.shape[2:])
    height, width = center_heatmap_pred.shape[2:]
    inp_h, inp_w, _ = img_shape

    center_heatmap_pred = get_local_maximum(
        center_heatmap_pred, kernel=kernel)

    *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
        center_heatmap_pred, k=k)
    batch_scores, batch_index, batch_topk_labels = batch_dets

    # keypoints
    keypoint_wh = transpose_and_gather_feat(keypoint_wh_pred, batch_index)
    keypoint_wh[:, :, 0::2] = keypoint_wh[:, :, 0::2] + topk_xs[:, :, None]
    keypoint_wh[:, :, 1::2] = keypoint_wh[:, :, 1::2] + topk_ys[:, :, None]
    keypoint_wh[:, :, 0::2] = keypoint_wh[:, :, 0::2] * (inp_w / width)
    keypoint_wh[:, :, 1::2] = keypoint_wh[:, :, 1::2] * (inp_h / height)
    batch_keypoint_wh = keypoint_wh

    # boxes
    wh = transpose_and_gather_feat(wh_pred, batch_index)
    offset = transpose_and_gather_feat(offset_pred, batch_index)

    topk_xs = topk_xs + offset[..., 0]
    topk_ys = topk_ys + offset[..., 1]
    tl_x = (topk_xs - wh[..., 0] / 2) * (inp_w / width)
    tl_y = (topk_ys - wh[..., 1] / 2) * (inp_h / height)
    br_x = (topk_xs + wh[..., 0] / 2) * (inp_w / width)
    br_y = (topk_ys + wh[..., 1] / 2) * (inp_h / height)

    batch_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], dim=2)
    batch_bboxes = torch.cat((batch_bboxes, batch_scores[..., None]),
                             dim=-1)

    # keypoints finetune
    if 1:
        keypoint_heatmap_pred = get_local_maximum(
            keypoint_heatmap_pred, kernel=kernel)
        batch, cat, height, width = keypoint_heatmap_pred.size()

        thresh = 0.1
        num_joints = cat
        batch_keypoint_wh = batch_keypoint_wh
        batch_keypoint_wh = batch_keypoint_wh.view(batch, k, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = batch_keypoint_wh.unsqueeze(3).expand(batch, num_joints, k, k, 2)
        hm_score, hm_inds, hm_ys, hm_xs = topk_channel(keypoint_heatmap_pred, K=k)
        hp_offset = transpose_and_gather_feat(keypoint_offset_pred, hm_inds.view(batch, -1))
        hp_offset = hp_offset.view(batch, num_joints, k, 2)
        hm_xs = (hm_xs + hp_offset[:, :, :, 0]) * (inp_w / width)
        hm_ys = (hm_ys + hp_offset[:, :, :, 1]) * (inp_h / height)

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, k, k, 2)

        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, k, 1, 1).expand(
            batch, num_joints, k, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, k, 2)

        l = batch_bboxes[:, :, 0].view(batch, 1, k, 1).expand(batch, num_joints, k, 1)
        t = batch_bboxes[:, :, 1].view(batch, 1, k, 1).expand(batch, num_joints, k, 1)
        r = batch_bboxes[:, :, 2].view(batch, 1, k, 1).expand(batch, num_joints, k, 1)
        b = batch_bboxes[:, :, 3].view(batch, 1, k, 1).expand(batch, num_joints, k, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, k, 2)
        kps = (1 - mask) * hm_kps + mask * batch_keypoint_wh
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, k, num_joints * 2)

        batch_keypoint_wh = kps

    # orients
    if 1:
        orients_score = transpose_and_gather_feat(orients_heatmap_pred, batch_index)
        orients = transpose_and_gather_feat(orients_pred, batch_index)
        orients = np.arctan2(orients[:, :, 0], orients[:, :, 1]) / np.pi * 180

    return batch_bboxes, batch_topk_labels, batch_keypoint_wh, orients_score, orients


def get_bboxes_pose_single(
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       keypoint_heatmap_pred,
                       keypoint_wh_pred,
                       keypoint_offset_pred,
                       orients_heatmap_pred,
                       orients_pred,
                       img_meta,
                       rescale=False,
                       with_nms=True):
    """Transform outputs of a single image into bbox results.

    Args:
        center_heatmap_pred (Tensor): Center heatmap for current level with
            shape (1, num_classes, H, W).
        wh_pred (Tensor): WH heatmap for current level with shape
            (1, num_classes, H, W).
        offset_pred (Tensor): Offset for current level with shape
            (1, corner_offset_channels, H, W).
        img_meta (dict): Meta information of current image, e.g.,
            image size, scaling factor, etc.
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        tuple[Tensor, Tensor]: The first item is an (n, 5) tensor, where
            5 represent (tl_x, tl_y, br_x, br_y, score) and the score
            between 0 and 1. The shape of the second tensor in the tuple
            is (n,), and each element represents the class label of the
            corresponding box.
    """
    # print(img_meta)
    # exit(1)
    batch_det_bboxes, batch_labels, batch_keypoint_wh, batch_orients_score, batch_orients = decode_heatmap_pose(
        center_heatmap_pred,
        wh_pred,
        offset_pred,
        keypoint_heatmap_pred,
        keypoint_wh_pred,
        keypoint_offset_pred,
        orients_heatmap_pred,
        orients_pred,
        img_meta['pad_shape'],
        k=100,
        kernel=3)

    det_bboxes = batch_det_bboxes.view([-1, 5])
    det_labels = batch_labels.view(-1)

    keypoint_wh = batch_keypoint_wh.view([-1, 34])
    batch_orients_score = batch_orients_score.view([-1, 1])
    batch_orients = batch_orients.view([-1, 1])

    # 这里的decode为padding的img解析，不进行padding的去除和返回原图大小的结果
    # batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
    #                                                          [2, 0, 2, 0]]
    # det_bboxes[..., :4] -= batch_border
    # keypoint_wh[:, 0::2] = keypoint_wh[:, 0::2] - batch_border[0]
    # keypoint_wh[:, 1::2] = keypoint_wh[:, 1::2] - batch_border[1]

    # if rescale:
    #     det_bboxes[..., :4] /= det_bboxes.new_tensor(
    #         img_meta['scale_factor'])
    #     keypoint_wh[:, 0::2] /= keypoint_wh.new_tensor(
    #         img_meta['scale_factor'])[0]
    #     keypoint_wh[:, 1::2] /= keypoint_wh.new_tensor(
    #         img_meta['scale_factor'])[1]

    # if with_nms:
    #     det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
    #                                               self.test_cfg)
    return det_bboxes, det_labels, keypoint_wh, batch_orients_score, batch_orients


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

def debug_coco_dataset():
    cfg = Config.fromfile('./checkpoints/centernet_resnet18_dcnv2_140e_coco_pose_orient.py')
    # cfg = Config.fromfile('./checkpoints/centernet_resnet18_dcnv2_140e_coco_pose.py')
    # cfg = Config.fromfile('./checkpoints/centernet_resnet18_dcnv2_140e_coco.py')

    bbox_head_cfg = cfg["model"]['bbox_head']
    bbox_head = build_head(bbox_head_cfg)

    # Build dataset
    datasets = [build_dataset(cfg.data.train)]
    print("CLASSES:", datasets[0].CLASSES)

    # prepare data loaders
    dataset = datasets if isinstance(datasets, (list, tuple)) else [datasets]

    distributed = False
    cfg.gpu_ids = [0]
    cfg.seed = 233
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        # `num_gpus` will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,
        runner_type=runner_type,
        persistent_workers=False)

    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }

    data_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in dataset]

    for i, data_batch in enumerate(data_loaders[0]):
        # debug dataset
        img_metas = data_batch["img_metas"]  # list [[dict], [dict]],
        img = data_batch["img"]     #  tensor, [2, 3, 512, 512]
        gt_bboxes = data_batch["gt_bboxes"]  # list, [[4, 4], [3, 4]]
        gt_labels = data_batch["gt_labels"]  # list, [[4], [3]]
        gt_keypoints = data_batch["gt_keypoints"]  # list, [[4], [3]]
        gt_orients = data_batch["gt_orients"]  # list, [[4], [3]]
        gt_orients_mask = data_batch["gt_orients_mask"]  # list, [[4], [3]]

        img_padding = img.data[0][0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, :3]
        img_h, img_w, _ = img_padding.shape
        print("input img shape:", img_padding.shape)

        mean_one = img_metas.data[0][0]["img_norm_cfg"]["mean"]
        std_one = img_metas.data[0][0]["img_norm_cfg"]["std"]
        file_name = img_metas.data[0][0]["filename"]

        img_padding = (img_padding * std_one + mean_one)
        img_padding = img_padding.astype(np.uint8).copy()

        # target_result, avg_factor = bbox_head.get_targets(gt_bboxes.data[0], gt_labels.data[0], [2, 1, int(img_h/8), int(img_w/8)], img_metas.data[0][0]['pad_shape'])
        target_result, avg_factor = bbox_head.get_pose_orient_targets(gt_bboxes.data[0],
                                                          gt_keypoints.data[0],
                                                          gt_labels.data[0],
                                                          gt_orients.data[0],
                                                          gt_orients_mask.data[0],
                                                          [2, 1, int(img_h/8), int(img_w/8)],
                                                          img_metas.data[0][0]['pad_shape'])

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        keypoints_center_heatmap_target = target_result['keypoints_center_heatmap_target']
        keypoints_offset_target = target_result['keypoints_offset_target']
        keypoints_offset_target_weight = target_result['keypoints_offset_target_weight']

        keypoints_wh_target = target_result['keypoints_wh_target']
        keypoints_wh_target_weight = target_result['keypoints_wh_target_weight']

        orients_center_heatmap_target = target_result['orients_center_heatmap_target']
        orients_target = target_result['orients_target']
        orients_target_weight = target_result['orients_target_weight']


        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                    get_bboxes_pose_single(
                    center_heatmap_target[img_id:img_id + 1, ...],
                    wh_target[img_id:img_id + 1, ...],
                    offset_target[img_id:img_id + 1, ...],
                    keypoints_center_heatmap_target[img_id:img_id + 1, ...],
                    keypoints_wh_target[img_id:img_id + 1, ...],
                    keypoints_offset_target[img_id:img_id + 1, ...],
                    orients_center_heatmap_target,   # orients
                    orients_target,

                    img_metas.data[0][img_id],
                    rescale=False,
                    with_nms=False))

        det_bboxes, det_labels, keypoints_wh, orients_score, orients = result_list[0]
        if isinstance(det_bboxes, torch.Tensor):
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_labels = det_labels.detach().cpu().numpy()
            keypoints_wh = keypoints_wh.detach().cpu().numpy()
            orients_score = orients_score.detach().cpu().numpy()
            orients = orients.detach().cpu().numpy()

        for det_bbox, det_label, keypoint_wh, orient_score,\
            orient in zip(det_bboxes, det_labels, keypoints_wh, orients_score, orients):
            if det_bbox[4] > 0.3:
                x1, y1, x2, y2 = det_bbox[:4]
                cv2.rectangle(img_padding, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2)
                for i in range(17):
                    kp_x = keypoint_wh[2 * i + 0]
                    kp_y = keypoint_wh[2 * i + 1]
                    cv2.circle(img_padding, (int(kp_x), int(kp_y)), 5, (0, 255, 0), -1)

                # 转回到MEBOW的表示
                if 180 >= orient >= 0:
                    orient = 90 + (180 - orient)
                elif orient >= -90:
                    orient = 270 + abs(orient)
                elif orient >= -180:
                    orient = abs(orient) - 90
                else:
                    orient = None

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
                    cv2.line(img_padding, (int(x1), int(y1)), (int(c_x), int(c_y)), (0, 0, 255), thickness=1)
                    cv2.arrowedLine(img_padding, (int(c_x), int(c_y)), (int(x1), int(y1)), (0, 0, 255), 3, 2, 0, 0.3)

                    img_padding = draw_orient_bins_in_cv_img(img_padding, c_x, c_y, c_r, bins=36, color=tuple(reversed((0, 0, 255))),
                                                    thickness=1)
        plt.imshow(img_padding)
        plt.show()
        exit(1)


if __name__ == "__main__":
    print("fff")
    debug_coco_dataset()
    print("End")

