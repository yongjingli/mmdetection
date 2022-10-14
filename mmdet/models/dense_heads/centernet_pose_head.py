# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob, normal_init
from mmcv.ops import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models import HEADS, build_loss
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from ..utils.gaussian_target import (get_local_maximum, get_topk_from_heatmap,
                                     transpose_and_gather_feat)
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class CenterNetPoseHead(BaseDenseHead, BBoxTestMixin):
    """Objects as Points Head. CenterHead use center_point to indicate object's
    position. Paper link <https://arxiv.org/abs/1904.07850>

    Args:
        in_channel (int): Number of channel in the input feature map.
        feat_channel (int): Number of channel in the intermediate feature map.
        num_classes (int): Number of categories excluding the background
            category.
        loss_center_heatmap (dict | None): Config of center heatmap loss.
            Default: GaussianFocalLoss.
        loss_wh (dict | None): Config of wh loss. Default: L1Loss.
        loss_offset (dict | None): Config of offset loss. Default: L1Loss.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channel,
                 feat_channel,
                 num_classes,
                 loss_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_offset=dict(type='L1Loss', loss_weight=1.0),
                 loss_keypoints_center_heatmap=dict(
                     type='GaussianFocalLoss', loss_weight=1.0),
                 loss_keypoints_wh=dict(type='L1Loss', loss_weight=0.1),
                 loss_keypoints_offset=dict(type='L1Loss', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 num_keypoints=17):
        super(CenterNetPoseHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_classes)
        self.wh_head = self._build_head(in_channel, feat_channel, 2)
        self.offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_center_heatmap = build_loss(loss_center_heatmap)
        self.loss_wh = build_loss(loss_wh)
        self.loss_offset = build_loss(loss_offset)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False

        # keypoints
        self.num_keypoints = num_keypoints
        self.keypoint_heatmap_head = self._build_head(in_channel, feat_channel,
                                             num_keypoints)
        self.keypoint_wh_head = self._build_head(in_channel, feat_channel, 2*num_keypoints)
        self.keypoint_offset_head = self._build_head(in_channel, feat_channel, 2)

        self.loss_keypoints_center_heatmap = build_loss(loss_keypoints_center_heatmap)
        self.loss_keypoints_wh = build_loss(loss_keypoints_wh)
        self.loss_keypoints_offset = build_loss(loss_keypoints_offset)

    def _build_head(self, in_channel, feat_channel, out_channel):
        """Build head for each branch."""
        layer = nn.Sequential(
            nn.Conv2d(in_channel, feat_channel, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channel, out_channel, kernel_size=1))
        return layer

    def init_weights(self):
        """Initialize weights of the head."""
        bias_init = bias_init_with_prob(0.1)
        self.heatmap_head[-1].bias.data.fill_(bias_init)

        keypoint_bias_init = bias_init_with_prob(0.1)
        self.keypoint_heatmap_head[-1].bias.data.fill_(keypoint_bias_init)
        for head in [self.wh_head, self.offset_head, self.keypoint_wh_head, self.keypoint_offset_head]:
            for m in head.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)

    def forward(self, feats):
        """Forward features. Notice CenterNet head does not use FPN.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            center_heatmap_preds (List[Tensor]): center predict heatmaps for
                all levels, the channels number is num_classes.
            wh_preds (List[Tensor]): wh predicts for all levels, the channels
                number is 2.
            offset_preds (List[Tensor]): offset predicts for all levels, the
               channels number is 2.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, feat):
        """Forward feature of a single level.

        Args:
            feat (Tensor): Feature of a single level.

        Returns:
            center_heatmap_pred (Tensor): center predict heatmaps, the
               channels number is num_classes.
            wh_pred (Tensor): wh predicts, the channels number is 2.
            offset_pred (Tensor): offset predicts, the channels number is 2.
        """
        center_heatmap_pred = self.heatmap_head(feat).sigmoid()
        wh_pred = self.wh_head(feat)
        offset_pred = self.offset_head(feat)

        keypoint_heatmap_pred = self.keypoint_heatmap_head(feat).sigmoid()
        keypoint_wh_pred = self.keypoint_wh_head(feat)
        keypoint_offset_pred = self.keypoint_offset_head(feat)

        return center_heatmap_pred, wh_pred, offset_pred, keypoint_heatmap_pred, keypoint_wh_pred, keypoint_offset_pred

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds'))
    def loss(self,
             center_heatmap_preds,
             wh_preds,
             offset_preds,
             keypoint_heatmap_preds,
             keypoint_wh_preds,
             keypoint_offset_preds,
             gt_bboxes,
             gt_labels,
             gt_keypoints,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            center_heatmap_preds (list[Tensor]): center predict heatmaps for
               all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): wh predicts for all levels with
               shape (B, 2, H, W).
            offset_preds (list[Tensor]): offset predicts for all levels
               with shape (B, 2, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: which has components below:
                - loss_center_heatmap (Tensor): loss of center heatmap.
                - loss_wh (Tensor): loss of hw heatmap
                - loss_offset (Tensor): loss of offset heatmap.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        center_heatmap_pred = center_heatmap_preds[0]
        wh_pred = wh_preds[0]
        offset_pred = offset_preds[0]

        assert len(keypoint_heatmap_preds) == len(keypoint_wh_preds) == len(
            keypoint_offset_preds) == 1
        keypoint_heatmap_pred = keypoint_heatmap_preds[0]
        keypoint_wh_pred = keypoint_wh_preds[0]
        keypoint_offset_pred = keypoint_offset_preds[0]

        # target_result, avg_factor = self.get_targets(gt_bboxes, gt_labels,
        #                                              center_heatmap_pred.shape,
        #                                              img_metas[0]['pad_shape'])
        target_result, avg_factor = self.get_pose_targets(gt_bboxes, gt_keypoints, gt_labels,
                                                     center_heatmap_pred.shape,
                                                     img_metas[0]['pad_shape'])

        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        # keypoints GT
        keypoints_center_heatmap_target = target_result['keypoints_center_heatmap_target']
        keypoints_offset_target = target_result['keypoints_offset_target']
        keypoints_offset_target_weight = target_result['keypoints_offset_target_weight']

        keypoints_wh_target = target_result['keypoints_wh_target']
        keypoints_wh_target_weight = target_result['keypoints_wh_target_weight']

        # Since the channel of wh_target and offset_target is 2, the avg_factor
        # of loss_center_heatmap is always 1/2 of loss_wh and loss_offset.
        loss_center_heatmap = self.loss_center_heatmap(
            center_heatmap_pred, center_heatmap_target, avg_factor=avg_factor)
        loss_wh = self.loss_wh(
            wh_pred,
            wh_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)
        loss_offset = self.loss_offset(
            offset_pred,
            offset_target,
            wh_offset_target_weight,
            avg_factor=avg_factor * 2)

        # loss keypoints
        avg_keypoints_factor = max(1, keypoints_center_heatmap_target.eq(1).sum())
        loss_keypoints_center_heatmap = self.loss_keypoints_center_heatmap(
            keypoint_heatmap_pred, keypoints_center_heatmap_target, avg_factor=avg_keypoints_factor)

        loss_keypoints_wh = self.loss_keypoints_wh(
            keypoint_wh_pred,
            keypoints_wh_target,
            keypoints_wh_target_weight,
            avg_factor=avg_factor * 2)

        loss_keypoints_offset = self.loss_keypoints_offset(
            keypoint_offset_pred,
            keypoints_offset_target,
            keypoints_offset_target_weight,
            avg_factor=avg_keypoints_factor * 2)

        return dict(
            loss_center_heatmap=loss_center_heatmap,
            loss_wh=loss_wh,
            loss_offset=loss_offset,
            loss_keypoints_center_heatmap=loss_keypoints_center_heatmap,
            loss_keypoints_wh=loss_keypoints_wh,
            loss_keypoints_offset=loss_keypoints_offset)

    def get_targets(self, gt_bboxes, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            for j, ct in enumerate(gt_centers):
                ctx_int, cty_int = ct.int()
                ctx, cty = ct
                scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                radius = gaussian_radius([scale_box_h, scale_box_w],
                                         min_overlap=0.3)
                radius = max(0, int(radius))
                ind = gt_label[j]
                gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                    [ctx_int, cty_int], radius)

                wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight)
        return target_result, avg_factor

    def get_pose_targets(self, gt_bboxes, gt_keypoints, gt_labels, feat_shape, img_shape):
        """Compute regression and classification targets in multiple images.

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.
            feat_shape (list[int]): feature map shape with value [B, _, H, W]
            img_shape (list[int]): image shape in [h, w] format.

        Returns:
            tuple[dict,float]: The float value is mean avg_factor, the dict has
               components below:
               - center_heatmap_target (Tensor): targets of center heatmap, \
                   shape (B, num_classes, H, W).
               - wh_target (Tensor): targets of wh predict, shape \
                   (B, 2, H, W).
               - offset_target (Tensor): targets of offset predict, shape \
                   (B, 2, H, W).
               - wh_offset_target_weight (Tensor): weights of wh and offset \
                   predict, shape (B, 2, H, W).
        """
        img_h, img_w = img_shape[:2]
        bs, _, feat_h, feat_w = feat_shape

        width_ratio = float(feat_w / img_w)
        height_ratio = float(feat_h / img_h)

        center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_classes, feat_h, feat_w])
        wh_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        wh_offset_target_weight = gt_bboxes[-1].new_zeros(
            [bs, 2, feat_h, feat_w])

        # keypoints target
        keypoints_center_heatmap_target = gt_bboxes[-1].new_zeros(
            [bs, self.num_keypoints, feat_h, feat_w])

        # heatmap offset regress
        keypoints_offset_target = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])
        keypoints_offset_target_weight = gt_bboxes[-1].new_zeros([bs, 2, feat_h, feat_w])

        # keypoints offset regress
        keypoints_wh_target = gt_bboxes[-1].new_zeros([bs, 2 * self.num_keypoints, feat_h, feat_w])
        keypoints_wh_target_weight = gt_bboxes[-1].new_zeros([bs, 2 * self.num_keypoints, feat_h, feat_w])

        for batch_id in range(bs):
            gt_bbox = gt_bboxes[batch_id]
            gt_label = gt_labels[batch_id]
            center_x = (gt_bbox[:, [0]] + gt_bbox[:, [2]]) * width_ratio / 2
            center_y = (gt_bbox[:, [1]] + gt_bbox[:, [3]]) * height_ratio / 2
            gt_centers = torch.cat((center_x, center_y), dim=1)

            gt_keypoint = gt_keypoints[batch_id]

            assert gt_bbox.shape[0] == gt_keypoint.shape[0], "gt_bbox[0] == gt_keypoint.shape[0]"

            for j, (ct, _gt_keypoint) in enumerate(zip(gt_centers, gt_keypoint)):
                ctx_int, cty_int = ct.int()
                if 0 <= ctx_int < feat_w and 0 <= cty_int < feat_h:
                    ctx, cty = ct
                    scale_box_h = (gt_bbox[j][3] - gt_bbox[j][1]) * height_ratio
                    scale_box_w = (gt_bbox[j][2] - gt_bbox[j][0]) * width_ratio
                    radius = gaussian_radius([scale_box_h, scale_box_w],
                                             min_overlap=0.3)
                    radius = max(0, int(radius))
                    ind = gt_label[j]
                    gen_gaussian_target(center_heatmap_target[batch_id, ind],
                                        [ctx_int, cty_int], radius)

                    wh_target[batch_id, 0, cty_int, ctx_int] = scale_box_w
                    wh_target[batch_id, 1, cty_int, ctx_int] = scale_box_h

                    offset_target[batch_id, 0, cty_int, ctx_int] = ctx - ctx_int
                    offset_target[batch_id, 1, cty_int, ctx_int] = cty - cty_int

                    wh_offset_target_weight[batch_id, :, cty_int, ctx_int] = 1

                    for j, kps in enumerate(_gt_keypoint):
                        kp_x, kp_y, kp_flag = kps
                        if kp_flag > 0:
                            kp_x = kp_x * width_ratio
                            kp_y = kp_y * height_ratio

                            kp_x_int = int(kp_x)
                            kp_y_int = int(kp_y)

                            if 0 <= kp_x < feat_w and 0 <= kp_y < feat_h:
                                gen_gaussian_target(keypoints_center_heatmap_target[batch_id, j],
                                                    [kp_x_int, kp_y_int], radius)

                                keypoints_offset_target[batch_id, 0, kp_y_int, kp_x_int] = kp_x - kp_x_int
                                keypoints_offset_target[batch_id, 1, kp_y_int, kp_x_int] = kp_y - kp_y_int
                                keypoints_offset_target_weight[batch_id, :, kp_y_int, kp_x_int] = 1

                                keypoints_wh_target[batch_id, 2*j + 0, cty_int, ctx_int] = kp_x - ctx_int
                                keypoints_wh_target[batch_id, 2*j + 1, cty_int, ctx_int] = kp_y - cty_int
                                keypoints_wh_target_weight[batch_id, 2*j + 0, cty_int, ctx_int] = 1
                                keypoints_wh_target_weight[batch_id, 2*j + 1, cty_int, ctx_int] = 1

        avg_factor = max(1, center_heatmap_target.eq(1).sum())
        target_result = dict(
            center_heatmap_target=center_heatmap_target,
            wh_target=wh_target,
            offset_target=offset_target,
            wh_offset_target_weight=wh_offset_target_weight,
            keypoints_center_heatmap_target=keypoints_center_heatmap_target,
            keypoints_offset_target=keypoints_offset_target,
            keypoints_offset_target_weight=keypoints_offset_target_weight,
            keypoints_wh_target=keypoints_wh_target,
            keypoints_wh_target_weight=keypoints_wh_target_weight)

        return target_result, avg_factor

    @force_fp32(apply_to=('center_heatmap_preds', 'wh_preds', 'offset_preds',
                          'keypoint_heatmap_pred', 'keypoint_wh_pred', 'keypoint_offset_pred'))
    def get_bboxes(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   keypoint_heatmap_pred,
                   keypoint_wh_pred,
                   keypoint_offset_pred,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    wh_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_bboxes_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           offset_pred,
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
        batch_det_bboxes, batch_labels = self.decode_heatmap(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
                                                                 [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor'])

        if with_nms:
            det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
                                                      self.test_cfg)
        return det_bboxes, det_labels

    def decode_heatmap(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
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
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

        center_heatmap_pred = get_local_maximum(
            center_heatmap_pred, kernel=kernel)

        *batch_dets, topk_ys, topk_xs = get_topk_from_heatmap(
            center_heatmap_pred, k=k)
        batch_scores, batch_index, batch_topk_labels = batch_dets

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
        return batch_bboxes, batch_topk_labels

    def _bboxes_nms(self, bboxes, labels, cfg):
        if labels.numel() > 0:
            max_num = cfg.max_per_img
            bboxes, keep = batched_nms(bboxes[:, :4], bboxes[:,
                                                             -1].contiguous(),
                                       labels, cfg.nms)
            if max_num > 0:
                bboxes = bboxes[:max_num]
                labels = labels[keep][:max_num]

        return bboxes, labels

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_keypoints=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            # loss_inputs = outs + (gt_bboxes, img_metas)
            if gt_keypoints is None:
                loss_inputs = outs + (gt_bboxes, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_keypoints, img_metas)
        else:
            # loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            if gt_keypoints is None:
                loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
            else:
                loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, img_metas)

        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def simple_test(self, feats, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n, ).
        """
        return self.simple_test_bboxes(feats, img_metas, rescale=rescale)

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        outs = self.forward(feats)
        results_list = self.get_bboxes(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def simple_test_pose(self, feats, img_metas, rescale=False):
        outs = self.forward(feats)
        results_list = self.get_bboxes_pose(
            *outs, img_metas=img_metas, rescale=rescale)
        return results_list

    def get_bboxes_pose(self,
                   center_heatmap_preds,
                   wh_preds,
                   offset_preds,
                   keypoint_heatmap_pred,
                   keypoint_wh_pred,
                   keypoint_offset_pred,
                   img_metas,
                   rescale=True,
                   with_nms=False):
        """Transform network output for a batch into bbox predictions.

        Args:
            center_heatmap_preds (list[Tensor]): Center predict heatmaps for
                all levels with shape (B, num_classes, H, W).
            wh_preds (list[Tensor]): WH predicts for all levels with
                shape (B, 2, H, W).
            offset_preds (list[Tensor]): Offset predicts for all levels
                with shape (B, 2, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Default: True.
            with_nms (bool): If True, do nms before return boxes.
                Default: False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(center_heatmap_preds) == len(wh_preds) == len(
            offset_preds) == 1
        result_list = []
        for img_id in range(len(img_metas)):
            result_list.append(
                self._get_bboxes_pose_single(
                    center_heatmap_preds[0][img_id:img_id + 1, ...],
                    wh_preds[0][img_id:img_id + 1, ...],
                    offset_preds[0][img_id:img_id + 1, ...],
                    keypoint_heatmap_pred[0][img_id:img_id + 1, ...],
                    keypoint_wh_pred[0][img_id:img_id + 1, ...],
                    keypoint_offset_pred[0][img_id:img_id + 1, ...],
                    img_metas[img_id],
                    rescale=rescale,
                    with_nms=with_nms))
        return result_list

    def _get_bboxes_pose_single(self,
                           center_heatmap_pred,
                           wh_pred,
                           offset_pred,
                           keypoint_heatmap_pred,
                           keypoint_wh_pred,
                           keypoint_offset_pred,
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
        batch_det_bboxes, batch_labels, batch_keypoint_wh = self.decode_heatmap_pose(
            center_heatmap_pred,
            wh_pred,
            offset_pred,
            keypoint_heatmap_pred,
            keypoint_wh_pred,
            keypoint_offset_pred,
            img_meta['batch_input_shape'],
            k=self.test_cfg.topk,
            kernel=self.test_cfg.local_maximum_kernel)

        det_bboxes = batch_det_bboxes.view([-1, 5])
        det_labels = batch_labels.view(-1)

        keypoint_wh = batch_keypoint_wh.view([-1, 34])

        batch_border = det_bboxes.new_tensor(img_meta['border'])[...,
                                                                 [2, 0, 2, 0]]
        det_bboxes[..., :4] -= batch_border
        keypoint_wh[:, 0::2] = keypoint_wh[:, 0::2] - batch_border[0]
        keypoint_wh[:, 1::2] = keypoint_wh[:, 1::2] - batch_border[1]

        if rescale:
            det_bboxes[..., :4] /= det_bboxes.new_tensor(
                img_meta['scale_factor'])
            keypoint_wh[:, 0::2] /= keypoint_wh.new_tensor(
                img_meta['scale_factor'])[0]
            keypoint_wh[:, 1::2] /= keypoint_wh.new_tensor(
                img_meta['scale_factor'])[1]

        # if with_nms:
        #     det_bboxes, det_labels = self._bboxes_nms(det_bboxes, det_labels,
        #                                               self.test_cfg)
        return det_bboxes, det_labels, keypoint_wh

    def decode_heatmap_pose(self,
                       center_heatmap_pred,
                       wh_pred,
                       offset_pred,
                       keypoint_heatmap_pred,
                       keypoint_wh_pred,
                       keypoint_offset_pred,
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
        height, width = center_heatmap_pred.shape[2:]
        inp_h, inp_w = img_shape

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
            hm_score, hm_inds, hm_ys, hm_xs = self._topk_channel(keypoint_heatmap_pred, K=k)
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

        return batch_bboxes, batch_topk_labels, batch_keypoint_wh

    def _topk_channel(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds.to(dtype=torch.float32) / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        return topk_scores, topk_inds, topk_ys, topk_xs