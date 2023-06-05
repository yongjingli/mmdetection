import cv2
import numpy as np
import math
import torch.nn as nn
import torch
import os

def _coco_box_to_bbox(box):
    bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                    dtype=np.float32)
    return bbox

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds.to(dtype=torch.float32) / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind.to(dtype=torch.float32) / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _topk_channel(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds.to(dtype=torch.float32) / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_ys, topk_xs


def multi_pose_decode_local(
        heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, hm_kps_vis=None, K=100,
        ltrb=False, orients=None, hm_orients=None, orients_front=None):

    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)

    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    if orients is not None:
        orients = _transpose_and_gather_feat(orients, inds)
        orients = orients.view(batch, K, 2)

        if orients_front is not None:
            orients_front = _transpose_and_gather_feat(orients_front, inds)
            orients_front = orients_front.view(batch, K, 4)
        # orients = math.atan2(orients[:, :, 0], orients[:, :, 1]) / math.pi * 180
        # orients = orients * 360
        # print(orients.shape)
        # exit(1)

    if hm_orients is not None:
        hm_orients = _transpose_and_gather_feat(hm_orients, inds)
        hm_orients = hm_orients.view(batch, K, 1)
        # print(hm_orients.shape)
        # exit(1)

    wh = _transpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)
    else:
        wh = wh.view(batch, K, 2)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5

        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)

        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)

    detections = torch.cat([bboxes, scores, kps, clses], dim=2)

    # debug_save
    # kps_np = kps.detach().cpu().numpy()
    # hm_kps_vis_np = hm_kps_vis.detach().cpu().numpy()
    # save_path = "/home/dev/data_disk/liyj/data/collect_data/debug/debug_hm_kps_vis"
    # np.save(os.path.join(save_path, "kps_np.npy"), kps_np)
    # np.save(os.path.join(save_path, "hm_kps_vis_np.npy"), hm_kps_vis_np)

    # decode keypoint visibility
    if hm_kps_vis is not None:

        hm_kps_vis = hm_kps_vis.permute(0, 2, 3, 1).contiguous()  # 1，17，120，216 ->  1，120，216，17
        hm_kps_vis = hm_kps_vis.view(hm_kps_vis.size(0), -1, hm_kps_vis.size(3)) #1，120，216，17 -> 1，(120，216)，17

        kps_ind = kps.reshape(batch, K, num_joints, 2).to(torch.int64)  # 1，100，34 -> 1,100,17,2

        # height, width
        kps_ind[:, :, :, 0] = torch.clamp(kps_ind[:, :, :, 0], 0, width - 1) # 1,100,17,2 -> 1,100,17
        kps_ind[:, :, :, 1] = torch.clamp(kps_ind[:, :, :, 1], 0, height - 1)

        kps_ind = kps_ind[:, :, :, 1] * width + kps_ind[:, :, :, 0]  # 1,100,17,2 - > 1,100,17
        kps_vis = hm_kps_vis.gather(1, kps_ind)
        detections = torch.cat([detections, kps_vis], dim=2)


        # dim = feat.size(2)
        # ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        # feat = feat.gather(1, ind)
        # hm_kps_vis = torch.sigmoid(_transpose_and_gather_feat(kps_vis, inds))
        # detections = torch.cat([detections, kps_vis], dim=2)

    if orients is not None:
        detections = torch.cat([detections, orients], dim=2)

    if hm_orients is not None:
        detections = torch.cat([detections, hm_orients], dim=2)

    if orients_front is not None:
        orients_front_cls = torch.max(orients_front, dim=2)[1].unsqueeze(-1)
        orients_front_score = torch.max(orients_front, dim=2)[0].unsqueeze(-1)
        # print(orients_front[0, 0])
        # print(orients_front_cls[0, 0])
        # print(orients_front_score[0, 0])
        # detections = torch.cat([detections, orients_front_cls, orients_front_score], dim=2)
        detections = torch.cat([detections, orients_front], dim=2)

    return detections



def save_frame_from_video():
    video_path = '/home/dev/data_disk/liyj/data/collect_data/orients/test_data/6_outdoor.avi'
    video_cap = cv2.VideoCapture(video_path)
    output_dir = os.path.splitext(video_path)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    jump = 1
    count = 0
    while 1:
        rval, frame = video_cap.read()
        if not rval:
            break
        if count % jump == 0:
            img_name = video_path.split("/")[-1].split('.')[0] + "_" + str(count) + '.jpg'
            img_path = os.path.join(os.path.splitext(video_path)[0], img_name)
            print(img_path)
            cv2.imwrite(img_path, frame)
            # print('frame count:{}'.format(count))
            # cv2.namedWindow('img', 0)
            # cv2.imshow('img', frame)
            # wait_key = cv2.waitKey(0)
            # if wait_key == 27:
            #     break
        count += 1

def convert_orientation_pair_2_orientation(orientation_results):
    orientation_results_convert = np.zeros((orientation_results.shape[0], 2))

    for i, orientation_result in enumerate(orientation_results):
        orients_pair = orientation_result[0:2]
        orientation = math.atan2(orients_pair[0], orients_pair[1]) / math.pi * 180
        # print(orients_pair[0], orients_pair[1])

        if 180 >= orientation >= 0:
            orientation = 90 + (180 - orientation)
        elif orientation >= -90:
            orientation = 270 + abs(orientation)
        elif orientation >= -180:
            orientation = abs(orientation) - 90
        else:
            orientation = None

        orientation_confident = orientation_result[2]
        # orientation_result[0] = orientation
        # orientation_result[1] = orientation_confident

        orientation_results_convert[i, 0] = orientation
        orientation_results_convert[i, 1] = orientation_confident

    return orientation_results_convert


def draw_orientation_in_cv_img(img, boxes, orientation_results, bins=36, color=(255, 0, 0), thinkness=1, draw_bins=False):
    for box, orientation_result in zip(boxes, orientation_results):

        orientation = orientation_result[0]
        orientation_confident = orientation_result[1]
        if orientation_confident < 0.3:
            color = tuple(reversed(color))
            continue

        # if orientation_confident < 0.4:
        #     color = tuple(reversed(color))
        # print(orientation_confident)

        x1, y1, w, h = box[:4]

        # c_x = int(x1 + w * 0.5)
        # c_y = int(y1 + h * 0.5)
        c_x = int(x1 + w * 1/2)
        c_y = int(y1 + h * 1/3)
        c_r = int(min(w, h) * 0.5)

        # draw center
        cv2.circle(img, (c_x, c_y), int(c_r * 0.05), color, -1)

        # draw main direction
        # for angle, direction in zip([0, 90, 180, 270], ['Back', 'Left', 'Font', 'Right']):
        #     margin = c_r * 0.1
        #     angle_show = 90 + angle
        #     if angle_show > 360:
        #         angle_show = angle_show - 360
        #
        #     x1 = c_x + (c_r - margin) * math.cos(angle_show * np.pi / 180.0)
        #     y1 = c_y - (c_r - margin) * math.sin(angle_show * np.pi / 180.0)
        #
        #     x2 = int(c_x + (c_r) * math.cos(angle_show * np.pi / 180.0))
        #     y2 = int(c_y - (c_r) * math.sin(angle_show * np.pi / 180.0))
        #     cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thinkness)
        #     img = cv2.putText(img, direction, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

        # draw orientation
        orientation_show = 90 + orientation
        if orientation_show > 360:
            orientation_show = orientation_show - 360
            # print(orientation_show)
        x1 = c_x + c_r * math.cos(orientation_show * np.pi / 180.0)
        y1 = c_y - c_r * math.sin(orientation_show * np.pi / 180.0)
        cv2.line(img, (int(x1), int(y1)), (int(c_x), int(c_y)), color, thickness=thinkness)
        cv2.arrowedLine(img, (int(c_x), int(c_y)), (int(x1), int(y1)), color, 3, 2, 0, 0.3)

        if draw_bins:
            draw_orientation_bins_in_cv_img(img, c_x, c_y, c_r, bins=bins, color=tuple(reversed(color)), thickness=thinkness)
    return img


def draw_orientation_in_cv_img2(img, boxes, orientation_results, orientation_fronts=None, bins=36, color=(255, 0, 0), thinkness=1, draw_bins=False):
    for box, orientation_result, orientation_front in zip(boxes, orientation_results, orientation_fronts):

        orientation = orientation_result[0]
        orientation_confident = orientation_result[1]
        if orientation_confident < 0.3:
            color = tuple(reversed(color))
            continue

        # if orientation_confident < 0.4:
        #     color = tuple(reversed(color))
        # print(orientation_confident)

        x1, y1, w, h = box[:4]

        # c_x = int(x1 + w * 0.5)
        # c_y = int(y1 + h * 0.5)
        c_x = int(x1 + w * 1/2)
        c_y = int(y1 + h * 1/3)
        c_r = int(min(w, h) * 0.5)

        # draw center
        cv2.circle(img, (c_x, c_y), int(c_r * 0.05), color, -1)

        # draw main direction
        # for angle, direction in zip([0, 90, 180, 270], ['Back', 'Left', 'Font', 'Right']):
        #     margin = c_r * 0.1
        #     angle_show = 90 + angle
        #     if angle_show > 360:
        #         angle_show = angle_show - 360
        #
        #     x1 = c_x + (c_r - margin) * math.cos(angle_show * np.pi / 180.0)
        #     y1 = c_y - (c_r - margin) * math.sin(angle_show * np.pi / 180.0)
        #
        #     x2 = int(c_x + (c_r) * math.cos(angle_show * np.pi / 180.0))
        #     y2 = int(c_y - (c_r) * math.sin(angle_show * np.pi / 180.0))
        #     cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thinkness)
        #     img = cv2.putText(img, direction, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        # orient_cls_info = ["back", 'left', 'front', 'right']
        # orient_cls_id, orient_cls_score = orientation_front.cpu().detach().numpy()
        # orient_cls_txt = orient_cls_info[int(orient_cls_id)]
        # img = cv2.putText(img, orient_cls_txt, (int(x1), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)
        #
        # if orientation_front > 0.5:
        #     # img = cv2.putText(img, "front", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        #     img = cv2.putText(img, "front", (int(x1), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        # else:
        #     # img = cv2.putText(img, "back", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        #     img = cv2.putText(img, "back", (int(x1), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)


        # draw orientation
        orientation_show = 90 + orientation
        if orientation_show > 360:
            orientation_show = orientation_show - 360
            # print(orientation_show)
        x1 = c_x + c_r * math.cos(orientation_show * np.pi / 180.0)
        y1 = c_y - c_r * math.sin(orientation_show * np.pi / 180.0)
        cv2.line(img, (int(x1), int(y1)), (int(c_x), int(c_y)), color, thickness=thinkness)
        cv2.arrowedLine(img, (int(c_x), int(c_y)), (int(x1), int(y1)), color, 3, 2, 0, 0.3)

        if draw_bins:
            draw_orientation_bins_in_cv_img(img, c_x, c_y, c_r, bins=bins, color=tuple(reversed(color)), thickness=thinkness)
    return img


def draw_orientation_in_cv_img3(img, boxes, orientation_results, orientation_fronts=None, bins=36, color=(255, 0, 0), thinkness=1, draw_bins=False):
    for box, orientation_result, orientation_front in zip(boxes, orientation_results, orientation_fronts):

        orientation = orientation_result[0]
        orientation_confident = orientation_result[1]

        conf_back = round(float(orientation_front[0]), 2)
        conf_front = round(float(orientation_front[2]), 2)

        if orientation_confident < 0.3:
            color = tuple(reversed(color))
            continue

        # if orientation_confident < 0.4:
        #     color = tuple(reversed(color))
        # print(orientation_confident)

        x1, y1, w, h = box[:4]

        # c_x = int(x1 + w * 0.5)
        # c_y = int(y1 + h * 0.5)
        c_x = int(x1 + w * 1/2)
        c_y = int(y1 + h * 1/3)
        c_r = int(min(w, h) * 0.5)

        # draw center
        cv2.circle(img, (c_x, c_y), int(c_r * 0.05), color, -1)

        # draw main direction
        # for angle, direction in zip([0, 90, 180, 270], ['Back', 'Left', 'Font', 'Right']):
        #     margin = c_r * 0.1
        #     angle_show = 90 + angle
        #     if angle_show > 360:
        #         angle_show = angle_show - 360
        #
        #     x1 = c_x + (c_r - margin) * math.cos(angle_show * np.pi / 180.0)
        #     y1 = c_y - (c_r - margin) * math.sin(angle_show * np.pi / 180.0)
        #
        #     x2 = int(c_x + (c_r) * math.cos(angle_show * np.pi / 180.0))
        #     y2 = int(c_y - (c_r) * math.sin(angle_show * np.pi / 180.0))
        #     cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thinkness)
        #     img = cv2.putText(img, direction, (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        # orient_cls_info = ["back", 'left', 'front', 'right']
        #
        # orient_cls_id, orient_cls_score = orientation_front.cpu().detach().numpy()
        # orient_cls_txt = orient_cls_info[int(orient_cls_id)]
        front_orient_info = "back_" + str(conf_back) + "_front_" + str(conf_front)
        img = cv2.putText(img, front_orient_info, (int(x1), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)
        # img = cv2.putText(img, orient_cls_txt, (int(x1), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 2)
        #
        # if orientation_front > 0.5:
        #     # img = cv2.putText(img, "front", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        #     img = cv2.putText(img, "front", (int(x1), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        # else:
        #     # img = cv2.putText(img, "back", (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
        #     img = cv2.putText(img, "back", (int(x1), int(c_y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)


        # draw orientation
        orientation_show = 90 + orientation
        if orientation_show > 360:
            orientation_show = orientation_show - 360
            # print(orientation_show)
        x1 = c_x + c_r * math.cos(orientation_show * np.pi / 180.0)
        y1 = c_y - c_r * math.sin(orientation_show * np.pi / 180.0)
        cv2.line(img, (int(x1), int(y1)), (int(c_x), int(c_y)), color, thickness=thinkness)
        cv2.arrowedLine(img, (int(c_x), int(c_y)), (int(x1), int(y1)), color, 3, 2, 0, 0.3)

        if draw_bins:
            draw_orientation_bins_in_cv_img(img, c_x, c_y, c_r, bins=bins, color=tuple(reversed(color)), thickness=thinkness)
    return img


def draw_orientation_bins_in_cv_img(img, c_x, c_y, c_r, bins=72, color=(0, 0, 255), thickness=1):
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


def draw_boxes_in_cv_img(img, boxes, color=(255, 0, 0), thinkness=1):
    for box in boxes:
        x1, y1, w, h = [int(tmp) for tmp in box[:4]]
        x2 = x1 + w
        y2 = y1 + h
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thinkness)
    return img


def draw_kps_in_cv_img(img, kps, color=(255, 0, 0), radius=5):
    for kp in kps:
        kp = kp.reshape(-1, 2)
        for _kp in kp:
            x, y = [int(tmp) for tmp in _kp]
            cv2.circle(img, (x, y), radius, color, -1)

        # _kp = kp[5]
        # x, y = [int(tmp) for tmp in _kp]
        # cv2.circle(img, (x, y), radius * 5, (255, 0, 255), -1)
        #
        # _kp = kp[11]
        # x, y = [int(tmp) for tmp in _kp]
        # cv2.circle(img, (x, y), radius * 5, (255, 0, 255), -1)
        #
        # _kp = kp[13]
        # x, y = [int(tmp) for tmp in _kp]
        # cv2.circle(img, (x, y), radius * 5, (255, 0, 255), -1)
    return img


def draw_skeleton_in_cv_img(img, kps, kps_score=None):
    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    pose_limb_color = palette[[0, 0, 0, 0, 7, 7, 7, 9, 9, 9, 9, 9, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0]]
    for i, kp in enumerate(kps):
        kp = kp.reshape(-1, 2)
        if kps_score is not None:
            # print(kps_score.shape)
            kp_score = kps_score[i]

        for i in range(len(skeleton)):
            idx_0 = skeleton[i][0] - 1
            idx_1 = skeleton[i][1] - 1
            limb_color = (int(pose_limb_color[i][0]), int(pose_limb_color[i][1]), int(pose_limb_color[i][2]))
            # if maxval[idx_0] > self.conf and maxval[idx_1] > self.conf:
            cv2.line(img, (int(kp[idx_0][0]), int(kp[idx_0][1])), (int(kp[idx_1][0]),
                     int(kp[idx_1][1])), limb_color, 2)
    return img

def show_det_result_on_img(img, dets):
    convert_det_boxes = dets[:, :4]
    convert_det_boxes[:, 2:4] = convert_det_boxes[:, 2:4] - convert_det_boxes[:, 0:2]

    orientation_results = dets[:, 57:60]
    # print(orientation_results)

    orientation_results = convert_orientation_pair_2_orientation(orientation_results)

    # orientation_results = dets[:, 57:59]
    # print(orientation_results)
    # print(convert_det_boxes)

    kps = dets[:, 5:39]
    kps_score = dets[:, 39:56]

    orientation_fronts = dets[:, 60:64]
    # orientation_fronts = None

    img = draw_boxes_in_cv_img(img, convert_det_boxes, color=(0, 255, 0), thinkness=2)
    img = draw_kps_in_cv_img(img, kps, color=(0, 255, 0), radius=5)
    # img = draw_orientation_in_cv_img(img, convert_det_boxes, orientation_results, color=(0, 0, 255), thinkness=8)
    img = draw_skeleton_in_cv_img(img, kps, kps_score=kps_score)
    # img = draw_orientation_in_cv_img(img, convert_det_boxes, orientation_results,
    #                                   orientation_fronts, color=(0, 0, 255), thinkness=8)
    # img = draw_orientation_in_cv_img2(img, convert_det_boxes, orientation_results,
    #                                   orientation_fronts, color=(0, 0, 255), thinkness=8)

    img = draw_orientation_in_cv_img3(img, convert_det_boxes, orientation_results,
                                      orientation_fronts, color=(0, 0, 255), thinkness=8)
    return img
