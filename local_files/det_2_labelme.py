import sys
import torch
import cv2
import copy
import numpy as np
import time
import os
from tqdm import tqdm
import shutil
import json

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/data10/liyj/programs/centernet_mot/src/lib")
from utils.arg_helper import argument_parser, init_constant
from opts import opts
from datasets.dataset_factory import get_dataset

from pose_infer_pt import PoseInferPt
from local_utils import save_frame_from_video
from local_utils import draw_boxes_in_cv_img
from local_utils import draw_orientation_in_cv_img
from local_utils import draw_orientation_in_cv_img2
from local_utils import draw_orientation_in_cv_img3
from local_utils import draw_kps_in_cv_img
from local_utils import convert_orientation_pair_2_orientation
from local_utils import draw_skeleton_in_cv_img


def get_closest_box(dets):
    if 0 not in dets.shape:
        boxes = dets[:, :4]
        kps = dets[:, 5:39]
        kps_score = dets[:, 39:56]
        areas = abs(boxes[:, 2] - boxes[:, 0]) * abs(boxes[:, 3] - boxes[:, 1])
        max_area_indx = torch.argmax(areas)
        max_boxes = dets[max_area_indx: max_area_indx + 1, :4]
        max_boxes_kps = dets[max_area_indx: max_area_indx + 1, 5:39]
        max_box_kps_score = dets[max_area_indx: max_area_indx + 1, 39:56]
    else:
        max_boxes, max_boxes_kps, max_box_kps_score = None, None, None
    return max_boxes, max_boxes_kps, max_box_kps_score


def save_labelme_json(save_json_path, img, dets):
    img_name = os.path.split(save_json_path)[-1][:-4] + "jpg"
    img_h, img_w, _ = img.shape
    final_coor = {"imagePath": img_name, "imageData": None, "shapes": [], "version": "3.5.0",
                  "flags": None, "fillColor": [255, 0, 0, 128], "lineColor": [0, 255, 0, 128], \
                  "imageWidth": img_w, "imageHeight": img_h}
    # boxes = dets[:, :4]
    # kps = dets[:, 5:39]
    # kps_score = dets[:, 39:56]

    boxes, kps, kps_score = get_closest_box(dets)
    if None not in [boxes, kps, kps_score]:
        for i in range(boxes.shape[0]):
            box = boxes[i]
            # conf = target[4]
            # cls = target[5]
            kps_one = kps[i]
            kps_one = kps_one.reshape(-1, 2)
            kps_score_one = kps_score[i]

            # num_kpts = len(kps) // kp_steps
            # if conf > 0.2:
            # box anns
            cls = 0
            box_name = "_".join(["cls", str(int(cls)), "id", str(i)])
            labelme_box = {"shape_type": "rectangle", "line_color": None, "points": [], \
                            "fill_color": None, "label": box_name}
            labelme_box["points"].append([int(box[0]), int(box[1])])
            labelme_box["points"].append([int(box[2]), int(box[3])])

            final_coor["shapes"].append(labelme_box)

            # points
            # for k in [11, 12]:
            for k in range(0, 17):
                kp = kps_one[k]
                kp_score = kps_score_one[k]
                kp_x, kp_y = kp[0], kp[1]
                point_name = "_".join(["kp", str(int(k)), "id", str(i)])
                labelme_point = {"shape_type": "point", "line_color": None, "points": [], \
                               "fill_color": None, "label": point_name, "score": float(kp_score)}
                labelme_point["points"].append([int(kp_x), int(kp_y)])
                final_coor["shapes"].append(labelme_point)

    # print(final_coor["shapes"])
    save_label_me_path = save_json_path
    with open(save_label_me_path, 'w') as fp:
        json.dump(final_coor, fp)

    cv2.imwrite(save_json_path[:-4] + "jpg", img)
    # print(save_json_path[:-4] + ".jpg")
    # exit(1)


def show_imgs_result(model, root_dir, save_dir=None):
    # img_names = [name for name in os.listdir(root_dir) if name.split('.')[-1] in ['jpg', 'png']]
    img_names = [name for name in os.listdir(root_dir) if name.split('.')[-1] in ['png', 'jpg']]

    labelme_root = "/mnt/data10/liyj/data/test_data/depth/test_indoor_person_caijiche_static_20230109/det_infos_centernet_20230209"
    if os.path.exists(labelme_root):
        shutil.rmtree(labelme_root)
    os.mkdir(labelme_root)

    for img_name in tqdm(img_names):
        # img_name = "1562.377140.png"
        img_path = os.path.join(root_dir, img_name)

        # img_path = "/mnt/data/liyj/data/1618591795_10943.jpg"
        # img_path = "/home/dev/data_disk/liyj/programs/centernet_mot/data/img_with_depth/image_rect_color_2021-12-17-15-33-34_frame_1639726426387279138.png"
        # img_path = "/home/dev/data_disk/liyj/data/open_dataset/coco/val2017/000000397133.jpg"
        img = cv2.imread(img_path)
        img_origin = copy.deepcopy(img)

        if img is None:
            continue

        # img = img[:, :1920, :]
        # r_image = img[:, 1920:, :]

        dets = model.infer(img)

        # save labelme info
        save_json_path = os.path.join(labelme_root, img_name[:-4] + ".json")
        save_labelme_json(save_json_path, img_origin, dets)


        convert_det_boxes = copy.deepcopy(dets[:, :4])
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
        img = draw_orientation_in_cv_img(img, convert_det_boxes, orientation_results, color=(0, 0, 255), thinkness=8)
        img = draw_skeleton_in_cv_img(img, kps, kps_score=kps_score)
        # img = draw_orientation_in_cv_img(img, convert_det_boxes, orientation_results,
        #                                   orientation_fronts, color=(0, 0, 255), thinkness=8)
        # img = draw_orientation_in_cv_img2(img, convert_det_boxes, orientation_results,
        #                                   orientation_fronts, color=(0, 0, 255), thinkness=8)

        # img = draw_orientation_in_cv_img3(img, convert_det_boxes, orientation_results,
        #                                   orientation_fronts, color=(0, 0, 255), thinkness=8)

        if save_dir is not None:
            save_img_path = os.path.join(save_dir, img_name.replace('.png', '.jpg'))
            cv2.imwrite(save_img_path, img)
            # exit(1)
        else:
            plt.imshow(img[:, :, ::-1])
            # plt.imshow(model.inp_resize.astype(np.uint8)[:, :, ::-1])
            # print(model.inp_resize.shape)
            plt.show()
            exit(1)
        # exit(1)



if __name__ == '__main__':

    # save_frame_from_video()
    # exit(1)
    # Parse argument
    args = argument_parser()

    model_root = "/mnt/data10/liyj/programs/centernet_mot/exp/pred_vis_new_aug_liyj_hm_kps_orientation_size512_0419_merge_local_dcn2"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/pred_vis_new_aug_liyj_hm_kps_orientation_size512_0414_merge_local"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/pred_vis_new_aug_liyj_hm_kps_orientation_size_0905_merge_master_no_dcn"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0908"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0910"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0914_orient_weight_4.0/"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/pred_vis_new_aug_liyj_hm_kps_orientation_size512_0419_merge_local_dcn2"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0919_orient_weight_4.0_with_orient_front"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0919_orient_weight_4.0_with_orient_4_orients"

    # check data
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0928_orient_weight_4.0_with_orient_4_orients"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0929_orient_weight_4.0_with_orient_4_orients_with_mask_head"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0929_orient_weight_4.0_with_orient_4_orients_with_mask_head_only_foot"

    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_0929_coco_px1_pose_old_0914_mask_head_only_foot"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/test_test_coco_px1_pose_old_1008_mask_head"
    # model_root = "/userdata/liyj/programs/centernet_mot/exp/pred_vis_new_aug_liyj_hm_kps_orientation_size_0905_merge_master_no_dcn"

    args.config = model_root + "/config.yaml"
    args.save = model_root
    args.init = model_root

    # Pt Model Initial constant
    opt = init_constant(args)
    Dataset = get_dataset(opt.dataset, opt.task, opt)
    opt = opts().update_dataset_info_and_set_heads(opt, Dataset)

    # opt.load_model = model_root + "/model_last.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0919/pose_with_old2.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0919/pose_with_old_px1_px2.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0919/pose_with_px1_px22.pth"

    # best for tmp
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0920/coco_old_corner_case.pth"   # best
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0920/pose_coco_old.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0920/pose_coco_old2.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0920/pose_coco_old3.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0920/pose_coco_old3_2.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0920/pose_coco_old4.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0920/pose_coco_old5.pth"

    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0921/pose_old_px1_px2_2.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0922/pose_pld_px1_px2_corner_case_cls_3.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/test_0926/pose_pld_px1_px2_corner_case_cls_weight_4.0_1.pth"
    # opt.load_model = "/userdata/liyj/programs/centernet_mot/exp/pred_vis_new_aug_liyj_hm_kps_orientation_size512_0414_merge_local/model_last.pth"

    # opt.load_model = model_root + "/model_270.pth"

    opt.load_model = "/mnt/data10/liyj/programs/centernet_mot/exp/pred_vis_new_aug_liyj_hm_kps_orientation_size512_0419_merge_local_dcn2/model_last.pth"

    # opt.vis_thresh = 0.01
    Pose_Infer_Pt = PoseInferPt(opt)

    # show img infer result
    # root_dir = "/home/dev/data_disk/liyj/data/collect_data/orients/test_data/6_outdoor"
    # root_dir = '/mnt/data/10.66.33.111_data/dp/dataset/data/2022/0424/b495d646-c3c8-11ec-8c56-0ac964faefb0/adu_camera_body_front_center_left_image_raw'
    # root_dir = "/userdata/liyj/data/jira_data/image_rect_color_left_0506"
    # save_dir = "/mnt/data/liyj/data/debug_data/b495d646-c3c8-11ec-8c56-0ac964faefb0-model-visualize-result"
    # save_dir = None

    # root_dir = '/userdata/liyj/data/test_data/pose/2022-08-29-17-59-49/image_rect_color'
    # save_dir = "/userdata/liyj/data/test_data/pose/2022-08-29-17-59-49/result"

    # root_dir = '/userdata/liyj/data/test_data/pose/2022-08-29-18-13-54/image_raw_rear'
    # save_dir = "/userdata/liyj/data/test_data/pose/2022-08-29-18-13-54/result"

    # root_dir = "/userdata/liyj/data/test_data/reid/2022-09-05-16-25-28/image_raw_rear_left"
    # root_dir = "/userdata/liyj/data/open_dataset/coco/val2017"
    # save_dir = "/userdata/liyj/data/test_data/pose/2022-08-29-17-59-49/result"

    # root_dir = "/userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/image_rect_color"
    # save_dir = "/userdata/liyj/data/test_data/pose/2022-09-09-15-41-06/result"

    # root_dir = "/userdata/nas01/training_data/posture_recognition/pose_corner_case_liyj_0916/png"
    root_dir = "/mnt/data10/liyj/data/test_data/depth/test_indoor_person_caijiche_static_20230109/left_rectify"
    # root_dir = "/userdata/liyj/data/train_data/pose_det/convet_pose_old_0914/train2017"
    # root_dir = "/userdata/liyj/data/train_data/pose_det/convet_pose_old_0914_corner_case/images"
    # root_dir = "/userdata/nas01/training_data/posture_recognition/pose_px2_crop_liyj_0928/png"
    # save_dir = "/userdata/liyj/data/test_data/depth/debug"

    # root_dir = "/userdata/liyj/data/test_data/pose/tmp"
    # root_dir = "/userdata/liyj/data/train_data/pose_det/convet_pose_old_0914_mask_head/train2017"

    save_dir = "/mnt/data10/liyj/data/debug"
    # save_dir = None

    if save_dir is not None:
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)

    show_imgs_result(Pose_Infer_Pt, root_dir, save_dir)


