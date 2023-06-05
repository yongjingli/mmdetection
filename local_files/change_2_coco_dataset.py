import cv2
import json
import os
import shutil
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from local_utils2 import vis_annot_info

# basic info
coco_joint_names = ['nose', 'left_eye', 'right_eye', 'left_ear',
                    'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',
                    'right_elbow', 'left_wrist', 'right_wrist', 'left_hip',
                    'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

coco_joint_name_dict = {}
for i, name in enumerate(coco_joint_names):
    coco_joint_name_dict.update({name: i})

# COCO关键点的标注格式为[x1,y1,v1,x2,y2,v2,…,x17,y17,v17],大小必为17×3
# visable 等于0 代表在屏幕外或者没标注的，或者不确定，同样设置为不可见属性
# visable 等于1的时候代表在屏幕内，但是由于遮挡属于不可见属性
# visable 等于2的时候代表在屏幕内，而且可见

# local define: 0 位置不确定 1 位置确定但不可见  2位置确定且可见


def convert_big_data_2_coco():
    # root = "/userdata/nas01/training_data/posture_recognition/pose_old_liyj_0914"
    # dst_root = "/userdata/liyj/data/train_data/pose_det/convet_pose_old_0914"

    # root = "/userdata/nas01/training_data/posture_recognition/pose_liyj_px2_0913"
    # dst_root = "/userdata/liyj/data/train_data/pose_det/pose_liyj_px2_0913"

    # root = "/userdata/nas01/training_data/posture_recognition/pose_liyj_0913"
    # dst_root = "/userdata/liyj/data/train_data/pose_det/pose_liyj_px1_0913"

    # corner case
    # root = "/userdata/nas01/training_data/posture_recognition/pose_corner_case_liyj_0927_2"
    # dst_root = "/userdata/liyj/data/train_data/pose_det/pose_corner_case_liyj_0927"

    # mask head
    # root = "/userdata/liyj/data/test_data/pose/pose_old_liyj_0914_mask_head"
    # dst_root = "/userdata/liyj/data/train_data/pose_det/convet_pose_old_0914_mask_head"

    # mask up half body
    root = "/userdata/liyj/data/test_data/pose/pose_old_liyj_0914_only_foot"
    dst_root = "/userdata/liyj/data/train_data/pose_det/convet_pose_old_0914_only_foot"

    has_kp_label = 1
    has_orient_label = 1

    # val_sample_step = 10  # number of images to sample a validation data
    val_sample_step = -1  # -1 mean no validation data

    json_root = os.path.join(root, 'json')
    png_root = os.path.join(root, 'png')
    img_names = [name for name in os.listdir(png_root) if name.endswith(".png")]

    # init output
    out_train = {}
    out_train['info'] = {}
    out_train['license'] = []
    out_train['images'] = []
    out_train['annotations'] = []
    out_train['categories'] = []

    out_val = {}
    out_val['info'] = {}
    out_val['license'] = []
    out_val['images'] = []
    out_val['annotations'] = []
    out_val['categories'] = []

    for mode in ['train', 'val']:
        path = os.path.join(dst_root, mode + '2017')
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    path = os.path.join(dst_root, 'annotations')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    image_id = 0
    annot_id = 0

    for img_name in tqdm(img_names):
        # print(img_name)
        # img_name = "5adb89bb-6013-4a02-a58c-b24750035bc6.png"

        img_path = os.path.join(png_root, img_name)
        json_path = os.path.join(json_root, img_name.replace(".png", ".json"))

        img_info = {}
        img = cv2.imread(img_path)
        height, width, _ = img.shape
        image_id += 1

        mode = 'val' if (val_sample_step != -1 and image_id % val_sample_step == 0) else 'train'

        dst_img_path = os.path.join(dst_root, 'train' + '2017', img_name)
        shutil.copy(img_path, dst_img_path)

        img_info['file_name'] = dst_img_path
        img_info['height'] = height
        img_info['width'] = width
        img_info['id'] = image_id

        if mode == 'train':
            out_train['images'].append(img_info)
        else:
            out_val['images'].append(img_info)

        # save img path in img_info['file_name'], no copy img
        # shutil.copy(img_file_path, osp.join(out_path_base, mode + '2017', file))

        with open(json_path, 'r') as fp:
            annos = json.load(fp)

        objects = annos['objects']
        for object in objects:
            annot_id += 1
            box_info = {}

            # angle_line = object['angle_line']
            angle_theta = object['body_angle_theta']
            # box_attribute = object['box_attribute']
            box_position = object['body_box_position']

            points = []
            if 'points' in object:
                points = object['points']

            point_attribute = []
            if 'point_attribute' in object:
                point_attribute = object['point_attribute']

            # clip box
            x1, y1, w, h = box_position
            x2 = x1 + w
            y2 = y1 + h
            x1 = min(max(0, x1), width - 1)
            x2 = min(max(0, x2), width - 1)
            y1 = min(max(0, y1), height - 1)
            y2 = min(max(0, y2), height - 1)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            assert w > 0, "box_w < = 0, {}".format(img_name)
            assert h > 0, "box_h < = 0, {}".format(img_name)

            # box_annot = box_position  # ltwh
            box_annot = [x1, y1, w, h]

            # add orientation
            if angle_theta is None:
                angle_theta = -1
                print("set angle_theta = -1")
            else:
                if type(angle_theta) == float or type(angle_theta) == int:
                    if angle_theta < 0 or angle_theta > 360:
                        print(angle_theta)
                        angle_theta = -1
                        print("set angle_theta = -1")
                else:
                    print(angle_theta)
                    angle_theta = -1
                    print("set angle_theta = -1")

            box_info['orientation'] = angle_theta    # -1 mean can not judge orientation
            box_info['segmentation'] = []
            box_info['num_keypoints'] = 0
            box_info['area'] = float(box_annot[2] * box_annot[3])
            box_info['iscrowd'] = 0
            box_info['keypoints'] = np.zeros((51,)).tolist()
            box_info['image_id'] = image_id
            # box_info['bbox'] = box_annot.tolist()
            box_info['bbox'] = box_annot
            box_info['category_id'] = 1
            box_info['id'] = annot_id

            # if has_no_kp_label = 0
            box_info["has_kp_label"] = has_kp_label
            box_info["has_orient_label"] = has_orient_label

            if len(points) != 0 or len(point_attribute) != 0:
                for kp, point_attr in zip(points, point_attribute):
                    x, y = kp
                    x = min(max(0, x), width - 1)
                    y = min(max(0, y), height - 1)

                    kp_name = point_attr["key_point"]
                    # no use palm
                    if kp_name in coco_joint_names:
                        # print(kp_name)
                        if kp_name not in ['left_palm', 'right_palm']:
                            kp_index = coco_joint_name_dict[kp_name]

                            if 'visuable' in point_attr:
                                kp_vis = point_attr['visuable']
                            else:
                                kp_vis == "sure_visuable"

                            if kp_vis == "sure_visuable":
                                kp_vis_value = 2
                            elif kp_vis == "sure_unvisuable":
                                kp_vis_value = 1
                            elif kp_vis == "unsure":
                                kp_vis_value = 0
                            else:
                                kp_vis_value = 0
                                print("unknow vis attr:", kp_vis_value, img_name)

                            box_info['keypoints'][3 * kp_index: 3 * (kp_index + 1)] = [x, y, kp_vis_value]
                # exit(1)
            if mode == 'train':
                out_train['annotations'].append(box_info)
            else:
                out_val['annotations'].append(box_info)

    catg_info = {}
    catg_info['supercategory'] = 'person'
    catg_info['id'] = 1
    catg_info['keypoints'] = ['nose', 'left_eye', 'right_eye',
                              'left_ear', 'right_ear', 'left_shoulder',
                              'right_shoulder', 'left_elbow', 'right_elbow',
                              'left_wrist', 'right_wrist', 'left_hip',
                              'right_hip', 'left_knee', 'right_knee',
                              'left_ankle', 'right_ankle']
    catg_info['skeleton'] = []
    out_train['categories'].append(catg_info)
    out_val['categories'].append(catg_info)

    for mode in ['train', 'val']:
        out_f = open(os.path.join(dst_root, 'annotations', 'person_keypoints_' + mode + '2017.json'), 'w+')
        if mode == 'train':
            json.dump(out_train, out_f)
        else:
            json.dump(out_val, out_f)


def check_data():
    # root = "/userdata/nas01/training_data/posture_recognition/pose_corner_case_liyj_0927_2"
    # root = "/userdata/nas01/training_data/posture_recognition/pose_old_liyj_0914"
    # root = "/userdata/nas01/training_data/posture_recognition/pose_old_liyj_0914"
    # root = "/userdata/nas01/training_data/posture_recognition/pose_corner_case_liyj_0927_3"
    # root = "/userdata/liyj/data/test_data/pose/pose_old_liyj_0914_mask_head"
    root = "/userdata/liyj/data/test_data/pose/pose_old_liyj_0914_only_foot"

    dst_root = "/userdata/liyj/data/test_data/depth/debug"
    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
    os.mkdir(dst_root)

    json_root = os.path.join(root, 'json')
    png_root = os.path.join(root, 'png')
    img_names = [name for name in os.listdir(png_root) if name.endswith(".png")]
    for img_name in tqdm(img_names):
        print(img_name)
        # img_name = "5e785a86-074d-11ed-9144-d6e9ae32ef6c.png"
        # img_name = "5ed9ff5c-074d-11ed-9144-d6e9ae32ef6c.png"
        # img_name = "05368d3a-9512-11ec-95d4-000c293913c8.png"

        # 标注中存在点，但是下载的文件没有关键点的信息
        # img_name = "170a35a0-2a9f-11ed-98ff-6ad8cbacf734.png"

        CHECK_ERR = False

        img_path = os.path.join(png_root, img_name)
        json_path = os.path.join(json_root, img_name.replace(".png", ".json"))

        img = cv2.imread(img_path)
        height, width, _ = img.shape

        with open(json_path, 'r') as fp:
            annos = json.load(fp)

        objects = annos['objects']
        for object in objects:
            box_info = {}

            box_position = object['body_box_position']
            # angle_line = object['angle_line']
            # angle_theta = object['angle_theta']
            # points = object['points']
            # point_attribute = object['point_attribute']
            # box_attribute = object['box_attribute']
            angle_theta = object['body_angle_theta']

            points = []
            if 'points' in object:
                points = object['points']

            point_attribute = []
            if 'point_attribute' in object:
                point_attribute = object['point_attribute']

            kps = np.zeros((51,)).tolist()

            # clip box
            x1, y1, w, h = box_position
            x2 = x1 + w
            y2 = y1 + h
            x1 = min(max(0, x1), width - 1)
            x2 = min(max(0, x2), width - 1)
            y1 = min(max(0, y1), height - 1)
            y2 = min(max(0, y2), height - 1)
            w = x2 - x1
            h = y2 - y1
            assert w > 0, "box_w < = 0, {}".format(img_name)
            assert h > 0, "box_h < = 0, {}".format(img_name)

            # box_annot = box_position  # ltwh
            box_annot = [x1, y1, w, h]

            if angle_theta is None:
                angle_theta = -1
                print("set angle_theta = -1")
            else:
                if type(angle_theta) == float or type(angle_theta) == int:
                    if angle_theta < 0 or angle_theta > 360:
                        angle_theta = -1
                        print("set angle_theta = -1")
                else:
                    angle_theta = -1
                    print("set angle_theta = -1")

            if len(points) != 0 or len(point_attribute) != 0:
                for kp, point_attr in zip(points, point_attribute):
                    x, y = kp
                    x = min(max(0, x), width - 1)
                    y = min(max(0, y), height - 1)

                    kp_name = point_attr["key_point"]
                    # no use palm
                    # if kp_name not in ['left_palm', 'right_palm']:
                    if kp_name in coco_joint_names:
                        if 'visuable' not in point_attr.keys():
                            print("no visuable attr")
                            CHECK_ERR = True
                            continue

                        kp_vis = point_attr['visuable']
                        kp_index = coco_joint_name_dict[kp_name]
                        if kp_vis == "sure_visuable":
                            kp_vis_value = 2
                        elif kp_vis == "sure_unvisuable":
                            kp_vis_value = 1
                        elif kp_vis == "unsure":
                            kp_vis_value = 0
                        else:
                            print(kp_name)
                            print("unknow vis attr:", kp_vis_value, img_name)
                            CHECK_ERR = True

                        kps[3 * kp_index: 3 * (kp_index + 1)] = [x, y, kp_vis_value]

                # vis annot info
                img = vis_annot_info(img, box_annot, angle_theta, kps)
            else:
                img = vis_annot_info(img, box_annot, angle_theta, None)
                print("no points")
                print(object)
        dst_img_path = os.path.join(dst_root, img_name.replace(".png", ".jpg"))
        cv2.imwrite(dst_img_path, img)

        plt.imshow(img)
        plt.show()
        exit(1)

        if CHECK_ERR:
            print("err anno path:", img_name)
            # shutil.move(img_path, os.path.join(dst_root, img_name))
            # shutil.move(json_path, os.path.join(dst_root, img_name.replace(".png", ".json")))


if __name__ == "__main__":
    print("Start Porc...")
    convert_big_data_2_coco()
    # check_data()
    print("End Porc...")