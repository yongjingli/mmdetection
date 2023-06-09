import os
# set current dir
os.chdir("../")
cwd_dir = os.getcwd()
print("current dir:", cwd_dir)

import numpy as np
import cv2
import matplotlib.pyplot as plt

from mmcv import Config
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models.builder import DETECTORS, build_backbone, build_head


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

        # debug bbox_head
        # img, img_metas, gt_bboxes, gt_labels = (**data_batch)
        # target_result, avg_factor = bbox_head.get_targets(gt_bboxes, gt_labels, [2, 1, 512/8, 512/8], img_metas[0]['pad_shape'])


        # debug dataset
        img_metas = data_batch["img_metas"]  # list [[dict], [dict]],
        img = data_batch["img"]     #  tensor, [2, 3, 512, 512]
        gt_bboxes = data_batch["gt_bboxes"]  # list, [[4, 4], [3, 4]]
        gt_labels = data_batch["gt_labels"]  # list, [[4], [3]]
        gt_keypoints = data_batch["gt_keypoints"]  # list, [[4], [3]]
        gt_orients = data_batch["gt_orients"]  # list, [[4], [3]]
        gt_orients_mask = data_batch["gt_orients_mask"]  # list, [[4], [3]]

        img_one = img.data[0][0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, :3]
        img_h, img_w, _ = img_one.shape

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

        center_heatmap_target_one = center_heatmap_target[0].detach().cpu().numpy()
        center_heatmap_target_one = center_heatmap_target_one.max(axis=0)

        keypoints_center_heatmap_one = keypoints_center_heatmap_target[0].detach().cpu().numpy()
        keypoints_center_heatmap_one = keypoints_center_heatmap_one.max(axis=0)

        wh_offset_target_weight_one = wh_offset_target_weight[0].detach().cpu().numpy()
        wh_offset_target_weight_one = wh_offset_target_weight_one.max(axis=0)

        keypoints_offset_target_weight_one = keypoints_offset_target_weight[0].detach().cpu().numpy()
        print("offset_keypoints_target_weight_one:", keypoints_offset_target_weight_one.shape)
        keypoints_offset_target_weight_one = keypoints_offset_target_weight_one.max(axis=0)

        keypoints_wh_target_weight_one = keypoints_wh_target_weight[0].detach().cpu().numpy()
        print("wh_keypoints_target_weight_one:", keypoints_wh_target_weight_one.shape)
        keypoints_wh_target_weight_one = keypoints_wh_target_weight_one.max(axis=0)

        keypoints_wh_target_one = keypoints_wh_target[0].detach().cpu().numpy()
        print("keypoints_wh_target_one:", keypoints_wh_target_one.shape)

        # orients
        if 1:
            orients_center_heatmap_target = target_result['orients_center_heatmap_target']
            orients_target = target_result['orients_target']
            orients_target_weight = target_result['orients_target_weight']

            orients_center_heatmap_target = orients_center_heatmap_target[0].detach().cpu().numpy()
            print("orients_center_heatmap_target:", orients_center_heatmap_target.shape)

            orients_target = orients_target[0].detach().cpu().numpy()
            print("orients_target:", orients_target.shape)

            orients_target_weight = orients_target_weight[0].detach().cpu().numpy()
            print("orients_target_weight:", orients_target_weight.shape)

        mean_one = img_metas.data[0][0]["img_norm_cfg"]["mean"]
        std_one = img_metas.data[0][0]["img_norm_cfg"]["std"]
        file_name = img_metas.data[0][0]["filename"]

        file_path = os.path.join("./", file_name)
        img_ori = cv2.imread(file_path)

        img_one = (img_one * std_one + mean_one)
        img_one = img_one.astype(np.uint8).copy()

        gt_bboxes_one = gt_bboxes.data[0][0]
        gt_labels_one = gt_labels.data[0][0]
        gt_keypoints_one = gt_keypoints.data[0][0]

        for gt_bbox, gt_label in zip(gt_bboxes_one, gt_labels_one):
            x1, y1, x2, y2 = [int(tmp) for tmp in gt_bbox]
            # print((x1 + x2)//2, (y1 + y2)//2)
            cv2.rectangle(img_one, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img_one = cv2.putText(img_one, str(int(gt_label)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        for gt_keypoint in gt_keypoints_one:
            for point in gt_keypoint:
                x, y, flag = point
                if flag > 0:
                    cv2.circle(img_one, (int(x), int(y)), 5, (0, 255, 0), -1)

        # print(wh_keypoints_target_weight_one.shape)
        # print(wh_offset_target_weight_one.shape)
        # non_ids = np.nonzero(wh_keypoints_target_weight_one)
        # for non_id in non_ids:
        #     print(wh_keypoints_target_weight_one[non_id[0], non_id[1]])
        #     print(non_id)

        non_ids = np.nonzero(keypoints_wh_target_weight_one)
        # non_ids = np.nonzero(wh_offset_target_weight_one)
        # print(non_ids)
        for non_id_x, non_id_y in zip(non_ids[0], non_ids[1]):
            # print(keypoints_wh_target_one[:, non_id[0], non_id[1]])
            # print(keypoints_wh_target_one[:, non_id_x, non_id_y])
            print(non_id_x, non_id_y)

        plt.subplot(4, 2, 1)
        plt.imshow(img_ori[:, :, ::-1])
        plt.title("raw img")

        plt.subplot(4, 2, 2)
        plt.imshow(img_one)
        plt.title("input img")
        print("img shape:", img_one.shape)

        plt.subplot(4, 2, 3)
        plt.imshow(center_heatmap_target_one)
        plt.title("center_heatmap_target_one")
        print("center_heatmap_target_one:", center_heatmap_target_one.shape)
        # print(center_heatmap_target_one.shape)

        plt.subplot(4, 2, 4)
        plt.imshow(wh_offset_target_weight_one)
        plt.title("wh_offset_target_weight_one")

        plt.subplot(4, 2, 5)
        plt.imshow(keypoints_center_heatmap_one)
        plt.title("center_heatmap_keypoints_one")

        plt.subplot(4, 2, 6)
        # plt.imshow(keypoints_offset_target_weight_one)
        # plt.title("keypoints_offset_target_weight_one")
        plt.imshow(keypoints_wh_target_weight_one)
        plt.title("keypoints_wh_target_weight_one")

        plt.subplot(4, 2, 7)
        plt.imshow(orients_center_heatmap_target.max(axis=0))
        plt.title("orients_center_heatmap_target")

        plt.subplot(4, 2, 8)
        plt.imshow(orients_target_weight.max(axis=0))
        plt.title("orients_target_weight")

        plt.show()

        exit(1)


if __name__ == "__main__":
    print("Start Proc...")
    debug_coco_dataset()
    print("End Proc...")