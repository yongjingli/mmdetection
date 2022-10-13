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
    cfg = Config.fromfile('./checkpoints/centernet_resnet18_dcnv2_140e_coco.py')

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

        target_result, avg_factor = bbox_head.get_targets(gt_bboxes.data[0], gt_labels.data[0], [2, 1, int(512/8), int(512/8)], img_metas.data[0][0]['pad_shape'])
        center_heatmap_target = target_result['center_heatmap_target']
        wh_target = target_result['wh_target']
        offset_target = target_result['offset_target']
        wh_offset_target_weight = target_result['wh_offset_target_weight']

        center_heatmap_target_one = center_heatmap_target[0].detach().cpu().numpy()
        center_heatmap_target_one = center_heatmap_target_one.max(axis=0)

        wh_offset_target_weight_one = wh_offset_target_weight[0].detach().cpu().numpy()
        wh_offset_target_weight_one = wh_offset_target_weight_one.max(axis=0)


        img_one = img.data[0][0].detach().cpu().numpy().transpose(1, 2, 0)[:, :, :3]
        mean_one = img_metas.data[0][0]["img_norm_cfg"]["mean"]
        std_one = img_metas.data[0][0]["img_norm_cfg"]["std"]
        file_name = img_metas.data[0][0]["filename"]

        file_path = os.path.join("./", file_name)
        img_ori = cv2.imread(file_path)

        img_one = (img_one * std_one + mean_one)
        img_one = img_one.astype(np.uint8).copy()

        gt_bboxes_one = gt_bboxes.data[0][0]
        gt_labels_one = gt_labels.data[0][0]

        for gt_bbox, gt_label in zip(gt_bboxes_one, gt_labels_one):
            x1, y1, x2, y2 = [int(tmp) for tmp in gt_bbox]
            cv2.rectangle(img_one, (x1, y1), (x2, y2), (255, 0, 0), 2)
            img_one = cv2.putText(img_one, str(int(gt_label)), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

        plt.subplot(2, 2, 1)
        plt.imshow(img_ori[:, :, ::-1])
        plt.title("raw img")

        plt.subplot(2, 2, 2)
        plt.imshow(img_one)
        plt.title("input img")

        plt.subplot(2, 2, 3)
        plt.imshow(center_heatmap_target_one)
        plt.title("center_heatmap_target_one")
        # print(center_heatmap_target_one.shape)

        plt.subplot(2, 2, 4)
        plt.imshow(wh_offset_target_weight_one)
        plt.title("wh_offset_target_weight_one")
        plt.show()

        exit(1)


if __name__ == "__main__":
    print("Start Proc...")
    debug_coco_dataset()
    print("End Proc...")