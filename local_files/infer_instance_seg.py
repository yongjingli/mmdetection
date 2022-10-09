import sys
sys.path.insert(0, "../")

import os
import cv2
import mmcv
import torch
import numpy as np
import matplotlib.pyplot as plt
from mmcv.runner import load_checkpoint
from matplotlib.patches import Polygon

from mmdet.apis import inference_detector, show_result_pyplot
from mmdet.models import build_detector
from mmdet.core.visualization.image import imshow_det_bboxes
from mmdet.core.mask.structures import bitmap_to_polygon

def demo_infer_instance_seg():
    # Choose to use a config and initialize the detector
    config = '../configs/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = '../checkpoints/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth'

    # Set the device to be used for evaluation
    device='cuda:6'

    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    # Use the detector to do inference
    img = '../demo/demo.jpg'
    result = inference_detector(model, img)

    # Let's plot the result
    show_result_pyplot(model, img, result, score_thr=0.3)


def show_result(class_name,
                img,
                result,
                score_thr=0.3,
                bbox_color=(72, 101, 241),
                text_color=(72, 101, 241),
                mask_color=None,
                thickness=2,
                font_size=13,
                win_name='',
                show=False,
                wait_time=0,
                out_file=None):
    """Draw `result` over `img`.

    Args:
        img (str or Tensor): The image to be displayed.
        result (Tensor or tuple): The results to draw over `img`
            bbox_result or (bbox_result, segm_result).
        score_thr (float, optional): Minimum score of bboxes to be shown.
            Default: 0.3.
        bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
           The tuple of color should be in BGR order. Default: 'green'
        text_color (str or tuple(int) or :obj:`Color`):Color of texts.
           The tuple of color should be in BGR order. Default: 'green'
        mask_color (None or str or tuple(int) or :obj:`Color`):
           Color of masks. The tuple of color should be in BGR order.
           Default: None
        thickness (int): Thickness of lines. Default: 2
        font_size (int): Font size of texts. Default: 13
        win_name (str): The window name. Default: ''
        wait_time (float): Value of waitKey param.
            Default: 0.
        show (bool): Whether to show the image.
            Default: False.
        out_file (str or None): The filename to write the image.
            Default: None.

    Returns:
        img (Tensor): Only if not `show` or `out_file`
    """
    img = mmcv.imread(img)
    img = img.copy()
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
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False

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
            cv2.rectangle(img, ([bbox_int[0], bbox_int[1]]), ([bbox_int[2], bbox_int[3]]),
                          (255, 0, 255), 2)

        # gen mask image
        mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        polygons = []
        for i, segm in enumerate(segms):
            contours, _ = bitmap_to_polygon(segm)
            polygons += [Polygon(c) for c in contours]
            for contour in contours:
                cv2.polylines(img=img, pts=[contour], isClosed=True, color=(0, 0, 255), thickness=3)
                cv2.fillPoly(mask_img, [contour], 255)

        plt.imshow(mask_img)
        plt.show()

    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        class_names=class_name, #class_names=self.CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file)

    if not (show or out_file):
        return img

def show_imgs_instance_seg():
    # Choose to use a config and initialize the detector
    config = '../configs/queryinst/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = '../checkpoints/queryinst_r101_fpn_300_proposals_crop_mstrain_480-800_3x_coco_20210904_153621-76cce59f.pth'

    # Set the device to be used for evaluation
    device='cuda:6'

    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()

    root = "/home/dev/data_disk/liyj/data/collect_data/reid/custom_front_rear_camera_20220324_test/bounding_box_test/p00041/front_a0cf1778-28b2-11ec-92ea-000c293913c8"
    img_names = [name for name in os.listdir(root) if name.endswith(".png")]
    for img_name in img_names:
        # img_path = os.path.join(root, img_name)
        img_path = "/home/dev/data_disk/liyj/data/collect_data/reid/label_data/20220322/camera_body_right_up_rgb/p00037/0699793d-28b2-11ec-92ea-000c293913c8/png/13149520-28b2-11ec-92ea-000c293913c8.png"


        # Use the detector to do inference
        result = inference_detector(model, img_path)

        score_thr = 0.3
        title = 'result'
        wait_time = 0
        palette = None
        if hasattr(model, 'module'):
            model = model.module
        class_name = model.CLASSES

        show_result(class_name,
                    img_path,
                    result,
                    score_thr=score_thr,
                    bbox_color=palette,
                    text_color=(200, 200, 200),
                    mask_color=palette,
                    thickness=2,
                    font_size=13,
                    win_name=title,
                    show=True,
                    wait_time=wait_time,
                    out_file=None)
        exit(1)

if __name__ == "__main__":
    print("Start...")
    demo_infer_instance_seg()
    # show_imgs_instance_seg()
    print("Done...")
