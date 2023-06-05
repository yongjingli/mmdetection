import cv2
import numpy as np
import math
import torch.nn as nn
import torch
import os


def vis_annot_info(img, box, angle_theta, kps):
    # draw box
    color = (0, 255, 0)
    thinkness = 1
    x1, y1, w, h = box[:4]
    x2 = x1 + w
    y2 = y1 + h
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thinkness)

    # draw angle
    color = (0, 0, 255)
    if angle_theta >= 0:
        c_x = int(x1 + w * 1/2)
        c_y = int(y1 + h * 1/3)
        c_r = int(min(w, h) * 0.5)

        angle_theta = 90 + angle_theta
        if angle_theta > 360:
            angle_theta = angle_theta - 360
        x1 = c_x + c_r * math.cos(angle_theta * np.pi / 180.0)
        y1 = c_y - c_r * math.sin(angle_theta * np.pi / 180.0)
        cv2.line(img, (int(x1), int(y1)), (int(c_x), int(c_y)), color, thickness=thinkness)
        cv2.arrowedLine(img, (int(c_x), int(c_y)), (int(x1), int(y1)), color, 3, 2, 0, 0.3)

    # draw kps
    if kps is not None:
        radius = 5
        kps = np.array(kps)
        kps = kps.reshape(-1, 3)
        colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0)]
        for kp in kps:
            x, y, vis = [int(tmp) for tmp in kp]
            color = colors[int(vis)]
            cv2.circle(img, (x, y), radius, color, -1)
    return img


