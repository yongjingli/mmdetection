# 记录训练orient的基本信息
if orientation >= 0:
    # orients[k] = orientation/360.0
    if flipped:
        # degree = (72 - degree) % 72
        orientation = (360 - orientation) % 360

    # orient is front or back
    # left
    if orientation >= 45 and orientation < 135:
        orients_front[k][1] = 1
    # front
    elif orientation >= 135 and orientation < 225:
        orients_front[k][2] = 1
    # right
    elif orientation >= 225 and orientation < 315:
        orients_front[k][3] = 1
    # back
    else:
        orients_front[k][0] = 1

    # convert to [-pi, pi], 将角度的范围转换为kitti数据的格式
    if orientation > 180:
        orientation = orientation - 360

    orientation = -orientation
    orientation = orientation - 90
    if orientation > 180:
        orientation = orientation - 360
    if orientation < -180:
        orientation = orientation + 360

    orientation = orientation / 180 * math.pi

    orients[k][0] = math.sin(orientation)
    orients[k][1] = math.cos(orientation)
    orients_mask[k] = 1

# if flipped:
#     # degree = (72 - degree) % 72
#     orientation = (360 - orientation) % 360
# if orient > 180:
#     orient = orient - 360
#
# orient = -orient
# orient = orient - 90
# if orient > 180:
#     orient = orient - 360
# if orient < -180:
#     orient = orient + 360
#
# orient = orient / 180 * math.pi



