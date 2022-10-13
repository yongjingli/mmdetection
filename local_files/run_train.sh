# 指导链接
#https://mmdetection.readthedocs.io/zh_CN/latest/1_exist_data_model.html

# 训练的基本流程

# ***************************************************************
# load config
#from mmcv import Config
#cfg = Config.fromfile('./configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')

# modify config
# use load_from to set the path of checkpoints.
#cfg.load_from = 'checkpoints/faster_rcnn_r50_caffe_fpn_mstrain_3x_coco_20210526_095054-1f77628b.pth'

# Set up working dir to save files and logs.
#cfg.work_dir = './tutorial_exps'

# process
# from mmdet.datasets import build_dataset
#from mmdet.models import build_detector
#from mmdet.apis import train_detector
#
#
## Build dataset
#datasets = [build_dataset(cfg.data.train)]
#
## Build the detector
#model = build_detector(cfg.model)
## Add an attribute for visualization convenience
#model.CLASSES = datasets[0].CLASSES
#
## Create work_dir
#mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
#train_detector(model, datasets, cfg, distributed=False, validate=True)
# ***************************************************************


# 不想采用分布式的训练方式，或者只有一块显卡
# python tools/train.py ${CONFIG_FILE}

# 分布式训练
# sh ./tools/dist_train.sh <CONFIG_FILE> <GPU_NUM> [optional arguments]

# centernet
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ..
sh ./tools/dist_train.sh ./checkpoints/centernet_resnet18_dcnv2_140e_coco.py 4