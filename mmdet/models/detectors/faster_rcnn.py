# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# 【Faster R-CNN：两阶段经典检测器】
# 论文：Ren et al., NeurIPS 2015  https://arxiv.org/abs/1506.01497
#
# 核心思想：
#   阶段一（RPN）：在 FPN 特征图上密集预测候选框（region proposals）
#   阶段二（RoI Head）：对候选框提取 RoI 特征，精细分类和回归
#
# 整体流程：
#   batch_inputs (N,3,H,W)
#     ↓ backbone + neck（ResNet + FPN）
#   多尺度特征 P2~P5
#     ↓ RPNHead（每个 FPN 层级）
#   anchor 分类 (前景/背景) + anchor 回归
#     ↓ NMS（每张图保留约 1000 个 proposals，测试时 300 个）
#   proposals: (num_per_img, 5)  [x1,y1,x2,y2,score]
#     ↓ SingleRoIExtractor（ROI Align，将任意大小 proposal 裁剪为 7×7）
#   roi_feats: (num_rois, 256, 7, 7)
#     ↓ Shared2FCBBoxHead（2 个全连接层）
#   cls_score: (num_rois, num_classes+1)  + bbox_pred: (num_rois, 4×num_classes)
#     ↓ 解码 + 类 NMS
#   最终检测结果: (num_det, 5)  [x1,y1,x2,y2,score] + labels
#
# Faster R-CNN 本身只是 TwoStageDetector 的简单包装，核心逻辑在 two_stage.py。
# ============================================================
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class FasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
