# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# 【FCOS：Fully Convolutional One-Stage Object Detection】
# 论文：Tian et al., ICCV 2019  https://arxiv.org/abs/1904.01355
#
# 核心思想：
#   - 无锚框（Anchor-Free），在每个 FPN 特征点直接预测边界框
#   - 每个特征点预测：
#       ① cls_score：该位置属于每个类别的概率（sigmoid）
#       ② bbox_pred：到 GT box 四条边的距离 (l, t, r, b)
#       ③ centerness：中心度分支，抑制远离目标中心的低质量预测
#   - 正负样本分配：
#       特征点在 GT box 内 + 预测距离在 FPN 层的 regress_range 内 → 正样本
#       多个 GT box 重叠时，选择面积最小的 → 避免 ambiguity
#   - 中心度（centerness）：
#       centerness = sqrt(min(l,r)/max(l,r) × min(t,b)/max(t,b))
#       越靠近 GT 中心，centerness 越接近 1
#       推理时 score = cls_score × centerness，抑制偏移预测
#
# 网络结构（SingleStageDetector）：
#   backbone（ResNet-50/101）→ neck（FPN）→ FCOSHead
#
# FCOS 本身很简单，仅继承 SingleStageDetector 并指定 FCOSHead。
# ============================================================
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class FCOS(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone config.
        neck (:obj:`ConfigDict` or dict): The neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of FCOS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of FCOS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or list[:obj:`ConfigDict`] or dict or
            list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
