# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# 【SingleRoIExtractor：FPN 多尺度 ROI 特征提取器】
#
# 核心：ROI Align（Mask R-CNN 论文提出）
#   对每个 proposal（无论大小），双线性插值到固定大小 (7×7 或 14×14)
#   相比 ROI Pooling，消除了量化误差，对精确位置敏感的任务效果更好
#
# 【FPN 层级分配规则（map_roi_levels）】
#   根据 proposal 面积大小，选择合适的 FPN 层级：
#   level = floor( log2( sqrt(area) / 56 ) ) + 2（finest_scale=56，P2 起始）
#   小目标（area≈32²）→ P2（stride=4）精细特征
#   大目标（area≈256²）→ P5（stride=32）语义特征
#   分配公式等价于：
#     area < 112²  → P2
#     area < 224²  → P3
#     area < 448²  → P4
#     area >= 448² → P5
#
# 【forward 输出 shape】
#   rois:      (num_rois, 5)  [batch_id, x1, y1, x2, y2]
#   roi_feats: (num_rois, out_channels, output_h, output_w)
#              = (num_rois, 256, 7, 7)  对于 bbox head
# ============================================================
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptMultiConfig
from .base_roi_extractor import BaseRoIExtractor


@MODELS.register_module()
class SingleRoIExtractor(BaseRoIExtractor):
    """Extract RoI features from a single level feature map.

    If there are multiple input feature levels, each RoI is mapped to a level
    according to its scale. The mapping rule is proposed in
    `FPN <https://arxiv.org/abs/1612.03144>`_.

    Args:
        roi_layer (:obj:`ConfigDict` or dict): Specify RoI layer type and
            arguments.
        out_channels (int): Output channels of RoI layers.
        featmap_strides (List[int]): Strides of input feature maps.
        finest_scale (int): Scale threshold of mapping to level 0.
            Defaults to 56.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 roi_layer: ConfigType,
                 out_channels: int,
                 featmap_strides: List[int],
                 finest_scale: int = 56,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            roi_layer=roi_layer,
            out_channels=out_channels,
            featmap_strides=featmap_strides,
            init_cfg=init_cfg)
        self.finest_scale = finest_scale

    def map_roi_levels(self, rois: Tensor, num_levels: int) -> Tensor:
        """根据 RoI 面积大小将其分配到对应的 FPN 层级。

        分配规则（finest_scale=56 默认）：
          scale = sqrt(roi_w × roi_h)       ← RoI 的几何平均边长
          target_lvl = floor(log2(scale/56)) ← 以 56 为基准，每翻倍升一级

          scale < 112  → level 0（最精细，P2/P3）
          112 <= scale < 224 → level 1
          224 <= scale < 448 → level 2
          scale >= 448 → level 3（最粗糙，P5）

        【示例】
          一个 32×32 的小目标：scale=32，target_lvl = floor(log2(32/56)) ≈ -1，clamp→0
          一个 512×512 的大目标：scale=512，target_lvl = floor(log2(512/56)) ≈ 3

        Args:
            rois (Tensor): Input RoIs, shape (k, 5)  [batch_id, x1, y1, x2, y2]
            num_levels (int): FPN 层级数（通常为 4，P2~P5）

        Returns:
            Tensor: 每个 RoI 的 FPN 层级索引 (0-based), shape (k,)
        """
        scale = torch.sqrt(
            (rois[:, 3] - rois[:, 1]) * (rois[:, 4] - rois[:, 2]))
        target_lvls = torch.floor(torch.log2(scale / self.finest_scale + 1e-6))
        target_lvls = target_lvls.clamp(min=0, max=num_levels - 1).long()
        return target_lvls

    def forward(self,
                feats: Tuple[Tensor],
                rois: Tensor,
                roi_scale_factor: Optional[float] = None):
        """对每个 RoI 从对应 FPN 层级提取固定大小的特征。

        【shape 变化】
          feats:     tuple of (N, 256, H_l, W_l)  FPN 多层特征（P2~P5）
          rois:      (num_rois, 5)  [batch_id, x1, y1, x2, y2]  绝对坐标（模型输入尺寸）
          roi_feats: (num_rois, 256, 7, 7)        每个 RoI 的固定大小特征

        ROI Align 操作：
          对每个 RoI 在对应 FPN 层的特征图上进行双线性采样
          将任意大小的 proposal 映射到统一的 (7, 7) 输出
          output_size=(7,7), spatial_scale=1/stride_of_level

        Args:
            feats (Tuple[Tensor]): Multi-scale features, each (N, C, H_l, W_l)
            rois (Tensor): (num_rois, 5)  [batch_id, x1, y1, x2, y2]
            roi_scale_factor (Optional[float]): RoI 扩展因子（通常不使用）

        Returns:
            Tensor: (num_rois, out_channels, output_h, output_w) RoI 特征
        """
        # convert fp32 to fp16 when amp is on
        rois = rois.type_as(feats[0])
        out_size = self.roi_layers[0].output_size       # 通常 (7, 7)
        num_levels = len(feats)
        # 初始化全零输出，后续按层级填充
        roi_feats = feats[0].new_zeros(
            rois.size(0), self.out_channels, *out_size)  # (num_rois, 256, 7, 7)

        # TODO: remove this when parrots supports
        if torch.__version__ == 'parrots':
            roi_feats.requires_grad = True

        if num_levels == 1:
            if len(rois) == 0:
                return roi_feats
            return self.roi_layers[0](feats[0], rois)

        # 根据 RoI 尺寸分配对应 FPN 层级（小目标→精细层，大目标→粗糙层）
        target_lvls = self.map_roi_levels(rois, num_levels)

        if roi_scale_factor is not None:
            rois = self.roi_rescale(rois, roi_scale_factor)

        for i in range(num_levels):
            mask = target_lvls == i
            inds = mask.nonzero(as_tuple=False).squeeze(1)
            if inds.numel() > 0:
                rois_ = rois[inds]
                roi_feats_t = self.roi_layers[i](feats[i], rois_)  # ROI Align
                roi_feats[inds] = roi_feats_t
            else:
                # 当某一层级没有 RoI 时，补充 0 梯度避免 DDP 卡死
                # （DDP 要求所有 GPU 的计算图结构一致）
                roi_feats += sum(
                    x.view(-1)[0]
                    for x in self.parameters()) * 0. + feats[i].sum() * 0.
        return roi_feats  # (num_rois, 256, 7, 7)
