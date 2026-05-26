# Copyright (c) OpenMMLab. All rights reserved.
# ============================================================
# 【FCOSHead：FCOS 检测头】
#
# 结构（每个 FPN 层级共享同一组权重）：
#   输入特征 x: (N, 256, H_l, W_l)
#     ↓ 4层 cls conv（3×3, GN, ReLU） → cls_feat: (N, 256, H_l, W_l)
#     ↓ 4层 reg conv（3×3, GN, ReLU） → reg_feat: (N, 256, H_l, W_l)
#     ├─ conv_cls（1×1）→ cls_score:  (N, num_classes, H_l, W_l)  sigmoid 激活
#     ├─ conv_reg（1×1）→ bbox_pred:  (N, 4, H_l, W_l)   exp 激活后为 (l,t,r,b) 像素距离
#     └─ conv_centerness（由 reg_feat 或 cls_feat）→ centerness: (N, 1, H_l, W_l)
#
# 【损失函数】
#   loss_cls:        Focal Loss（处理正负样本极度不均衡，α=0.25, γ=2.0）
#   loss_bbox:       IoU Loss（直接优化 IoU，比 L1 Loss 对大小更鲁棒）
#   loss_centerness: Binary Cross Entropy（监督中心度分支）
#
# 【FPN 层级 ↔ regress_range 对应关系（默认配置）】
#   P3 (stride=8):   regress_range = [-1, 64]   负责预测小目标
#   P4 (stride=16):  regress_range = [64, 128]
#   P5 (stride=32):  regress_range = [128, 256]
#   P6 (stride=64):  regress_range = [256, 512]
#   P7 (stride=128): regress_range = [512, INF] 负责预测大目标
# ============================================================
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Scale
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from ..utils import multi_apply
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@MODELS.register_module()
class FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (Sequence[int] or Sequence[Tuple[int, int]]): Strides of points
            in multiple feature levels. Defaults to (4, 8, 16, 32, 64).
        regress_ranges (Sequence[Tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling.
            Defaults to False.
        center_sample_radius (float): Radius of center sampling.
            Defaults to 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets with
            FPN strides. Defaults to False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Defaults to False.
        conv_bias (bool or str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Defaults to "auto".
        loss_cls (:obj:`ConfigDict` or dict): Config of classification loss.
        loss_bbox (:obj:`ConfigDict` or dict): Config of localization loss.
        loss_centerness (:obj:`ConfigDict`, or dict): Config of centerness
            loss.
        norm_cfg (:obj:`ConfigDict` or dict): dictionary to construct and
            config norm layer.  Defaults to
            ``norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)``.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict]): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 norm_on_bbox: bool = False,
                 centerness_on_reg: bool = False,
                 loss_cls: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        self.loss_centerness = MODELS.build(loss_centerness)

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(
            self, x: Tuple[Tensor]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: A tuple of each level outputs.

            - cls_scores (list[Tensor]): Box scores for each scale level, \
            each is a 4D-tensor, the channel number is \
            num_points * num_classes.
            - bbox_preds (list[Tensor]): Box energies / deltas for each \
            scale level, each is a 4D-tensor, the channel number is \
            num_points * 4.
            - centernesses (list[Tensor]): centerness for each scale level, \
            each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, x, self.scales, self.strides)

    def forward_single(self, x: Tensor, scale: Scale,
                       stride: int) -> Tuple[Tensor, Tensor, Tensor]:
        """单层 FPN 特征的前向推理。

        【输入输出 shape（以 P3 层为例，stride=8，输入图 800×1333）】
          x:          (N, 256, 100, 167)    ← FPN P3 特征
          cls_feat:   (N, 256, 100, 167)    ← 经过 4 个 3×3 conv（分类分支）
          reg_feat:   (N, 256, 100, 167)    ← 经过 4 个 3×3 conv（回归分支）
          cls_score:  (N, num_classes, 100, 167)  ← 1×1 conv，sigmoid 后为类别概率
          bbox_pred:  (N, 4, 100, 167)       ← exp 后为 (l,t,r,b) 像素距离（stride 未归一化）
          centerness: (N, 1, 100, 167)       ← sigmoid 后为 [0,1] 中心度

        norm_on_bbox=True 时，bbox_pred 存储归一化后的值（÷stride），
        推理阶段乘回 stride 还原到像素单位。

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj:`mmcv.cnn.Scale`): 可学习缩放因子，对不同层级 bbox 幅度进行校正。
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness
            predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        # centerness_on_reg=True 时从回归分支提取中心度（效果更好），否则从分类分支
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            # bbox_pred needed for gradient computation has been modified
            # by F.relu(bbox_pred) when run with PyTorch 1.10. So replace
            # F.relu(bbox_pred) with bbox_pred.clamp(min=0)
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                bbox_pred *= stride  # 推理时乘回 stride，恢复像素单位
        else:
            bbox_pred = bbox_pred.exp()  # exp 保证距离值为正
        return cls_score, bbox_pred, centerness

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """基于预测特征计算 FCOS 三路损失。

        【关键 shape 变化】
          all_level_points: list of (H_l*W_l, 2)，每个特征点的 (x, y) 坐标
          labels:           list of (H_l*W_l*N,)，每个点的类别标签
          bbox_targets:     list of (H_l*W_l*N, 4)，每个点的 (l,t,r,b) 目标值

          flatten 后（所有层级、所有图像合并）：
          flatten_cls_scores:  (sum_l(H_l*W_l) * N, num_classes)
          flatten_bbox_preds:  (sum_l(H_l*W_l) * N, 4)
          flatten_centerness:  (sum_l(H_l*W_l) * N,)

          pos_inds: 正样本索引（在 GT box 内且 regress_range 合适的特征点）
          pos_bbox_preds:   (num_pos, 4)  正样本回归预测
          pos_bbox_targets: (num_pos, 4)  正样本回归目标

        【损失计算细节】
          loss_cls:        Focal Loss，所有点参与，avg_factor=num_pos（有效正样本数）
          loss_bbox:       IoU Loss，仅正样本参与，权重为 centerness_target（中心点权重更大）
          loss_centerness: BCE，仅正样本参与

        Args:
            cls_scores (list[Tensor]): 各 FPN 层的分类得分，每项 (N, num_classes, H_l, W_l)
            bbox_preds (list[Tensor]): 各 FPN 层的回归预测，每项 (N, 4, H_l, W_l)
            centernesses (list[Tensor]): 各 FPN 层的中心度，每项 (N, 1, H_l, W_l)
            batch_gt_instances: GT 实例列表，每项含 bboxes (K,4) 和 labels (K,)
            batch_img_metas: 图像元信息（尺寸、scale_factor 等）

        Returns:
            dict[str, Tensor]: 含 loss_cls / loss_bbox / loss_centerness 的字典
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # 生成各层级特征图上所有点的 (x, y) 坐标（特征图坐标 × stride）
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        # labels: list of (H_l*W_l*N,), bbox_targets: list of (H_l*W_l*N, 4)
        labels, bbox_targets = self.get_targets(all_level_points,
                                                batch_gt_instances)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        # 将 (N, C, H, W) 转为 (N*H*W, C) 形式，便于后续直接取 pos_inds
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        # 合并所有层级 → (sum_l(H_l*W_l)*N, ...)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        # pos_inds: 正样本的索引（label < num_classes 即前景）
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)  # DDP 多卡同步正样本数量
        # Focal Loss 对所有位置计算（包括背景），avg_factor=正样本数
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]    # (num_pos, 4)
        pos_centerness = flatten_centerness[pos_inds]    # (num_pos,)
        pos_bbox_targets = flatten_bbox_targets[pos_inds]  # (num_pos, 4)
        # centerness_target 由 GT box 各边距离的比值计算：sqrt(min/max × min/max)
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)  # (num_pos,)
        # centerness weighted iou loss：以中心度为权重，近中心点的损失贡献更大
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            # decode: 将 (x,y) + (l,t,r,b) → 转为 xyxy 格式的绝对坐标框
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,  # 中心度加权，近中心预测更重要
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    def get_targets(
            self, points: List[Tensor], batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes`` and ``labels``
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_targets_single,
            batch_gt_instances,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_targets_single(
            self, gt_instances: InstanceData, points: Tensor,
            regress_ranges: Tensor,
            num_points_per_lvl: List[int]) -> Tuple[Tensor, Tensor]:
        """为单张图像的所有特征点计算分类和回归目标（FCOS 标签分配策略）。

        【FCOS 标签分配规则】
        一个特征点被分配为正样本，需同时满足：
          条件1：该点在某个 GT box 内部（x1<xs<x2, y1<ys<y2）
          条件2：该点的最大预测距离 max(l,t,r,b) 在当前 FPN 层级的 regress_range 内
        多个 GT box 重叠时：选面积最小的 GT（避免一个点被多个 box 分配）

        如果 center_sampling=True，条件1 收紧为：点需在 GT 中心的 radius×stride 范围内

        【shape 说明（以单张图 num_gts=5，num_points=22223 为例）】
          points:         (22223, 2)   所有层级特征点坐标
          regress_ranges: (22223, 2)   每个点对应的 (min_range, max_range)
          gt_bboxes:      (5, 4) → 广播到 (22223, 5, 4)
          left/right/top/bottom: (22223, 5)  各点到各 GT box 四条边的距离
          bbox_targets:   (22223, 5, 4)      (l,t,r,b)

        返回：
          labels:       (22223,)   每个点的类别标签（背景=num_classes）
          bbox_targets: (22223, 4) 每个正样本点的 (l,t,r,b) 回归目标
        """
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)  # (num_points, num_gts) GT box 面积
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)  # (num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        # 计算每个特征点到每个 GT box 四条边的距离（像素单位）
        left = xs - gt_bboxes[..., 0]    # 点到左边界
        right = gt_bboxes[..., 2] - xs   # 点到右边界
        top = ys - gt_bboxes[..., 1]     # 点到上边界
        bottom = gt_bboxes[..., 3] - ys  # 点到下边界
        # bbox_targets: (num_points, num_gts, 4) — 四条边距离
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets

    def centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """计算中心度目标值。

        中心度衡量特征点与 GT box 中心的距离：
          centerness = sqrt( min(l,r)/max(l,r) × min(t,b)/max(t,b) )

        直觉理解：当特征点正好在 GT box 中心时，l=r, t=b，centerness=1.0
                 越偏离中心，min/max 比值越小，centerness 越趋近 0.0

        推理时将 cls_score 与 centerness 相乘作为最终得分，
        使远离中心的低质量预测被抑制（类似 soft-NMS 的效果）。

        Args:
            pos_bbox_targets (Tensor): 正样本的边界框目标 (num_pos, 4)，
                                       格式为 (l, t, r, b)（到四条边的像素距离）

        Returns:
            Tensor: centerness 目标值 (num_pos,)，值域 [0, 1]
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]   # (num_pos, 2) → (l, r)
        top_bottom = pos_bbox_targets[:, [1, 3]]   # (num_pos, 2) → (t, b)
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
