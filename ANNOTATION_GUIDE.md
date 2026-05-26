# 目标检测算法注释阅读指南

本文档为 FCOS、Faster-RCNN、Deformable-DETR 三大经典检测算法的注释阅读路径，
按照**数据流动方向**逐步讲解，帮助你建立从原始图像到最终检测结果的完整认知。

---

## 一、注释覆盖范围总览

| 阶段 | 文件路径 | 内容 |
|------|----------|------|
| 数据加载 | `mmdet/datasets/transforms/loading.py` | 图像读取、标注解析 |
| 数据变换 | `mmdet/datasets/transforms/transforms.py` | Resize、RandomFlip、Pad |
| 数据打包 | `mmdet/datasets/transforms/formatting.py` | PackDetInputs |
| 批预处理 | `mmdet/models/data_preprocessors/data_preprocessor.py` | 归一化、批 Pad |
| FCOS 检测器 | `mmdet/models/detectors/fcos.py` | 算法总述 |
| 单阶段基类 | `mmdet/models/detectors/single_stage.py` | 特征提取流程 |
| FCOS 检测头 | `mmdet/models/dense_heads/fcos_head.py` | 预测 + 损失 + 标签分配 |
| Faster-RCNN 检测器 | `mmdet/models/detectors/faster_rcnn.py` | 算法总述 |
| 两阶段基类 | `mmdet/models/detectors/two_stage.py` | RPN + ROI Head 流程 |
| RPN 检测头 | `mmdet/models/dense_heads/rpn_head.py` | Anchor 预测 + NMS |
| ROI 特征提取 | `mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py` | FPN 层级分配 + ROI Align |
| ROI 检测头 | `mmdet/models/roi_heads/standard_roi_head.py` | 采样 + 特征提取 + 损失 |
| BBox 分类回归头 | `mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py` | FC 分类回归网络 |
| Deformable-DETR 检测器 | `mmdet/models/detectors/deformable_detr.py` | 算法总述 + 前向流程 |
| Transformer 层 | `mmdet/models/layers/transformer/deformable_detr_layers.py` | Encoder/Decoder 细节 |
| COCO 评估 | `mmdet/evaluation/metrics/coco_metric.py` | AP 计算原理 |

---

## 二、推荐阅读步骤

### 第一步：理解数据 Pipeline（所有算法共用）

**目标**：搞清楚训练数据从磁盘到模型输入的全过程。

```
reading order:
1. mmdet/datasets/transforms/loading.py
   ├─ 文件顶部注释：了解 pipeline 全景图
   ├─ LoadAnnotations.__init__  → 参数含义
   ├─ _load_bboxes              → gt_bboxes 的 shape 和格式
   └─ _load_labels              → gt_bboxes_labels 的 shape

2. mmdet/datasets/transforms/transforms.py
   ├─ 文件顶部注释：典型 COCO pipeline 步骤
   ├─ Resize.transform          → 图像和 bbox 的缩放 (scale_factor)
   └─ Pad.transform             → pad 到 32 倍数的原因

3. mmdet/datasets/transforms/formatting.py
   ├─ 文件顶部注释：结果打包为 DetDataSample
   └─ PackDetInputs.transform   → 输出 inputs Tensor 和 data_sample

4. mmdet/models/data_preprocessors/data_preprocessor.py
   ├─ 文件顶部注释：归一化 + 批 Pad 流程
   └─ DetDataPreprocessor.forward → 输出 batch_inputs (N,3,H_batch,W_batch)
```

**关键概念**：
- `img_shape`：Resize 后的尺寸（网络实际处理的尺寸）
- `ori_shape`：原始图像尺寸（推理后坐标要 rescale 回来）
- `scale_factor`：Resize 的缩放比例 `(w_scale, h_scale)`
- `batch_input_shape`：批内统一的 Pad 后尺寸

---

### 第二步：阅读 FCOS（单阶段无锚框检测器）

**目标**：理解如何在每个特征点直接预测框，以及中心度（centerness）的设计动机。

```
reading order:
1. mmdet/models/detectors/fcos.py
   └─ 文件顶部注释：FCOS 算法核心思想概览

2. mmdet/models/detectors/single_stage.py
   ├─ 文件顶部注释：单阶段检测器前向流程
   └─ extract_feat              → Backbone + FPN，详细标注了每层输出 shape

3. mmdet/models/dense_heads/fcos_head.py
   ├─ 文件顶部注释：head 结构、损失函数、FPN层级对应关系
   ├─ forward_single            → 单层 FPN 的预测，shape 变化注释详细
   ├─ _get_targets_single       → ★ 最重要：理解 FCOS 标签分配策略
   ├─ loss_by_feat              → 三路损失的计算，含 centerness 加权技巧
   └─ centerness_target         → centerness 公式及直观理解
```

**重点关注**：
- `forward_single` 中 `exp()` vs `clamp(min=0)` 的差异（`norm_on_bbox` 参数）
- `_get_targets_single` 中 `inside_gt_bbox_mask` 和 `inside_regress_range` 两个条件
- `loss_by_feat` 中 Focal Loss 的 `avg_factor=num_pos`（正样本数均衡）
- 为什么 centerness 可以抑制低质量预测（推理时 `score = cls × centerness`）

---

### 第三步：阅读 Faster-RCNN（两阶段 Anchor 检测器）

**目标**：理解两阶段架构的设计：为什么需要 RPN？ROI Align 如何解决量化误差？

```
reading order:
1. mmdet/models/detectors/faster_rcnn.py
   └─ 文件顶部注释：两阶段流程全景图，含各步骤 shape

2. mmdet/models/detectors/two_stage.py
   ├─ 文件顶部注释：训练/推理调用链
   └─ loss()                    → ★ 两阶段损失计算的核心，注释说明了
                                    RPN 如何将所有 GT 类别设为 0

3. mmdet/models/dense_heads/rpn_head.py
   ├─ 文件顶部注释：Anchor 设置、样本分配规则、后处理流程
   ├─ forward_single            → anchor 分类和回归的 shape 注释
   └─ _predict_by_feat_single   → NMS 后处理细节

4. mmdet/models/roi_heads/roi_extractors/single_level_roi_extractor.py
   ├─ 文件顶部注释：FPN 层级分配规则（按面积分配）
   ├─ map_roi_levels             → ★ 理解 log2 公式：小目标→精细层
   └─ forward                   → ROI Align 操作，输出 (num_rois, 256, 7, 7)

5. mmdet/models/roi_heads/standard_roi_head.py
   ├─ 文件顶部注释：第二阶段职责概述
   └─ _bbox_forward             → shape 变化注释：roi_feats → cls/reg 预测

6. mmdet/models/roi_heads/bbox_heads/convfc_bbox_head.py
   ├─ 文件顶部注释：Shared2FCBBoxHead 网络结构图
   └─ forward                   → 完整 shape 变化：(512,256,7,7) → (512,81)
```

**重点关注**：
- RPN 的 `num_base_priors=3`（3 种长宽比）vs FCOS 的无锚框
- `map_roi_levels` 中的 `finest_scale=56` 设计理由
- `_bbox_forward` 中 `cls_score` 含背景类（`num_classes+1`）
- ROI Align 输出固定 `(7,7)` 的意义（可以用同一个 FC 层处理所有大小的 proposal）

---

### 第四步：阅读 Deformable-DETR（端到端 Transformer 检测器）

**目标**：理解 DETR 系列如何用 Transformer 替代 NMS，以及可变形注意力如何提升效率。

```
reading order:
1. mmdet/models/detectors/deformable_detr.py
   ├─ 文件顶部注释：★ 最重要，完整前向流程图，含所有中间 shape
   ├─ _init_layers              → 理解 query_embedding 的维度为什么是 embed_dims*2
   ├─ pre_transformer           → FPN 特征展平拼接，mask 构建逻辑
   ├─ pre_decoder               → 单阶段 vs 两阶段的 query 初始化差异
   └─ forward_decoder           → with_box_refine 的 iterative refinement

2. mmdet/models/layers/transformer/deformable_detr_layers.py
   ├─ 文件顶部注释：可变形注意力原理，计算复杂度对比
   └─ DeformableDetrTransformerDecoder.forward
      → ★ 核心：逐层更新 reference_points 的逻辑
         注意 .detach() 的作用（防止梯度循环）
```

**重点关注**：
- `query_embedding` 前 256 维是位置信息（query_pos），后 256 维是内容信息（query）
- `reference_points` 的语义：每个 query 在空间中"关注"哪个位置
- `valid_ratios` 的作用：处理 batch 内图像尺寸不同时的 padding 问题
- `with_box_refine=True` 时：为什么在 sigmoid 空间做残差修正（`inverse_sigmoid + delta`）
- 为什么 Deformable-DETR 不需要 NMS：HungarianMatcher 保证一对一匹配

---

### 第五步：阅读性能评估（所有算法共用）

**目标**：理解 COCO AP 的计算原理，能解读实验结果表格。

```
reading order:
1. mmdet/evaluation/metrics/coco_metric.py
   ├─ 文件顶部注释：AP 指标体系和计算原理
   ├─ process                   → 收集每张图的预测结果（bboxes/scores/labels）
   └─ compute_metrics           → ★ COCOeval 调用链 + stats 索引含义
```

**重点关注**：
- `mAP` vs `mAP_50`：前者更严格，需要在多个 IoU 阈值下都有高精度
- `mAP_s/m/l`：理解为什么 FCOS 在小目标（s）上通常优于 Faster-RCNN
- `xyxy2xywh`：坐标格式转换，是 bug 的高发区

---

## 三、算法对比速查

| 维度 | FCOS | Faster-RCNN | Deformable-DETR |
|------|------|-------------|-----------------|
| **类型** | 单阶段 Anchor-Free | 两阶段 Anchor-Based | 端到端 Transformer |
| **正负样本分配** | 点在框内 + regress_range | MaxIoU Assigner | Hungarian Matching（一对一） |
| **预测格式** | (l,t,r,b) 距离 | Delta XYWH 偏移 | 归一化 (cx,cy,w,h) |
| **NMS** | 需要（推理后处理） | 需要（RPN + 第二阶段） | 不需要 |
| **多尺度** | FPN + regress_range | FPN + ROI Align | 多尺度可变形注意力 |
| **关键 shape** | `(N,num_cls,H_l,W_l)` × 5 层 | `(num_rois,81)` 分类 | `(6,N,300,num_cls)` 6 层解码 |
| **损失** | Focal+IoU+BCE | CE+SmoothL1×2 | Focal+L1+GIoU |
| **优势** | 快速、无 anchor 超参 | 精度高、两阶段精化 | 无 NMS、全局感受野 |

---

## 四、关键 Tensor Shape 速查

### FCOS（输入图 800×1333，ResNet50+FPN，80 类 COCO）

```
batch_inputs:     (N, 3, 800, 1333)
FPN P3:           (N, 256, 100, 167)
FPN P4:           (N, 256,  50,  84)
FPN P5:           (N, 256,  25,  42)
FPN P6:           (N, 256,  13,  21)
FPN P7:           (N, 256,   7,  11)

cls_score P3:     (N, 80, 100, 167)
bbox_pred P3:     (N,  4, 100, 167)   ← (l, t, r, b)
centerness P3:    (N,  1, 100, 167)

loss_cls:         scalar  (Focal Loss)
loss_bbox:        scalar  (IoU Loss，centerness 加权)
loss_centerness:  scalar  (BCE)
```

### Faster-RCNN（同上配置）

```
batch_inputs:     (N, 3, 800, 1333)
FPN P2~P5:        (N, 256, H_l, W_l)

── RPN ──
rpn_cls_score P3: (N, 3, 100, 167)   ← 3 anchors × 1 class
rpn_bbox_pred P3: (N, 12, 100, 167)  ← 3 anchors × 4 delta
proposals/img:    ≈2000 (train), 1000 (test)
proposals:        (num_per_img, 5)   [x1,y1,x2,y2,score]

── ROI Head ──
rois:             (all_proposals, 5) [batch_id, x1,y1,x2,y2]
bbox_feats:       (all_proposals, 256, 7, 7)   ROI Align 输出
cls_score:        (all_proposals, 81)           80 类 + 背景
bbox_pred:        (all_proposals, 320)          80 × 4

final detections: (num_det, 5) per image
```

### Deformable-DETR（同上配置，num_queries=300）

```
batch_inputs:     (N, 3, 800, 1333)
FPN 4 levels:     (N, 256, H_l, W_l)  l=1..4

feat_flatten:     (N, ≈22223, 256)    ← 4 层展平拼接
memory:           (N, ≈22223, 256)    ← Encoder 输出

query:            (N, 300, 256)       ← 可学习 object queries
reference_points: (N, 300, 2)         ← 参考点 (cx, cy) ∈ [0,1]

hidden_states:    (6, N, 300, 256)    ← 6 个 Decoder 层输出
cls_scores:       (6, N, 300, 80)     ← 每层解码器的分类预测
bbox_preds:       (6, N, 300, 4)      ← 归一化 (cx,cy,w,h)
```

---

## 五、常见疑问解答

**Q1：FCOS 中 centerness 为什么用 sqrt？**
sqrt 使 centerness 分布更均匀（不用 sqrt 时靠近中心点的值接近 1，但大多数点值接近 0，梯度消失）。

**Q2：ROI Align 中的 spatial_scale 怎么理解？**
`spatial_scale = 1/stride`，将原图坐标映射到特征图坐标。例如 P3 的 stride=8，一个 (x1,y1,x2,y2) 的框在 P3 特征图上对应 (x1/8, y1/8, x2/8, y2/8)。

**Q3：Deformable-DETR 中 valid_ratios 的作用？**
批内不同图像 pad 到同一大小后，有效区域比例不同。`valid_ratios[n,l] = (valid_H/H, valid_W/W)` 用于将参考点坐标从有效区域的归一化坐标转换到特征图坐标，避免 attention 聚焦到 padding 区域。

**Q4：为什么 mAP 使用 10 个 IoU 阈值的平均，而不只用 IoU=0.5？**
单一 IoU 阈值无法反映检测器的定位精度。IoU=0.5 只要求框大致正确，而 mAP（COCO）在 [0.5, 0.55, ..., 0.95] 10 个阈值取平均，高 IoU 阈值（如 0.95）需要近乎完美的定位，迫使模型优化精确位置。

**Q5：FCOS 和 Faster-RCNN 哪个在小目标上更好？**
通常 FCOS 更好，因为 P3（stride=8）提供了更密集的特征点，且无需 anchor 的 aspect ratio 限制。Faster-RCNN 在 P2（stride=4）上也有 RPN，但 ROI Align 步骤引入了轻微的量化误差。

---

## 六、延伸阅读

如果你已经理解了上述内容，推荐继续阅读：

- **FPN 实现**：`mmdet/models/necks/fpn.py`（理解多尺度特征融合）
- **HungarianMatcher**：`mmdet/models/task_modules/assigners/hungarian_assigner.py`（DETR 系列的一对一匹配）
- **ATSS Assigner**：`mmdet/models/task_modules/assigners/atss_assigner.py`（自适应正样本分配，解决 anchor 超参问题）
- **AnchorFreeHead**：`mmdet/models/dense_heads/anchor_free_head.py`（FCOS Head 的父类）
- **BBoxHead**：`mmdet/models/roi_heads/bbox_heads/bbox_head.py`（ConvFCBBoxHead 的父类，含损失计算）
