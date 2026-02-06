# EfficientSAM3 Decoder 蒸馏与微调完整方案

## 目录

- [1. 方案概述](#1-方案概述)
- [2. 训练阶段设计](#2-训练阶段设计)
- [3. 数据集准备](#3-数据集准备)
- [4. Student Decoder 架构设计](#4-student-decoder-架构设计)
- [5. 蒸馏策略详解](#5-蒸馏策略详解)
- [6. 训练配置](#6-训练配置)
- [7. 实施步骤](#7-实施步骤)
- [8. 评估与验证](#8-评估与验证)
- [9. 预期结果](#9-预期结果)

---

## 1. 方案概述

### 1.1 目标

将 EfficientSAM3 的重型 Decoder（29.32M 参数，111 MB ONNX）蒸馏为轻量级版本，在保持文本引导检测能力的同时：

- **参数减少**: 29.32M → 7.73M（**-74%**）
- **ONNX 大小**: 111 MB → ~31 MB（**-72%**）
- **推理速度**: 提升 2.5-3.0×
- **精度损失**: 控制在 2-4 mAP points 以内

### 1.2 训练范式

参考 **EfficientSAM3 原作者的三阶段渐进式蒸馏策略**：

```
EfficientSAM3 原版训练范式:
┌─────────────────────────────────────────────────────────────┐
│ Stage 1: Encoder Distillation (Image + Text)               │
│   - Image Encoder: SAM3 ViT-H → RepViT/TinyViT/EfficientViT│
│   - Text Encoder: SAM3 Text → MobileCLIP S0/S1/2-L         │
│   - Dataset: 1% SA-1B + 1% Recap-DataComp-1B                │
│   - 方法: Feature-level MSE + Cosine Similarity             │
├─────────────────────────────────────────────────────────────┤
│ Stage 2: Memory Distillation (Video Tracking)              │
│   - Perceiver-based Memory Module                           │
│   - Dataset: SA-V (Video)                                   │
│   - 方法: Memory-conditioned Mask Prediction                │
├─────────────────────────────────────────────────────────────┤
│ Stage 3: End-to-End Fine-Tuning (Concept Segmentation)     │
│   - Joint Optimization: Encoder + Memory + Decoder          │
│   - Dataset: SAM3 Official (SA-Co Gold+Silver)             │
│   - 方法: Task-specific Fine-tuning                         │
└─────────────────────────────────────────────────────────────┘
```

**本方案（Decoder 蒸馏）对应 Stage 1.5**：

在 Stage 1（Encoder 已蒸馏）和 Stage 2（Memory 模块）之间，专门针对 **Text-Grounding Decoder** 进行蒸馏。

---

## 2. 训练阶段设计

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 0: 准备阶段（Prerequisites）                          │
│   - 使用 Stage 1 已训练的 Image Encoder + Text Encoder      │
│   - 冻结 Encoder 权重                                        │
│   - 保存 Teacher Decoder 的中间特征                         │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: Teacher Feature Extraction（1-2 天）              │
│   - 在训练集上运行 Teacher Decoder                          │
│   - 保存所有中间层特征 (memory, hs, attention maps)         │
│   - 保存到磁盘供 Student 训练使用                           │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 2: Student Warm-up（5-10 epochs）                    │
│   - 纯监督学习，不使用蒸馏损失                              │
│   - Loss: Task Loss (Detection + Segmentation)             │
│   - 目标: 快速收敛到合理的检测性能                          │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 3: Feature Distillation（20-30 epochs）              │
│   - 加载预存的 Teacher features                             │
│   - Loss: Task Loss + Feature MSE + Logits KL + Attention   │
│   - 权重调度: 逐步降低蒸馏损失权重                          │
└────────────────┬────────────────────────────────────────────┘
                 ▼
┌─────────────────────────────────────────────────────────────┐
│ Phase 4: Task Fine-tuning（10 epochs）                     │
│   - 降低蒸馏权重，强化 Task Loss                            │
│   - 在 SA-Co Gold+Silver 上进行文本对齐微调                 │
│   - 最终优化检测精度                                        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 训练时长估算

| 阶段 | Epochs | 数据集大小 | 单 epoch 时间 (A100) | 总时长 |
|------|--------|-----------|---------------------|--------|
| Phase 1: Feature Extraction | 1 pass | 1% SA-1B (~110K) | ~3-4h | **3-4h** |
| Phase 2: Warm-up | 10 | 1% SA-1B | ~3h | **30h** |
| Phase 3: Distillation | 30 | 1% SA-1B | ~3.5h | **105h** |
| Phase 4: Fine-tuning | 10 | SA-Co (~150K) | ~1h | **10h** |
| **总计** | **51** | - | - | **~148h (6.2天)** |

---

## 3. 数据集准备

### 3.1 训练数据集

参考 EfficientSAM3 原作者的 Stage 1 策略，使用以下数据集：

#### **A. SA-1B (1% Subset)** — 主要训练数据

**用途**: Phase 2 Warm-up + Phase 3 Distillation

**数据规模**:
- 原始 SA-1B: 11M 图像，1B+ masks
- 1% subset: **~110K 图像**

**下载方法**:
```bash
# 1. 下载 1% subset 文件列表
# EfficientSAM3 提供的 1% subset 文件列表
cat data/sa-1b-1p.txt  # 包含 1% 数据的 URL 列表

# 2. 批量下载
bash data/download_sa1b_subset.sh

# 3. 重组文件结构
python data/reorg_sa1b.py
```

**最终结构**:
```
data/sa-1b/
├── images/
│   ├── train/
│   │   ├── sa_000000.jpg
│   │   ├── sa_000001.jpg
│   │   └── ...
│   └── val/
│       └── ...
└── annotations/
    ├── train/
    │   ├── sa_000000.json
    │   └── ...
    └── val/
        └── ...
```

**Annotation 格式**:
```json
{
  "image": {"image_id": 0, "width": 1500, "height": 2250},
  "annotations": [
    {
      "bbox": [x, y, w, h],  // COCO format
      "area": 12345,
      "segmentation": {...},   // RLE or polygon
      "predicted_iou": 0.85,
      "stability_score": 0.92
    }
  ]
}
```

---

#### **B. Recap-DataComp-1B (1% Subset)** — 文本理解增强

**用途**: Phase 3 Distillation（增强文本-图像对齐）

**数据规模**:
- 原始 Recap-DataComp-1B: 1B 图像-文本对
- 1% subset: **~10M 样本**

**下载方法**:
```bash
# 下载 parquet 文件
python data/download_datacomp.py --subset 0.01 --output data/recap_subset
```

**数据格式** (Parquet):
```
columns: ['image_path', 'caption', 'url', 'similarity']
```

**使用策略**:
- 随机采样 10% Recap + 90% SA-1B
- 对 Recap 样本，使用 caption 作为 text prompt
- 对 SA-1B 样本，使用自动生成的 noun phrases（见 3.1.C）

---

#### **C. SA-Co Gold + Silver** — 概念分割微调

**用途**: Phase 4 Fine-tuning

**数据规模**:
- SA-Co Gold: ~25K images, ~150K text annotations
- SA-Co Silver: ~125K images (额外标注)

**下载方法**:
```bash
# SA-Co Gold
git clone https://huggingface.co/datasets/facebook/SACo-Gold
cd SACo-Gold
git lfs pull

# SA-Co Silver (如果需要)
git clone https://huggingface.co/datasets/facebook/SACo-Silver
```

**Annotation 格式**:
```json
{
  "image_id": "sa_123456",
  "width": 1024,
  "height": 768,
  "annotations": [
    {
      "category_name": "person",        // 文本标签
      "bbox": [x, y, w, h],
      "segmentation": {...},
      "noun_phrase": "person wearing red shirt"  // 详细描述
    }
  ]
}
```

**关键特性**:
- 高质量的文本-物体对齐标注
- 包含 noun phrases（短语级别描述）
- 适合 text-grounding 任务

---

### 3.2 验证数据集

#### **A. COCO 2017 val** — 标准检测评估

**用途**: 每个 epoch 后的快速验证

```bash
# 下载
bash data/download_coco.sh
```

**评估指标**: mAP, mAP@50, mAP@75

---

#### **B. LVIS v1.0** — Long-tail 类别评估

**用途**: 评估 open-vocabulary 能力

```bash
bash data/download_lvis.sh
```

**评估指标**: mAP, mAP_rare, mAP_common, mAP_frequent

---

#### **C. SA-Co VEval** — 文本引导分割基准

**用途**: 评估 text-grounding 质量

- 5184 noun phrases
- 与 SAM3 Teacher 直接对比

---

### 3.3 数据生成策略（Text Prompts）

由于 SA-1B 原始标注**没有文本标签**，需要为训练样本生成 text prompts。

#### **方案 A: 使用 CLIP Zero-shot 分类**

```python
# 对每个 mask，使用 CLIP 预测类别
import clip

model, preprocess = clip.load("ViT-L/14")
prompts = ["person", "car", "tree", "building", ...]  # 1000+ classes

# 对每个 bbox crop
crop = image.crop(bbox)
crop_feat = model.encode_image(preprocess(crop))

# 计算相似度
text_feats = model.encode_text(clip.tokenize(prompts))
similarities = (crop_feat @ text_feats.T).softmax(dim=-1)

# 选择 top-1
category = prompts[similarities.argmax()]
```

**优点**: 简单，不需要额外标注
**缺点**: 可能不准确

---

#### **方案 B: 使用已有 COCO/LVIS 标注映射**

SA-1B 中部分图像来自 COCO，可以通过 `image_id` 映射到 COCO 标注。

```python
# 加载 COCO annotations
coco_anns = load_coco_annotations("instances_train2017.json")

# 对 SA-1B mask，找最近的 COCO box (IoU > 0.7)
sa_box = mask_to_bbox(sa_mask)
matched_coco = find_matched_coco_box(sa_box, coco_anns, iou_thresh=0.7)

if matched_coco:
    category = coco_categories[matched_coco["category_id"]]
else:
    category = None  # 跳过或使用方案 A
```

**优点**: 高质量标注
**缺点**: 只覆盖部分数据

---

#### **方案 C: 混合策略（推荐）**

```python
def get_text_prompt(image_id, mask):
    # 1. 优先使用 COCO/LVIS 标注（如果有）
    if image_id in coco_mapping:
        return coco_mapping[image_id]["category"]

    # 2. 使用 SA-Co noun phrases（如果有）
    if image_id in saco_mapping:
        return saco_mapping[image_id]["noun_phrase"]

    # 3. Fallback: CLIP zero-shot
    return clip_classify(image, mask)
```

**数据分布（推荐）**:
- 30% 来自 COCO/LVIS（高质量）
- 20% 来自 SA-Co（noun phrases）
- 40% 来自 CLIP（zero-shot）
- 10% 来自 Recap-DataComp-1B（caption）

---

### 3.4 数据预处理 Pipeline

```python
class DecoderDistillDataset(Dataset):
    def __init__(self, sa1b_path, teacher_feat_path, text_source):
        self.images = load_sa1b_images(sa1b_path)
        self.teacher_feats = load_teacher_features(teacher_feat_path)
        self.text_source = text_source  # 'coco', 'clip', 'saco', etc.

    def __getitem__(self, idx):
        # 1. Load image
        image = self.load_image(idx)  # [H, W, 3]

        # 2. Load ground-truth masks
        masks = self.load_masks(idx)   # List of masks

        # 3. Generate text prompts
        text_prompts = []
        for mask in masks:
            text = self.get_text_prompt(idx, mask)
            text_prompts.append(text)

        # 4. Load teacher features (if distilling)
        teacher_dict = self.teacher_feats[idx] if self.teacher_feats else None

        return {
            'image': image,
            'masks': masks,
            'text_prompts': text_prompts,
            'teacher_features': teacher_dict,  # None in Phase 2
        }
```

---

## 4. Student Decoder 架构设计

### 4.1 Lite-EfficientSAM3 Decoder

参考 [DECODER_COMPARISON_AND_DISTILLATION.md](DECODER_COMPARISON_AND_DISTILLATION.md) 第 6.4 节的设计：

```python
class LiteDecoderWrapper(nn.Module):
    """
    轻量级 Decoder: 7.73M 参数（vs Teacher 29.32M, -74%）
    """
    def __init__(self, config):
        super().__init__()

        # ❌ 组件 1: 移除 Geometry Encoder (节省 4.80M)
        # Teacher 有 3 层 Transformer，Student 不使用

        # ✅ 组件 2: Encoder Fusion (3 层，FFN 1024)
        self.encoder = TransformerEncoderFusion(
            num_layers=3,           # 6 → 3
            d_model=256,
            nhead=8,
            dim_feedforward=1024,   # 2048 → 1024
            add_pooled_text_to_img_feat=False,
        )  # ~2.37M

        # ✅ 组件 3: Decoder (3 层，FFN 1024)
        self.decoder = TransformerDecoder(
            num_layers=3,           # 6 → 3
            num_queries=200,
            d_model=256,
            nhead=8,
            dim_feedforward=1024,   # 2048 → 1024
            use_text_cross_attention=True,
            box_refine=True,
            dac=True,
            boxRPB="log",
            presence_token=True,
        )  # ~3.21M

        # ✅ 组件 4: Scoring (FFN 1024)
        prompt_mlp = MLP(256, 1024, 256, 2)  # 2048 → 1024
        self.dot_prod_scoring = DotProductScoring(
            d_model=256,
            d_proj=256,
            prompt_mlp=prompt_mlp,
        )  # ~0.65M

        # ✅ 组件 5: Simplified Seg Head (2 Conv layers)
        self.segmentation_head = SimplifiedSegHead(
            hidden_dim=256,
            num_upsampling_stages=2,  # 3 → 2
            use_cross_attention=True,
        )  # ~1.50M

    def forward(self, feat_4x, feat_2x, feat_1x, pos_1x,
                text_features, text_mask):
        # ---- 步骤 1: 跳过 Geometry Encoder ----
        # 直接使用 text_features，不生成 geo CLS

        # ---- 步骤 2: 构建 Prompt ----
        prompt = text_features  # [77, 1, 256]
        prompt_mask = text_mask  # [1, 77]

        # ---- 步骤 3: Encoder Fusion (3 层) ----
        src_seq = feat_1x.flatten(2).permute(2, 0, 1)
        pos_seq = pos_1x.flatten(2).permute(2, 0, 1)

        memory_dict = self.encoder(
            src=[src_seq],
            src_pos=[pos_seq],
            prompt=prompt,
            prompt_pos=torch.zeros_like(prompt),
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=[(72, 72)],
        )
        memory = memory_dict["memory"]
        prompt_after_enc = memory_dict.get("memory_text", prompt)

        # ---- 步骤 4: Decoder (3 层) ----
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1)

        hs, reference_boxes, dec_presence_out, _ = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_text=prompt_after_enc,
            text_attention_mask=prompt_mask,
            # ... 其他参数
        )
        hs = hs.transpose(1, 2)  # [3, 1, 200, 256]

        # ---- 步骤 5: Scoring ----
        outputs_class = self.dot_prod_scoring(
            hs, prompt_after_enc, prompt_mask
        )

        # ---- 步骤 6: Boxes ----
        anchor_box_offsets = self.decoder.bbox_embed(hs)
        outputs_coord = (inverse_sigmoid(reference_boxes) +
                         anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        # ---- 步骤 7: Segmentation Head (简化) ----
        seg_out = self.segmentation_head(
            backbone_feats=[feat_4x, feat_2x, feat_1x],
            obj_queries=hs,
            prompt=prompt_after_enc,
            prompt_mask=prompt_mask,
        )
        mask_logits = seg_out["pred_masks"]

        # ---- 步骤 8: 提取最后一层结果 ----
        scores = outputs_class[-1, :, :, 0].sigmoid()
        if dec_presence_out is not None:
            presence = dec_presence_out[-1].sigmoid()
            scores = scores * presence
        boxes_xyxy = outputs_boxes_xyxy[-1]

        return scores, boxes_xyxy, mask_logits
```

### 4.2 参数分布

```
Student Decoder (Lite-EfficientSAM3): 7.73M
┌──────────────────────┬─────────┬─────────┐
│ Component            │ Params  │ %       │
├──────────────────────┼─────────┼─────────┤
│ Encoder Fusion (3层) │ 2.37M   │ 30.7%   │
│ Decoder (3层)        │ 3.21M   │ 41.5%   │
│ DotProductScoring    │ 0.65M   │  8.4%   │
│ Segmentation Head    │ 1.50M   │ 19.4%   │
└──────────────────────┴─────────┴─────────┘

vs Teacher: 29.32M → 7.73M (-74%)
ONNX size: 111 MB → ~31 MB (-72%)
```

### 4.3 Layer Mapping 策略

**Teacher 6 层 → Student 3 层映射**:

```
Encoder Fusion:
  Teacher [L0, L1, L2, L3, L4, L5]
          ↓   ↓       ↓       ↓
  Student [L0,        L1,     L2]
  映射:    0→0,       2→1,    4→2

Decoder:
  Teacher [L0, L1, L2, L3, L4, L5]
          ↓   ↓       ↓       ↓
  Student [L0,        L1,     L2]
  映射:    0→0,       2→1,    4→2
```

**初始化方法**:

```python
def init_from_teacher(student, teacher):
    """
    注意: Teacher FFN_dim=2048, Student FFN_dim=1024
    只能复制维度匹配的权重（attention 部分），FFN 需要重新训练
    """
    # Encoder: 复制 0, 2, 4 层（仅 attention 和 LayerNorm）
    for s_idx, t_idx in enumerate([0, 2, 4]):
        t_layer = teacher.encoder.layers[t_idx]
        s_layer = student.encoder.layers[s_idx]
        # 复制 attention 权重（维度匹配）
        s_layer.self_attn.load_state_dict(t_layer.self_attn.state_dict())
        s_layer.cross_attn_image.load_state_dict(t_layer.cross_attn_image.state_dict())
        # 复制 LayerNorm 权重
        s_layer.norm1.load_state_dict(t_layer.norm1.state_dict())
        s_layer.norm2.load_state_dict(t_layer.norm2.state_dict())
        s_layer.norm3.load_state_dict(t_layer.norm3.state_dict())
        # FFN 维度不同 (2048 vs 1024)，跳过

    # Decoder: 复制 0, 2, 4 层（仅 attention 和 LayerNorm）
    for s_idx, t_idx in enumerate([0, 2, 4]):
        t_layer = teacher.decoder.layers[t_idx]
        s_layer = student.decoder.layers[s_idx]
        # 复制 attention 权重
        s_layer.self_attn.load_state_dict(t_layer.self_attn.state_dict())
        s_layer.cross_attn.load_state_dict(t_layer.cross_attn.state_dict())
        s_layer.ca_text.load_state_dict(t_layer.ca_text.state_dict())
        # 复制 LayerNorm 权重
        s_layer.norm1.load_state_dict(t_layer.norm1.state_dict())
        s_layer.norm2.load_state_dict(t_layer.norm2.state_dict())
        s_layer.norm3.load_state_dict(t_layer.norm3.state_dict())
        s_layer.catext_norm.load_state_dict(t_layer.catext_norm.state_dict())
        # FFN 维度不同，跳过

    # Scoring: 复制 prompt_proj, hs_proj（维度匹配）
    student.dot_prod_scoring.prompt_proj.load_state_dict(
        teacher.dot_prod_scoring.prompt_proj.state_dict()
    )
    student.dot_prod_scoring.hs_proj.load_state_dict(
        teacher.dot_prod_scoring.hs_proj.state_dict()
    )
    # prompt_mlp 维度不同 (2048 vs 1024)，跳过

    # Seg Head: 复制可复制的部分
    # 简化后的结构需要部分重新训练
```

---

## 5. 蒸馏策略详解

### 5.1 参考 EfficientSAM3 Stage 1 策略

EfficientSAM3 原作者在 Stage 1 Encoder 蒸馏中使用的损失：

```python
# stage1/train_image_encoder_stage1.py
loss = (
    mse_loss(student_embed, teacher_embed, valid_mask) +
    1.0 * cosine_similarity_loss(student_embed, teacher_embed, valid_mask)
)
```

**关键点**:
1. **Feature-level MSE**: 逐像素特征对齐
2. **Cosine Similarity**: 保持方向一致性
3. **Valid Mask**: 过滤 padding 区域

### 5.2 Decoder 蒸馏损失设计

```python
def distillation_loss(student_outputs, teacher_outputs, targets, config):
    """
    完整蒸馏损失函数
    """
    total_loss = 0.0
    losses_dict = {}

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. Task Loss (Detection + Segmentation)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    task_loss = compute_task_loss(student_outputs, targets)
    losses_dict['task'] = task_loss
    total_loss += config.task_weight * task_loss

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. Feature Distillation (中间层对齐)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    feat_loss = 0.0

    # 2a. Encoder Fusion 输出蒸馏
    feat_loss += F.mse_loss(
        student_outputs['memory'],      # [5184, 1, 256]
        teacher_outputs['memory']
    )

    # 2b. Decoder 中间层蒸馏 (layer-wise)
    # Teacher 6 层 [0,1,2,3,4,5], Student 3 层 [0,1,2]
    # 映射: T0→S0, T2→S1, T4→S2
    s_hs = student_outputs['decoder_features']  # [3, 200, 1, 256]
    t_hs = teacher_outputs['decoder_features']  # [6, 200, 1, 256]
    for s_idx, t_idx in enumerate([0, 2, 4]):
        feat_loss += F.mse_loss(s_hs[s_idx], t_hs[t_idx])

    # 2c. Query embeddings 蒸馏（最后一层）
    feat_loss += F.mse_loss(
        s_hs[-1],  # Student 最后一层
        t_hs[-1]   # Teacher 最后一层
    )

    # 2d. 加入 Cosine Similarity（参考 Stage 1）
    feat_loss += 1.0 * cosine_similarity_loss(
        s_hs[-1].flatten(0, 2),
        t_hs[-1].flatten(0, 2)
    )

    losses_dict['feature'] = feat_loss
    total_loss += config.feat_weight * feat_loss

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. Logits Distillation (soft targets)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # 3a. Classification scores
    # 注意: scores 是独立的二分类检测分数 [1, 200]，不应该用 softmax
    # 应该用 MSE 或 BCE 蒸馏（每个 query 是独立的检测，不是互斥分类）
    s_scores = student_outputs['scores']  # [1, 200]
    t_scores = teacher_outputs['scores']
    logits_loss = F.mse_loss(s_scores, t_scores)

    # 3b. Box predictions (L1 loss)
    logits_loss += F.l1_loss(
        student_outputs['boxes_xyxy'],
        teacher_outputs['boxes_xyxy']
    )

    # 3c. Mask logits (MSE)
    logits_loss += F.mse_loss(
        student_outputs['mask_logits'],
        teacher_outputs['mask_logits']
    )

    losses_dict['logits'] = logits_loss
    total_loss += config.logit_weight * logits_loss

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. Attention Map Distillation
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if config.distill_attention:
        attn_loss = 0.0
        s_attns = student_outputs['attention_maps']  # List[Tensor]
        t_attns = teacher_outputs['attention_maps']

        for s_idx, t_idx in enumerate([0, 2, 4]):
            # Text cross-attention
            attn_loss += F.mse_loss(
                s_attns['text_cross'][s_idx],
                t_attns['text_cross'][t_idx]
            )
            # Image cross-attention
            attn_loss += F.mse_loss(
                s_attns['image_cross'][s_idx],
                t_attns['image_cross'][t_idx]
            )

        losses_dict['attention'] = attn_loss
        total_loss += config.attn_weight * attn_loss

    losses_dict['total'] = total_loss
    return total_loss, losses_dict


def compute_task_loss(outputs, targets):
    """
    标准检测 + 分割损失
    """
    # 分类损失（Focal Loss 或 BCE）
    cls_loss = focal_loss(
        outputs['scores'],
        targets['labels'],
        alpha=0.25,
        gamma=2.0
    )

    # Box 损失（GIoU + L1）
    box_loss = (
        giou_loss(outputs['boxes_xyxy'], targets['boxes']) +
        F.l1_loss(outputs['boxes_xyxy'], targets['boxes'])
    )

    # Mask 损失（Dice + BCE）
    mask_loss = (
        dice_loss(outputs['masks'], targets['masks']) +
        F.binary_cross_entropy_with_logits(
            outputs['mask_logits'], targets['masks']
        )
    )

    return cls_loss + 5.0 * box_loss + 2.0 * mask_loss


def cosine_similarity_loss(feat1, feat2):
    """
    参考 Stage 1 的 Cosine Similarity Loss
    """
    # Normalize
    feat1_norm = F.normalize(feat1, p=2, dim=-1)
    feat2_norm = F.normalize(feat2, p=2, dim=-1)

    # Cosine similarity
    cos_sim = (feat1_norm * feat2_norm).sum(dim=-1)

    # Loss = 1 - cos_sim (最大化相似度)
    return (1.0 - cos_sim).mean()
```

### 5.3 损失权重调度

参考 **EfficientSAM3 的渐进式训练**，损失权重随 epoch 动态调整：

```python
def get_loss_weights(epoch, phase):
    """
    损失权重调度策略
    """
    if phase == 'warmup':
        # Phase 2: 纯监督学习
        return {
            'task_weight': 1.0,
            'feat_weight': 0.0,
            'logit_weight': 0.0,
            'attn_weight': 0.0,
        }

    elif phase == 'distillation':
        # Phase 3: 蒸馏阶段，逐步降低蒸馏权重
        progress = epoch / 30  # 30 epochs

        return {
            'task_weight': 1.0,
            'feat_weight': 0.5 * (1 - progress * 0.5),  # 0.5 → 0.25
            'logit_weight': 0.3 * (1 - progress * 0.5), # 0.3 → 0.15
            'attn_weight': 0.2 * (1 - progress),        # 0.2 → 0.0
        }

    elif phase == 'finetuning':
        # Phase 4: 微调阶段，只保留少量蒸馏
        return {
            'task_weight': 1.0,
            'feat_weight': 0.1,
            'logit_weight': 0.05,
            'attn_weight': 0.0,
        }
```

---

## 6. 训练配置

### 6.1 Phase 1: Teacher Feature Extraction

**目标**: 在训练集上一次性保存所有 Teacher 中间特征，供后续蒸馏使用。

**配置文件**: `configs/phase1_save_teacher_features.yaml`

```yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Save Teacher Decoder Features
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL:
  # Teacher checkpoint (Stage 1 已训练的 Image + Text Encoder)
  RESUME: "checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth"
  BACKBONE_TYPE: "repvit"
  MODEL_NAME: "m0_9"
  TEXT_ENCODER_TYPE: "MobileCLIP-S1"

DATA:
  DATASET: "sa1b"
  DATA_PATH: "data/sa-1b"
  BATCH_SIZE: 8
  NUM_WORKERS: 8
  RESOLUTION: 1008

OUTPUT:
  FEATURE_SAVE_PATH: "output/teacher_decoder_features"
  SAVE_FORMAT: "pkl"  # pickle format for fast loading

EXTRACT:
  # 需要保存的特征
  SAVE_MEMORY: true            # Encoder Fusion 输出
  SAVE_DECODER_FEATURES: true  # 每层 decoder hs
  SAVE_ATTENTION_MAPS: true    # Attention weights
  SAVE_LOGITS: true            # Scores, boxes, masks
```

**运行脚本**: `scripts/phase1_save_teacher_features.sh`

```bash
#!/bin/bash

# Phase 1: Save Teacher Decoder Features
PYTHONPATH=. python stage1_decoder/save_teacher_decoder_features.py \
  --config configs/phase1_save_teacher_features.yaml \
  --num-gpus 4 \
  --output output/teacher_decoder_features

# 预期输出结构:
# output/teacher_decoder_features/
# ├── config.json
# ├── log_rank0.txt
# └── features/
#     ├── sa_000000.pkl
#     ├── sa_000001.pkl
#     └── ...
```

**内存优化**:
- 使用 `float16` 存储特征（减少磁盘空间）
- 按 batch 存储，避免一次性加载全部数据
- 预期磁盘空间: ~500 GB (110K samples × ~5 MB/sample)

---

### 6.2 Phase 2: Student Warm-up

**目标**: 纯监督训练，快速收敛到合理性能。

**配置文件**: `configs/phase2_warmup.yaml`

```yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Student Warm-up (Supervised)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL:
  TYPE: "LiteDecoderWrapper"
  INIT_FROM_TEACHER: true  # 从 Teacher 初始化 (0,2,4 层)
  TEACHER_CHECKPOINT: "checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth"

  # Student 配置
  ENCODER_LAYERS: 3
  DECODER_LAYERS: 3
  FFN_DIM: 1024
  NUM_QUERIES: 200

DATA:
  DATASET: "sa1b"
  DATA_PATH: "data/sa-1b"
  BATCH_SIZE: 4
  NUM_WORKERS: 8
  RESOLUTION: 1008

  # Text prompt 生成策略
  TEXT_SOURCE:
    - "coco_mapping"     # 30%
    - "saco_mapping"     # 20%
    - "clip_classify"    # 40%
    - "recap_caption"    # 10%

TRAIN:
  EPOCHS: 10
  BASE_LR: 1e-4
  WEIGHT_DECAY: 0.01
  WARMUP_EPOCHS: 1
  LR_SCHEDULER: "cosine"

  # Phase 2 只用 Task Loss
  LOSS:
    TASK_WEIGHT: 1.0
    FEAT_WEIGHT: 0.0
    LOGIT_WEIGHT: 0.0
    ATTN_WEIGHT: 0.0

OPTIMIZER:
  TYPE: "AdamW"
  BETAS: [0.9, 0.999]
  EPS: 1e-8

OUTPUT:
  SAVE_DIR: "output/phase2_warmup"
  SAVE_FREQ: 1  # 每个 epoch 保存一次
  EVAL_FREQ: 1  # 每个 epoch 评估一次

EVAL:
  DATASET: "coco_val2017"
  DATA_PATH: "data/coco"
```

**运行脚本**: `scripts/phase2_warmup.sh`

```bash
#!/bin/bash

# Phase 2: Warm-up Training
PYTHONPATH=. torchrun --nproc_per_node=4 \
  stage1_decoder/train_student_decoder.py \
  --config configs/phase2_warmup.yaml \
  --phase warmup

# 监控训练进度
tensorboard --logdir output/phase2_warmup/logs
```

**预期结果**:
- Epoch 1: mAP ~30%
- Epoch 5: mAP ~38%
- Epoch 10: mAP ~40%

---

### 6.3 Phase 3: Feature Distillation

**目标**: 加载 Teacher features，蒸馏中间层表示。

**配置文件**: `configs/phase3_distillation.yaml`

```yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Feature Distillation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL:
  TYPE: "LiteDecoderWrapper"
  RESUME: "output/phase2_warmup/ckpt_epoch_10.pth"  # 从 Warm-up 继续

DATA:
  DATASET: "sa1b_with_teacher_features"
  DATA_PATH: "data/sa-1b"
  TEACHER_FEAT_PATH: "output/teacher_decoder_features/features"
  BATCH_SIZE: 4
  NUM_WORKERS: 8

  # 混合数据源
  RECAP_RATIO: 0.1  # 10% Recap-DataComp-1B

TRAIN:
  EPOCHS: 30
  BASE_LR: 5e-5  # 降低学习率
  WEIGHT_DECAY: 0.01
  LR_SCHEDULER: "cosine"

  # Phase 3 蒸馏权重（动态调度）
  LOSS:
    TASK_WEIGHT: 1.0
    FEAT_WEIGHT: 0.5    # 初始 0.5，线性降至 0.25
    LOGIT_WEIGHT: 0.3   # 初始 0.3，线性降至 0.15
    ATTN_WEIGHT: 0.2    # 初始 0.2，线性降至 0.0
    TEMPERATURE: 2.0

    # 特征蒸馏配置
    DISTILL_LAYERS: [0, 2, 4]  # Teacher 层映射
    DISTILL_ATTENTION: true
    USE_COSINE_SIM: true       # 参考 Stage 1

OUTPUT:
  SAVE_DIR: "output/phase3_distillation"
  SAVE_FREQ: 2  # 每 2 个 epoch 保存
  EVAL_FREQ: 1

EVAL:
  DATASETS:
    - "coco_val2017"
    - "lvis_val"      # 额外评估 long-tail
```

**运行脚本**: `scripts/phase3_distillation.sh`

```bash
#!/bin/bash

# Phase 3: Feature Distillation
PYTHONPATH=. torchrun --nproc_per_node=4 \
  stage1_decoder/train_student_decoder.py \
  --config configs/phase3_distillation.yaml \
  --phase distillation

# 使用混合精度训练（AMP）
# PYTHONPATH=. torchrun --nproc_per_node=4 \
#   stage1_decoder/train_student_decoder.py \
#   --config configs/phase3_distillation.yaml \
#   --phase distillation \
#   --use-amp
```

**预期结果**:
- Epoch 5: mAP ~41%
- Epoch 15: mAP ~42%
- Epoch 30: mAP ~42.5%

---

### 6.4 Phase 4: Task Fine-tuning

**目标**: 在 SA-Co Gold+Silver 上微调，强化文本对齐。

**配置文件**: `configs/phase4_finetuning.yaml`

```yaml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: Task Fine-tuning on SA-Co
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MODEL:
  TYPE: "LiteDecoderWrapper"
  RESUME: "output/phase3_distillation/ckpt_epoch_30.pth"

DATA:
  DATASET: "saco_gold_silver"
  DATA_PATH: "data/SACo-Gold"
  SILVER_PATH: "data/SACo-Silver"  # 可选
  BATCH_SIZE: 8
  NUM_WORKERS: 8

  # SA-Co 特有的文本标注
  USE_NOUN_PHRASES: true

TRAIN:
  EPOCHS: 10
  BASE_LR: 1e-5  # 更低的学习率
  WEIGHT_DECAY: 0.01
  LR_SCHEDULER: "cosine"

  # Phase 4: 降低蒸馏权重
  LOSS:
    TASK_WEIGHT: 1.0
    FEAT_WEIGHT: 0.1   # 保留少量蒸馏
    LOGIT_WEIGHT: 0.05
    ATTN_WEIGHT: 0.0   # 关闭 attention 蒸馏

OUTPUT:
  SAVE_DIR: "output/phase4_finetuning"
  SAVE_FREQ: 1
  EVAL_FREQ: 1

EVAL:
  DATASETS:
    - "coco_val2017"
    - "lvis_val"
    - "saco_veval"   # SA-Co VEval benchmark
```

**运行脚本**: `scripts/phase4_finetuning.sh`

```bash
#!/bin/bash

# Phase 4: Fine-tuning on SA-Co
PYTHONPATH=. torchrun --nproc_per_node=4 \
  stage1_decoder/train_student_decoder.py \
  --config configs/phase4_finetuning.yaml \
  --phase finetuning
```

**预期结果**:
- Epoch 5: mAP ~43%
- Epoch 10: mAP ~43.5%
- SA-Co VEval: Avg Cos Sim > 0.90（与 Teacher 文本对齐）

---

## 7. 实施步骤

### 7.1 环境准备

```bash
# 1. Clone 仓库
cd /Users/wanghao/efficientsam3

# 2. 激活环境
conda activate efficientsam3

# 3. 安装额外依赖
pip install tensorboard wandb tqdm rich
```

### 7.2 数据集下载与准备

```bash
# 步骤 1: 下载 SA-1B (1% subset)
bash data/download_sa1b_subset.sh
python data/reorg_sa1b.py

# 步骤 2: 下载 Recap-DataComp-1B (1% subset)
python data/download_datacomp.py --subset 0.01

# 步骤 3: 下载 SA-Co Gold+Silver
git clone https://huggingface.co/datasets/facebook/SACo-Gold data/SACo-Gold
cd data/SACo-Gold && git lfs pull

# 步骤 4: 下载 COCO / LVIS
bash data/download_coco.sh
bash data/download_lvis.sh

# 步骤 5: 生成 text prompts
python scripts/generate_text_prompts.py \
  --sa1b-path data/sa-1b \
  --coco-path data/coco \
  --saco-path data/SACo-Gold \
  --output data/sa1b_text_prompts.json
```

### 7.3 创建 Student 模型

```bash
# 创建 Student Decoder 代码目录
mkdir -p stage1_decoder

# 实现 LiteDecoderWrapper
# 参考 export_image_model_onnx.py 中的 DecoderWrapper
# 修改层数和 FFN 维度

# 文件结构:
# stage1_decoder/
# ├── __init__.py
# ├── lite_decoder.py              # LiteDecoderWrapper 定义
# ├── save_teacher_decoder_features.py  # Phase 1 脚本
# ├── train_student_decoder.py     # Phase 2-4 训练脚本
# ├── distill_loss.py              # 蒸馏损失函数
# └── dataset.py                   # 数据集加载器
```

### 7.4 执行训练

```bash
# Phase 1: Save Teacher Features (一次性)
bash scripts/phase1_save_teacher_features.sh

# Phase 2: Warm-up (10 epochs, ~30h)
bash scripts/phase2_warmup.sh

# Phase 3: Distillation (30 epochs, ~105h)
bash scripts/phase3_distillation.sh

# Phase 4: Fine-tuning (10 epochs, ~10h)
bash scripts/phase4_finetuning.sh
```

### 7.5 模型合并与导出

```bash
# 合并 Student Decoder 到完整 checkpoint
python stage1_decoder/merge_student_decoder.py \
  --student-ckpt output/phase4_finetuning/ckpt_epoch_10.pth \
  --base-ckpt checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
  --output checkpoints/efficient_sam3_repvit_m0_9_mobileclip_s1_lite_decoder.pth

# 导出 ONNX
python export_image_model_onnx.py \
  --checkpoint checkpoints/efficient_sam3_repvit_m0_9_mobileclip_s1_lite_decoder.pth \
  --output exports_lite/

# 验证 ONNX
python run_onnx_text_grounding.py \
  --image test_image/person.jpg \
  --prompt "person" \
  --onnx-dir exports_lite/ \
  --output result_lite.png
```

---

## 8. 评估与验证

### 8.1 定量评估

```bash
# COCO mAP
python eval/eval_coco.py \
  --checkpoint checkpoints/efficient_sam3_lite_decoder.pth \
  --coco-root data/coco \
  --output-dir eval_results/coco

# LVIS mAP (Long-tail)
python eval/eval_lvis.py \
  --checkpoint checkpoints/efficient_sam3_lite_decoder.pth \
  --lvis-root data/lvis \
  --output-dir eval_results/lvis

# SA-Co VEval (Text-Image Alignment)
python eval/eval_saco_veval.py \
  --checkpoint checkpoints/efficient_sam3_lite_decoder.pth \
  --saco-root data/SACo-Gold \
  --output-dir eval_results/saco
```

### 8.2 定性评估

```bash
# 可视化检测结果
python scripts/visualize_detections.py \
  --checkpoint checkpoints/efficient_sam3_lite_decoder.pth \
  --image-dir test_images/ \
  --prompts "person,car,dog,cat,tree,building" \
  --output-dir visualizations/

# 对比 Teacher vs Student
python scripts/compare_teacher_student.py \
  --teacher checkpoints/efficient_sam3_teacher.pth \
  --student checkpoints/efficient_sam3_lite_decoder.pth \
  --image test_images/person.jpg \
  --prompt "person" \
  --output comparison.png
```

### 8.3 效率评估

```bash
# 推理速度对比
python scripts/benchmark_inference.py \
  --teacher checkpoints/efficient_sam3_teacher.pth \
  --student checkpoints/efficient_sam3_lite_decoder.pth \
  --device cuda \
  --num-runs 100

# ONNX Runtime 速度测试
python scripts/benchmark_onnx.py \
  --onnx-dir exports_lite/ \
  --image test.jpg \
  --prompt "person" \
  --num-runs 100
```

---

## 9. 预期结果

### 9.1 参数与大小

| 模型 | Decoder 参数 | ONNX 大小 | 压缩率 |
|------|-------------|-----------|--------|
| **Teacher** | 29.32M | 111 MB | - |
| **Student (Lite)** | 7.73M | ~31 MB | **74%** |
| **Student + INT8** | 7.73M | **~8 MB** | **93%** |

### 9.2 精度指标

| 数据集 | Teacher mAP | Student mAP | Gap |
|--------|-------------|-------------|-----|
| **COCO val2017** | 42.2% | 38 - 40% | **-2~4%** |
| **LVIS v1.0** | 38.5% | 35 - 37% | **-2~3%** |
| **SA-Co VEval (Cos Sim)** | 0.947 | 0.90 - 0.92 | **-0.03** |

> **注意**: 由于压缩比例较大（74%），精度损失可能比之前估计的更大。实际结果需要通过实验验证。

### 9.3 推理速度

| 平台 | Teacher | Student | 加速比 |
|------|---------|---------|--------|
| **A100 GPU** | 45 ms | 18 ms | **2.5×** |
| **4070 Ti** | 80 ms | 30 ms | **2.7×** |
| **CPU (12 cores)** | 850 ms | 300 ms | **2.8×** |
| **ONNX (INT8, CPU)** | 650 ms | **120 ms** | **5.4×** |

### 9.4 各阶段精度演进（预期）

| 阶段 | Epoch | COCO mAP | LVIS mAP | 说明 |
|------|-------|----------|----------|------|
| Phase 0 (Teacher) | - | 42.2% | 38.5% | 基线 |
| Phase 2 (Warm-up) | 10 | 36.0% | 33.0% | 纯监督（参数少，收敛较难） |
| Phase 3 (Distillation) | 30 | 39.0% | 35.5% | 蒸馏提升 |
| Phase 4 (Fine-tuning) | 10 | 40.0% | 36.0% | 文本对齐微调 |

**关键发现**:
- Phase 3 蒸馏能够显著弥补精度损失（36% → 39%）
- Phase 4 微调进一步提升文本理解（39% → 40%）
- **最终 Student 精度约为 Teacher 的 95%**，符合知识蒸馏的典型效果

---

## 总结

### 关键要点

1. **参考 EfficientSAM3 原作者的训练范式**
   - Stage 1: Encoder 蒸馏（已完成）
   - **Stage 1.5**: Decoder 蒸馏（本方案）
   - Stage 2-3: Memory + End-to-End（未来工作）

2. **数据集选择**
   - SA-1B (1%): 主要训练数据
   - Recap-DataComp-1B (1%): 文本增强
   - SA-Co Gold+Silver: 高质量微调

3. **蒸馏策略**
   - Feature MSE + Cosine Similarity（参考 Stage 1）
   - Layer-wise 映射（6 层 → 3 层）
   - 渐进式权重调度

4. **预期收益**
   - 参数减少 74%（29.32M → 7.73M）
   - ONNX 大小减少 72%（111 MB → ~31 MB）
   - 速度提升 2.5-3.0×
   - 精度损失约 2-4 mAP points

5. **实施建议**
   - 从 Warm-up 开始，确保监督训练收敛
   - 预存 Teacher features，避免重复计算
   - 使用混合精度训练（AMP）节省显存
   - 多 GPU 训练加速（4× A100 约 6 天完成）
   - 初始化时只复制维度匹配的权重（attention + LayerNorm），FFN 需重新训练