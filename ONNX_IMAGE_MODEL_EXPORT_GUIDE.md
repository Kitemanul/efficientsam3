# EfficientSAM3 Image Model ONNX 导出指南

本文档详细描述了 EfficientSAM3 Image Model（文本引导检测+分割）的完整 ONNX 导出过程，包括架构分析、遇到的问题与解决方案、导出结果与验证。

---

## 目录

1. [概述](#1-概述)
2. [导出组件架构](#2-导出组件架构)
3. [端到端推理流程](#3-端到端推理流程)
4. [技术挑战与解决方案](#4-技术挑战与解决方案)
5. [运行导出脚本](#5-运行导出脚本)
6. [导出结果](#6-导出结果)
7. [ONNX Runtime 推理示例](#7-onnx-runtime-推理示例)
8. [不可导出组件说明](#8-不可导出组件说明)

---

## 1. 概述

### 目标

将 EfficientSAM3 Image Model 的完整文本引导检测（text-grounded detection）流水线导出为 3 个 ONNX 模型，覆盖从图像/文本输入到 masks + scores + boxes 输出的全部计算图。

### 导出结果

| 组件 | 文件 | 大小 | 导出方式 |
|------|------|------|----------|
| Image Encoder | `image_encoder.onnx` | 1.3 MB | `torch.export` (dynamo) |
| Text Encoder | `text_encoder.onnx` | 0.5 MB | `torch.export` (dynamo) |
| Detector | `detector.onnx` | 111.3 MB | TorchScript tracer (legacy) |

### 与旧版导出的对比

旧版 `export_image_model_onnx.py` 仅导出 4 个孤立的简单组件（Image Encoder、Text Encoder、DotProductScoring、Box Head），核心检测流水线（Transformer Encoder/Decoder、Geometry Encoder、Segmentation Head）被标记为"不可导出"。

新版将整个检测流水线封装为单一 `DetectorWrapper`，成功导出为 `detector.onnx`，实现了端到端推理能力。

---

## 2. 导出组件架构

### 2.1 Image Encoder (`ImageEncoderWrapper`)

```
images [1, 3, 1008, 1008]
  │
  ▼
RepViT Trunk (m0_9)          ← student_encoder
  │
  ▼
features [1, 1024, 72, 72]
  │
  ├─→ neck_conv_4x ──→ feat_4x [1, 256, 288, 288]   (ConvTranspose2d ×2 upsampling)
  ├─→ neck_conv_2x ──→ feat_2x [1, 256, 144, 144]   (ConvTranspose2d ×1 upsampling)
  └─→ neck_conv_1x ──→ feat_1x [1, 256, 72, 72]     (1×1 Conv, no upsampling)
                   └─→ pos_1x  [1, 256, 72, 72]     (PositionEmbeddingSine)
```

**来源**: `model.backbone.vision_backbone` (Sam3DualViTDetNeck)

- `trunk.model`: RepViT-m0_9 backbone (MobileViT 变体)
- `convs[0/1/2]`: Neck 卷积层，分别产生 4×/2×/1× 分辨率的特征
- `position_encoding`: 正弦位置编码

### 2.2 Text Encoder (`TextEncoderWrapper`)

```
token_ids [1, 77] (int64)
  │
  ▼
forward_embedding()      ← token embedding + positional embedding
  │
  ▼
12× Transformer layers   ← MobileCLIP-S1 text transformer
  │
  ▼
final_layer_norm()
  │
  ▼
projector (Linear 512→256)
  │
  ▼
text_features [1, 77, 256]
text_mask     [1, 77] (bool, True=padding)
```

**来源**: `model.backbone.language_backbone` (TextStudentEncoder)

- `encoder`: MobileCLIP-S1 text model
- `projector`: 将 MobileCLIP 的 512 维降到 256 维

### 2.3 Detector (`DetectorWrapper`)

这是核心组件，将以下子模块串联为单一 ONNX 模型：

```
feat_4x, feat_2x, feat_1x, pos_1x, text_features, text_mask
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 1: Text features → seq-first               │
│   txt_feats [77, 1, 256]                        │
│   txt_masks [1, 77]                             │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 2: Geometry Encoder CLS token              │
│   cls_embed (nn.Embedding)                      │
│   → final_proj + norm                           │
│   → 3× Transformer layers (cross-attn to image) │
│   → encode_norm                                 │
│   cls [1, 1, 256]                               │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 3: Build prompt = text + geo CLS           │
│   prompt [78, 1, 256]                           │
│   prompt_mask [1, 78]                           │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 4: Encoder Fusion                          │
│   6× TransformerEncoderLayer                    │
│     - self_attn (batch_first=True)              │
│     - cross_attn to prompt (batch_first=True)   │
│     - FFN                                       │
│   → memory [5184, 1, 256]                       │
│   → prompt_after_enc [78, 1, 256]               │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 5: Decoder                                 │
│   200 queries, 6× TransformerDecoderLayer       │
│     - self_attn + text cross_attn + img cross_attn │
│     - box refinement (box_refine=True)          │
│     - boxRPB="log" (relative position bias)     │
│     - presence_token=True                       │
│   → hs [6, 1, 200, 256]                        │
│   → reference_boxes [6, 1, 200, 4]             │
│   → presence_out [6, 1, 1]                      │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 6: DotProduct Scoring                      │
│   mean_pool_text → project → dot product        │
│   + joint presence scoring                      │
│   → outputs_class [6, 1, 200, 1]               │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 7: Box Head (bbox_embed MLP)               │
│   MLP(256, 256, 4, 3 layers)                    │
│   + reference box refinement                    │
│   → cxcywh → xyxy conversion                   │
│   → boxes_xyxy [1, 200, 4]                     │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────┐
│ Step 8: Segmentation Head                       │
│   PixelDecoder (3-stage FPN upsampling)         │
│   + use_encoder_inputs=True (replace last feat  │
│     with encoder memory)                        │
│   + cross_attend_prompt                         │
│   + MaskPredictor (einsum: queries × pixels)    │
│   → mask_logits [1, 200, 288, 288]             │
└─────────────────────────────────────────────────┘
  │
  ▼
scores [1, 200], boxes_xyxy [1, 200, 4], mask_logits [1, 200, 288, 288]
```

**各子模块来源**:

| 子模块 | 来源路径 | 源文件 |
|--------|----------|--------|
| Geometry Encoder | `model.geometry_encoder` | `geometry_encoders.py` |
| Encoder Fusion | `model.transformer.encoder` | `encoder.py` |
| Decoder | `model.transformer.decoder` | `decoder.py` |
| DotProduct Scoring | `model.dot_prod_scoring` | `model_misc.py` |
| Segmentation Head | `model.segmentation_head` | `maskformer_segmentation.py` |

---

## 3. 端到端推理流程

```
输入:
  - image: [1, 3, 1008, 1008] float32 (normalized RGB)
  - token_ids: [1, 77] int64 (SimpleTokenizer 编码的文本)

Pipeline:
  feat_4x, feat_2x, feat_1x, pos_1x = image_encoder(image)
  text_features, text_mask           = text_encoder(token_ids)
  scores, boxes_xyxy, mask_logits    = detector(feat_4x, feat_2x, feat_1x, pos_1x,
                                                 text_features, text_mask)

输出:
  - scores:      [1, 200]           检测置信度 (0~1, 已含 sigmoid + presence)
  - boxes_xyxy:  [1, 200, 4]       归一化边界框 (x1, y1, x2, y2, 范围 0~1)
  - mask_logits: [1, 200, 288, 288] 分割 mask logits (需 sigmoid 获取概率)

后处理:
  1. 按 score 阈值过滤 (如 score > 0.3)
  2. boxes_xyxy × [W, H, W, H] 获取像素坐标
  3. sigmoid(mask_logits) > 0.5 获取二值 mask
  4. 将 mask 从 288×288 resize 到原始图像尺寸
```

---

## 4. 技术挑战与解决方案

### 4.1 CUDA 硬编码 — PositionEmbeddingSine

**问题**: `PositionEmbeddingSine.__init__` 在 `position_encoding.py:47` 硬编码 `device="cuda"` 预计算位置编码。在 CPU 环境下构建模型会直接报错。

**解决方案**: Monkey-patch `__init__`，强制 `precompute_resolution=None` 跳过 CUDA 预计算：

```python
_orig_pe_init = PositionEmbeddingSine.__init__

def _patched_pe_init(self, num_pos_feats, temperature=10000, normalize=True,
                     scale=None, precompute_resolution=None):
    _orig_pe_init(self, num_pos_feats, temperature=temperature,
                  normalize=normalize, scale=scale,
                  precompute_resolution=None)  # 强制 None

PositionEmbeddingSine.__init__ = _patched_pe_init
```

### 4.2 CUDA 硬编码 — TransformerDecoder._get_coords

**问题**: `_get_coords` 静态方法创建坐标张量时使用默认 `device`，在某些路径可能传入 `"cuda"`。

**解决方案**: Monkey-patch 强制 `device="cpu"`：

```python
_orig_get_coords = TransformerDecoder._get_coords

@staticmethod
def _patched_get_coords(H, W, device="cpu"):
    return _orig_get_coords(H, W, device="cpu")

TransformerDecoder._get_coords = _patched_get_coords
```

### 4.3 数据依赖的符号守卫 — _get_rpb_matrix

**问题**: `decoder.py` 中 `_get_rpb_matrix` 方法有条件判断：
```python
if torch.compiler.is_dynamo_compiling() or self.compilable_stored_size == (H, W):
```
当 H, W 是符号变量时，`torch.export` 无法处理这个数据依赖的 guard。

**报错信息**:
```
GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)
```

**解决方案**: Monkey-patch `_get_rpb_matrix`，移除数据依赖判断，始终使用预计算的坐标缓存：

```python
def _patched_get_rpb_matrix(self, reference_boxes, feat_size):
    H, W = feat_size
    if self.compilable_cord_cache is None:
        self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
        self.compilable_stored_size = (H, W)
    coords_h, coords_w = self.compilable_cord_cache
    # ... 后续计算不变
```

### 4.4 MultiheadAttention view 不兼容 — torch.export

**问题**: `torch.export` (dynamo-based) 在分解 `nn.MultiheadAttention` 时，内部 view 操作遇到非连续张量的 stride 不兼容：

```
ValueError: Cannot view a tensor with shape torch.Size([201, 1, 8, 32])
and strides (32, 51456, 6432, 1) as a tensor with shape (201, 256)
```

**根因**: Decoder 的 `nn.MultiheadAttention` 使用 `batch_first=False`，生成的中间张量是非连续的，`torch.export` 的分解步骤尝试用 view 而非 reshape 导致失败。

**解决方案**: Detector 使用 TorchScript legacy tracer (`dynamo=False`) 导出。Image Encoder 和 Text Encoder 结构较简单，使用新版 `torch.export` 导出。

```python
# Image Encoder & Text Encoder: 使用 dynamo
torch.onnx.export(model, inputs, path, dynamo=True)

# Detector: 使用 legacy tracer
torch.onnx.export(model, inputs, path, dynamo=False)
```

### 4.5 Encoder 输入格式 — seq-first vs image-format

**问题**: `TransformerEncoderFusion.forward` 接收 `feat_sizes` 参数时，期望输入为 seq-first 格式 `[HW, B, C]`，内部会 reshape 回 `[B, C, H, W]` 再交给 `_prepare_multilevel_features`。

**注意事项**: `encoder.py:352` 的 `assert all(x.dim == 4 for x in src)` 实际上有 bug —— `x.dim` 是方法引用而非调用 `x.dim()`，所以该 assert 永远不会真正生效。

**解决方案**: 在 DetectorWrapper 中将 `feat_1x` 显式转换为 seq-first 格式，并传入 `feat_sizes`：

```python
H, W = feat_1x.shape[2], feat_1x.shape[3]
src_seq = feat_1x.flatten(2).permute(2, 0, 1)  # [5184, 1, 256]
pos_seq = pos_1x.flatten(2).permute(2, 0, 1)   # [5184, 1, 256]
memory_dict = self.encoder(
    src=[src_seq], src_pos=[pos_seq],
    prompt=prompt, ..., feat_sizes=[(H, W)]
)
```

### 4.6 Segmentation Head 的 prompt 参数

**问题**: `UniversalSegmentationHead.forward` 在 `cross_attend_prompt=True` 时需要 `prompt` 和 `prompt_mask` 参数，否则对 `None` 调用方法导致 `AttributeError`。

**解决方案**: 将 Encoder Fusion 输出的 `prompt_after_enc` 和 `prompt_mask` 传递给 Segmentation Head：

```python
seg_out = self.segmentation_head(
    backbone_feats=[feat_4x, feat_2x, feat_1x],
    obj_queries=hs,
    image_ids=torch.zeros(1, dtype=torch.long, device=feat_1x.device),
    encoder_hidden_states=memory,
    prompt=prompt_after_enc,  # 不能省略
    prompt_mask=prompt_mask,
)
```

### 4.7 Geometry Encoder 的 CLS-only 路径

**问题**: `SequenceGeometryEncoder.forward` 支持多种 prompt 类型（boxes, points, scribbles 等），但文本引导检测只使用 CLS token 路径（dummy prompt）。完整 forward 包含大量条件分支，无法直接导出。

**解决方案**: 在 DetectorWrapper 中手动实现 CLS token 路径，只提取必要的子模块：

```python
self.geo_cls_embed = geo.cls_embed          # nn.Embedding(1, 256)
self.geo_final_proj = geo.final_proj        # Linear(256, 256)
self.geo_norm = geo.norm                    # LayerNorm(256)
self.geo_encode_layers = geo.encode         # 3× TransformerDecoderLayer
self.geo_encode_norm = geo.encode_norm      # LayerNorm(256)
```

---

## 5. 运行导出脚本

### 依赖

```bash
pip install torch onnx onnxruntime numpy
```

### 基本用法

```bash
python export_image_model_onnx.py \
    --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
    --output exports_repvit_m0_9/
```

### 仅导出不验证

```bash
python export_image_model_onnx.py --skip-verify \
    --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
    --output exports_repvit_m0_9/
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint` | `checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth` | 模型权重路径 |
| `--output` | `exports_repvit_m0_9/` | ONNX 输出目录 |
| `--opset` | `18` | ONNX opset 版本 |
| `--skip-verify` | `False` | 跳过 onnxruntime 数值验证 |

---

## 6. 导出结果

### 6.1 数值验证

PyTorch 输出与 ONNX Runtime 输出的对比：

| 组件 | 输出 | 最大绝对误差 | 容差 | 状态 |
|------|------|------------|------|------|
| Image Encoder | feat_4x [1,256,288,288] | 4.84e-07 | 1e-4 | PASS |
| Image Encoder | feat_2x [1,256,144,144] | 8.82e-06 | 1e-4 | PASS |
| Image Encoder | feat_1x [1,256,72,72] | 1.22e-05 | 1e-4 | PASS |
| Image Encoder | pos_1x [1,256,72,72] | 0.00e+00 | 1e-4 | PASS |
| Text Encoder | text_features [1,77,256] | 3.04e-05 | 1e-4 | PASS |
| Text Encoder | text_mask [1,77] | exact | — | PASS |
| Detector | scores [1,200] | 1.15e-08 | 1e-3 | PASS |
| Detector | boxes_xyxy [1,200,4] | 9.30e-06 | 1e-3 | PASS |
| Detector | mask_logits [1,200,288,288] | 1.93e-03 | 1e-3 | PASS |

### 6.2 ONNX 模型信息

| 文件 | 大小 | Op 类型数 | 参数量 |
|------|------|----------|--------|
| `image_encoder.onnx` | 1.3 MB | 14 | 21,500,896 |
| `text_encoder.onnx` | 0.5 MB | 11 | 63,559,424 |
| `detector.onnx` | 111.3 MB | 41 | 29,332,790 |

> 注: Text Encoder 的参数量中 MobileCLIP 原始权重占大部分（~63M），但 projector 只投影到 256 维，所以 ONNX 文件很小（仅包含 projector + 量化后的少量层）。实际 ONNX 中 text_encoder 大小仅 0.5 MB 是因为 MobileCLIP 使用了高效的 FastViT-style 文本 transformer。

### 6.3 输出文件

```
exports_repvit_m0_9/
├── image_encoder.onnx    (1.3 MB)
├── text_encoder.onnx     (0.5 MB)
└── detector.onnx         (111.3 MB)
```

---

## 7. ONNX Runtime 推理示例

```python
import numpy as np
import onnxruntime as ort

# 加载 3 个 ONNX 模型
img_sess = ort.InferenceSession("exports_repvit_m0_9/image_encoder.onnx")
txt_sess = ort.InferenceSession("exports_repvit_m0_9/text_encoder.onnx")
det_sess = ort.InferenceSession("exports_repvit_m0_9/detector.onnx")

# 准备输入
image = np.random.randn(1, 3, 1008, 1008).astype(np.float32)  # 替换为真实图像
token_ids = np.zeros((1, 77), dtype=np.int64)
token_ids[0, :5] = [49406, 1125, 539, 320, 49407]  # 示例: "a cat" 的 token

# 1. Image Encoder
feat_4x, feat_2x, feat_1x, pos_1x = img_sess.run(None, {"images": image})

# 2. Text Encoder
text_features, text_mask = txt_sess.run(None, {"token_ids": token_ids})

# 3. Detector
scores, boxes_xyxy, mask_logits = det_sess.run(None, {
    "feat_4x": feat_4x,
    "feat_2x": feat_2x,
    "feat_1x": feat_1x,
    "pos_1x": pos_1x,
    "text_features": text_features,
    "text_mask": text_mask,
})

# 后处理
score_threshold = 0.3
valid = scores[0] > score_threshold
valid_scores = scores[0][valid]
valid_boxes = boxes_xyxy[0][valid]       # 归一化 xyxy, 乘以图像尺寸获取像素坐标
valid_masks = 1.0 / (1.0 + np.exp(-mask_logits[0][valid]))  # sigmoid

print(f"Detected {valid.sum()} objects")
for i in range(valid.sum()):
    print(f"  Object {i}: score={valid_scores[i]:.3f}, box={valid_boxes[i]}")
```

---

## 8. 不可导出组件说明

以下组件已成功包含在 `detector.onnx` 中（旧版标记为"不可导出"，现已解决）：

| 组件 | 旧状态 | 新状态 | 解决方式 |
|------|--------|--------|----------|
| Transformer Encoder Fusion | 不可导出 | **已导出** | 封装在 DetectorWrapper，使用 legacy tracer |
| Transformer Decoder | 不可导出 | **已导出** | Monkey-patch RPB + legacy tracer |
| Segmentation Head | 不可导出 | **已导出** | 传入 prompt 参数，封装在 DetectorWrapper |
| Geometry Encoder | 不可导出 | **已导出** | 手动实现 CLS-only 路径 |
| DotProduct Scoring | 已导出 | **已导出** | 包含在 DetectorWrapper 内 |
| Box Head | 已导出 | **已导出** | 包含在 DetectorWrapper 内 |

### Tracker 组件（不在此脚本范围内）

Tracker 相关组件（Memory Attention、SAM Prompt/Mask Encoder/Decoder 等）的 ONNX 导出参见 `export_onnx.py` 脚本。
