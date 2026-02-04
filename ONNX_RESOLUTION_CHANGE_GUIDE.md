# EfficientSAM3 输入分辨率修改指南

本文档分析将输入分辨率从 1008 改为 512 或 256 对模型和 ONNX 导出脚本的影响。

---

## 目录

1. [结论](#1-结论)
2. [分辨率传播链路](#2-分辨率传播链路)
3. [三种分辨率的完整 shape 对比](#3-三种分辨率的完整-shape-对比)
4. [需要修改的位置](#4-需要修改的位置)
5. [不需要修改的位置（自动适配）](#5-不需要修改的位置自动适配)
6. [分辨率对齐问题](#6-分辨率对齐问题)
7. [导出脚本的改动](#7-导出脚本的改动)
8. [精度影响评估](#8-精度影响评估)

---

## 1. 结论

**导出脚本本身不需要大改**。整个流水线对分辨率几乎完全是动态适配的，只需要：

1. 修改 **1 个输入参数**：`torch.randn(1, 3, 1008, 1008)` → `torch.randn(1, 3, NEW_RES, NEW_RES)`
2. 修改 `model_builder.py` 中 **3 个硬编码值**：`embed_size`、`img_size`、`resolution`
3. 注释中的 shape 说明需要更新

但需要注意**分辨率对齐问题**：RepViT stride=32，512 和 256 都能被 32 整除，不存在对齐问题。

---

## 2. 分辨率传播链路

```
输入图像 [1, 3, R, R]
    │
    ▼
RepViT-m0.9 backbone (stride=32)
    │  R ÷ 32 = S    (1008→31, 512→16, 256→8)
    ▼
ImageStudentEncoder.head (Linear 投影 + interpolate)
    │  interpolate(S×S → E×E)    E = embed_size
    ▼
backbone output [1, 1024, E, E]
    │
    ├─→ neck_conv_4x (ConvTranspose2d ×2) → feat_4x [1, 256, 4E, 4E]
    ├─→ neck_conv_2x (ConvTranspose2d ×1) → feat_2x [1, 256, 2E, 2E]
    └─→ neck_conv_1x (Conv 1×1)           → feat_1x [1, 256, E, E]
                                            → pos_1x [1, 256, E, E]
    │
    ▼
Encoder Fusion (序列长度 = E²)
    │  memory [E², 1, 256]
    ▼
Decoder (RPB 基于 E×E 网格)
    │  hs [6, 1, 200, 256]
    ▼
Segmentation Head (PixelDecoder FPN 上采样)
    │  mask_logits [1, 200, 4E, 4E]
    ▼
输出
```

**关键参数 `embed_size` (E) 决定了整个下游的 shape。**

---

## 3. 三种分辨率的完整 shape 对比

### 推荐的 embed_size 选择

| 输入分辨率 | RepViT 输出 (R÷32) | 推荐 embed_size | 理由 |
|-----------|-------------------|-----------------|------|
| 1008 | 31×31 | **72** | 当前默认值，对齐 ViT stride-14 |
| 512 | 16×16 | **36** | 与 1008 同比例缩放 (72×512/1008≈36) |
| 256 | 8×8 | **18** | 与 1008 同比例缩放 (72×256/1008≈18) |

> 也可以选择其他 embed_size（如 32, 16），但建议保持偶数以便 FPN 上采样对齐。

### 完整 shape 表

| 张量 | 1008 (E=72) | 512 (E=36) | 256 (E=18) |
|------|------------|------------|------------|
| 输入 | [1,3,1008,1008] | [1,3,512,512] | [1,3,256,256] |
| backbone 原始 | [1,1024,31,31] | [1,1024,16,16] | [1,1024,8,8] |
| interpolate 后 | [1,1024,72,72] | [1,1024,36,36] | [1,1024,18,18] |
| **feat_4x** | [1,256,**288**,288] | [1,256,**144**,144] | [1,256,**72**,72] |
| **feat_2x** | [1,256,**144**,144] | [1,256,**72**,72] | [1,256,**36**,36] |
| **feat_1x** | [1,256,**72**,72] | [1,256,**36**,36] | [1,256,**18**,18] |
| pos_1x | [1,256,72,72] | [1,256,36,36] | [1,256,18,18] |
| encoder seq_len | **5184** (72²) | **1296** (36²) | **324** (18²) |
| memory | [5184,1,256] | [1296,1,256] | [324,1,256] |
| RPB matrix | [8,200,5184] | [8,200,1296] | [8,200,324] |
| hs (不变) | [6,1,200,256] | [6,1,200,256] | [6,1,200,256] |
| scores (不变) | [1,200] | [1,200] | [1,200] |
| boxes (不变) | [1,200,4] | [1,200,4] | [1,200,4] |
| **mask_logits** | [1,200,**288**,288] | [1,200,**144**,144] | [1,200,**72**,72] |
| **decoder.onnx 预估大小** | 111 MB | ~111 MB | ~111 MB |
| **image_encoder.onnx** | 1.3 MB | ~1.3 MB | ~1.3 MB |

> decoder.onnx 大小基本不变——参数量不受分辨率影响，中间激活不存储在 ONNX 中。

---

## 4. 需要修改的位置

### 4.1 model_builder.py — 3 处硬编码

```python
# (1) ImageStudentEncoder 的 embed_size 和 img_size
# model_builder.py:877-883
student_encoder = ImageStudentEncoder(
    backbone=wrapped_backbone,
    in_channels=in_channels,
    embed_dim=1024,
    embed_size=72,    # ← 改为 36 (512) 或 18 (256)
    img_size=1008,    # ← 改为 512 或 256（仅用于日志/断言，不影响计算）
)

# (2) Decoder 的 resolution（用于预计算 RPB 坐标缓存）
# model_builder.py:183-184
resolution=1008,  # ← 改为 512 或 256
stride=14,        # ← 改为 512//36=14.2... 不整除！见下文"对齐问题"
```

**注意**: `resolution` 和 `stride` 仅用于 Decoder `__init__` 中**预计算** RPB 坐标缓存：
```python
# decoder.py:278-284
feat_size = resolution // stride   # = 1008 // 14 = 72
coords = self._get_coords(feat_size, feat_size, device="cuda")
```

我们的 monkey-patch 已经跳过了这个预计算（强制 `device="cpu"` + 延迟初始化），所以 `resolution/stride` 实际上**在导出时不生效**。RPB 坐标会在第一次 forward 时根据实际 spatial_shapes 动态计算。

因此实际上只需要改 `embed_size`。

### 4.2 export_image_model_onnx.py — 1 处输入 + 注释

```python
# 当前:
torch.randn(1, 3, 1008, 1008)   # ← 改为 (1, 3, 512, 512) 或 (1, 3, 256, 256)

# 以下仅是注释中的 shape 说明，不影响运行（可选更新）:
# [1, 256, 288, 288] → [1, 256, 144, 144] (512) 或 [1, 256, 72, 72] (256)
# [5184, 1, 256] → [1296, 1, 256] (512) 或 [324, 1, 256] (256)
```

### 4.3 完整修改清单

| 文件 | 行号 | 当前值 | 512 | 256 | 必须改 |
|------|-----|--------|-----|-----|--------|
| `model_builder.py:881` | `embed_size=72` | 36 | 18 | **是** |
| `model_builder.py:882` | `img_size=1008` | 512 | 256 | 否（仅日志） |
| `model_builder.py:183` | `resolution=1008` | 512 | 256 | 否（被 patch 跳过） |
| `model_builder.py:184` | `stride=14` | 14 | 14 | 否（被 patch 跳过） |
| `export_*.py` | `torch.randn(1,3,1008,1008)` | 512 | 256 | **是** |
| `export_*.py` | 注释中的 shape | 更新 | 更新 | 否（仅注释） |

**最少只需改 2 处**：`embed_size` 和输入 tensor 大小。

---

## 5. 不需要修改的位置（自动适配）

以下组件全部基于输入张量的实际 shape 动态计算，**无需任何修改**：

### 5.1 ImageEncoderWrapper

```python
feats = self.student_encoder(images)     # 自动适配任意输入尺寸
feat_4x = self.neck_conv_4x(feats)       # ConvTranspose2d，输出 shape 由输入决定
feat_2x = self.neck_conv_2x(feats)       # 同上
feat_1x = self.neck_conv_1x(feats)       # Conv2d 1×1，保持空间尺寸
pos_1x = self.position_encoding(feat_1x) # PositionEmbeddingSine 动态计算
```

### 5.2 TextEncoderWrapper

完全不涉及图像分辨率，无影响。

### 5.3 DecoderWrapper

```python
# Step 2: Geometry Encoder — 使用 feat_1x 的实际 shape
img_feat_seq = feat_1x.flatten(2).permute(2, 0, 1)   # 自动: [E², 1, 256]

# Step 4: Encoder — 使用实际 H, W
H, W = feat_1x.shape[2], feat_1x.shape[3]             # 动态获取
src_seq = feat_1x.flatten(2).permute(2, 0, 1)          # 自动: [E², 1, 256]
memory_dict = self.encoder(..., feat_sizes=[(H, W)])    # 传入实际尺寸

# Step 5: Decoder — spatial_shapes 由 encoder 返回，RPB 基于实际尺寸计算
spatial_shapes = memory_dict["spatial_shapes"]          # tensor([[E, E]])

# Step 9: Segmentation Head — PixelDecoder 使用 interpolate(size=curr_fpn.shape[-2:])
seg_out = self.segmentation_head(backbone_feats=[feat_4x, feat_2x, feat_1x], ...)
# mask 输出尺寸 = feat_4x 的空间尺寸 = 4E × 4E
```

### 5.4 PositionEmbeddingSine

```python
# position_encoding.py
def forward(self, tensor, mask=None):
    # 完全基于输入 tensor 的 shape 动态计算正弦位置编码
    # precompute_resolution 仅在 __init__ 时使用，我们的 patch 已跳过
    not_mask = ~mask if mask is not None else torch.ones(...)
    y_embed = not_mask.cumsum(1)
    x_embed = not_mask.cumsum(2)
    # ... 后续全部基于 y_embed, x_embed 动态计算
```

### 5.5 TransformerEncoderFusion

```python
# encoder.py:525-530
for i, (h, w) in enumerate(feat_sizes):   # 使用传入的实际 feat_sizes
    src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
```

### 5.6 TransformerDecoder (boxRPB)

```python
# decoder.py:516-521 (在 forward 中)
memory_mask = self._get_rpb_matrix(
    reference_boxes,
    (spatial_shapes[0, 0], spatial_shapes[0, 1]),  # 使用实际的空间尺寸
)
```

我们的 monkey-patch 确保坐标缓存在第一次 forward 时按实际尺寸初始化。

### 5.7 PixelDecoder (FPN 上采样)

```python
# maskformer_segmentation.py:211-218
prev_fpn = backbone_feats[-1]   # feat_1x: [1, 256, E, E]
for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
    curr_fpn = bb_feat          # feat_2x 或 feat_4x
    prev_fpn = curr_fpn + F.interpolate(
        prev_fpn, size=curr_fpn.shape[-2:],  # ← 动态匹配上一级的尺寸
        mode=self.interpolation_mode
    )
```

---

## 6. 分辨率对齐问题

### RepViT stride = 32

RepViT-m0.9 的实际 stride 是 **32**（4×patch_embed + 3×stride-2 block）。

| 输入 R | R ÷ 32 | 整除 | backbone 输出 |
|--------|--------|------|--------------|
| 1008 | 31.5 | **否** | ~31×31（padding 处理） |
| 512 | 16 | **是** | 16×16 |
| 256 | 8 | **是** | 8×8 |
| 224 | 7 | **是** | 7×7 |

512 和 256 都能被 32 整除，**比 1008 更干净**。1008 实际上会产生非整数（31.5），RepViT 内部通过 padding 处理。

### embed_size 选择建议

| 输入 R | 推荐 embed_size | 说明 |
|--------|----------------|------|
| 512 | **36** | 同比例：72×512/1008≈36.6→36。偶数，FPN 友好 |
| 512 | **32** | backbone 原始输出 16×16 的 2 倍。偶数 |
| 256 | **18** | 同比例：72×256/1008≈18.3→18。偶数 |
| 256 | **16** | backbone 原始输出 8×8 的 2 倍。偶数 |

embed_size 越大，细节越多但计算量越大。可以根据精度/速度需求调整。

---

## 7. 导出脚本的改动

### 方案 A：最简改法（推荐）

只改 2 处，其他全部自动适配：

```python
# model_builder.py:881
embed_size=36,   # 原来 72，改为 36（512 输入）

# export_image_model_onnx.py:
torch.randn(1, 3, 512, 512)   # 原来 1008
```

### 方案 B：参数化（更灵活）

给导出脚本添加 `--resolution` 参数，自动计算 embed_size：

```python
parser.add_argument("--resolution", type=int, default=1008)

# 在 build_model 之前动态修改 embed_size
resolution = args.resolution
embed_size = round(72 * resolution / 1008)
if embed_size % 2 != 0:
    embed_size += 1   # 保持偶数
```

但这需要修改 `model_builder.py` 接受 `embed_size` 参数（当前是硬编码的）。

### 方案 C：不改 model_builder，只改导出脚本

在导出脚本中 monkey-patch `ImageStudentEncoder.embed_size`：

```python
# 构建模型后修改 embed_size
model.backbone.vision_backbone.trunk.model.embed_size = 36
```

但这种方式不够干净，需要确认 `embed_size` 没有在 `__init__` 中用于其他初始化。

---

## 8. 精度影响评估

### 分辨率降低的影响

| 方面 | 1008→512 | 1008→256 |
|------|----------|----------|
| backbone 特征图 | 31×31→16×16 | 31×31→8×8 |
| encoder 序列长度 | 5184→1296 (↓75%) | 5184→324 (↓94%) |
| 小物体检测 | 中度下降 | 严重下降 |
| mask 精度 | 288×288→144×144 | 288×288→72×72 |
| 推理速度 | **提升 ~3-4×** | **提升 ~10-15×** |
| 内存占用 | **减少 ~60%** | **减少 ~90%** |

### 关键注意事项

1. **权重通用性**: RepViT 的卷积权重不依赖输入分辨率，可以直接使用原始 checkpoint。`ImageStudentEncoder.head`（Linear 投影）也不依赖空间尺寸。唯一的变化是 interpolate 的目标尺寸。

2. **RPB 坐标**: boxRPB 的坐标网格是 `linspace(0, 1, E)` 归一化的，分辨率变化只改变网格密度，不需要重新训练。

3. **query 数量不变**: 200 个 object query 与分辨率无关。但当特征图很小时（如 8×8=64 个 token），200 个 query 可能过多，可以适当减少。

4. **无需重新训练**: 纯推理层面改分辨率是可行的，模型应该能在不同分辨率下工作（精度有损失）。如果要恢复精度，建议在目标分辨率上 finetune。
