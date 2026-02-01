# EfficientSAM3 ONNX 导出指南

## 概述

本文档记录 EfficientSAM3 Detector 和 Tracker 组件的 ONNX 导出现状、配置参数和使用指南。

**导出状态**：
- ✅ **Detector**：4/4 组件可导出，全部验证通过
- ✅ **Tracker**：6/6 组件可导出，全部验证通过
- ❌ **不可导出**：Memory Attention、Geometry Encoder、Transformer Fusion

---

## Detector 组件导出

### 导出脚本
**文件**：`export_detector_components.py`

```bash
python export_detector_components.py \
    --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
    --output exports_repvit_m0_9/
```

### 1. Image Encoder (RepViT)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `detector.backbone.vision_backbone.` |
| 架构 | RepViT(384ch) → Head(384→1024) → Neck(1024→256) |
| 输入 | `images [B, 3, 1152, 1152]` |
| 输出 | `image_embeddings [B, 256, 72, 72]` |
| 参数量 | ~8.4M |
| 文件大小 | 1105.8 KB |
| 验证误差 | max_abs=3.42e-05 (atol=1e-4) |
| opset | 18 |

**关键配置**：
- Backbone: RepViT-m0.9（3 层 RepVGG block）
- Head: embed_dim=1024（不是 256！）
- Neck: 1024→256 投影，来自 `detector.backbone.vision_backbone.convs.2.*`
- 输入分辨率 1152×1152（不是 1024×1024）

**权重加载**：
```python
# Trunk 权重
trunk_weights = {k[len("detector.backbone.vision_backbone.trunk.model."):]: v
                 for k, v in state_dict.items()
                 if k.startswith("detector.backbone.vision_backbone.trunk.model.")}

# Neck 权重映射：convs.2.* → neck.*
neck_weights = {}
for k, v in state_dict.items():
    if k.startswith("detector.backbone.vision_backbone.convs.2."):
        new_key = k.replace("detector.backbone.vision_backbone.convs.2.", "")
        neck_weights[new_key] = v
```

---

### 2. Text Encoder (MobileCLIP-S1)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `detector.backbone.language_backbone.` |
| 架构 | MobileCLIP-S1（12 层 Transformer） |
| 输入 | `texts [B, max_text_len=77]`（token IDs） |
| 输出 | `text_embeddings [B, 77, 256]` |
| 参数量 | ~13.6M |
| 文件大小 | 659.5 KB |
| 验证误差 | max_abs=7.51e-05 (atol=1e-4) |
| opset | 18 |

**关键配置**：
- 12 层 Transformer encoder，8 head
- Embed dim = 256
- Max sequence length = 77（CLIP 标准）

**权重加载**：
```python
weights = {k[len("detector.backbone.language_backbone."):]: v
           for k, v in state_dict.items()
           if k.startswith("detector.backbone.language_backbone.")}
```

---

### 3. DotProductScoring (评分模块)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `detector.dot_prod_scoring.` |
| 架构 | 均值池化 + 投影 + 点积 + clamp |
| 输入 | `image_feats [B, 256, 72, 72]`，`text_feats [B, 77, 256]` |
| 输出 | `scores [B, 77]` |
| 参数量 | ~66K |
| 文件大小 | 23.6 KB |
| 验证误差 | max_abs=9.54e-07 (atol=1e-6) |
| opset | 18 |

**关键配置**：
- Image feat 均值池化：[B, 256, 72, 72] → [B, 256]
- Text feat 已是 [B, 77, 256]
- 投影：256 → 256
- 最终点积 + clamp(-inf, 100)

---

### 4. Box Head (边界框回归头)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `detector.transformer.decoder.bbox_embed.` |
| 架构 | MLP(256 → 256 → 256 → 4) |
| 输入 | `decoder_output [B, 256]` |
| 输出 | `bbox_logits [B, 4]` |
| 参数量 | ~197K |
| 文件大小 | 8.8 KB |
| 验证误差 | max_abs=1.19e-07 (atol=1e-6) |
| opset | 18 |

**关键配置**：
- 3 层 MLP，隐层 256 维
- ReLU 激活
- 输出 4 个值（x, y, w, h 或相似坐标）

---

### 导出结果

```
Detector ONNX 文件:
  image_encoder_repvit_m0_9.onnx     1105.8 KB
  text_encoder_mobileclip_s1.onnx     659.5 KB
  dot_prod_scoring.onnx                23.6 KB
  bbox_head.onnx                        8.8 KB
```

**验证总结**：4/4 通过 ✅

---

## Tracker 组件导出

### 导出脚本
**文件**：`export_tracker_components.py`

```bash
python export_tracker_components.py \
    --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
    --output exports_repvit_m0_9/
```

### 1. SAM Prompt Encoder

| 属性 | 值 |
|------|-----|
| 权重前缀 | `tracker.sam_prompt_encoder.` |
| 架构 | SAM PromptEncoder（固定 points 输入） |
| 输入 | `point_coords [B, N, 2]`（归一化 0-1），`point_labels [B, N]`（1=前景，0=背景） |
| 输出 | `sparse_embeddings [B, N+1, 256]`，`dense_embeddings [B, 256, 64, 64]` |
| 参数量 | 6.2K |
| 文件大小 | 44.0 KB |
| 验证误差 | max_abs=1.19e-07 (atol=1e-5) |
| opset | 18 |

**关键设计**：
- 内部调用 `pe(points=(coords, labels), boxes=None, masks=None)`
- Wrapper 处理 None 检查，使 ONNX 兼容
- Dense embedding 使用位置编码，分辨率 64×64

---

### 2. SAM Mask Decoder

| 属性 | 值 |
|------|-----|
| 权重前缀 | `tracker.sam_mask_decoder.` |
| 架构 | TwoWayTransformer(depth=2) + 双head输出 |
| 输入 | `image_embeddings [B, 256, 64, 64]`，`sparse_embeddings [B, N, 256]`，`dense_embeddings [B, 256, 64, 64]` |
| 输出 | `masks [B, 4, 256, 256]`（1个单mask + 3个multi-mask），`iou_predictions [B, 4]` |
| 参数量 | 4.19M |
| 文件大小 | 546.2 KB |
| 验证误差 | max_abs=1.91e-05 (atol=1e-4) |
| opset | 18 |

**关键配置**：
```python
MaskDecoder(
    num_multimask_outputs=3,
    transformer=TwoWayTransformer(depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048),
    transformer_dim=256,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    pred_obj_scores=True,      # ✅ 必须
    pred_obj_scores_mlp=True,  # ✅ 必须
    use_high_res_features=False,# ✅ 简单导出时 False
)
```

**权重加载**：
- Missing: 0，Unexpected: 4（conv_s0/s1，仅用于高分辨率路径）
- 固定 `multimask_output=True, repeat_image=False`

---

### 3. Memory Encoder (maskmem_backbone)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `tracker.maskmem_backbone.` |
| 架构 | SimpleMaskEncoder = 下采样 + pix_feat投影 + CXBlock融合 + 位置编码 + 输出投影 |
| 输入 | `pix_feat [B, 256, 72, 72]`，`masks [B, 1, 256, 256]`（任意分辨率） |
| 输出 | `vision_features [B, 64, 72, 72]`，`vision_pos_enc [B, 64, 72, 72]` |
| 参数量 | 1.38M |
| 文件大小 | 204.8 KB |
| 验证误差 | max_abs=5.70e-05 (atol=1e-4) |
| opset | 18 |

**关键配置**：
```python
SimpleMaskEncoder(
    out_dim=64,
    mask_downsampler=SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1,
        interpol_size=[1152, 1152],  # 掩码插值到 1152×1152
    ),
    position_encoding=PositionEmbeddingSine(
        num_pos_feats=64, normalize=True,
        precompute_resolution=None,  # ⚠️ CPU-only：避免 CUDA 硬编码
    ),
    fuser=SimpleFuser(CXBlock(dim=256, kernel_size=7), num_layers=2),
)
```

**输入分辨率链**：
- 掩码：[B, 1, 256, 256] → 插值到 [B, 1, 1152, 1152]
- 下采样：stride 2^4 = 16 → [B, *, 72, 72]
- pix_feat：需要 [B, 256, 72, 72]

---

### 4. obj_ptr_proj (MLP)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `tracker.obj_ptr_proj.` |
| 架构 | MLP(256 → 256 → 256 → 256)，ReLU |
| 输入 | `sam_tokens [B, 256]` |
| 输出 | `obj_ptr [B, 256]` |
| 参数量 | 197K |
| 文件大小 | 6.7 KB |
| 验证误差 | max_abs=1.19e-07 (atol=1e-6) |
| opset | 18 |

---

### 5. obj_ptr_tpos_proj (Linear)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `tracker.obj_ptr_tpos_proj.` |
| 架构 | Linear(256 → 64) |
| 输入 | `tpos_embed [B, 256]` |
| 输出 | `tpos_proj [B, 64]` |
| 参数量 | 16.4K |
| 文件大小 | 2.0 KB |
| 验证误差 | max_abs=3.58e-07 (atol=1e-6) |
| opset | 18 |

---

### 6. mask_downsample (Conv2d)

| 属性 | 值 |
|------|-----|
| 权重前缀 | `tracker.mask_downsample.` |
| 架构 | Conv2d(1, 1, kernel_size=4, stride=4) |
| 输入 | `mask [B, 1, 256, 256]` |
| 输出 | `downsampled [B, 1, 64, 64]` |
| 参数量 | 17 |
| 文件大小 | 1.8 KB |
| 验证误差 | max_abs=9.54e-07 (atol=1e-6) |
| opset | 18 |

---

### 导出结果

```
Tracker ONNX 文件:
  prompt_encoder.onnx                 44.0 KB
  mask_decoder.onnx                  546.2 KB
  memory_encoder.onnx                204.8 KB
  obj_ptr_proj.onnx                    6.7 KB
  obj_ptr_tpos_proj.onnx               2.0 KB
  mask_downsample.onnx                 1.8 KB
```

**验证总结**：6/6 通过 ✅

---

## 不可导出的组件

### 1. Memory Attention (tracker.transformer.encoder)

**权重**：106 keys

**阻挡因素**：
1. `isinstance(src, list)` — 运行时类型检查
2. `isinstance(cross_attn, RoPEAttention)` — 动态分发
3. RoPEAttention 动态频率缓存重新计算
4. `num_k_exclude_rope` — 变长切片
5. Dict 返回类型

**替代方案**：在目标推理框架（如 NPU SDK）中实现 TransformerEncoderCrossAttention

---

### 2. Geometry Encoder (SequenceGeometryEncoder)

**权重**：~300+ keys

**阻挡因素**：
- 动态分支：`if geometry_type == "roi"`
- ROI 操作不兼容 ONNX（Detectron2 特定）
- 对象类别、轨迹索引的动态列表处理

---

### 3. Transformer Fusion (detector.transformer)

**权重**：~1000+ keys

**阻挡因素**：
- List[Tensor] 作为输入/输出
- 动态特征融合分支
- DAC（动态自适应计算）条件分支

---

### 参数张量（直接加载，无需 ONNX）

| Key | Shape | 说明 |
|-----|-------|------|
| `tracker.maskmem_tpos_enc` | [7, 1, 1, 64] | 内存时间位置编码 |
| `tracker.no_mem_embed` | [1, 1, 256] | 无内存嵌入 token |
| `tracker.no_mem_pos_enc` | [1, 1, 256] | 无内存位置编码 |
| `tracker.no_obj_ptr` | [1, 256] | 无对象指针 |
| `tracker.no_obj_embed_spatial` | [1, 64] | 无对象空间嵌入 |

---

## 验证精度指南

### 浮点误差容差

| 组件类型 | atol | rtol | 说明 |
|---------|------|------|------|
| 单层 MLP/Linear | 1e-6 | 1e-4 | 最小误差 |
| Conv2d | 1e-6 | 1e-4 | Conv 操作精度高 |
| Prompt Encoder | 1e-5 | 1e-3 | 位置编码精度 |
| Mask Decoder | 1e-4 | 1e-3 | Transformer 累积误差 |
| Memory Encoder | 1e-4 | 1e-3 | 多层 Conv + CXBlock + LayerNorm |

**原因**：
- ONNX 运行时和 PyTorch 的浮点运算顺序不同
- Transformer 层中自注意力、LayerNorm 累积精度损失
- 多层卷积 + 激活函数的级联误差

---

## 使用示例

### Python 推理

```python
import onnxruntime as ort
import numpy as np

# 加载会话
sess = ort.InferenceSession("exports_repvit_m0_9/image_encoder_repvit_m0_9.onnx")

# 准备输入
images = np.random.randn(1, 3, 1152, 1152).astype(np.float32)

# 推理
outputs = sess.run(None, {"images": images})
image_embeddings = outputs[0]  # [1, 256, 72, 72]

print(f"Image embeddings shape: {image_embeddings.shape}")
```

### 端到端流程

```
1. Image Encoder: [B, 3, 1152, 1152] → [B, 256, 72, 72]
2. Text Encoder: [B, 77] (tokens) → [B, 77, 256]
3. DotProductScoring: [B, 256, 72, 72] + [B, 77, 256] → [B, 77]
4. Box Head: [B, 256] → [B, 4]

Tracker:
1. Prompt Encoder: coords + labels → sparse_emb [B, N+1, 256] + dense_emb [B, 256, 64, 64]
2. Mask Decoder: image_emb + prompt_emb → masks [B, 4, 256, 256] + iou [B, 4]
3. Memory Encoder: pix_feat + masks → vision_feat [B, 64, 72, 72] + pos_enc [B, 64, 72, 72]
```

---

## 故障排查

### 权重加载缺失 (Missing keys)

**问题**：导出时输出 `Missing: N` 个权重

**原因**：权重前缀不匹配或模型架构配置错误

**解决**：
1. 检查 checkpoint 中的实际权重前缀（用 `torch.load()` 打印 `state_dict.keys()`）
2. 验证模型架构参数是否匹配（如 embed_dim、num_layers 等）
3. 更新前缀或架构配置

**示例**：
```python
checkpoint = torch.load("path/to/ckpt.pth", map_location="cpu")
state_dict = checkpoint.get("model", checkpoint)
prefixes = set()
for k in state_dict.keys():
    prefix = k.split('.')[0] + '.' + k.split('.')[1] + '.'
    prefixes.add(prefix)
print("Available prefixes:", sorted(prefixes))
```

### Memory Encoder 形状不匹配

**问题**：`pix_feat` 形状与 mask downsampler 输出不对齐

**原因**：分辨率链错误

**解决**：
- 掩码输入 → 插值到 1152×1152
- stride 2^4 = 16 下采样 → 72×72
- pix_feat 必须 [B, 256, 72, 72]

### ONNX 检查失败

**问题**：`onnx.checker.check_model()` 失败

**原因**：通常是算子不兼容或维度错误

**解决**：
1. 检查 opset 版本（推荐 18）
2. 用 `onnx.helper.printable_graph()` 查看 ONNX 图
3. 确保所有输入维度正确

---

## NPU 部署注意事项

### 不友好的算子

- `LayerNormalization`（Memory Encoder、Mask Decoder）
- `Gelu`（文本编码器中如有）
- `Attention`（SAM 解码器内）

**建议**：
- 将这些算子转换为等效的 Conv + Gemm 组合
- 或在 NPU SDK 中实现原生支持

### 优化技巧

1. **权重量化**：Int8 量化 CNN 编码器
2. **融合**：Conv + BN + ReLU 融合
3. **剪枝**：移除高分辨率特征路径（use_high_res_features=False）

---

## 文件清单

```
导出脚本:
  export_detector_components.py      导出 Detector 4 个组件
  export_tracker_components.py       导出 Tracker 6 个组件

生成的 ONNX 文件 (exports_repvit_m0_9/):
  Detector (4 files, ~1.8 MB):
    image_encoder_repvit_m0_9.onnx
    text_encoder_mobileclip_s1.onnx
    dot_prod_scoring.onnx
    bbox_head.onnx

  Tracker (6 files, ~805 KB):
    prompt_encoder.onnx
    mask_decoder.onnx
    memory_encoder.onnx
    obj_ptr_proj.onnx
    obj_ptr_tpos_proj.onnx
    mask_downsample.onnx
```

---

## 总结

| 类别 | 可导出 | 已验证 | 文件大小 |
|------|--------|--------|---------|
| Detector 组件 | 4/4 | 4/4 ✅ | ~1.8 MB |
| Tracker 组件 | 6/6 | 6/6 ✅ | ~805 KB |
| 总计 | 10/10 | 10/10 ✅ | ~2.6 MB |

所有可导出组件已通过 PyTorch ↔ ONNX Runtime 数值验证。
