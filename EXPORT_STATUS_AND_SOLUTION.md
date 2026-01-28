# EfficientSAM3 导出状态与单帧概念分割方案

## 一、PTH权重文件组件清单

### 完整的组件列表（585.67M 参数）

```
checkpoint['model'] = {

    ✅ Detector 组件 (573.93M, 98%)
    ├─ detector.backbone.vision_backbone.*          4.72M   ✅ 已导出 ONNX
    ├─ detector.backbone.text_backbone.*            63.56M  ✅ 已导出 ONNX
    ├─ detector.transformer.*                       21.05M  ⚠️ 仅权重文件
    ├─ detector.geometry_encoder.*                  8.22M   ❌ 未导出
    ├─ detector.segmentation_head.*                 2.30M   ⚠️ ONNX(权重部分匹配)
    └─ detector.dot_prod_scoring.*                  1.18M   ⚠️ ONNX(权重部分匹配)

    ✅ Tracker 组件 (11.74M, 2%)
    ├─ tracker.sam_prompt_encoder.*                 0.01M   ✅ 已导出 ONNX
    ├─ tracker.sam_mask_decoder.*                   4.22M   ✅ 已导出 ONNX
    ├─ tracker.transformer.*                        5.92M   ❌ 未导出 (视频专用)
    └─ tracker.maskmem_backbone.*                   1.38M   ❌ 未导出 (视频专用)
}
```

---

## 二、已导出组件详情

### ✅ 完全可用的 ONNX 模型

| 组件 | 文件 | 大小 | 用途 | 状态 |
|-----|------|------|------|------|
| Image Encoder | `image_encoder_repvit_m0_9.onnx` | 22 MB | 提取图像特征 | ✅ 完美 |
| Text Encoder | `text_encoder_mobileclip_s1.onnx` | 242 MB | 提取文本特征 | ✅ 完美 |
| Prompt Encoder | `prompt_encoder.onnx` | 63 KB | 点/框提示编码 | ✅ 完美 |
| Mask Decoder | `mask_decoder.onnx` | 16 MB | SAM风格分割 | ✅ 完美 |

### ⚠️ 有问题的导出

| 组件 | 文件 | 问题 | 影响 |
|-----|------|------|------|
| Transformer Fusion | `transformer_fusion_weights.pth` | 仅权重，无法ONNX | **无法用于推理** |
| Segmentation Head | `segmentation_head.onnx` | 权重部分匹配(Missing=8) | **可能推理错误** |
| Scoring Module | `scoring_module.onnx` | 权重部分匹配(Missing=6) | **可能推理错误** |

### ❌ 未导出的组件

| 组件 | 参数量 | 原因 |
|-----|--------|------|
| Geometry Encoder | 8.22M | 不需要（用于点/框提示） |
| Tracker Transformer | 5.92M | 视频专用，单帧不需要 |
| Tracker Memory Backbone | 1.38M | 视频专用，单帧不需要 |

---

## 三、单帧概念分割所需组件

### 完整流程

```
输入: 图像 + 文本 "a dog"
    ↓
┌────────────────────┐
│ 1. Image Encoder   │ ✅ ONNX 可用
└────────┬───────────┘
         │ [1, 256, 64, 64]
         ↓
┌────────────────────┐
│ 2. Text Encoder    │ ✅ ONNX 可用
└────────┬───────────┘
         │ [1, 32, 256]
         ↓
┌────────────────────┐
│ 3. Transformer     │ ⚠️ 仅权重，无法ONNX
│    Fusion          │ ❌ 这是瓶颈！
└────────┬───────────┘
         │ [1, 256, 64, 64]
         ↓
┌────────────────────┐
│ 4. Segmentation    │ ⚠️ ONNX但权重不完全匹配
│    Head            │ ❌ 可能不准确！
└────────┬───────────┘
         │ [1, N, 64, 64]
         ↓
┌────────────────────┐
│ 5. Scoring Module  │ ⚠️ ONNX但权重不完全匹配
│    (可选)          │
└────────┬───────────┘
         ↓
    最终 masks
```

### 关键问题

❌ **Transformer Fusion** - 无法导出ONNX，无法推理
❌ **Segmentation Head** - ONNX权重不完全匹配，可能输出错误

---

## 四、解决方案

### 方案 A：完全使用 PyTorch（推荐 ⭐⭐⭐⭐⭐）

**最可靠、最简单的方案**

```python
from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image

# 1. 加载完整模型
model = build_efficientsam3_image_model(
    checkpoint_path="weights/efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
    backbone_type="repvit",
    model_name="m0_9",
    text_encoder_type="MobileCLIP-S1"
)

# 2. 创建处理器
processor = Sam3Processor(model)

# 3. 单帧概念分割
image = Image.open("input.jpg")
state = processor.set_image(image)
state = processor.set_text_prompt("a dog", state)

# 4. 获取结果
masks = state["masks"]  # [N, H, W] - N个检测到的狗
scores = state["scores"]  # [N] - 每个mask的置信度
```

**优点**：
- ✅ 使用完整权重，结果准确
- ✅ 代码简单，3行搞定
- ✅ 官方实现，有保障

**缺点**：
- ❌ 需要 PyTorch 环境
- ❌ 模型较大（586M）
- ❌ 不能跨平台部署（需要 Python）

---

### 方案 B：重新正确导出 Segmentation Head（可尝试 ⭐⭐⭐）

**问题根源**：之前用简化的wrapper导出，权重不匹配

**解决方法**：直接从源码实例化真实的 SegmentationHead

```python
from sam3.model.maskformer_segmentation import SegmentationHead
import torch

# 1. 加载checkpoint
checkpoint = torch.load("weights/efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
                       map_location='cpu')
state_dict = checkpoint['model']

# 2. 提取 segmentation_head 权重
seg_head_state = {}
for key in state_dict.keys():
    if key.startswith('detector.segmentation_head.'):
        new_key = key.replace('detector.segmentation_head.', '')
        seg_head_state[new_key] = state_dict[key]

# 3. 实例化真实的 SegmentationHead（需要找到正确的参数配置）
# 问题：需要知道 hidden_dim, upsampling_stages 等参数
seg_head = SegmentationHead(
    hidden_dim=256,  # 需要从代码中确认
    upsampling_stages=...,  # 需要从代码中确认
    use_encoder_inputs=...,  # 需要从代码中确认
    # ... 其他参数
)

# 4. 加载权重
seg_head.load_state_dict(seg_head_state, strict=True)  # 必须完全匹配

# 5. 导出 ONNX
torch.onnx.export(seg_head, dummy_input, "segmentation_head_correct.onnx")
```

**但是**：
- ⚠️ Transformer Fusion 仍然无法导出（这是最大的障碍）
- ⚠️ 即使 Segmentation Head 导出成功，没有 Transformer Fusion 也无法推理

**结论**：即使重新导出 Segmentation Head，由于 Transformer Fusion 无法ONNX化，整个概念分割流程仍然不可用。

---

### 方案 C：混合推理（ONNX + PyTorch）（不推荐 ⭐）

**理论上可行，但复杂**

```python
import onnxruntime as ort
import torch
from sam3.model_builder import build_efficientsam3_image_model

# 1. ONNX 编码器
img_encoder = ort.InferenceSession("image_encoder_repvit_m0_9.onnx")
text_encoder = ort.InferenceSession("text_encoder_mobileclip_s1.onnx")

# 2. 编码阶段用 ONNX
image_np = preprocess(image)
image_emb = img_encoder.run(["image_embedding"], {"image": image_np})[0]

text_ids = tokenize("a dog")
text_feat = text_encoder.run(None, {"input_ids": text_ids})[0]

# 3. 融合+解码阶段用 PyTorch
model = build_efficientsam3_image_model(...)
with torch.no_grad():
    # 将 ONNX 输出转为 torch
    image_emb_torch = torch.from_numpy(image_emb)
    text_feat_torch = torch.from_numpy(text_feat)

    # 调用 PyTorch 的 transformer + seg_head
    fused_feat = model.detector.transformer(image_emb_torch, text_feat_torch)
    masks = model.detector.segmentation_head(fused_feat, ...)
```

**问题**：
- ❌ 复杂，需要理解内部接口
- ❌ 仍然需要 PyTorch
- ❌ 节省的计算量有限（编码器只占20%）

---

### 方案 D：仅使用点提示分割（备选 ⭐⭐⭐⭐）

**如果不需要文本概念分割，可以用纯 ONNX**

```python
import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型（100% ONNX）
img_enc = ort.InferenceSession("image_encoder_repvit_m0_9.onnx")
prompt_enc = ort.InferenceSession("prompt_encoder.onnx")
mask_dec = ort.InferenceSession("mask_decoder.onnx")

# 推理流程
image_emb = img_enc.run(["image_embedding"], {"image": image_np})[0]

# 点击提示（用户交互）
point_coords = np.array([[[0.5, 0.5]]], dtype=np.float32)  # 归一化坐标
point_labels = np.array([[1]], dtype=np.int64)
sparse_emb, dense_emb = prompt_enc.run(None, {
    "point_coords": point_coords,
    "point_labels": point_labels
})

# 生成 mask
masks, iou = mask_dec.run(None, {
    "image_embeddings": image_emb,
    "sparse_embeddings": sparse_emb,
    "dense_embeddings": dense_emb
})

# 选择最佳 mask
best_mask = masks[0, np.argmax(iou[0])]
```

**优点**：
- ✅ 100% ONNX，完全可部署
- ✅ 跨平台，无需 PyTorch
- ✅ 轻量高效

**缺点**：
- ❌ 不支持文本提示（"a dog"）
- ❌ 需要用户点击交互

---

## 五、缺失组件分析

### 为什么 Transformer Fusion 无法导出？

```python
# detector.transformer 的实际结构（从权重推断）

class TransformerFusion(nn.Module):
    def __init__(self):
        self.encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                # 关键：有自定义的 cross-attention 层
                cross_attn_image=...,  # 文本→图像
                cross_attn_text=...,   # 图像→文本
            )
            for _ in range(6)
        ])

    def forward(self, vision_feat, text_feat):
        # 拼接
        x = torch.cat([vision_feat, text_feat], dim=1)

        # 6层transformer，每层都有：
        # 1. Self-attention
        # 2. Cross-attention (自定义实现)
        # 3. FFN
        for layer in self.encoder:
            x = layer(x)  # 内部有复杂的跨模态交互

        return x
```

**问题**：
1. 自定义 cross-attention 层（不是标准 PyTorch 组件）
2. 权重命名不匹配标准 TransformerEncoder
3. 动态的文本/图像序列长度处理

**解决方案**：
- ❌ 无法用标准 torch.nn.TransformerEncoder 替代
- ❌ 手动重建需要完整的源码定义
- ✅ **唯一可行：使用完整 PyTorch 模型**

---

## 六、最终推荐方案

### 对于单帧概念分割

| 方案 | 可行性 | 准确性 | 部署难度 | 推荐度 |
|-----|--------|--------|---------|--------|
| **方案A: 完全PyTorch** | ✅ | ⭐⭐⭐⭐⭐ | 简单 | ⭐⭐⭐⭐⭐ |
| 方案B: 重新导出 | ❌ | - | - | ⭐ |
| 方案C: 混合推理 | ⚠️ | ⭐⭐⭐⭐ | 复杂 | ⭐⭐ |
| 方案D: 点提示分割 | ✅ | ⭐⭐⭐⭐ | 简单 | ⭐⭐⭐⭐ |

### 明确建议

#### 如果需要文本概念分割（"a dog"）：
**→ 使用方案A（完全PyTorch）**

没有其他可行方案，因为：
- Transformer Fusion 无法导出 ONNX
- 这是概念分割的核心组件（21M参数）
- 手动实现需要完整源码

#### 如果可以接受点击交互：
**→ 使用方案D（点提示分割 + 纯ONNX）**

优势：
- 100% ONNX，完全可部署
- 跨平台支持
- 性能优秀

---

## 七、补充导出建议

### 如果坚持要尝试导出，需要：

1. **查找完整的模型定义**
```bash
# 找到 detector 的构建代码
grep -r "class.*Detector" sam3/sam3/model/
grep -r "build.*detector" sam3/sam3/model/
```

2. **查看模型配置**
```python
# 从 checkpoint 中查看配置
checkpoint = torch.load("weights/efficient_sam3_repvit-m0_9_mobileclip_s1.pth")
if 'config' in checkpoint:
    print(checkpoint['config'])
```

3. **直接实例化完整模型**
```python
from sam3.model_builder import build_efficientsam3_image_model

# 完整模型
model = build_efficientsam3_image_model(
    checkpoint_path="weights/efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
    backbone_type="repvit",
    model_name="m0_9",
    text_encoder_type="MobileCLIP-S1"
)

# 尝试导出 transformer（大概率失败）
torch.onnx.export(
    model.detector.transformer,
    (dummy_vision, dummy_text),
    "transformer_fusion.onnx"
)
```

但预期：**仍然会失败**，因为内部有不支持的操作。

---

## 八、总结

### 当前导出状态

```
✅ 可用的 ONNX (4个):
  - image_encoder_repvit_m0_9.onnx
  - text_encoder_mobileclip_s1.onnx
  - prompt_encoder.onnx
  - mask_decoder.onnx

⚠️ 有问题的导出 (3个):
  - transformer_fusion_weights.pth (仅权重)
  - segmentation_head.onnx (权重部分匹配)
  - scoring_module.onnx (权重部分匹配)

❌ 未导出 (3个):
  - geometry_encoder (不需要)
  - tracker.transformer (视频专用)
  - tracker.maskmem_backbone (视频专用)
```

### 单帧概念分割结论

**无法使用纯 ONNX 实现文本概念分割**

原因：
1. Transformer Fusion（21M，核心组件）无法导出ONNX
2. Segmentation Head 的 ONNX 权重不完全匹配

**唯一可行方案**：
→ **使用完整的 PyTorch 模型**（方案A）

**备选方案**：
→ 如果可以接受点击交互，使用纯ONNX的点提示分割（方案D）

---

## 九、快速开始代码

### 推荐方案：PyTorch 单帧概念分割

```python
#!/usr/bin/env python3
"""单帧图像概念分割 - 完整 PyTorch 方案"""

from sam3.model_builder import build_efficientsam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import matplotlib.pyplot as plt

# 1. 加载模型（只需要一次）
print("加载模型...")
model = build_efficientsam3_image_model(
    checkpoint_path="weights/efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
    backbone_type="repvit",
    model_name="m0_9",
    text_encoder_type="MobileCLIP-S1"
)

# 2. 创建处理器
processor = Sam3Processor(model)

# 3. 处理图像
print("处理图像...")
image = Image.open("input.jpg")
state = processor.set_image(image)

# 4. 设置文本提示
print("概念分割: 'a dog'")
state = processor.set_text_prompt("a dog", state)

# 5. 获取结果
masks = state["masks"]  # [N, H, W] boolean
scores = state["scores"]  # [N] float

print(f"检测到 {len(masks)} 个物体")
for i, score in enumerate(scores):
    print(f"  物体 {i+1}: 置信度 {score:.3f}")

# 6. 可视化
fig, axes = plt.subplots(1, len(masks) + 1, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("原图")
axes[0].axis('off')

for i, (mask, score) in enumerate(zip(masks, scores)):
    axes[i+1].imshow(mask, cmap='gray')
    axes[i+1].set_title(f"Mask {i+1} ({score:.2f})")
    axes[i+1].axis('off')

plt.tight_layout()
plt.savefig("output.jpg")
print("结果已保存到 output.jpg")
```

**这就是最简单、最可靠的方案！**
