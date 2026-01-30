# Transformer Fusion 详细分析：Encoder 和 Decoder

本文档详细讲解 EfficientSAM3 中 Transformer Fusion 的两个核心部分：**TransformerEncoderFusion** 和 **TransformerDecoder**。

---

## 整体架构

```
                    Transformer Fusion 内部
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌────────────────────────────────────────────────────┐     │
│   │            TransformerEncoderFusion                 │     │
│   │            (encoder.py:462)                        │     │
│   │                                                    │     │
│   │  输入:                                              │     │
│   │    src = 图像特征 (H*W, B, 256)                     │     │
│   │    prompt = text+geo 拼接 (seq, B, 256)             │     │
│   │                                                    │     │
│   │  Step 0: 文本池化融合                                │     │
│   │    pooled_text = mean(prompt)                       │     │
│   │    src = src + linear(pooled_text)  ← 全局语义注入   │     │
│   │                                                    │     │
│   │  Step 1~N: TransformerEncoderLayer × N层            │     │
│   │    每层做:                                          │     │
│   │    ┌──────────────────────────────────────────┐    │     │
│   │    │ 1. Self-Attention:                       │    │     │
│   │    │    Q=K=V = src (图像token之间互相看)       │    │     │
│   │    │                                         │    │     │
│   │    │ 2. Cross-Attention:                     │    │     │
│   │    │    Q = src (图像特征)                     │    │     │
│   │    │    K = V = prompt (text+geo)             │    │     │
│   │    │    → 图像去"读"文本语义                   │    │     │
│   │    │                                         │    │     │
│   │    │ 3. FFN                                  │    │     │
│   │    └──────────────────────────────────────────┘    │     │
│   │                                                    │     │
│   │  输出: memory (H*W, B, 256) ← 融合了文本语义的图像特征│     │
│   └────────────────────────────────────────────────────┘     │
│                          │                                   │
│                          ▼                                   │
│   ┌────────────────────────────────────────────────────┐     │
│   │            TransformerDecoder                       │     │
│   │            (decoder.py:190)                        │     │
│   │                                                    │     │
│   │  输入:                                              │     │
│   │    tgt = obj_queries (N, B, 256) ← nn.Embedding     │     │
│   │    memory = Encoder输出 (H*W, B, 256)               │     │
│   │    memory_text = prompt (seq, B, 256)               │     │
│   │    presence_token (1, B, 256) ← nn.Embedding        │     │
│   │                                                    │     │
│   │  每层 TransformerDecoderLayer:                       │     │
│   │    ┌──────────────────────────────────────────┐    │     │
│   │    │ 1. Self-Attention:                       │    │     │
│   │    │    Q=K=V = [presence, obj_queries]       │    │     │
│   │    │    → token之间互相看                      │    │     │
│   │    │                                         │    │     │
│   │    │ 2. Text Cross-Attention (可选):          │    │     │
│   │    │    Q = obj_queries                       │    │     │
│   │    │    K = V = prompt (text+geo)             │    │     │
│   │    │    → 再次读文本语义                       │    │     │
│   │    │                                         │    │     │
│   │    │ 3. Image Cross-Attention:               │    │     │
│   │    │    Q = [presence, obj_queries]           │    │     │
│   │    │    K = V = memory (Encoder输出)          │    │     │
│   │    │    → 去图像中"找"物体                    │    │     │
│   │    │                                         │    │     │
│   │    │ 4. FFN                                  │    │     │
│   │    │                                         │    │     │
│   │    │ 5. 拆分: presence=tgt[:1], obj=tgt[1:]  │    │     │
│   │    └──────────────────────────────────────────┘    │     │
│   │                                                    │     │
│   │  每层之后:                                          │     │
│   │    reference_boxes 迭代精修 (box refine)            │     │
│   │    presence_logit = MLP(norm(presence_out))         │     │
│   │                                                    │     │
│   │  输出:                                              │     │
│   │    hs: (layers, N, B, 256) ← 每层obj_queries输出   │     │
│   │    reference_boxes: (layers, N, B, 4) ← 精修后的框  │     │
│   │    presence_logit: (layers, B) ← 存在性判断         │     │
│   └────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

---

## 1. TransformerEncoderFusion 详解

### 文件位置
- **定义**: `sam3/sam3/model/encoder.py:462-577`
- **使用**: `sam3/sam3/model/sam3_image.py:228`

### 输入参数

```python
encoder.forward(
    src=img_feats,              # 图像特征列表: List[(B, C, H, W)]
    src_key_padding_mask=None,  # 填充掩码
    src_pos=img_pos_embeds,     # 位置编码列表
    prompt=prompt,              # Text+Geo 拼接: (seq, B, 256)
    prompt_pos=prompt_pos_embed,# Prompt 位置编码
    prompt_key_padding_mask=prompt_mask,  # Prompt 掩码
    feat_sizes=vis_feat_sizes,  # 特征空间维度 [(H1, W1), ...]
)
```

### 执行流程

#### Step 0: 文本池化融合

**代码**: `encoder.py:543-551`

```python
if self.add_pooled_text_to_img_feat:
    # 对所有文本 token 求均值（可选用掩码加权）
    pooled_text = pool_text_feat(
        prompt, prompt_key_padding_mask, self.pool_text_with_mask
    )
    # 投影到图像特征维度，扩展为空间维度
    pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]

    # 全局融合：把均值文本特征加到每个图像特征
    src = [x.add_(pooled_text) for x in src]
```

**作用**: 在 Encoder 的最开始，用全局文本语义（均值）"初始化"图像特征。

#### Step 1~N: 多层 Encoder Layer

**代码**: `encoder.py:428-446` (外层) 和 `encoder.py:82-137` (内层)

每个 `TransformerEncoderLayer` 执行：

```python
def forward_post(self, tgt, memory, query_pos, pos, ...):
    # tgt = 图像特征 (H*W, B, 256)
    # memory = prompt (seq, B, 256)

    # 1. Self-Attention: 图像 token 之间互相看
    q = k = tgt + query_pos  # 加位置编码
    tgt2 = self.self_attn(q, k, value=tgt, ...)
    tgt = tgt + tgt2
    tgt = self.norm1(tgt)

    # 2. Cross-Attention: 图像去读 prompt（text+geo）
    tgt2 = self.cross_attn_image(
        query=tgt + query_pos,      # Q: 图像特征
        key=memory + pos,           # K: prompt + 位置编码
        value=memory,               # V: prompt
        ...
    )
    tgt = tgt + tgt2
    tgt = self.norm2(tgt)

    # 3. FFN (Feed-Forward Network)
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + tgt2
    tgt = self.norm3(tgt)

    return tgt
```

### 输出

**代码**: `encoder.py:569-577`

```python
return {
    "memory": out,                      # (seq, B, 256) 融合特征
    "padding_mask": key_padding_masks_flatten,
    "pos_embed": lvl_pos_embed_flatten,
    "memory_text": prompt,              # 保留原始 prompt
    "level_start_index": level_start_index,
    "spatial_shapes": spatial_shapes,
    "valid_ratios": valid_ratios,
}
```

### Encoder 的本质

| 功能 | 说明 |
|------|------|
| **输入** | 图像特征 (被查询者) + 文本/几何 (提示) |
| **过程** | 图像通过 Self-Attn 理解自己的结构，通过 Cross-Attn 理解文本语义 |
| **输出** | 融合了文本语义的图像特征（内容没变，但学到了"要找什么"） |
| **类比** | 阅读理解：读题目，理解要求 |

---

## 2. TransformerDecoder 详解

### 文件位置
- **定义**: `sam3/sam3/model/decoder.py:190-599`
- **使用**: `sam3/sam3/model/sam3_image.py:269-284`

### 输入参数

```python
decoder.forward(
    tgt=obj_queries,            # (N, B, 256) - 可学习 query
    memory=memory,              # (H*W, B, 256) - Encoder 输出
    memory_text=prompt,         # (seq, B, 256) - 原始 prompt
    text_attention_mask=prompt_mask,
    pos=pos_embed,              # 位置编码
    reference_boxes=None,       # 初始框（第一层为空）
    level_start_index=...,
    spatial_shapes=...,
    valid_ratios=...,
)
```

### 初始化关键组件

**代码**: `decoder.py:241, 299-301`

```python
# obj_queries: 可学习的物体查询 embedding
self.query_embed = nn.Embedding(tot_num_queries, d_model)  # (N, 256)

# presence_token: 可学习的存在性查询
self.presence_token = nn.Embedding(1, d_model)             # (1, 256)
self.presence_token_head = MLP(d_model, d_model, 1, 3)     # 3层MLP
self.presence_token_out_norm = nn.LayerNorm(d_model)
```

### 执行流程

#### Step 0: 初始化

**代码**: `decoder.py:489-493`

```python
output = tgt  # (N, B, 256) 直接使用 obj_queries

# presence_token 展开到 batch 维度
if self.presence_token is not None:
    presence_out = self.presence_token.weight[None].expand(1, bs, -1)
    # shape: (1, B, 256)
```

#### Step 1~L: 多层 Decoder Layer (box refine 循环)

**代码**: `decoder.py:503-593`

每个 `TransformerDecoderLayer` 执行：

```python
def forward(self, tgt, tgt_query_pos, memory, memory_text, presence_token, ...):
    # tgt: (N, B, 256)
    # presence_token: (1, B, 256)

    # === Step 1: Self-Attention ===
    if presence_token is not None:
        tgt_o2o = torch.cat([presence_token, tgt], dim=0)  # (1+N, B, 256)
        tgt_query_pos = torch.cat([zeros, tgt_query_pos], dim=0)

    q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos)
    tgt2 = self.self_attn(q, k, tgt_o2o)
    tgt_o2o = tgt_o2o + tgt2

    if presence_token is not None:
        tgt = tgt_o2o[1:]  # 去掉 presence_token

    # === Step 2: Text Cross-Attention (可选) ===
    if self.use_text_cross_attention:
        tgt2 = self.ca_text(
            self.with_pos_embed(tgt, tgt_query_pos),
            memory_text,  # K, V = text+geo prompt
            memory_text,
        )
        tgt = tgt + tgt2

    # === Step 3: Image Cross-Attention ===
    # 重新拼接 presence_token
    if presence_token is not None:
        tgt_concat = torch.cat([presence_token, tgt], dim=0)
    else:
        tgt_concat = tgt

    tgt2 = self.cross_attn(
        query=self.with_pos_embed(tgt_concat, tgt_query_pos),
        key=self.with_pos_embed(memory, memory_pos),      # K = Encoder 输出
        value=memory,                                      # V = Encoder 输出
    )
    tgt_concat = tgt_concat + tgt2

    # === Step 4: FFN ===
    tgt = self.forward_ffn(tgt_concat)

    # === Step 5: 分离 presence_token ===
    if presence_token is not None:
        presence_token_out = tgt[:1]
        tgt = tgt[1:]

    return tgt, presence_token_out
```

#### Step 2: 每层之后的处理

**代码**: `decoder.py:554-594`

```python
# === Box Refine ===
# 用 MLP 预测边界框偏移
reference_before_sigmoid = inverse_sigmoid(reference_boxes)
delta_unsig = self.bbox_embed(output)  # (N, B, 4)
outputs_unsig = delta_unsig + reference_before_sigmoid
reference_boxes = outputs_unsig.sigmoid()  # 迭代精修

# === Presence Logit 计算 ===
# 通过 3 层 MLP 预测存在性
intermediate_layer_presence_logits = self.presence_token_head(
    self.presence_token_out_norm(presence_out)  # (1, B, 256) → norm → (1, B, 256)
).squeeze(-1)  # → (1, B) → (B)

# 可选的 clamp（防止数值溢出）
if self.clamp_presence_logits:
    intermediate_layer_presence_logits.clamp(
        min=-self.clamp_presence_logit_max_val,
        max=self.clamp_presence_logit_max_val,
    )

intermediate_presence_logits.append(intermediate_layer_presence_logits)
```

### 输出

**代码**: `decoder.py:600-650` (省略但从使用推断)

```python
hs: (L, N, B, 256)              # 每层的 obj_queries 输出
reference_boxes: (L, N, B, 4)   # 每层精修后的边界框
dec_presence_out: (L, B)        # 每层的存在性 logit
dec_presence_feats: (1, B, 256) # 最后的 presence_token 特征
```

### Decoder 的本质

| 功能 | 说明 |
|------|------|
| **输入** | 可学习 query + 融合后的图像特征 (Encoder 输出) |
| **过程** | Query 通过多层 Attention 去图像中"定位"物体，同时精修边界框 |
| **输出** | 物体位置、类别、存在性（三重预测） |
| **类比** | 答题：根据理解，在图像中找出答案 |

---

## 3. Encoder 和 Decoder 的对比

| 对比维度 | Encoder | Decoder |
|---------|---------|---------|
| **Q 是谁** | 图像特征 (src) | obj_queries (可学习) + presence_token |
| **K/V 是谁** | prompt (text+geo) | Encoder 输出 (memory) |
| **处理对象** | 图像（被提示者） | Query（提问者） |
| **做什么** | 图像理解文本语义 | Query 在图像中定位物体 |
| **重复次数** | L 层 | L 层 (box refine 循环) |
| **输出维度** | (H*W, B, 256) | (N, B, 256) |
| **辅助输出** | 无 | 边界框、存在性 logit |
| **ONNX导出** | ❌ 困难 | ❌ 困难 |

---

## 4. Issue #17 的根本原因：低置信度问题

### 问题描述

当用 EfficientSAM3 的权重进行推理时，即使是明显的文本提示（如"a dog"），最终的置信度分数也很低（接近 0）。

### 根本原因链

```
Stage 1 蒸馏策略
  │
  ├─ ✅ 蒸馏了: Vision Encoder (RepViT)
  ├─ ✅ 蒸馏了: Text Encoder (MobileCLIP)
  │
  └─ ❌ 没有蒸馏: Transformer Fusion
      │
      ├── Encoder 的权重 ← 来自 SAM3 Teacher (Hiera + CLIP)
      ├── Decoder 的权重 ← 来自 SAM3 Teacher
      ├── obj_queries ← 来自 SAM3 Teacher (可学习)
      ├── presence_token ← 来自 SAM3 Teacher (可学习)
      ├── presence_token_head ← 来自 SAM3 Teacher (3层MLP)
      └── 所有 Attention 权重 ← 来自 SAM3 Teacher

特征分布不匹配
  │
  ├─ Teacher Vision Encoder (Hiera) 输出特征分布 A
  ├─ Student Vision Encoder (RepViT) 输出特征分布 B
  │
  └─ A ≠ B (虽然蒸馏但不完全相同)
     │
     └─ Decoder 的所有参数都在分布 A 上训练
        面对分布 B，行为系统性偏移
```

### presence_logit 为什么特别低

**代码**: `sam3_image_processor.py:194-196`

```python
out_probs = out_logits.sigmoid()                          # 分类分数 [0, 1]
presence_score = outputs["presence_logit_dec"].sigmoid()  # 存在性分数 [0, 1]
out_probs = out_probs * presence_score                    # 最终 = 两者相乘
```

**分布特性**:

```
教师模型 (SAM3):
  presence_logit ≈ -5.89
  sigmoid(-5.89) ≈ 0.003
  → 虽然低，但其他分数高，整体还行

学生模型 (EfficientSAM3):
  presence_logit ≈ -11.05          ← 系统性更负！
  sigmoid(-11.05) ≈ 0.000016       ← 接近零！
  → 最终分数 = 0.95 × 0.000016 = 0.0000152 ≈ 0
```

**原因分析**:

1. Encoder 把分布 B 的特征转换为假的分布 A
2. Decoder 中的 `presence_token_head` MLP 是在真实分布 A 上训练的
3. 面对转换后的"假分布 A"，MLP 输出更负的 logit
4. sigmoid 在负值区域非常陡峭：
   ```
   logit 从 -5 变到 -11 (只差 6)
   但 sigmoid 输出从 0.007 跌到 0.000016 (下降约 400 倍！)
   ```
5. "乘法"关系放大了问题：即使分类分数 0.95，乘以 0.000016 ≈ 0

---

## 5. 解决方案

### 方案 1: 跳过 presence_score（最简单但粗暴）

**代码修改**: `sam3_image_processor.py:195-196`

```python
out_probs = out_logits.sigmoid()
# 删除或注释掉以下两行
# presence_score = outputs["presence_logit_dec"].sigmoid()
# out_probs = out_probs * presence_score
```

**优缺点**:
- ✅ 简单快速
- ✅ 立即解决低分数问题
- ❌ 失去"存在性判断"能力
- ❌ 可能增加误检

**适用场景**: 快速验证、debug

### 方案 2: 微调 presence 相关组件（折中方案）

只微调 `presence_token`、`presence_token_head`、`presence_token_out_norm`，冻结其他参数。

```python
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只解冻 presence 相关参数
for name, param in model.named_parameters():
    if 'presence_token' in name:
        param.requires_grad = True

# 用少量数据微调
```

**优缺点**:
- ✅ 保留存在性判断功能
- ✅ 只需少量数据
- ⚠️ 需要有标注数据

**适用场景**: 有少量标注数据的应用场景

### 方案 3: Stage 2/3 完整蒸馏（根本解决）

在 Stage 1 的基础上，继续蒸馏 Transformer Decoder（包括所有权重、presence_token、presence_token_head）。

这是项目 roadmap 中明确列出的下一步工作。

**优缺点**:
- ✅ 彻底解决分布不匹配问题
- ✅ 保持所有功能
- ❌ 需要大量训练数据和计算
- ⏳ 工作量大，时间长

**适用场景**: 长期方案，追求最优效果

### 方案 4: 手动偏移修正（临时 hack）

在 presence_logit 上加一个固定偏移：

```python
# 在 decoder.py 的 forward 方法中
intermediate_layer_presence_logits = self.presence_token_head(
    self.presence_token_out_norm(presence_out)
) + 5.0  # ← 加一个偏移补偿
```

**优缺点**:
- ✅ 无需微调或重训
- ❌ 硬编码，不同模型需要不同值
- ❌ 不科学，仅临时 debug 用

**适用场景**: 临时验证、debug

### 推荐优先级

| 优先级 | 方案 | 时间成本 | 效果 | 适用场景 |
|--------|------|---------|------|---------|
| 1️⃣ | 方案 1: 跳过 presence | 5分钟 | 中 | 快速验证推理 |
| 2️⃣ | 方案 2: 微调 presence | 1-2小时 | 好 | 有少量标注数据 |
| 3️⃣ | 方案 3: Stage 2/3 蒸馏 | 数小时~数天 | 最优 | 追求最终效果 |
| 4️⃣ | 方案 4: 手动偏移 | 5分钟 | 差 | 临时 debug |

---

## 总结

### Transformer Fusion 的两层结构

1. **Encoder**: 让图像理解文本语义（"读题"）
2. **Decoder**: 让 query 在图像中定位物体（"答题"）

### Issue #17 的核心

Stage 1 蒸馏只替换了 Vision 和 Text Encoder，没有重训 Transformer Fusion。导致 Decoder 面对新的特征分布时，输出的 presence_logit 系统性偏负，sigmoid 函数将其压缩到接近零，最终置信度崩溃。

### 最快的修复方案

- **快速验证**: 跳过 presence_score
- **有数据微调**: 只微调 presence 相关组件
- **长期方案**: 等待 Stage 2/3 蒸馏完成

