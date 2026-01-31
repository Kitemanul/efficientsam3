# ONNX 导出解决方案与 LayerNorm 替代方案

本文档为 EfficientSAM3 中无法导出 ONNX 的组件提供**具体的代码级解决方案**，并提供 LayerNorm 的 NPU 兼容替代方案。

---

## 目录

1. [Transformer Encoder 的 ONNX 解决方案](#1-transformer-encoder-的-onnx-解决方案)
2. [Transformer Decoder 的 ONNX 解决方案](#2-transformer-decoder-的-onnx-解决方案)
3. [Segmentation Head 的 ONNX 解决方案](#3-segmentation-head-的-onnx-解决方案)
4. [Memory Attention (Tracker) 的解决方案](#4-memory-attention-tracker-的解决方案)
5. [LayerNorm 替代方案](#5-layernorm-替代方案)
6. [完整混合部署架构](#6-完整混合部署架构)

---

## 1. Transformer Encoder 的 ONNX 解决方案

**源文件**: `sam3/sam3/model/encoder.py`

### 问题 1: 动态分支选择 (line 236)

**原代码**:

```python
# encoder.py:236
fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
return fwd_fn(...)
```

**问题**: ONNX 不支持运行时函数选择。

**解决**: 固定为 `pre_norm=True`，直接调用 `forward_pre`。

```python
# 修改后: 移除分支，固定调用 forward_pre
def forward(self, tgt, memory, **kwargs):
    return self.forward_pre(tgt, memory, **kwargs)
```

**权重兼容性**: ✅ 完全兼容（层结构不变）

---

### 问题 2: DAC 动态切分 (lines 174-187)

**原代码**:

```python
# encoder.py:174-187
if dac:
    assert tgt.shape[0] % 2 == 0
    other_tgt = tgt[tgt.shape[0] // 2 :]
    tgt = tgt[: tgt.shape[0] // 2]
    ...
    tgt = torch.cat((tgt, other_tgt), dim=0)
```

**问题**: 动态条件 + 动态切片 + 动态拼接。

**解决**: 推理时 DAC 不使用（`apply_dac = self.dac and self.training`），直接移除。

```python
# 修改后: 移除所有 dac 相关代码
def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
    # 直接处理，不做 DAC 切分
    tgt2 = self.norm1(tgt)
    q = k = tgt2 + query_pos  # 固定加位置编码
    tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                           key_padding_mask=tgt_key_padding_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    # ... 后续不变
```

**权重兼容性**: ✅ 完全兼容（DAC 只影响数据流）

---

### 问题 3: 条件位置编码 (lines 113, 180, 190-191)

**原代码**:

```python
# encoder.py:180
q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2

# encoder.py:190-191
query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
```

**问题**: 三元表达式在 ONNX 中是条件分支。

**解决**: 固定为 True（加位置编码是默认行为）。

```python
# 修改后: 固定加位置编码
q = k = tgt2 + query_pos

tgt2 = self.cross_attn_image(
    query=tgt2 + query_pos,
    key=memory + pos,
    value=memory,
    ...
)[0]
```

**权重兼容性**: ✅ 完全兼容（只改变计算方式）

---

### ~~问题 4: 自定义 Cross-Attention 模块~~ (已排除，非问题)

经过仔细排查 `model_builder.py` 中的构建代码，确认 Transformer Fusion 中所有 Attention 模块都是**标准的 `nn.MultiheadAttention`**，不存在自定义 Attention 结构。

**构建代码** (`model_builder.py:116-138`):

```python
def _create_transformer_encoder():
    encoder_layer = TransformerEncoderLayer(
        ...
        self_attention=MultiheadAttention(     # ← MultiheadAttentionWrapper
            num_heads=8, dropout=0.1, embed_dim=256, batch_first=True,
        ),
        cross_attention=MultiheadAttention(    # ← MultiheadAttentionWrapper
            num_heads=8, dropout=0.1, embed_dim=256, batch_first=True,
        ),
    )
```

**MultiheadAttentionWrapper 定义** (`model_misc.py:31-34`):

```python
class MultiheadAttentionWrapper(nn.MultiheadAttention):
    def forward(self, *args, **kwargs):
        kwargs["need_weights"] = False          # ← 唯一区别: 不返回注意力权重
        return super().forward(*args, **kwargs)
```

**结论**: `MultiheadAttentionWrapper` 继承自 `nn.MultiheadAttention`，只添加了 `need_weights=False`。

- ✅ 权重格式与标准 `nn.MultiheadAttention` 完全相同
- ✅ 计算逻辑完全相同
- ✅ ONNX 导出时直接使用 `nn.MultiheadAttention` 加载权重即可，**无需任何转换**

**Transformer Fusion 中所有 Attention 的实际类型**:

| 组件 | Attention | 实际类 | ONNX 兼容 |
|------|-----------|--------|-----------|
| Encoder Self-Attn | `self.self_attn` | `nn.MultiheadAttention` (Wrapper) | ✅ |
| Encoder Cross-Attn | `self.cross_attn_image` | `nn.MultiheadAttention` (Wrapper) | ✅ |
| Decoder Self-Attn | `self.self_attn` | `nn.MultiheadAttention` | ✅ |
| Decoder Image Cross-Attn | `self.cross_attn` | `nn.MultiheadAttention` (Wrapper) | ✅ |
| Decoder Text Cross-Attn | `self.ca_text` | `nn.MultiheadAttention` | ✅ |
| SegHead Cross-Attn | `self.cross_attend_prompt` | `nn.MultiheadAttention` (Wrapper) | ✅ |

> **注意**: 真正自定义的 Attention (`RoPEAttention`) 只在 **Tracker 的 SAM Mask Decoder** 中使用，不在 Transformer Fusion 内部。

---

### 问题 4: 文本池化条件 (lines 543-551)

**原代码**:

```python
# encoder.py:543-551
if self.add_pooled_text_to_img_feat:
    pooled_text = pool_text_feat(prompt, prompt_key_padding_mask, ...)
    pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
    src = [x.add_(pooled_text) for x in src]
```

**问题**: 条件分支 + 列表推导。

**解决**: 固定执行（此特性默认开启）。

```python
# 修改后: 固定执行文本池化
pooled_text = prompt.mean(dim=0)  # 简化池化
pooled_text = self.text_pooling_proj(pooled_text)[..., None, None]
# 对每个 src 特征都加（展开为固定操作）
src_0 = src_0 + pooled_text
src_1 = src_1 + pooled_text  # 如果有多级特征
```

**权重兼容性**: ✅ 完全兼容

---

### 完整的 ONNX 可导出 Encoder Layer

```python
class OnnxTransformerEncoderLayer(nn.Module):
    """ONNX 可导出的 Transformer Encoder Layer

    注意: 原模型中的 Attention 就是标准的 nn.MultiheadAttention
    (MultiheadAttentionWrapper 只加了 need_weights=False，不影响计算和权重)
    因此可以直接加载原始权重，无需转换。
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        # 与原模型完全相同的标准 Attention (权重可直接加载)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)  # 后续替换为 NPU 兼容方案
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt, memory, pos=None, query_pos=None,
                memory_key_padding_mask=None):
        """
        Args:
            tgt: 图像特征 (H*W, B, 256)
            memory: prompt (seq, B, 256)
            pos: memory 位置编码
            query_pos: tgt 位置编码
        """
        # Pre-norm Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)

        # Pre-norm Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos,
            key=memory + pos,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        # Pre-norm FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt
```

---

## 2. Transformer Decoder 的 ONNX 解决方案

**源文件**: `sam3/sam3/model/decoder.py`

### 问题 1: presence_token 动态拼接/拆分 (lines 125-132, 182-185)

**原代码**:

```python
# decoder.py:125-132
if presence_token is not None:
    tgt_o2o = torch.cat([presence_token, tgt_o2o], dim=0)
    tgt_query_pos_o2o = torch.cat(
        [torch.zeros_like(presence_token), tgt_query_pos_o2o], dim=0
    )

# decoder.py:182-185
if presence_token is not None:
    presence_token_out = tgt[:1]
    tgt = tgt[1:]
```

**问题**: 动态条件 + 动态拼接/切片。

**解决**: 固定拼接（推理时 presence_token 始终存在）。

```python
# 修改后: 固定拼接和拆分
# 拼接 (固定执行)
tgt = torch.cat([presence_token, tgt], dim=0)           # (1+N, B, 256)
tgt_query_pos = torch.cat(
    [torch.zeros(1, tgt.size(1), tgt.size(2), device=tgt.device),
     tgt_query_pos], dim=0
)                                                        # (1+N, B, 256)

# Self-Attention ...

# 拆分 (固定执行，用固定索引)
presence_token_out = tgt[0:1]  # (1, B, 256) — 固定取第一个
tgt = tgt[1:]                  # (N, B, 256) — 固定取剩余
```

**权重兼容性**: ✅ 完全兼容

---

### 问题 2: DAC 逻辑 (lines 114-145)

**原代码**:

```python
# decoder.py:114-120
if dac:
    num_o2o_queries = tgt.shape[0] // 2
    tgt_o2o = tgt[:num_o2o_queries]
    tgt_o2m = tgt[num_o2o_queries:]
```

**解决**: 与 Encoder 相同，推理时不使用 DAC，直接移除。

```python
# 修改后: 移除所有 dac 逻辑
tgt_o2o = tgt
tgt_query_pos_o2o = tgt_query_pos
```

**权重兼容性**: ✅ 完全兼容

---

### 问题 3: 可选 Text Cross-Attention (lines 147-155)

**原代码**:

```python
# decoder.py:147-155
if self.use_text_cross_attention:
    tgt2 = self.ca_text(
        self.with_pos_embed(tgt, tgt_query_pos),
        memory_text,
        memory_text,
        key_padding_mask=text_attention_mask,
    )[0]
    tgt = tgt + self.catext_dropout(tgt2)
    tgt = self.catext_norm(tgt)
```

**问题**: 条件分支。

**解决**: 根据模型配置，固定为执行或不执行。

```python
# 方案 A: 如果 use_text_cross_attention=True，固定执行
tgt2 = self.ca_text(
    tgt + tgt_query_pos,
    memory_text,
    memory_text,
    key_padding_mask=text_attention_mask,
)[0]
tgt = tgt + self.catext_dropout(tgt2)
tgt = self.catext_norm(tgt)

# 方案 B: 如果 use_text_cross_attention=False，直接删除这段代码
# (什么都不做，跳过)
```

**权重兼容性**: ✅ 完全兼容

---

### 问题 4: Box Refine 循环 (lines 503-577)

**原代码**:

```python
# decoder.py:503-577
for layer_idx, layer in enumerate(self.layers):
    # ... 计算 reference_points_input
    # ... 计算 query_pos
    output, presence_out = layer(...)

    # Box Refine
    delta_unsig = box_head(out_norm(output))
    outputs_unsig = delta_unsig + reference_before_sigmoid
    reference_boxes = outputs_unsig.sigmoid()

    intermediate.append(out_norm(output))
```

**问题**: 循环内有动态条件。

**解决**: 将循环展开为固定数量的层调用。

```python
# 修改后: 展开为固定层数 (假设 num_layers=6)
# Layer 0
output, presence_out = self.layer_0(tgt=output, memory=memory, ...)
delta_0 = self.bbox_embed(self.norm(output))
reference_boxes = (inverse_sigmoid(reference_boxes) + delta_0).sigmoid()
presence_logit_0 = self.presence_token_head(
    self.presence_token_out_norm(presence_out)
).squeeze(-1)

# Layer 1
output, presence_out = self.layer_1(tgt=output, memory=memory, ...)
delta_1 = self.bbox_embed(self.norm(output))
reference_boxes = (inverse_sigmoid(reference_boxes) + delta_1).sigmoid()
presence_logit_1 = self.presence_token_head(
    self.presence_token_out_norm(presence_out)
).squeeze(-1)

# ... Layer 2 ~ Layer 5 类似
```

**注意**: 虽然每层的权重不同（`self.layers[i]`），但 `bbox_embed` 和 `presence_token_head` 是所有层共享的。

**权重兼容性**: ✅ 完全兼容（只是展开循环）

---

### 问题 5: 条件返回值 (lines 600-610)

**原代码**:

```python
# decoder.py:600-610
return (
    torch.stack(intermediate),
    torch.stack(intermediate_ref_boxes),
    torch.stack(intermediate_presence_logits) if self.presence_token is not None else None,
    presence_feats,
)
```

**问题**: 条件返回 Tensor 或 None。

**解决**: 固定返回所有值。

```python
# 修改后: 始终返回所有值
return (
    torch.stack(intermediate),
    torch.stack(intermediate_ref_boxes),
    torch.stack(intermediate_presence_logits),
    presence_feats,
)
```

**权重兼容性**: ✅ 完全兼容

---

### 完整的 ONNX 可导出 Decoder Layer

```python
class OnnxTransformerDecoderLayer(nn.Module):
    """ONNX 可导出的 Transformer Decoder Layer"""

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.0,
                 use_text_cross_attention=True):
        super().__init__()
        # Image Cross-Attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # Text Cross-Attention
        self.use_text_cross_attention = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.catext_dropout = nn.Dropout(dropout)
            self.catext_norm = nn.LayerNorm(d_model)

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, tgt_query_pos, memory, memory_pos,
                presence_token, memory_text=None, text_attention_mask=None,
                cross_attn_mask=None, memory_key_padding_mask=None):
        """
        Args:
            tgt: obj_queries (N, B, 256)
            presence_token: (1, B, 256)
            memory: Encoder输出 (H*W, B, 256)
            memory_text: prompt (seq, B, 256)
        """
        # === Step 1: 固定拼接 presence_token ===
        tgt_with_presence = torch.cat([presence_token, tgt], dim=0)  # (1+N, B, 256)
        zero_pos = torch.zeros(1, tgt.size(1), tgt.size(2),
                               device=tgt.device, dtype=tgt.dtype)
        tgt_query_pos_with_presence = torch.cat([zero_pos, tgt_query_pos], dim=0)

        # === Step 2: Self-Attention ===
        q = k = tgt_with_presence + tgt_query_pos_with_presence
        tgt2 = self.self_attn(q, k, tgt_with_presence)[0]
        tgt_with_presence = tgt_with_presence + self.dropout2(tgt2)
        tgt_with_presence = self.norm2(tgt_with_presence)

        # === Step 3: Text Cross-Attention (固定执行) ===
        if self.use_text_cross_attention:
            tgt2 = self.ca_text(
                tgt_with_presence + tgt_query_pos_with_presence,
                memory_text,
                memory_text,
                key_padding_mask=text_attention_mask,
            )[0]
            tgt_with_presence = tgt_with_presence + self.catext_dropout(tgt2)
            tgt_with_presence = self.catext_norm(tgt_with_presence)

        # === Step 4: Image Cross-Attention ===
        # 扩展 cross_attn_mask 为 presence_token 添加零 mask
        presence_mask = torch.zeros(
            cross_attn_mask.size(0), 1, cross_attn_mask.size(2),
            device=cross_attn_mask.device, dtype=cross_attn_mask.dtype
        )
        full_mask = torch.cat([presence_mask, cross_attn_mask], dim=1)

        tgt2 = self.cross_attn(
            query=tgt_with_presence + tgt_query_pos_with_presence,
            key=memory + memory_pos,
            value=memory,
            attn_mask=full_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt_with_presence = tgt_with_presence + self.dropout1(tgt2)
        tgt_with_presence = self.norm1(tgt_with_presence)

        # === Step 5: FFN ===
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(
            tgt_with_presence
        ))))
        tgt_with_presence = tgt_with_presence + self.dropout4(tgt2)
        tgt_with_presence = self.norm3(tgt_with_presence)

        # === Step 6: 固定拆分 ===
        presence_token_out = tgt_with_presence[0:1]  # (1, B, 256)
        tgt_out = tgt_with_presence[1:]              # (N, B, 256)

        return tgt_out, presence_token_out
```

---

## 3. Segmentation Head 的 ONNX 解决方案

**源文件**: `sam3/sam3/model/maskformer_segmentation.py`

### 问题 1: List[Tensor] 输入 (line 147)

**原代码**:

```python
# maskformer_segmentation.py:147
def forward(
    self,
    backbone_feats: List[torch.Tensor],  # ← ONNX 不支持 List
    obj_queries: torch.Tensor,
    ...
)
```

**解决**: 展开为固定数量的独立参数。

```python
# 修改后: 展开 List 为固定参数
def forward(
    self,
    backbone_feat_0: torch.Tensor,  # (B, C, H/4, W/4)
    backbone_feat_1: torch.Tensor,  # (B, C, H/8, W/8)
    backbone_feat_2: torch.Tensor,  # (B, C, H/16, W/16)
    backbone_feat_3: torch.Tensor,  # (B, C, H/32, W/32)
    obj_queries: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
):
    backbone_feats = [backbone_feat_0, backbone_feat_1,
                      backbone_feat_2, backbone_feat_3]
    # ... 后续不变
```

---

### 问题 2: 动态索引 (lines 112-142)

**原代码**:

```python
# maskformer_segmentation.py:112-117
if backbone_feats[0].shape[0] > 1:
    for feat in backbone_feats:
        backbone_visual_feats.append(feat[image_ids_, ...])
else:
    backbone_visual_feats = [bb_feat.clone() for bb_feat in backbone_feats]
```

**解决**: 固定 batch=1，走简单分支。

```python
# 修改后: 固定 batch=1
backbone_visual_feats = [feat.clone() for feat in backbone_feats]
```

---

### 问题 3: 多分支 mask 预测 (lines 162-167)

**原代码**:

```python
# maskformer_segmentation.py:162-167
if self.no_dec:
    mask_pred = self.mask_predictor(pixel_embed)
elif self.aux_masks:
    mask_pred = self.mask_predictor(obj_queries, pixel_embed)
else:
    mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)
```

**解决**: 固定一条路径（根据模型配置选择）。

```python
# 修改后: 固定为 aux_masks=False, no_dec=False
mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)
```

---

### 问题 4: MaskPredictor 的 einsum 多分支 (lines 29-49)

**原代码**:

```python
if len(obj_queries.shape) == 3:
    if pixel_embed.ndim == 3:
        mask_preds = torch.einsum("bqc,chw->bqhw", ...)
    else:
        mask_preds = torch.einsum("bqc,bchw->bqhw", ...)
else:
    if pixel_embed.ndim == 3:
        mask_preds = torch.einsum("lbqc,chw->lbqhw", ...)
    else:
        mask_preds = torch.einsum("lbqc,bchw->lbqhw", ...)
```

**解决**: 固定为推理时的形状（3维 obj_queries, 3维 pixel_embed）。

```python
# 修改后: 固定为推理时的路径
# obj_queries: (B, N, 256), pixel_embed: (C, H, W)
mask_preds = torch.einsum("bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed)
```

---

### 完整的 ONNX 可导出 Segmentation Head

```python
class OnnxSegmentationHead(nn.Module):
    """ONNX 可导出的 Segmentation Head"""

    def __init__(self, hidden_dim=256):
        super().__init__()
        self.pixel_decoder = OnnxPixelDecoder(hidden_dim)
        self.mask_embed = nn.Linear(hidden_dim, hidden_dim)
        self.instance_seg_head = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.semantic_seg_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, backbone_feat_0, backbone_feat_1,
                backbone_feat_2, backbone_feat_3,
                obj_queries, encoder_hidden_states):
        """
        Args:
            backbone_feat_i: (B, C, H_i, W_i) — 各级 backbone 特征
            obj_queries: (B, N, 256) — Decoder 输出
            encoder_hidden_states: (H*W, B, 256) — Encoder 输出
        """
        # Step 1: Pixel Decoder (FPN 上采样)
        pixel_embed = self.pixel_decoder(
            backbone_feat_0, backbone_feat_1,
            backbone_feat_2, backbone_feat_3,
            encoder_hidden_states
        )  # (C, H, W) — batch=1 时 squeeze 掉 B 维

        # Step 2: Instance embedding
        instance_embeds = self.instance_seg_head(pixel_embed)  # (C, H, W)

        # Step 3: Mask 预测 (query × pixel 点积)
        query_embeds = self.mask_embed(obj_queries)  # (B, N, 256)
        mask_preds = torch.einsum("bqc,chw->bqhw", query_embeds, instance_embeds)

        # Step 4: 语义分割头
        semantic_seg = self.semantic_seg_head(pixel_embed)

        return mask_preds, semantic_seg


class OnnxPixelDecoder(nn.Module):
    """ONNX 可导出的 FPN Pixel Decoder"""

    def __init__(self, hidden_dim=256, num_stages=3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1)
            for _ in range(num_stages)
        ])
        self.norms = nn.ModuleList([
            nn.GroupNorm(8, hidden_dim)  # 注意: 这里已经用 GroupNorm 而非 LayerNorm
            for _ in range(num_stages)
        ])

    def forward(self, feat_0, feat_1, feat_2, feat_3, encoder_hidden_states):
        # 自底向上融合
        prev = feat_3  # 最低分辨率

        prev = feat_2 + torch.nn.functional.interpolate(
            prev, size=feat_2.shape[-2:], mode='nearest'
        )
        prev = torch.nn.functional.relu(self.norms[0](self.convs[0](prev)))

        prev = feat_1 + torch.nn.functional.interpolate(
            prev, size=feat_1.shape[-2:], mode='nearest'
        )
        prev = torch.nn.functional.relu(self.norms[1](self.convs[1](prev)))

        prev = feat_0 + torch.nn.functional.interpolate(
            prev, size=feat_0.shape[-2:], mode='nearest'
        )
        prev = torch.nn.functional.relu(self.norms[2](self.convs[2](prev)))

        return prev
```

---

## 4. Memory Attention (Tracker) 的解决方案

**源文件**: `sam3/sam3/model/sam3_tracker_base.py`

Tracker 的 Memory Attention 是**最难导出 ONNX** 的组件，因为它依赖动态循环、字典访问和动态长度拼接。推荐使用 **C++ 实现**。

### 问题总览

| 问题 | 行号 | 描述 |
|------|------|------|
| 动态循环 | 615-650 | `for t_pos in range(1, self.num_maskmem)` |
| 字典访问 | 645 | `output_dict["non_cond_frame_outputs"].get(...)` |
| 条件跳过 | 648 | `if out is None: continue` |
| 动态拼接 | 780 | `torch.cat(to_cat_prompt, dim=0)` |
| 动态指针堆叠 | 735-765 | `torch.stack(ptrs_list, dim=0)` |

### 解决方案: 预分配固定大小记忆 + C++ 实现

**核心思路**: 将动态的 Python dict + list 操作转换为固定大小的 Tensor 操作。

```python
class OnnxMemoryBank:
    """ONNX 兼容的固定大小记忆库"""

    def __init__(self, max_frames=7, mem_dim=64, d_model=256,
                 spatial_size=64*64, max_obj_ptrs=16):
        # 预分配固定大小记忆
        self.max_frames = max_frames

        # 记忆特征: (max_frames, B, mem_dim, spatial_size)
        self.memory_feats = torch.zeros(max_frames, 1, mem_dim, spatial_size)

        # 记忆位置编码: (max_frames, spatial_size, B, d_model)
        self.memory_pos = torch.zeros(max_frames, spatial_size, 1, d_model)

        # 记忆有效性掩码: (max_frames,) — 0=无效, 1=有效
        self.memory_valid = torch.zeros(max_frames, dtype=torch.bool)

        # 对象指针: (max_obj_ptrs, B, d_model)
        self.obj_ptrs = torch.zeros(max_obj_ptrs, 1, d_model)
        self.obj_ptrs_valid = torch.zeros(max_obj_ptrs, dtype=torch.bool)

        self.write_idx = 0

    def update(self, frame_feat, frame_pos, obj_ptr):
        """更新记忆库 (循环覆盖)"""
        idx = self.write_idx % self.max_frames
        self.memory_feats[idx] = frame_feat
        self.memory_pos[idx] = frame_pos
        self.memory_valid[idx] = True

        ptr_idx = self.write_idx % self.obj_ptrs.size(0)
        self.obj_ptrs[ptr_idx] = obj_ptr
        self.obj_ptrs_valid[ptr_idx] = True

        self.write_idx += 1

    def get_memory(self):
        """获取所有有效记忆 (固定形状输出，用 mask 标记无效)"""
        return (
            self.memory_feats,      # (max_frames, B, mem_dim, spatial)
            self.memory_pos,        # (max_frames, spatial, B, d_model)
            self.memory_valid,      # (max_frames,)
            self.obj_ptrs,          # (max_obj_ptrs, B, d_model)
            self.obj_ptrs_valid,    # (max_obj_ptrs,)
        )
```

### C++ 伪代码实现

```cpp
// memory_attention.cpp
// 用 C++ 实现 Tracker 的 Memory Attention 逻辑

struct MemoryBank {
    // 固定大小记忆
    Tensor memory_feats;    // [max_frames, B, mem_dim, spatial]
    Tensor memory_pos;      // [max_frames, spatial, B, d_model]
    Tensor memory_valid;    // [max_frames]
    Tensor obj_ptrs;        // [max_obj_ptrs, B, d_model]
    Tensor obj_ptrs_valid;  // [max_obj_ptrs]
    int write_idx = 0;
    int max_frames;
};

Tensor run_memory_attention(
    Tensor current_feat,         // 当前帧特征
    Tensor current_pos,          // 当前帧位置编码
    MemoryBank& memory,          // 记忆库
    OnnxModel& attn_model        // ONNX 注意力模型
) {
    // Step 1: 收集有效记忆
    std::vector<Tensor> valid_feats, valid_pos;
    for (int i = 0; i < memory.max_frames; i++) {
        if (memory.memory_valid[i]) {
            valid_feats.push_back(memory.memory_feats[i]);
            valid_pos.push_back(memory.memory_pos[i]);
        }
    }

    // Step 2: Padding 到固定长度
    Tensor padded_feats = pad_to_fixed_length(valid_feats, memory.max_frames);
    Tensor padded_pos = pad_to_fixed_length(valid_pos, memory.max_frames);
    Tensor padding_mask = create_padding_mask(valid_feats.size(), memory.max_frames);

    // Step 3: 收集有效对象指针
    std::vector<Tensor> valid_ptrs;
    for (int i = 0; i < memory.obj_ptrs.size(0); i++) {
        if (memory.obj_ptrs_valid[i]) {
            valid_ptrs.push_back(memory.obj_ptrs[i]);
        }
    }
    Tensor padded_ptrs = pad_to_fixed_length(valid_ptrs, MAX_OBJ_PTRS);

    // Step 4: 拼接为 ONNX 模型输入
    Tensor prompt = concat({padded_feats, padded_ptrs}, dim=0);
    Tensor prompt_pos = concat({padded_pos, ptr_pos}, dim=0);
    Tensor prompt_mask = concat({padding_mask, ptr_mask}, dim=0);

    // Step 5: 调用 ONNX 注意力模型
    Tensor output = attn_model.run({
        {"current_feat", current_feat},
        {"prompt", prompt},
        {"prompt_pos", prompt_pos},
        {"prompt_mask", prompt_mask},
    });

    // Step 6: 更新记忆库
    memory.update(current_feat, current_pos, output.obj_ptr);

    return output;
}
```

---

## 5. LayerNorm 替代方案

### 问题

NPU 不支持 `nn.LayerNorm`。项目中共有 **28+ 处** LayerNorm 需要替换。

### LayerNorm 的位置清单

| 文件 | 组件 | LayerNorm 数量 | 行号 |
|------|------|---------------|------|
| `encoder.py` | TransformerEncoderLayer | 3 | 65, 66, 67 |
| `decoder.py` | TransformerDecoderLayer | 4 | 47, 54, 59, 67 |
| `decoder.py` | TransformerDecoder | 3 | 238, 248, 301 |
| `decoder.py` | TransformerEncoderCrossAttention | 1 | 632 |
| `maskformer_segmentation.py` | UniversalSegmentationHead | 1 | 261 |
| `geometry_encoders.py` | SequenceGeometryEncoder | 3 | 576, 580, 588 |
| `student_sam/transformer.py` | TwoWayTransformer/Block | 5 | 63, 128, 133, 136, 138 |
| `text_encoder_ve.py` | ResidualAttentionBlock | 2 | 28, 29 |
| **合计** | | **22+** | |

### 方案 1: GroupNorm (推荐)

**优点**: 项目中已有使用（PixelDecoder, line 190），NPU 普遍支持。

**原理**: LayerNorm 对最后一维做归一化，GroupNorm 将通道分组后归一化。当 `num_groups=1` 时，GroupNorm 在数学上等价于 LayerNorm（对 2D+ 输入）。

```python
class GroupNormAsLayerNorm(nn.Module):
    """用 GroupNorm 替代 LayerNorm

    LayerNorm: 对 (d_model,) 维度归一化
    GroupNorm(1, d_model): 对整个通道归一化（等价于 LayerNorm）

    注意: GroupNorm 要求输入至少是 3D: (B, C, *)
    而 LayerNorm 通常接收: (seq, B, C)
    所以需要做 shape 转换
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=1, num_channels=d_model, eps=eps)
        self.d_model = d_model

    def forward(self, x):
        # x: (seq, B, d_model) — Transformer 的标准输入格式
        # GroupNorm 要求: (B, C, *) 格式

        if x.dim() == 3:
            # (seq, B, d_model) → (B, d_model, seq)
            x = x.permute(1, 2, 0)
            x = self.norm(x)
            # (B, d_model, seq) → (seq, B, d_model)
            x = x.permute(2, 0, 1)
        elif x.dim() == 2:
            # (B, d_model) → (B, d_model, 1)
            x = x.unsqueeze(-1)
            x = self.norm(x)
            x = x.squeeze(-1)

        return x
```

**权重转换**:

```python
def convert_layernorm_to_groupnorm(state_dict):
    """将 LayerNorm 权重转换为 GroupNorm 格式"""
    new_state_dict = {}
    for key, value in state_dict.items():
        if '.weight' in key and any(norm_name in key for norm_name in
            ['norm1', 'norm2', 'norm3', 'norm', 'catext_norm',
             'presence_token_out_norm', 'instance_norm',
             'cross_attn_norm', 'ln_1', 'ln_2',
             'img_pre_norm', 'encode_norm', 'norm_final_attn']):
            # LayerNorm.weight → GroupNorm.weight (格式相同)
            new_key = key.replace('.weight', '.norm.weight')
            new_state_dict[new_key] = value
        elif '.bias' in key and any(norm_name in key for norm_name in
            ['norm1', 'norm2', 'norm3', 'norm', 'catext_norm',
             'presence_token_out_norm', 'instance_norm',
             'cross_attn_norm', 'ln_1', 'ln_2',
             'img_pre_norm', 'encode_norm', 'norm_final_attn']):
            new_key = key.replace('.bias', '.norm.bias')
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict
```

---

### 方案 2: BatchNorm1d + shape 转换

**优点**: NPU 对 BatchNorm 支持非常好，速度快。

**缺点**: BatchNorm 在推理时使用 running mean/var，需要先 calibration。

```python
class BatchNormAsLayerNorm(nn.Module):
    """用 BatchNorm1d 替代 LayerNorm

    注意: 需要先在训练数据上 calibrate running stats
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model, eps=eps)

    def forward(self, x):
        # x: (seq, B, d_model)

        if x.dim() == 3:
            seq, B, C = x.shape
            # (seq, B, C) → (seq*B, C) — BatchNorm 需要 (N, C) 或 (N, C, L) 格式
            x = x.reshape(seq * B, C)
            x = self.bn(x)
            x = x.reshape(seq, B, C)
        elif x.dim() == 2:
            x = self.bn(x)

        return x
```

**权重初始化**:

```python
def init_batchnorm_from_layernorm(bn_module, ln_state):
    """用 LayerNorm 的 weight/bias 初始化 BatchNorm"""
    bn_module.weight.data = ln_state['weight'].clone()
    bn_module.bias.data = ln_state['bias'].clone()
    # running_mean 和 running_var 需要通过数据 calibration 获得
    bn_module.running_mean.zero_()
    bn_module.running_var.fill_(1.0)
```

---

### 方案 3: RMSNorm (手动实现)

**优点**: 比 LayerNorm 简单（不需要 mean），只用基础算子。

**缺点**: 没有 bias，与 LayerNorm 不完全等价。

```python
class RMSNorm(nn.Module):
    """RMSNorm: 只用 RMS 归一化，不减均值

    LayerNorm: y = (x - mean) / sqrt(var + eps) * weight + bias
    RMSNorm:   y = x / sqrt(mean(x^2) + eps) * weight
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        # 归一化
        x = x / rms
        # 缩放
        x = x * self.weight
        return x
```

**权重转换**:

```python
def convert_layernorm_to_rmsnorm(ln_state):
    """将 LayerNorm 权重转换为 RMSNorm
    注意: bias 会丢失!"""
    return {'weight': ln_state['weight'].clone()}
```

---

### 方案 4: 手动分解为基础算子

**优点**: 完全用基础算子实现，任何 NPU 都支持。

**缺点**: 性能可能不如原生 LayerNorm。

```python
class ManualLayerNorm(nn.Module):
    """手动实现 LayerNorm，只用基础算子

    分解为: mean → sub → square → mean → add(eps) → sqrt → div → mul → add
    这些都是 ONNX 标准算子
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # Step 1: 计算均值
        mean = x.mean(dim=-1, keepdim=True)       # ONNX ReduceMean

        # Step 2: 减去均值
        x_centered = x - mean                      # ONNX Sub

        # Step 3: 计算方差
        variance = (x_centered * x_centered).mean(dim=-1, keepdim=True)  # ONNX Mul + ReduceMean

        # Step 4: 归一化
        x_normed = x_centered / torch.sqrt(variance + self.eps)          # ONNX Add + Sqrt + Div

        # Step 5: 缩放和偏移
        output = x_normed * self.weight + self.bias  # ONNX Mul + Add

        return output
```

**权重转换**: 直接复制（格式完全相同）。

```python
def convert_layernorm_to_manual(manual_module, ln_state):
    """直接复制权重"""
    manual_module.weight.data = ln_state['weight'].clone()
    manual_module.bias.data = ln_state['bias'].clone()
```

---

### 方案对比

| 方案 | NPU兼容 | 精度保持 | 性能 | 权重兼容 | 推荐度 |
|------|---------|---------|------|---------|--------|
| GroupNorm | ✅ | 等价(groups=1) | 好 | 需转换格式 | ⭐⭐⭐⭐ |
| BatchNorm | ✅ | 近似 | 最快 | 需 calibration | ⭐⭐⭐ |
| RMSNorm | ✅ | 近似(无bias) | 好 | 丢失 bias | ⭐⭐ |
| **手动分解** | **✅** | **等价** | **一般** | **直接复制** | **⭐⭐⭐⭐⭐** |

**推荐**:
- **首选**: 方案 4（手动分解）— 数学完全等价，权重直接复制，任何 NPU 都支持
- **备选**: 方案 1（GroupNorm）— 如果 NPU 支持 GroupNorm，性能更好

---

### 统一替换工具

```python
def replace_all_layernorms(model, replacement='manual'):
    """替换模型中所有 LayerNorm 为 NPU 兼容版本

    Args:
        model: PyTorch 模型
        replacement: 'manual' | 'groupnorm' | 'batchnorm' | 'rmsnorm'
    """
    replacements = {
        'manual': ManualLayerNorm,
        'groupnorm': GroupNormAsLayerNorm,
        'batchnorm': BatchNormAsLayerNorm,
        'rmsnorm': RMSNorm,
    }
    ReplacementClass = replacements[replacement]

    replaced_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # 获取 LayerNorm 的参数
            d_model = module.normalized_shape[0]
            eps = module.eps

            # 创建替代模块
            new_module = ReplacementClass(d_model, eps=eps)

            # 复制权重
            if hasattr(new_module, 'weight') and hasattr(module, 'weight'):
                new_module.weight.data = module.weight.data.clone()
            if hasattr(new_module, 'bias') and hasattr(module, 'bias'):
                if module.bias is not None:
                    new_module.bias.data = module.bias.data.clone()

            # 替换模块
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model
            for part in parent_name.split('.'):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, new_module)
            replaced_count += 1

    print(f"替换了 {replaced_count} 个 LayerNorm")
    return model


# 使用方式:
model = build_efficientsam3_model(...)
model = replace_all_layernorms(model, replacement='manual')

# 验证
for name, module in model.named_modules():
    assert not isinstance(module, nn.LayerNorm), f"未替换: {name}"
print("所有 LayerNorm 已替换!")
```

---

## 6. 完整混合部署架构

### 整体方案

```
┌─────────────────────────────────────────────────────────────┐
│                    NPU 部署架构                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         ONNX 模型部分 (NPU 加速)                     │    │
│  │                                                     │    │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────┐  │    │
│  │  │ Vision      │  │ Text         │  │ Geometry  │  │    │
│  │  │ Encoder     │  │ Encoder      │  │ Encoder   │  │    │
│  │  │ (RepViT)    │  │ (MobileCLIP) │  │           │  │    │
│  │  │ .onnx       │  │ .onnx        │  │ .onnx     │  │    │
│  │  └──────┬──────┘  └──────┬───────┘  └─────┬─────┘  │    │
│  │         │                │                │         │    │
│  │         │                └────────┬───────┘         │    │
│  │         │                         │                 │    │
│  │         │                    torch.cat              │    │
│  │         │                    (C++ 实现)              │    │
│  │         │                         │                 │    │
│  │         ▼                         ▼                 │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │  Transformer Encoder (简化版 ONNX)            │   │    │
│  │  │  - 固定 pre_norm=True                        │   │    │
│  │  │  - 移除 DAC                                  │   │    │
│  │  │  - Attention 已是标准 MHA (权重直接加载)      │   │    │
│  │  │  - LayerNorm → ManualLayerNorm               │   │    │
│  │  │  .onnx                                       │   │    │
│  │  └──────────────────┬───────────────────────────┘   │    │
│  │                     │                               │    │
│  │                     ▼                               │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │  Transformer Decoder (简化版 ONNX)            │   │    │
│  │  │  - 固定 presence_token 拼接                   │   │    │
│  │  │  - 移除 DAC                                  │   │    │
│  │  │  - 循环展开为固定层数                         │   │    │
│  │  │  - LayerNorm → ManualLayerNorm               │   │    │
│  │  │  .onnx                                       │   │    │
│  │  └──────────────────┬───────────────────────────┘   │    │
│  │                     │                               │    │
│  │         ┌───────────┼───────────┐                   │    │
│  │         ▼           ▼           ▼                   │    │
│  │  ┌───────────┐ ┌──────────┐ ┌──────────┐           │    │
│  │  │Seg Head   │ │ Scoring  │ │Box Head  │           │    │
│  │  │(简化版)   │ │ Module   │ │          │           │    │
│  │  │.onnx      │ │ .onnx    │ │ .onnx    │           │    │
│  │  └───────────┘ └──────────┘ └──────────┘           │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │         C++ 控制逻辑                                 │    │
│  │                                                     │    │
│  │  • 模型加载和初始化                                   │    │
│  │  • 图像预处理 / 后处理                                │    │
│  │  • prompt 拼接 (torch.cat → C++ concat)              │    │
│  │  • NMS 去重                                          │    │
│  │  • 置信度过滤                                        │    │
│  │  • Memory Bank 管理 (Tracker)                        │    │
│  │  • Memory Attention 循环逻辑                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │    SAM Tracker 部分 (混合)                           │    │
│  │                                                     │    │
│  │  ┌──────────────────┐  ┌──────────────────────┐     │    │
│  │  │ Memory Encoder   │  │ SAM Mask Decoder     │     │    │
│  │  │ .onnx (NPU)      │  │ .onnx (NPU)          │     │    │
│  │  └──────────────────┘  └──────────────────────┘     │    │
│  │                                                     │    │
│  │  ┌──────────────────────────────────────────────┐   │    │
│  │  │ Memory Attention                              │   │    │
│  │  │ C++ 实现 (CPU)                                │   │    │
│  │  │ - 记忆库管理 (固定大小)                        │   │    │
│  │  │ - 帧循环逻辑                                  │   │    │
│  │  │ - 对象指针管理                                │   │    │
│  │  └──────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### ONNX 导出流程

```python
import torch

# Step 1: 加载原始模型
model = build_efficientsam3_model(checkpoint_path="weights.pt")
model.eval()

# Step 2: 替换 LayerNorm
model = replace_all_layernorms(model, replacement='manual')

# Step 3: 创建 ONNX 可导出版本
onnx_encoder_layer = OnnxTransformerEncoderLayer(d_model=256)
onnx_decoder_layer = OnnxTransformerDecoderLayer(d_model=256)
onnx_seg_head = OnnxSegmentationHead(hidden_dim=256)

# Step 4: 转换权重
onnx_encoder_layer.load_state_dict(
    convert_encoder_weights(model.state_dict()), strict=False
)

# Step 5: 导出 ONNX
# Vision Encoder
dummy_img = torch.randn(1, 3, 1024, 1024)
torch.onnx.export(model.backbone, dummy_img, "vision_encoder.onnx",
                   input_names=["image"], output_names=["features"],
                   opset_version=17)

# Text Encoder
dummy_text = torch.randint(0, 1000, (1, 77))
torch.onnx.export(model.text_encoder, dummy_text, "text_encoder.onnx",
                   input_names=["tokens"], output_names=["text_features"],
                   opset_version=17)

# Transformer Encoder
dummy_src = torch.randn(4096, 1, 256)
dummy_prompt = torch.randn(20, 1, 256)
torch.onnx.export(onnx_encoder_layer,
                   (dummy_src, dummy_prompt),
                   "transformer_encoder_layer.onnx",
                   input_names=["src", "prompt"],
                   output_names=["output"],
                   opset_version=17)

# Transformer Decoder Layer
dummy_tgt = torch.randn(100, 1, 256)
dummy_presence = torch.randn(1, 1, 256)
dummy_memory = torch.randn(4096, 1, 256)
torch.onnx.export(onnx_decoder_layer,
                   (dummy_tgt, dummy_tgt, dummy_memory, dummy_memory,
                    dummy_presence),
                   "transformer_decoder_layer.onnx",
                   input_names=["tgt", "tgt_query_pos", "memory",
                               "memory_pos", "presence_token"],
                   output_names=["output", "presence_out"],
                   opset_version=17)

print("所有 ONNX 模型导出完成!")
```

### 实现优先级

| 优先级 | 组件 | 方案 | 难度 |
|--------|------|------|------|
| 1 | Vision Encoder | 直接导出 ONNX | 低 |
| 2 | Text Encoder | 直接导出 ONNX | 低 |
| 3 | Geometry Encoder | 直接导出 ONNX | 低 |
| 4 | LayerNorm 替换 | ManualLayerNorm 全局替换 | 低 |
| 5 | Transformer Encoder | 简化后导出 ONNX | 中 |
| 6 | Segmentation Head | 简化后导出 ONNX | 中 |
| 7 | Scoring Module | 直接导出 ONNX | 低 |
| 8 | Transformer Decoder | 简化+循环展开后导出 ONNX | 高 |
| 9 | Memory Attention | C++ 实现 | 高 |
| 10 | 端到端集成测试 | C++ 编排所有模型 | 高 |

---

## 总结

### 修改策略总表

| 修改类型 | 权重兼容性 | 涉及组件 |
|---------|-----------|---------|
| 移除 if/else 分支 | ✅ 完全兼容 | Encoder, Decoder |
| 移除 DAC 逻辑 | ✅ 完全兼容 | Encoder, Decoder |
| 固定条件表达式 | ✅ 完全兼容 | Encoder |
| 固定 presence_token 拼接 | ✅ 完全兼容 | Decoder |
| 展开循环 | ✅ 完全兼容 | Decoder |
| Attention 模块 | ✅ 已是标准 MHA，无需替换 | Encoder, Decoder |
| 替换 LayerNorm | ✅/⚠️ 取决于方案 | 全部 |
| 展开 List[Tensor] | ✅ 完全兼容 | Segmentation Head |
| C++ 实现 | N/A | Memory Attention |

> **重要发现**: 经排查，Transformer Fusion 中所有 Attention 模块都是标准的 `nn.MultiheadAttention`
> （`MultiheadAttentionWrapper` 只加了一行 `need_weights=False`，不影响计算和权重格式）。
> 这意味着 Attention 权重可以**直接加载**，无需任何转换脚本。
> 真正自定义的 `RoPEAttention` 只在 Tracker 的 SAM Mask Decoder 中使用。

### LayerNorm 推荐方案

**首选: ManualLayerNorm (方案 4)**
- 数学完全等价
- 权重直接复制
- 任何 NPU 都支持
- 只用 ONNX 标准算子: ReduceMean, Sub, Mul, Add, Sqrt, Div
