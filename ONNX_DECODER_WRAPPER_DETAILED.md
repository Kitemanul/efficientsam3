# DecoderWrapper ONNX 导出逐行对比详解

本文档逐行对比 `export_image_model_onnx.py` 中 `DecoderWrapper` 的每一行代码与原始模型代码的对应关系，详细说明每个变量的含义、张量形状、以及为何需要这样转写。

---

## 目录

1. [Monkey-Patch 详解](#1-monkey-patch-详解)
2. [构造函数 __init__ 对比](#2-构造函数-__init__-对比)
3. [Step 1: Text features → seq-first](#3-step-1-text-features--seq-first)
4. [Step 2: Geometry Encoder CLS token](#4-step-2-geometry-encoder-cls-token)
5. [Step 3: Build prompt](#5-step-3-build-prompt)
6. [Step 4: Encoder Fusion](#6-step-4-encoder-fusion)
7. [Step 5: Decoder](#7-step-5-decoder)
8. [Step 6: DotProduct Scoring](#8-step-6-dotproduct-scoring)
9. [Step 7: Joint Presence Scoring](#9-step-7-joint-presence-scoring)
10. [Step 8: Boxes](#10-step-8-boxes)
11. [Step 9: Segmentation Head](#11-step-9-segmentation-head)
12. [Step 10: Extract Last Layer](#12-step-10-extract-last-layer)
13. [省略了什么](#13-省略了什么)

---

## 1. Monkey-Patch 详解

导出脚本在 `import sam3` 之前应用了 3 个 monkey-patch，解决 CUDA 硬编码和 ONNX 不兼容问题。

### 1.1 PositionEmbeddingSine.__init__

**原始代码** (`position_encoding.py:47`):
```python
def __init__(self, num_pos_feats, temperature=10000, normalize=True,
             scale=None, precompute_resolution=None):
    super().__init__()
    ...
    if precompute_resolution is not None:
        # 在 CUDA 上预计算位置编码
        self._precompute(precompute_resolution, device="cuda")  # ← 硬编码 cuda
```

**问题**: 模型构建时 `model_builder.py` 传入 `precompute_resolution=1008`，导致在 CPU 环境报错。

**Patch 代码**:
```python
def _patched_pe_init(self, num_pos_feats, temperature=10000, normalize=True,
                     scale=None, precompute_resolution=None):
    _orig_pe_init(self, num_pos_feats, temperature=temperature,
                  normalize=normalize, scale=scale,
                  precompute_resolution=None)  # ← 强制 None，跳过预计算
```

**效果**: 位置编码不再预计算，改为 forward 时动态计算（结果完全相同，仅性能略慢）。

### 1.2 TransformerDecoder._get_coords

**原始代码** (`decoder.py:323-329`):
```python
@staticmethod
def _get_coords(H, W, device="cpu"):
    coords_h = torch.linspace(0, 1, H, device=device)
    coords_w = torch.linspace(0, 1, W, device=device)
    return coords_h, coords_w
```

**问题**: 虽然默认 `device="cpu"`，但在某些调用路径中 `reference_boxes.device` 可能是 `"cuda"`。

**Patch 代码**:
```python
@staticmethod
def _patched_get_coords(H, W, device="cpu"):
    return _orig_get_coords(H, W, device="cpu")  # ← 强制 CPU
```

### 1.3 TransformerDecoder._get_rpb_matrix

**原始代码** (`decoder.py:331-408`):
```python
def _get_rpb_matrix(self, reference_boxes, feat_size):
    H, W = feat_size
    boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
    bs, num_queries, _ = boxes_xyxy.shape
    if self.compilable_cord_cache is None:
        self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
        self.compilable_stored_size = (H, W)

    if torch.compiler.is_dynamo_compiling() or self.compilable_stored_size == (H, W):
        #                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #                                     这里 H, W 是符号变量时，torch.export 无法处理
        coords_h, coords_w = self.compilable_cord_cache
    else:
        # cache miss 路径...
        if feat_size not in self.coord_cache:
            self.coord_cache[feat_size] = self._get_coords(H, W, reference_boxes.device)
        coords_h, coords_w = self.coord_cache[feat_size]
    ...
```

**问题**: `torch.export` 将 H, W 视为符号变量，`self.compilable_stored_size == (H, W)` 是数据依赖的比较，触发 `GuardOnDataDependentSymNode` 错误。

**Patch 代码**:
```python
def _patched_get_rpb_matrix(self, reference_boxes, feat_size):
    H, W = feat_size
    if self.compilable_cord_cache is None:
        self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
        self.compilable_stored_size = (H, W)
    coords_h, coords_w = self.compilable_cord_cache  # ← 总是使用缓存，去掉数据依赖判断
    # 后续计算逻辑与原始完全相同 ...
```

**安全性**: 推理时特征图大小固定 (72×72)，不存在 cache miss 的情况。

---

## 2. 构造函数 __init__ 对比

### Wrapper __init__

```python
class DecoderWrapper(nn.Module):
    def __init__(self, model):     # model 是 Sam3Image 实例
        super().__init__()
        # Geometry encoder — only CLS token path
        geo = model.geometry_encoder                      # SequenceGeometryEncoder
        self.geo_cls_embed = geo.cls_embed                # nn.Embedding(1, 256)
        self.geo_final_proj = geo.final_proj              # nn.Linear(256, 256)
        self.geo_norm = geo.norm                          # nn.LayerNorm(256)
        self.geo_encode_layers = geo.encode               # nn.ModuleList of 3 TransformerDecoderLayer
        self.geo_encode_norm = geo.encode_norm            # nn.LayerNorm(256)

        # Encoder fusion
        self.encoder = model.transformer.encoder          # TransformerEncoderFusion

        # Decoder
        self.decoder = model.transformer.decoder          # TransformerDecoder

        # Scoring
        self.dot_prod_scoring = model.dot_prod_scoring    # DotProductScoring

        # Segmentation head
        self.segmentation_head = model.segmentation_head  # UniversalSegmentationHead

        # Config flags from the model
        self.supervise_joint_box_scores = model.supervise_joint_box_scores  # True
```

### 原始模型 Sam3Image.__init__ 对应属性

```python
# sam3_image.py, 构建时由 model_builder.py 设置:
self.geometry_encoder = SequenceGeometryEncoder(...)       # 完整几何编码器
self.transformer = TransformerContainer(encoder, decoder)  # 包含 encoder + decoder
self.dot_prod_scoring = DotProductScoring(...)             # 打分模块
self.segmentation_head = UniversalSegmentationHead(...)    # 分割头
self.supervise_joint_box_scores = True                     # 联合打分标志
```

### 为什么只提取部分子模块？

`SequenceGeometryEncoder` 的完整 `forward` 方法包含对 boxes, points, scribbles, masks 等多种 prompt 类型的处理（700+ 行代码），充满条件分支和动态数据结构，无法直接导出 ONNX。

但在**文本引导检测**模式下，几何 prompt 是 dummy 的（空 boxes），实际只走 **CLS token 路径**。Wrapper 只提取这条路径需要的 5 个子模块，手动实现该路径的计算逻辑。

---

## 3. Step 1: Text features → seq-first

### Wrapper 代码

```python
# export_image_model_onnx.py: DecoderWrapper.forward(), line 209-211
txt_feats = text_features.permute(1, 0, 2)   # [1, 77, 256] → [77, 1, 256]
txt_masks = text_mask                          # [1, 77]
```

### 原始代码

```python
# sam3_image.py: _encode_prompt(), line 176-178
txt_ids = find_input.text_ids                                    # 文本 prompt 索引
txt_feats = backbone_out["language_features"][:, txt_ids]        # [seq, bs, 256]
txt_masks = backbone_out["language_mask"][txt_ids]                # [bs, seq]
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `text_features.permute(1, 0, 2)` | `backbone_out["language_features"][:, txt_ids]` | 原始代码从 backbone_out 中按 text_ids 索引，已经是 seq-first `[seq, bs, C]`。Wrapper 的输入是 TextEncoder 输出的 batch-first `[bs, 77, 256]`，所以需要 permute 转为 seq-first |
| `text_mask` 直接使用 | `backbone_out["language_mask"][txt_ids]` | 原始代码按索引取 mask。Wrapper 直接接收 TextEncoder 输出的 mask |

### 变量含义

- **text_features** `[1, 77, 256]`: TextEncoder 输出的文本特征，77 = MobileCLIP 上下文长度，256 = 投影维度
- **txt_feats** `[77, 1, 256]`: seq-first 格式，所有后续 Transformer 操作都需要 seq-first
- **text_mask** `[1, 77]`: bool 型，`True` = padding token（token_id == 0 的位置），用于 attention 中忽略 padding

---

## 4. Step 2: Geometry Encoder CLS token

这是最核心的改写部分。原始 `SequenceGeometryEncoder.forward` 有 120+ 行代码，Wrapper 简化为 ~15 行。

### Wrapper 代码

```python
# export_image_model_onnx.py: line 213-233

# Convert image features to seq-first for geo encoder cross-attention
img_feat_seq = feat_1x.flatten(2).permute(2, 0, 1)   # [1,256,72,72] → [5184, 1, 256]
img_pos_seq = pos_1x.flatten(2).permute(2, 0, 1)     # [1,256,72,72] → [5184, 1, 256]

bs = feat_1x.shape[0]  # 1
cls = self.geo_cls_embed.weight.view(1, 1, -1).expand(1, bs, -1)   # [1, 1, 256]
cls_mask = torch.zeros(bs, 1, dtype=torch.bool, device=feat_1x.device)  # [1, 1]

# Project and normalize CLS
cls = self.geo_norm(self.geo_final_proj(cls))   # [1, 1, 256]

# 3-layer transformer with cross-attention to image features
for lay in self.geo_encode_layers:
    cls = lay(
        tgt=cls,                          # [1, 1, 256]
        memory=img_feat_seq,              # [5184, 1, 256]
        tgt_key_padding_mask=cls_mask,    # [1, 1]
        pos=img_pos_seq,                  # [5184, 1, 256]
    )
cls = self.geo_encode_norm(cls)   # [1, 1, 256]
```

### 原始代码

```python
# geometry_encoders.py: SequenceGeometryEncoder.forward(), line 729-850

def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
    # line 739-745: 提取各类 prompt (boxes, points, masks...)
    points = geo_prompt.point_embeddings       # dummy 时: [0, 1, 2] (空)
    boxes = geo_prompt.box_embeddings          # dummy 时: [0, 1, 4] (空)
    masks = geo_prompt.mask_embeddings         # None
    seq_first_img_feats = img_feats[-1]        # [5184, 1, 256] ← 与 wrapper 的 img_feat_seq 相同
    seq_first_img_pos_embeds = img_pos_embeds[-1]  # [5184, 1, 256]

    # line 749-758: ROI pooling 相关 (文本检测不走此路径)
    # line 760-800: encode points, encode boxes (dummy 时全部为空)

    # line 802-806: _encode_points 返回空 embeddings
    final_embeds, final_mask = self._encode_points(...)
    # dummy 时: final_embeds = [0, 1, 256], final_mask = [1, 0] (空序列)

    # line 821-830: CLS token 路径 ← Wrapper 手动实现的就是这一段
    bs = final_embeds.shape[1]               # 1
    if self.cls_embed is not None:
        cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)  # [1, 1, 256]
        cls_mask = torch.zeros(bs, 1, dtype=final_mask.dtype, device=...)      # [1, 1]
        final_embeds, final_mask = concat_padded_sequences(
            final_embeds, final_mask, cls, cls_mask
        )
        # dummy 时: concat([0,1,256], [1,1,256]) → [1, 1, 256] (空序列+CLS = 只有CLS)

    # line 832-833: 投影 + 归一化
    if self.final_proj is not None:
        final_embeds = self.norm(self.final_proj(final_embeds))  # [1, 1, 256]

    # line 835-844: 3 层 cross-attention to image
    if self.encode is not None:
        for lay in self.encode:
            final_embeds = activation_ckpt_wrapper(lay)(
                tgt=final_embeds,                          # [1, 1, 256]
                memory=seq_first_img_feats,                # [5184, 1, 256]
                tgt_key_padding_mask=final_mask,           # [1, 1]
                pos=seq_first_img_pos_embeds,              # [5184, 1, 256]
                act_ckpt_enable=self.training and ...,
            )
        final_embeds = self.encode_norm(final_embeds)      # [1, 1, 256]

    return final_embeds, final_mask
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `feat_1x.flatten(2).permute(2,0,1)` | `img_feats[-1]`（已是 seq-first） | 原始模型在 `_get_img_feats` 中已转换为 seq-first。Wrapper 从 image-format 输入重新转换 |
| `self.geo_cls_embed.weight.view(1,1,-1).expand(1,bs,-1)` | `self.cls_embed.weight.view(1,1,self.d_model).repeat(1,bs,1)` | 完全一致。expand vs repeat 在 bs=1 时等价 |
| `torch.zeros(bs,1,dtype=torch.bool)` | `torch.zeros(bs,1,dtype=final_mask.dtype)` | 一致。final_mask.dtype 就是 bool |
| `self.geo_norm(self.geo_final_proj(cls))` | `self.norm(self.final_proj(final_embeds))` | 一致。但原始对整个 final_embeds 操作，wrapper 只对 CLS 操作。因为 dummy prompt 时 final_embeds 在 concat 后就只有 CLS |
| `lay(tgt=cls, memory=..., pos=...)` | `activation_ckpt_wrapper(lay)(tgt=..., act_ckpt_enable=...)` | 一致。Wrapper 省略了 `activation_ckpt_wrapper`（eval 时 `act_ckpt_enable=False`，wrapper 直接不包裹） |
| `self.geo_encode_norm(cls)` | `self.encode_norm(final_embeds)` | 一致 |

### Wrapper 省略了什么

1. **_encode_points / _encode_boxes**: dummy prompt 时 points 和 boxes 都是空张量 `[0, bs, ...]`，编码结果为空
2. **concat_padded_sequences**: 空序列 + CLS = CLS，直接用 CLS 即可
3. **encode_boxes_as_points 路径**: dummy 时不执行
4. **mask_encoder**: 文本检测模式不使用 mask prompt
5. **activation_ckpt_wrapper**: eval 模式下等于直接调用

---

## 5. Step 3: Build prompt

### Wrapper 代码

```python
# export_image_model_onnx.py: line 235-237
prompt = torch.cat([txt_feats, cls], dim=0)           # [77+1, 1, 256] = [78, 1, 256]
prompt_mask = torch.cat([txt_masks, cls_mask], dim=1)  # [1, 77+1] = [1, 78]
```

### 原始代码

```python
# sam3_image.py: _encode_prompt(), line 196-211

# geo_feats, geo_masks 是 geometry_encoder 的返回值
geo_feats, geo_masks = self.geometry_encoder(geo_prompt=geometric_prompt, ...)
# dummy 时: geo_feats = [1, 1, 256] (CLS), geo_masks = [1, 1]

# visual_prompt_embed 在文本检测时为 None，创建空张量:
visual_prompt_embed = torch.zeros((0, *geo_feats.shape[1:]), device=...)  # [0, 1, 256]
visual_prompt_mask = torch.zeros((*geo_masks.shape[:-1], 0), device=...) # [1, 0]

# encode_text=True 时:
prompt = torch.cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)
# = cat([77,1,256], [1,1,256], [0,1,256]) = [78, 1, 256]
prompt_mask = torch.cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)
# = cat([1,77], [1,1], [1,0]) = [1, 78]
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `cat([txt_feats, cls], dim=0)` | `cat([txt_feats, geo_feats, visual_prompt_embed], dim=0)` | Wrapper 省略了 `visual_prompt_embed`（大小为 [0,...] 的空张量，cat 后不影响结果） |
| `cat([txt_masks, cls_mask], dim=1)` | `cat([txt_masks, geo_masks, visual_prompt_mask], dim=1)` | 同理省略了大小为 [1,0] 的空 mask |

### 变量含义

- **prompt** `[78, 1, 256]`: 完整的文本 + 几何 prompt，78 = 77 个文本 token + 1 个 CLS token
- **prompt_mask** `[1, 78]`: padding mask，text 的 padding 位为 True，CLS 位为 False
- 后续 Encoder 和 Decoder 都会对 prompt 做 cross-attention

---

## 6. Step 4: Encoder Fusion

### Wrapper 代码

```python
# export_image_model_onnx.py: line 239-260

H, W = feat_1x.shape[2], feat_1x.shape[3]                # 72, 72
src_seq = feat_1x.flatten(2).permute(2, 0, 1)             # [1,256,72,72] → [5184, 1, 256]
pos_seq = pos_1x.flatten(2).permute(2, 0, 1)              # [1,256,72,72] → [5184, 1, 256]
prompt_pos = torch.zeros_like(prompt)                       # [78, 1, 256] 全零

memory_dict = self.encoder(
    src=[src_seq],                        # List of [5184, 1, 256]
    src_pos=[pos_seq],                    # List of [5184, 1, 256]
    prompt=prompt,                        # [78, 1, 256]
    prompt_pos=prompt_pos,                # [78, 1, 256]
    prompt_key_padding_mask=prompt_mask,  # [1, 78]
    feat_sizes=[(H, W)],                  # [(72, 72)]
)

memory = memory_dict["memory"]                # [5184, 1, 256] — 编码后的图像特征 (seq-first)
pos_embed = memory_dict["pos_embed"]          # [5184, 1, 256] — 位置编码
padding_mask = memory_dict["padding_mask"]    # None（无 padding）
level_start_index = memory_dict["level_start_index"]  # tensor([0])
spatial_shapes = memory_dict["spatial_shapes"]         # tensor([[72, 72]])
valid_ratios = memory_dict["valid_ratios"]             # tensor([[[1., 1.]]])
prompt_after_enc = memory_dict.get("memory_text", prompt)  # [78, 1, 256] — 编码后的 prompt
```

### 原始代码

```python
# sam3_image.py: _run_encoder(), line 214-252

feat_tuple = self._get_img_feats(backbone_out, find_input.img_ids)
backbone_out, img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple
# img_feats:      List of [5184, 1, 256]  (已是 seq-first，在 _get_img_feats 中转换)
# img_pos_embeds: List of [5184, 1, 256]
# vis_feat_sizes: [(72, 72)]

prompt_pos_embed = torch.zeros_like(prompt)   # [78, 1, 256]

memory = self.transformer.encoder(
    src=img_feats.copy(),                         # ← .copy() 因为 encoder 可能 in-place 修改 list
    src_key_padding_mask=None,                    # 无 padding
    src_pos=img_pos_embeds.copy(),
    prompt=prompt,
    prompt_pos=prompt_pos_embed,
    prompt_key_padding_mask=prompt_mask,
    feat_sizes=vis_feat_sizes,                    # [(72, 72)]
    encoder_extra_kwargs=encoder_extra_kwargs,    # None（文本检测模式）
)
# memory 是 dict，与 wrapper 中 memory_dict 完全相同
```

### TransformerEncoderFusion.forward 内部流程

```python
# encoder.py: line 513-577

def forward(self, src, prompt, ..., feat_sizes=None):
    bs = src[0].shape[1]  # src[0] 是 [5184, 1, 256]，bs = 1

    # ---- 关键步骤：seq-first → image-format ----
    if feat_sizes is not None:
        for i, (h, w) in enumerate(feat_sizes):
            src[i] = src[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)
            # [5184, 1, 256] → reshape(72, 72, 1, 256) → permute → [1, 256, 72, 72]
            src_pos[i] = src_pos[i].reshape(h, w, bs, -1).permute(2, 3, 0, 1)

    # ---- 调用父类 TransformerEncoder.forward ----
    out, ... = super().forward(
        src,                                      # [[1, 256, 72, 72]]
        pos=src_pos,                              # [[1, 256, 72, 72]]
        prompt=prompt.transpose(0, 1),            # [78,1,256] → [1,78,256] batch-first
        prompt_key_padding_mask=prompt_mask,
    )
    # ---- TransformerEncoder.forward 内部 ----
    # 1. _prepare_multilevel_features:
    #    src.flatten(2).transpose(1,2) → [1, 5184, 256] (batch-first)
    #    pos_embed + level_embed → lvl_pos_embed
    #    spatial_shapes = tensor([[72, 72]])
    #    level_start_index = tensor([0])
    # 2. 6 × TransformerEncoderLayer:
    #    每层: self_attn(batch-first) → cross_attn to prompt (batch-first) → FFN
    # 3. 返回 output.transpose(0,1) → seq-first [5184, 1, 256]

    return {
        "memory": out,                # [5184, 1, 256]
        "padding_mask": ...,          # None
        "pos_embed": ...,             # [5184, 1, 256]
        "memory_text": prompt,        # [78, 1, 256] ← 注意: prompt 本身未被修改
        "level_start_index": ...,     # tensor([0])
        "spatial_shapes": ...,        # tensor([[72, 72]])
        "valid_ratios": ...,          # tensor([[[1., 1.]]])
    }
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `feat_1x.flatten(2).permute(2,0,1)` | `_get_img_feats` 中 `x[img_ids].flatten(2).permute(2,0,1)` | 转换方式相同，都是 `[B,C,H,W]→[HW,B,C]` |
| `feat_sizes=[(H,W)]` | `feat_sizes=vis_feat_sizes` | 都是 `[(72,72)]`。传入后 encoder 内部会 reshape 回 image-format |
| `src_key_padding_mask` 未传 | `src_key_padding_mask=None` | 默认为 None |
| `encoder_extra_kwargs` 未传 | `encoder_extra_kwargs=None` | 文本检测模式不使用 |

### 变量含义

- **memory** `[5184, 1, 256]`: 经过 6 层 Encoder 编码后的图像特征，5184 = 72×72
- **pos_embed** `[5184, 1, 256]`: 位置编码（加上了 level_embed）
- **prompt_after_enc** `[78, 1, 256]`: 当前实现中 prompt 没有被 encoder 修改（`memory_text` 直接返回原始 prompt）。但保留此变量以备将来 encoder 可能修改 prompt
- **level_start_index** `tensor([0])`: 只有 1 个 level，起始索引为 0
- **spatial_shapes** `tensor([[72, 72]])`: 特征图空间尺寸
- **valid_ratios** `tensor([[[1., 1.]]])`: 无 padding，全部有效

---

## 7. Step 5: Decoder

### Wrapper 代码

```python
# export_image_model_onnx.py: line 262-283

query_embed = self.decoder.query_embed.weight            # [200, 256]
tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)         # [200, 1, 256]

hs, reference_boxes, dec_presence_out, _ = self.decoder(
    tgt=tgt,                                  # [200, 1, 256] — 200 个 learnable query
    memory=memory,                            # [5184, 1, 256] — encoder 输出
    memory_key_padding_mask=padding_mask,     # None
    pos=pos_embed,                            # [5184, 1, 256] — 位置编码
    reference_boxes=None,                     # None → decoder 内部生成初始 reference points
    level_start_index=level_start_index,      # tensor([0])
    spatial_shapes=spatial_shapes,            # tensor([[72, 72]])
    valid_ratios=valid_ratios,                # tensor([[[1., 1.]]])
    memory_text=prompt_after_enc,             # [78, 1, 256]
    text_attention_mask=prompt_mask,           # [1, 78]
    apply_dac=False,                          # ← 推理时不用 DAC
)

# hs: [6, 200, 1, 256] → [6, 1, 200, 256]   (6层, batch-first)
hs = hs.transpose(1, 2)
reference_boxes = reference_boxes.transpose(1, 2)  # [6, 1, 200, 4] → [6, 1, 200, 4]
if dec_presence_out is not None:
    dec_presence_out = dec_presence_out.transpose(1, 2)  # [6, 1, 1]
```

### 原始代码

```python
# sam3_image.py: _run_decoder(), line 254-278

bs = memory.shape[1]   # 1
query_embed = self.transformer.decoder.query_embed.weight    # [200, 256]
tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)             # [200, 1, 256]

apply_dac = self.transformer.decoder.dac and self.training   # False（eval 时）

hs, reference_boxes, dec_presence_out, dec_presence_feats = (
    self.transformer.decoder(
        tgt=tgt,
        memory=memory,
        memory_key_padding_mask=src_mask,         # = encoder_out["padding_mask"] = None
        pos=pos_embed,
        reference_boxes=None,
        level_start_index=encoder_out["level_start_index"],
        spatial_shapes=encoder_out["spatial_shapes"],
        valid_ratios=encoder_out["valid_ratios"],
        tgt_mask=None,                            # Wrapper 未传 → 默认 None
        memory_text=prompt,                       # = encoder_out["prompt_after_enc"]
        text_attention_mask=prompt_mask,
        apply_dac=apply_dac,                      # False
    )
)
hs = hs.transpose(1, 2)                          # [6, 200, 1, 256] → [6, 1, 200, 256]
reference_boxes = reference_boxes.transpose(1, 2) # [6, 200, 1, 4] → [6, 1, 200, 4]
if dec_presence_out is not None:
    dec_presence_out = dec_presence_out.transpose(1, 2)
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `self.decoder.query_embed.weight` | `self.transformer.decoder.query_embed.weight` | 路径不同，同一对象 |
| `apply_dac=False` | `apply_dac = self.transformer.decoder.dac and self.training` = False | eval 时相同 |
| `_` (忽略第4返回值) | `dec_presence_feats` | Wrapper 不需要 presence_feats（仅训练用） |

### Decoder 内部关键流程 (`decoder.py:410-611`)

```python
# 1. box_refine=True 且 reference_boxes=None → 使用 learned reference_points
reference_boxes = self.reference_points.weight.unsqueeze(1)  # [200, 1, 4]
reference_boxes = reference_boxes.repeat(1, bs, 1).sigmoid()

# 2. 初始化 presence token (presence_token=True)
presence_out = self.presence_token.weight[None].expand(1, bs, -1)  # [1, 1, 256]

# 3. 逐层处理
for layer_idx, layer in enumerate(self.layers):   # 6 层
    # 3a. RPB (boxRPB="log"): 计算基于 box 的相对位置偏置
    memory_mask = self._get_rpb_matrix(reference_boxes, (72, 72))
    # memory_mask: [1*8, 200, 5184] = [8, 200, 5184]  (bs*n_heads, nq, HW)

    # 3b. TransformerDecoderLayer:
    #     - self_attn: 200 queries + 1 presence token 之间的自注意力
    #     - ca_text: cross_attn to text prompt [78, 1, 256]
    #     - cross_attn: cross_attn to image memory [5184, 1, 256] + RPB mask
    #     - FFN

    # 3c. Box refinement: 预测 delta → 更新 reference_boxes
    delta = box_head(output)
    new_ref = (inverse_sigmoid(reference_boxes) + delta).sigmoid()
    reference_boxes = new_ref.detach()

    # 3d. 收集中间层输出
    intermediate.append(out_norm(output))       # [200, 1, 256]
    intermediate_presence_logits.append(...)     # [1, 1]

# 4. 返回
return (
    torch.stack(intermediate),                   # [6, 200, 1, 256]
    torch.stack(intermediate_ref_boxes),          # [6+1, 200, 1, 4] → 取 [6, 200, 1, 4]
    torch.stack(intermediate_presence_logits),    # [6, 1, 1]
    presence_feats,                               # 训练用，推理忽略
)
```

### 变量含义

- **tgt** `[200, 1, 256]`: 200 个可学习的 object query（对应最多 200 个检测目标）
- **hs** `[6, 1, 200, 256]`: 6 层 decoder 的输出，每层都经过 norm，用于后续 scoring 和 box prediction（支持 auxiliary loss）
- **reference_boxes** `[6, 1, 200, 4]`: 每层 box refinement 后的 reference box（cxcywh 格式，sigmoid 归一化到 0~1）
- **dec_presence_out** `[6, 1, 1]`: presence logit，表示"画面中是否存在目标"的置信度

---

## 8. Step 6: DotProduct Scoring

### Wrapper 代码

```python
# export_image_model_onnx.py: line 285-288
outputs_class = self.dot_prod_scoring(hs, prompt_after_enc, prompt_mask)
# outputs_class: [6, 1, 200, 1]
```

### 原始代码

```python
# sam3_image.py: _update_scores_and_boxes(), line 319-326
if self.use_dot_prod_scoring:
    dot_prod_scoring_head = self.dot_prod_scoring
    if is_instance_prompt and self.instance_dot_prod_scoring is not None:
        dot_prod_scoring_head = self.instance_dot_prod_scoring
    outputs_class = dot_prod_scoring_head(hs, prompt, prompt_mask)
```

### DotProductScoring.forward 内部 (`model_misc.py:66-91`)

```python
def forward(self, hs, prompt, prompt_mask):
    # hs: [6, 1, 200, 256], prompt: [78, 1, 256], prompt_mask: [1, 78]

    if self.prompt_mlp is not None:
        prompt = self.prompt_mlp(prompt)          # MLP 投影 prompt

    # Mean pooling: 对非 padding 的 text token 取均值
    pooled_prompt = self.mean_pool_text(prompt, prompt_mask)  # [1, 256]

    # 分别投影到 d_proj 维
    proj_pooled_prompt = self.prompt_proj(pooled_prompt)  # [1, d_proj]
    proj_hs = self.hs_proj(hs)                            # [6, 1, 200, d_proj]

    # 点积打分
    scores = torch.matmul(proj_hs, proj_pooled_prompt.unsqueeze(-1))  # [6, 1, 200, 1]
    scores *= self.scale
    if self.clamp_logits:
        scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)
    return scores
```

### 对比说明

| Wrapper | 原始 | 说明 |
|---------|------|------|
| 直接调用 `self.dot_prod_scoring(...)` | 先检查 `use_dot_prod_scoring`，再检查 `is_instance_prompt` | Wrapper 省略了分支判断。文本检测模式下 `use_dot_prod_scoring=True` 且 `is_instance_prompt=False` |

### 变量含义

- **outputs_class** `[6, 1, 200, 1]`: 6 层 decoder 对应的 200 个 query 的分类 logit。这不是概率，还需要 sigmoid + presence 调整

---

## 9. Step 7: Joint Presence Scoring

### Wrapper 代码

```python
# export_image_model_onnx.py: line 290-295
if self.supervise_joint_box_scores and dec_presence_out is not None:
    prob_presence = dec_presence_out.sigmoid()  # [6, 1, 1]
    outputs_class = inverse_sigmoid(
        outputs_class.sigmoid() * prob_presence.unsqueeze(2)  # [6,1,200,1] * [6,1,1,1]
    ).clamp(min=-10.0, max=10.0)
```

### 原始代码

```python
# sam3_image.py: _update_scores_and_boxes(), line 348-357
if self.supervise_joint_box_scores:
    assert dec_presence_out is not None
    prob_dec_presence_out = dec_presence_out.clone().sigmoid()
    if self.detach_presence_in_joint_score:
        prob_dec_presence_out = prob_dec_presence_out.detach()

    outputs_class = inverse_sigmoid(
        outputs_class.sigmoid() * prob_dec_presence_out.unsqueeze(2)
    ).clamp(min=-10.0, max=10.0)
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `dec_presence_out.sigmoid()` | `dec_presence_out.clone().sigmoid()` | Wrapper 省略 `.clone()`。原始中 clone 是为了避免修改用于训练 loss 的原始张量，推理时不需要 |
| 无 detach | `if self.detach_presence_in_joint_score: ... .detach()` | Wrapper 省略了 detach 检查。推理时不计算梯度，detach 无意义 |

### 计算逻辑

这是 **joint box scoring** 的核心：将每个 query 的分类 logit 与整体 presence 概率相乘。

```
原始 logit → sigmoid → 概率 p ∈ (0,1)
presence logit → sigmoid → 概率 q ∈ (0,1)
联合概率 = p × q
结果 = inverse_sigmoid(p × q)   → 回到 logit 空间
clamp(-10, 10)                   → 防止数值溢出
```

---

## 10. Step 8: Boxes

### Wrapper 代码

```python
# export_image_model_onnx.py: line 297-301
anchor_box_offsets = self.decoder.bbox_embed(hs)        # [6, 1, 200, 4]
ref_inv_sig = inverse_sigmoid(reference_boxes)           # [6, 1, 200, 4]
outputs_coord = (ref_inv_sig + anchor_box_offsets).sigmoid()  # [6, 1, 200, 4]
outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)  # [6, 1, 200, 4]
```

### 原始代码

```python
# sam3_image.py: _update_scores_and_boxes(), line 336-343

box_head = self.transformer.decoder.bbox_embed
if is_instance_prompt and self.transformer.decoder.instance_bbox_embed is not None:
    box_head = self.transformer.decoder.instance_bbox_embed
anchor_box_offsets = box_head(hs)
reference_boxes_inv_sig = inverse_sigmoid(reference_boxes)
outputs_coord = (reference_boxes_inv_sig + anchor_box_offsets).sigmoid()
outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `self.decoder.bbox_embed(hs)` | `box_head(hs)` (box_head = self.transformer.decoder.bbox_embed) | 完全一致。省略了 `instance_bbox_embed` 分支判断 |
| 其他行 | 完全一致 | 纯数学操作，无简化 |

### 注意

这里的 box prediction 是**重复计算**：Decoder 内部的 box_refine 循环中已经做过同样的计算（`delta_unsig + reference_before_sigmoid`），但那里是为了更新 reference_boxes 供下一层使用（且做了 `.detach()`）。这里重新计算是为了得到带梯度的最终 box（训练时需要）。推理时两者数值一致。

### 变量含义

- **bbox_embed**: `MLP(256, 256, 4, num_layers=3)` — 3 层 MLP，预测 box offset (Δcx, Δcy, Δw, Δh)
- **reference_boxes** `[6, 1, 200, 4]`: 已 sigmoid 的参考框 (cx, cy, w, h)，范围 0~1
- **inverse_sigmoid(reference_boxes)**: 转回 logit 空间，与 offset 相加后再 sigmoid
- **outputs_coord** `[6, 1, 200, 4]`: 最终预测框 (cx, cy, w, h)，范围 0~1
- **outputs_boxes_xyxy** `[6, 1, 200, 4]`: 转为 (x1, y1, x2, y2) 格式

---

## 11. Step 9: Segmentation Head

### Wrapper 代码

```python
# export_image_model_onnx.py: line 303-312
seg_out = self.segmentation_head(
    backbone_feats=[feat_4x, feat_2x, feat_1x],   # 3 个尺度的特征图
    obj_queries=hs,                                 # [6, 1, 200, 256]
    image_ids=torch.zeros(1, dtype=torch.long, device=feat_1x.device),  # [1]
    encoder_hidden_states=memory,                   # [5184, 1, 256]
    prompt=prompt_after_enc,                        # [78, 1, 256]
    prompt_mask=prompt_mask,                        # [1, 78]
)
mask_logits = seg_out["pred_masks"]                 # [1, 200, 288, 288]
```

### 原始代码

```python
# sam3_image.py: _run_segmentation_heads(), line 388-420
num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)  # 200
obj_queries = hs if self.o2m_mask_predict else hs[:, :, :num_o2o]

seg_head_outputs = activation_ckpt_wrapper(self.segmentation_head)(
    backbone_feats=backbone_out["backbone_fpn"],    # [feat_4x, feat_2x, feat_1x]
    obj_queries=obj_queries,                         # [6, 1, 200, 256]
    image_ids=img_ids,                               # find_input.img_ids
    encoder_hidden_states=out["encoder_hidden_states"],  # = memory
    act_ckpt_enable=self.training and ...,
    prompt=prompt,                                   # 编码后的 prompt
    prompt_mask=prompt_mask,
)
# seg_head_outputs 包含 "pred_masks", "semantic_seg", "presence_logit"
```

### UniversalSegmentationHead.forward 内部流程 (`maskformer_segmentation.py:268-323`)

```python
def forward(self, backbone_feats, obj_queries, image_ids,
            encoder_hidden_states, prompt, prompt_mask, **kwargs):

    # 1. Cross-attention to prompt (如果配置了 cross_attend_prompt)
    tgt2 = self.cross_attn_norm(encoder_hidden_states)           # LayerNorm
    tgt2 = self.cross_attend_prompt(                             # nn.MultiheadAttention
        query=tgt2, key=prompt, value=prompt,
        key_padding_mask=prompt_mask
    )[0]
    encoder_hidden_states = tgt2 + encoder_hidden_states         # 残差连接

    # 2. Presence head (池化 encoder 输出 → 预测 presence logit)
    pooled_enc = encoder_hidden_states.mean(0)                   # [1, 256]
    presence_logit = self.presence_head(pooled_enc, prompt, prompt_mask)

    # 3. Pixel embedding (FPN 上采样)
    pixel_embed = self._embed_pixels(backbone_feats, image_ids, encoder_hidden_states)
    # _embed_pixels 内部:
    #   a. use_encoder_inputs=True 时，用 encoder_hidden_states 替换 backbone_feats[-1]
    #      encoder_hidden_states.permute(1,2,0) → [1, 256, 5184] → reshape → [1, 256, 72, 72]
    #      backbone_visual_feats[-1] = encoder_visual_embed
    #   b. pixel_decoder(backbone_visual_feats):
    #      从低分辨率到高分辨率逐级上采样 + 融合 (FPN)
    #      feat_1x [1,256,72,72] → +interpolate → feat_2x [1,256,144,144]
    #                             → +interpolate → feat_4x [1,256,288,288]
    #      每级: conv + groupnorm + relu
    # pixel_embed: [256, 288, 288] (bs=1 时 squeeze 了 batch 维)

    # 4. Instance segmentation head (可能有额外处理)
    instance_embeds = self.instance_seg_head(pixel_embed)

    # 5. Mask prediction
    mask_pred = self.mask_predictor(obj_queries[-1], instance_embeds)
    # mask_predictor 内部:
    #   mask_embed = self.mask_embed(obj_queries[-1])     # Linear(256, 256)
    #   mask_pred = einsum("bqc,chw->bqhw", mask_embed, pixel_embed)
    #   → [1, 200, 288, 288]

    return {"pred_masks": mask_pred, "semantic_seg": ..., "presence_logit": ...}
```

### 逐行对比

| Wrapper | 原始 | 说明 |
|---------|------|------|
| `backbone_feats=[feat_4x, feat_2x, feat_1x]` | `backbone_out["backbone_fpn"]` | 内容相同，都是 3 个尺度的特征图 |
| `image_ids=torch.zeros(1, dtype=torch.long)` | `img_ids=find_input.img_ids` | 单张图推理时 img_ids = [0] |
| `prompt=prompt_after_enc` | `prompt=prompt` | 都是编码后的 prompt。**注意**: 如果不传 prompt，cross_attend_prompt 会对 None 调用方法导致崩溃 |
| 无 `activation_ckpt_wrapper` | `activation_ckpt_wrapper(self.segmentation_head)(...)` | eval 时等价直接调用 |

### 变量含义

- **backbone_feats**: `[feat_4x, feat_2x, feat_1x]` 对应 3 个空间分辨率的特征图
  - feat_4x: `[1, 256, 288, 288]` — 4× 上采样分辨率
  - feat_2x: `[1, 256, 144, 144]` — 2× 上采样分辨率
  - feat_1x: `[1, 256, 72, 72]` — 原始分辨率
- **obj_queries** `[6, 1, 200, 256]`: 取最后一层 `obj_queries[-1]` = `[1, 200, 256]` 用于 mask 预测
- **encoder_hidden_states** `[5184, 1, 256]`: 替换 `backbone_feats[-1]`，使 mask 预测基于编码后的特征
- **mask_logits** `[1, 200, 288, 288]`: 每个 query 对应一个 288×288 的 mask logit（需 sigmoid 得到概率）

---

## 12. Step 10: Extract Last Layer

### Wrapper 代码

```python
# export_image_model_onnx.py: line 314-324

# Scores: apply sigmoid to get final detection probabilities
scores = outputs_class[-1, :, :, 0].sigmoid()   # [6,1,200,1] → 取最后一层 → [1,200] → sigmoid
# Also multiply by presence score (matching processor behavior)
if dec_presence_out is not None:
    presence = dec_presence_out[-1].sigmoid()     # [6,1,1] → 取最后一层 → [1,1] → sigmoid
    scores = scores * presence                    # [1,200] * [1,1] = [1,200]

boxes_xyxy = outputs_boxes_xyxy[-1]              # [6,1,200,4] → 取最后一层 → [1,200,4]

return scores, boxes_xyxy, mask_logits
```

### 原始代码 (后处理)

```python
# sam3_image_processor.py: _forward_grounding(), line 183-222

outputs = self.model.forward_grounding(...)
out_logits = outputs["pred_logits"]          # 最后一层的 logit [1, 200, 1]
out_probs = out_logits.sigmoid()             # sigmoid

presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)  # [1, 1, 1]
out_probs = (out_probs * presence_score).squeeze(-1)   # [1, 200]

out_bbox = outputs["pred_boxes"]             # cxcywh [1, 200, 4]
boxes = box_ops.box_cxcywh_to_xyxy(out_bbox) # xyxy [1, 200, 4]

out_masks = outputs["pred_masks"]            # [1, 200, 288, 288]
```

### 逐行对比

| Wrapper | 原始 (processor) | 说明 |
|---------|-----------------|------|
| `outputs_class[-1,:,:,0].sigmoid()` | `outputs["pred_logits"].sigmoid()` | `pred_logits` 就是最后一层的 logit。Wrapper 从完整 6 层结果 `[-1]` 取最后一层。`.squeeze(0)` 等于 `[:,:,0]` |
| `dec_presence_out[-1].sigmoid()` | `outputs["presence_logit_dec"].sigmoid()` | 相同操作 |
| `scores * presence` | `out_probs * presence_score` → `.squeeze(-1)` | 相同操作，维度处理方式略有不同 |
| `outputs_boxes_xyxy[-1]` | `box_cxcywh_to_xyxy(outputs["pred_boxes"])` | Wrapper 在 Step 8 已提前转为 xyxy，这里直接取最后一层 |

### 与原始后处理的差异

原始 processor 中还有额外的后处理步骤（score 阈值过滤、resize masks 到原始尺寸等），这些**不在 ONNX 模型中**，需要在 ONNX 推理端自行实现：

```python
# 这些在 ONNX 之外做:
keep = out_probs > confidence_threshold     # 阈值过滤
boxes = boxes * scale_fct                    # 归一化坐标 → 像素坐标
out_masks = interpolate(...).sigmoid()       # resize + sigmoid
masks = out_masks > 0.5                      # 二值化
```

---

## 13. 省略了什么

### 13.1 训练相关逻辑

| 省略内容 | 原始位置 | 原因 |
|---------|---------|------|
| DAC (Dual Assignment) | decoder.py: `apply_dac` 路径 | `dac and self.training = False` |
| activation_ckpt_wrapper | 所有模块 | eval 时 `act_ckpt_enable=False`，等于直接调用 |
| o2m queries | _update_scores_and_boxes | DAC 关闭时 `num_o2m = 0` |
| _compute_matching | forward_grounding | 训练用的 matcher |
| .clone() / .detach() | _update_scores_and_boxes | 推理时无梯度 |
| aux_loss 相关 | _update_scores_and_boxes | Wrapper 只取最后一层 |

### 13.2 多模态/多类型 prompt

| 省略内容 | 原始位置 | 原因 |
|---------|---------|------|
| Point encoding | geometry_encoders.py | 文本检测不使用 point prompt |
| Box encoding | geometry_encoders.py | 文本检测不使用 box prompt（dummy prompt 的 boxes 为空） |
| Mask encoding | geometry_encoders.py | 文本检测不使用 mask prompt |
| ROI pooling | geometry_encoders.py | 仅点/框 prompt 使用 |
| visual_prompt_embed | _encode_prompt | 文本检测时为空张量 [0,...] |
| instance_dot_prod_scoring | _update_scores_and_boxes | 不是 instance prompt |
| instance_bbox_embed | decoder.py | 不是 instance prompt |

### 13.3 后处理

| 省略内容 | 原始位置 | 原因 |
|---------|---------|------|
| Score 阈值过滤 | sam3_image_processor.py | ONNX 输出原始结果，由推理端过滤 |
| Mask resize + sigmoid | sam3_image_processor.py | 同上 |
| Box 坐标缩放 | sam3_image_processor.py | 同上 |

### 13.4 多尺度 / 多 batch

| 省略内容 | 原始位置 | 原因 |
|---------|---------|------|
| num_feature_levels > 1 | encoder.py | 当前模型配置 `num_feature_levels=1` |
| bs > 1 支持 | maskformer_segmentation.py | ONNX 固定 batch_size=1 |
| id_mapping | _get_img_feats | 单帧推理不需要 |
