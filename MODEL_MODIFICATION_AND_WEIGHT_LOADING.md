# 模型结构修改与权重加载兼容性指南

本文档说明在修改模型结构后，如何正确加载原始权重文件。

---

## 核心问题

**问题**: 修改模型结构后，能否正确加载预训练权重文件？

**答案**: 取决于修改的具体内容。

---

## 情况分析

### 情况 1: 只移除条件分支 ✅ 可以加载

```python
# 原代码
def forward(self, ...):
    if self.pre_norm:
        return self.forward_pre(...)
    else:
        return self.forward_post(...)

# 修改后 (固定 pre_norm=True)
def forward(self, ...):
    return self.forward_pre(...)  # 直接调用，不做选择
```

**结果**: 权重可以正常加载，因为层结构没变 (norm1, norm2, linear1 等都还在)

**原因**:
- 两个版本的模型都包含完全相同的参数
- `state_dict` 的 key 名称完全相同
- 权重的 shape 完全相同

---

### 情况 2: 移除 DAC 逻辑 ✅ 可以加载

```python
# 原代码
def forward_pre(self, tgt, memory, dac=False, ...):
    if dac:
        other_tgt = tgt[tgt.shape[0] // 2 :]
        tgt = tgt[: tgt.shape[0] // 2]
    ...

# 修改后 (移除 dac)
def forward_pre(self, tgt, memory, ...):
    # 直接处理，不做切分
    ...
```

**结果**: 权重可以正常加载，DAC 只是运行时逻辑，不影响参数

**原因**:
- DAC 是一个条件参数，影响数据流，不影响模型参数
- 模型的所有权重层保持不变
- 只是改变了使用这些权重的方式

---

### 情况 3: 替换自定义 Attention ⚠️ 需要转换

```python
# 原代码 (自定义模块)
self.cross_attn_image = CustomCrossAttention(...)
# 参数名可能是: cross_attn_image.q_proj, cross_attn_image.k_proj, ...

# 修改后 (标准模块)
self.cross_attn_image = nn.MultiheadAttention(...)
# 参数名变成: cross_attn_image.in_proj_weight, cross_attn_image.out_proj, ...
```

**结果**: 权重**无法直接加载**，因为参数名不匹配

**错误信息**:
```
RuntimeError: Error(s) in loading state_dict for Model:
    Missing key(s) in state_dict: ['cross_attn_image.in_proj_weight', ...]
    Unexpected key(s) in state_dict: ['cross_attn_image.q_proj', ...]
```

**解决方法**: 写权重转换脚本

```python
def convert_attention_weights(old_state_dict):
    """将自定义Attention权重转换为标准MultiheadAttention格式"""
    new_state_dict = {}

    for key, value in old_state_dict.items():
        if 'cross_attn_image' in key:
            # 转换逻辑（需要根据具体的旧格式来写）
            if 'q_proj' in key:
                # 从 q_proj, k_proj, v_proj 合并为 in_proj_weight
                # 这需要理解两个模块的具体参数结构
                new_key = key.replace('q_proj', 'in_proj_weight')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict

# 使用
old_state = torch.load('old_weights.pth')
converted_state = convert_attention_weights(old_state)
model.load_state_dict(converted_state)
```

---

### 情况 4: 固定条件表达式 ✅ 可以加载

```python
# 原代码
tgt2 = self.cross_attn_image(
    query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
    key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
    value=memory,
)

# 修改后 (固定为一种)
tgt2 = self.cross_attn_image(
    query=tgt + query_pos,  # 固定加位置编码
    key=memory + pos,       # 固定加位置编码
    value=memory,
)
```

**结果**: 权重可以正常加载

**原因**:
- 这只改变了数据的计算方式，不改变模型参数
- 两个模块的 `state_dict` 完全相同
- 只是改变了输入处理方式

---

### 情况 5: 删除某些层 ❌ 部分权重丢失

```python
# 原代码
self.presence_head = DotProductScoring(...)

# 修改后 (删除)
# self.presence_head = None  # 删除了
```

**结果**: 加载时会报 `Unexpected key(s)` 警告，但可以用 `strict=False` 忽略

**加载方式**:
```python
model.load_state_dict(state_dict, strict=False)  # 忽略多余的权重
```

**输出**:
```
UserWarning: Some weights of the model checkpoint at ... were not used when
initializing Model: ['presence_head.weight', 'presence_head.bias', ...]
```

---

### 情况 6: 添加新层 ⚠️ 需要初始化

```python
# 原代码 (没有这个层)
# self.new_layer = None

# 修改后 (添加)
self.new_layer = nn.Linear(256, 256)  # 新增
```

**结果**: 新层权重加载不了，需要额外处理

**解决方法 1**: 随机初始化新层
```python
model.load_state_dict(state_dict, strict=False)
# 新层会被 PyTorch 的初始化器初始化
```

**解决方法 2**: 从现有权重复制
```python
# 用现有层的权重初始化新层
model.new_layer.weight.data = model.existing_layer.weight.data.clone()
model.new_layer.bias.data = model.existing_layer.bias.data.clone()
```

---

### 情况 7: 改变层维度 ❌ 不行

```python
# 原代码
self.linear = nn.Linear(256, 256)

# 修改后 (改变维度)
self.linear = nn.Linear(256, 512)  # 输出维度改了
```

**结果**: shape 不匹配，**无法加载**

**错误信息**:
```
RuntimeError: Error(s) in loading state_dict for Model:
    size mismatch for linear.weight: copying a param with shape torch.Size([256, 256])
    from checkpoint, the shape in current model is torch.Size([512, 256])
```

**解决方法**: 无法直接转换，需要重新训练或微调

---

## 修改方案总结表

| 修改类型 | 能否加载 | 难度 | 说明 |
|---------|---------|------|------|
| 移除 if/else 分支 | ✅ 可以 | 无 | 层结构不变，只是执行路径固定 |
| 移除 DAC 逻辑 | ✅ 可以 | 无 | 运行时逻辑，不影响参数 |
| 固定条件表达式 | ✅ 可以 | 无 | 输入处理变化，不影响参数 |
| 替换 Attention 模块 | ⚠️ 需转换 | 高 | 参数名/结构不同，需要映射脚本 |
| 删除某些层 | ⚠️ strict=False | 低 | 多余权重被忽略 |
| 添加新层 | ⚠️ 需初始化 | 低 | 新层随机初始化 |
| 改变层维度 | ❌ 不行 | 极高 | shape 不匹配，无法转换 |

---

## 推荐的安全修改方式

### Step 1: 保存原始模型的权重信息

```python
# 加载原始模型
original_model = build_model(original_config)
original_state = torch.load('weights.pth')

# 打印权重信息，便于后续调试
print("原始权重键:")
for key in sorted(original_state.keys()):
    print(f"  {key}: {original_state[key].shape}")
```

### Step 2: 创建修改后的模型

```python
# 创建修改后的模型（采用保守的修改策略）
modified_model = build_modified_model(modified_config)
```

### Step 3: 尝试加载，检查不匹配

```python
# 尝试加载，记录缺失和意外的键
missing, unexpected = modified_model.load_state_dict(
    original_state,
    strict=False
)

print(f"缺失的键 (新模型有，旧权重没有): {missing}")
print(f"意外的键 (旧权重有，新模型没有): {unexpected}")
```

### Step 4: 根据情况处理

```python
if not unexpected:
    # 完美匹配！可以直接使用
    print("✅ 权重加载完美匹配！")
else:
    # 有不匹配
    if len(unexpected) < 10:
        # 量少，可能是删除的层，可以接受
        print("⚠️ 有少量意外键，可能是删除的层")
    else:
        # 量大，需要写转换脚本
        print("❌ 权重不兼容，需要写转换脚本")

        # 尝试手动转换
        converted_state = convert_weights(original_state)
        modified_model.load_state_dict(converted_state, strict=True)
```

---

## 针对 Transformer Fusion 的具体建议

如果你想修改 Transformer Fusion 使其可导出 ONNX：

```
推荐修改顺序:
├── [✅ 第1步] 移除 if pre_norm/post_norm 分支
│   └─ 方法: 固定 self.pre_norm = True
│   └─ 权重兼容性: ✅ 完全兼容
│
├── [✅ 第2步] 移除 if dac 逻辑
│   └─ 方法: 移除 DAC 相关代码，固定 dac=False
│   └─ 权重兼容性: ✅ 完全兼容
│
├── [✅ 第3步] 固定条件表达式
│   └─ 方法: q + pos (不再 if condition else)
│   └─ 权重兼容性: ✅ 完全兼容
│
└── [⚠️ 第4步] 替换 cross_attention (最复杂)
    └─ 如果自定义Attention能映射到标准Attention
    └─ 权重兼容性: ⚠️ 需要转换脚本
```

**前3步都是保险的**，权重完全兼容。
**第4步是困难的**，需要理解两个 Attention 模块的参数结构才能写转换脚本。

---

## 总结

**记住这个规则**:
```
权重加载能否成功 = 模型结构是否改变了参数
                     ↓
                只改变数据流: ✅ 能加载
                改变参数名称: ⚠️ 需转换
                改变参数形状: ❌ 不能加载
```
