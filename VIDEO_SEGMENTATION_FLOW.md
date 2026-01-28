# EfficientSAM3 视频分割流程

## Detector + Tracker 协作架构

```
视频输入: Frame 0, Frame 1, Frame 2, ..., Frame N
             │
             ▼
    ┌────────────────────────────────────────────────┐
    │      对每一帧执行 _det_track_one_frame         │
    └────────────────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
        ▼                                  ▼
┏━━━━━━━━━━━━━━━━━━━━━┓          ┏━━━━━━━━━━━━━━━━━━━━━┓
┃  Detector 分支       ┃          ┃  Tracker 分支        ┃
┃  (检测新物体)        ┃          ┃  (跟踪已知物体)      ┃
┗━━━━━━━━━━━━━━━━━━━━━┛          ┗━━━━━━━━━━━━━━━━━━━━━┛
```

---

## 完整的5步流程

### Frame t 的处理流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                       Frame t 输入                                   │
│                  [1, 3, 1024, 1024]                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              │
    ┌─────────────────────────┴─────────────────────────┐
    │                                                     │
    ▼                                                     ▼

═══════════════════════════════════    ═══════════════════════════════════
 Step 1: Detector 检测 (98%参数)          Step 2: Tracker 传播 (2%参数)
═══════════════════════════════════    ═══════════════════════════════════

┌─────────────────────────┐              ┌─────────────────────────┐
│ run_backbone_and_       │              │ run_tracker_            │
│ detection()             │              │ propagation()           │
│                         │              │                         │
│ • Image Encoder         │              │ 从 tracker_states_local  │
│   (RepViT-M0.9)         │              │ (上一帧的跟踪状态)       │
│                         │              │                         │
│ • Text Encoder          │              │ • Memory Attention      │
│   (MobileCLIP-S1)       │              │   (记忆过去6帧)          │
│                         │              │                         │
│ • Transformer Fusion    │              │ • SAM Mask Decoder      │
│   (跨模态注意力)         │              │   (生成mask)            │
│                         │              │                         │
│ • Segmentation Head     │              │ • Temporal Transformer  │
│   (生成mask)            │              │   (时序建模)            │
│                         │              │                         │
└──────────┬──────────────┘              └──────────┬──────────────┘
           │                                        │
           ▼                                        ▼
   det_out (新检测)                       tracker_masks (传播的masks)
   • pred_masks [M, H, W]                • pred_masks [N, H, W]
   • pred_boxes [M, 4]                   • obj_ids [N]
   • confidence_scores [M]                • confidence_scores [N]
           │                                        │
           M 个新物体候选                            N 个跟踪中的物体
           │                                        │
           └────────────┬───────────────────────────┘
                        │
                        ▼

═════════════════════════════════════════════════════════════════════
 Step 3: 匹配与规划 (run_tracker_update_planning_phase)
═════════════════════════════════════════════════════════════════════

           ┌─────────────────────────────────┐
           │  在 GPU 0 (主节点) 执行启发式     │
           │                                 │
           │  1. 匹配 (IoU Matching):        │
           │     - 计算 det_boxes 和         │
           │       tracker_boxes 的 IoU      │
           │     - IoU > threshold → 匹配    │
           │                                 │
           │  2. 去重 (NMS):                 │
           │     - 检测框之间的 NMS          │
           │     - 跟踪框之间的 NMS          │
           │                                 │
           │  3. 决策:                       │
           │     - 新物体: 未匹配的检测框     │
           │     - 保留物体: 匹配的跟踪      │
           │     - 删除物体: 丢失的跟踪      │
           │                                 │
           │  4. 负载均衡:                   │
           │     - 分配物体到不同GPU          │
           │     - 最多跟踪 max_obj_to_track │
           │                                 │
           └────────────┬────────────────────┘
                        │
                        ▼
              tracker_update_plan
              {
                "add": [obj_1, obj_2, ...],      # 新检测到的物体
                "keep": [obj_3, obj_4, ...],     # 继续跟踪
                "remove": [obj_5, obj_6, ...],   # 删除丢失的
                "match": {det_id: trk_id, ...}   # 检测-跟踪匹配
              }
                        │
                        ▼

═════════════════════════════════════════════════════════════════════
 Step 4: 执行更新 (run_tracker_update_execution_phase)
═════════════════════════════════════════════════════════════════════

           ┌─────────────────────────────────┐
           │  每个 GPU 更新本地 tracker_states│
           │                                 │
           │  For 新增物体:                  │
           │    • 初始化 SAM state           │
           │    • 编码 mask 到 memory        │
           │    • 添加到 tracker_states_local│
           │                                 │
           │  For 保留物体:                  │
           │    • 更新 memory bank           │
           │    • 编码当前帧 mask            │
           │    • 更新 temporal features     │
           │                                 │
           │  For 删除物体:                  │
           │    • 从 tracker_states_local    │
           │      中移除                     │
           │                                 │
           └────────────┬────────────────────┘
                        │
                        ▼
            tracker_states_local_new
            (更新后的跟踪状态 → 下一帧使用)
                        │
                        ▼

═════════════════════════════════════════════════════════════════════
 Step 5: 构建输出 (build_outputs)
═════════════════════════════════════════════════════════════════════

           ┌─────────────────────────────────┐
           │  合并 Detector + Tracker 的结果 │
           │                                 │
           │  优先级:                        │
           │  1. 匹配的跟踪 (高置信度)       │
           │  2. 新检测 (首次出现)           │
           │  3. 未匹配的跟踪 (待确认)       │
           │                                 │
           │  后处理:                        │
           │  • 上采样到原始分辨率           │
           │  • 二值化 mask                  │
           │  • 填充孔洞                     │
           │                                 │
           └────────────┬────────────────────┘
                        │
                        ▼
              obj_id_to_mask (Frame t)
              {
                obj_1: mask_1 [H, W],
                obj_2: mask_2 [H, W],
                ...
              }
                        │
                        ▼
              输出到用户 / 下一帧
```

---

## 关键概念解释

### 1. Detector 的角色

**检测新出现的物体**

- 输入：当前帧图像 + 文本提示（可选）
- 使用：98% 的模型参数（Image/Text Encoder + Transformer + Seg Head）
- 输出：M 个候选物体的 masks 和 boxes
- 特点：**无状态**，每帧独立检测

### 2. Tracker 的角色

**跟踪已知物体**

- 输入：当前帧图像 + 过去的 memory（最多7帧）
- 使用：2% 的模型参数（SAM Decoder + Memory Backbone + Temporal Transformer）
- 输出：N 个跟踪中物体的 masks
- 特点：**有状态**，记忆过去帧

### 3. Memory Mechanism

```
Tracker 的记忆机制 (最多7帧)

Frame t-6: mask → memory_6
Frame t-5: mask → memory_5
Frame t-4: mask → memory_4
Frame t-3: mask → memory_3
Frame t-2: mask → memory_2
Frame t-1: mask → memory_1
Frame t:   image_emb
           ↓
  ┌────────────────────┐
  │ Memory Attention   │
  │                    │
  │ Query: Frame t     │
  │ Key/Value:         │
  │  memory_1 ~ 6      │
  └────────┬───────────┘
           ↓
    Predicted mask t
```

**作用**：
- 通过 cross-attention 查询过去的 memory
- 实现时序一致性（同一物体在不同帧的mask一致）
- 处理遮挡和消失（短暂消失后重新出现）

### 4. 匹配策略（Step 3）

```python
# 伪代码
det_boxes = detector的检测框 [M, 4]
trk_boxes = tracker的预测框 [N, 4]

# 计算IoU矩阵
iou_matrix = compute_iou(det_boxes, trk_boxes)  # [M, N]

# 匹配规则
for det_i in range(M):
    for trk_j in range(N):
        if iou_matrix[det_i, trk_j] > 0.5:  # 匹配阈值
            match[det_i] = trk_j
            # 决策：使用tracker的结果（更稳定）
            # 用detector的结果更新tracker的memory

# 未匹配的detector → 新物体，添加到tracker
# 未匹配的tracker → 可能消失，继续跟踪几帧后删除
```

---

## 实际例子

### 场景：追踪视频中的狗

```
文本提示: "a dog"

═════════════════════════════════════════════════════════════════

Frame 0 (首帧):
  Detector: 检测到 2 只狗 → dog_1, dog_2
  Tracker: 空（无历史）
  → 输出: dog_1, dog_2
  → 初始化 tracker_states: [dog_1_state, dog_2_state]

═════════════════════════════════════════════════════════════════

Frame 1:
  Detector: 检测到 2 只狗 → det_1, det_2
  Tracker: 传播 dog_1, dog_2 → trk_1, trk_2
  匹配:
    - det_1 ↔ trk_1 (IoU=0.85) ✅ 匹配
    - det_2 ↔ trk_2 (IoU=0.78) ✅ 匹配
  → 输出: dog_1 (来自tracker), dog_2 (来自tracker)
  → 更新 memory: 编码 Frame 1 的 masks

═════════════════════════════════════════════════════════════════

Frame 5:
  Detector: 检测到 3 个物体 → det_1, det_2, det_3
  Tracker: 传播 dog_1, dog_2 → trk_1, trk_2
  匹配:
    - det_1 ↔ trk_1 (IoU=0.82) ✅ 匹配
    - det_2 ↔ trk_2 (IoU=0.75) ✅ 匹配
    - det_3: 无匹配 → 新物体！
  → 输出: dog_1, dog_2 (tracker), dog_3 (新增)
  → 添加 dog_3 到 tracker_states

═════════════════════════════════════════════════════════════════

Frame 10 (dog_2 被遮挡):
  Detector: 检测到 2 个物体 → det_1, det_3
  Tracker: 传播 dog_1, dog_2, dog_3 → trk_1, trk_2, trk_3
  匹配:
    - det_1 ↔ trk_1 ✅
    - det_3 ↔ trk_3 ✅
    - trk_2 (dog_2): 无匹配 → 可能被遮挡
  → 输出: dog_1, dog_2 (低置信度), dog_3
  → 保留 dog_2 在 tracker (给几帧缓冲期)

═════════════════════════════════════════════════════════════════

Frame 15 (dog_2 重新出现):
  Detector: 检测到 3 个物体 → det_1, det_2, det_3
  Tracker: 传播 dog_1, dog_2, dog_3 → trk_1, trk_2, trk_3
  匹配:
    - det_2 ↔ trk_2 (IoU=0.72) ✅ dog_2 重新匹配！
  → 输出: dog_1, dog_2 (恢复), dog_3
  → dog_2 的 ID 保持不变（时序一致性）
```

---

## 参数分布与职责

| 模块 | 参数量 | 占比 | 职责 | 使用频率 |
|-----|--------|------|------|---------|
| **Detector** | 573.93M | 98% | 检测新物体 | 每帧 |
| └─ Image Encoder | 4.72M | 0.8% | 图像特征 | 每帧 |
| └─ Text Encoder | 63.56M | 10.9% | 文本特征 | 首帧/更新提示时 |
| └─ Transformer Fusion | 21.05M | 3.6% | 跨模态融合 | 每帧 |
| └─ Segmentation Head | 2.30M | 0.4% | 生成mask | 每帧 |
| **Tracker** | 11.74M | 2% | 跟踪物体 | 每帧（第2帧起）|
| └─ Memory Backbone | 1.38M | 0.2% | 编码memory | 每帧 |
| └─ Temporal Transformer | 5.92M | 1.0% | 时序建模 | 每帧 |
| └─ SAM Mask Decoder | 4.22M | 0.7% | 生成mask | 每帧 |
| └─ SAM Prompt Encoder | 0.01M | 0.002% | 点/框提示 | 用户交互时 |

---

## 设计优势

### 1. **互补性**

- Detector：精准检测，但无时序一致性
- Tracker：时序平滑，但可能漂移
- 结合：Detector修正漂移，Tracker保持稳定

### 2. **效率**

- Detector 98% 参数，但检测准确
- Tracker 2% 参数，但跟踪流畅
- 视频中跟踪占主导 → 整体高效

### 3. **鲁棒性**

- 遮挡：Tracker 记忆保持ID
- 新物体：Detector 实时发现
- 消失/重现：匹配策略处理

---

## 总结

**Detector + Tracker = 完整的视频物体分割系统**

```
Detector (检测器)：
  "找出所有符合文本描述的物体"
  → 语义驱动，无状态

Tracker (跟踪器)：
  "记住这些物体，跟踪它们的运动"
  → 几何驱动，有状态（memory）

协作方式：
  1. Detector 每帧检测 → 发现新物体
  2. Tracker 每帧传播 → 保持ID一致性
  3. 匹配合并 → 最佳结果
  4. 更新 memory → 下一帧使用

结果：
  ✅ 准确的语义理解（Detector）
  ✅ 流畅的时序一致性（Tracker）
  ✅ 鲁棒的遮挡处理（Memory）
```
