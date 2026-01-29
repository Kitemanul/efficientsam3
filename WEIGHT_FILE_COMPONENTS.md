# EfficientSAM3 权重文件组件统计

**权重文件**: `efficient_sam3_repvit-m0_9_mobileclip_s1.pth`

**总参数量**: 585,670,291 (585.67M)

**总权重键数**: 2028

## 总体结构

| 模块 | 参数量 | 占比 |
|------|--------|------|
| Detector (概念分割) | 573,926,673 (573.93M) | 98.0% |
| Tracker (视频追踪) | 11,743,618 (11.74M) | 2.0% |
| **总计** | **585,670,291** | **100%** |

## Detector 详细组件 (单帧概念分割)

| 组件 | 参数量 | 参数量(M) | 权重键数 | 来源 | ONNX导出 |
|------|--------|-----------|----------|------|----------|
| Text Encoder (MobileCLIP) | 63,559,424 | 63.56M | 151 | Stage1蒸馏 | ✅ 可导出 |
| Vision Encoder (RepViT) | 477,621,979 | 477.62M | 1171 | Stage1蒸馏 | ✅ 可导出 |
| Scoring Module | 1,182,976 | 1.18M | 10 | SAM3原始 | ⚠️ 困难 |
| Geometry Encoder | 8,215,808 | 8.22M | 76 | SAM3原始 | ✅ 可导出 |
| Segmentation Head | 2,298,881 | 2.30M | 28 | SAM3原始 | ⚠️ 困难 |
| Transformer Fusion | 21,047,605 | 21.05M | 283 | SAM3原始 | ❌ 不可导出 |

## Tracker 详细组件 (视频追踪)

| 组件 | 参数量 | 参数量(M) | 权重键数 | 来源 | ONNX导出 |
|------|--------|-----------|----------|------|----------|
| Memory Encoder Backbone | 1,384,608 | 1.38M | 40 | SAM3原始 | ✅ 可导出 |
| Object Pointer | 213,824 | 0.21M | 8 | SAM3原始 | ✅ 可导出 |
| SAM Mask Decoder | 4,215,109 | 4.22M | 131 | SAM3原始 | ✅ 可导出 |
| SAM Prompt Encoder | 6,476 | 0.01M | 17 | SAM3原始 | ✅ 可导出 |
| Memory Attention | 5,922,304 | 5.92M | 106 | SAM3原始 | ⚠️ 困难 |
| Mask Downsample | 17 | 0.00M | 2 | SAM3原始 | 未知 |
| Maskmem Tpos Enc | 448 | 0.00M | 1 | SAM3原始 | 未知 |
| No Mem Embed | 256 | 0.00M | 1 | SAM3原始 | 未知 |
| No Mem Pos Enc | 256 | 0.00M | 1 | SAM3原始 | 未知 |
| No Obj Embed Spatial | 64 | 0.00M | 1 | SAM3原始 | 未知 |
| No Obj Ptr | 256 | 0.00M | 1 | SAM3原始 | 未知 |

## Stage1 蒸馏模型 vs SAM3原始组件

| 类型 | 参数量 | 占比 |
|------|--------|------|
| Stage1蒸馏模型 | 541,181,403 (541.18M) | 92.4% |
| SAM3原始组件 | 44,488,888 (44.49M) | 7.6% |

## ONNX导出状态总结

| 状态 | 参数量 | 组件 |
|------|--------|------|
| ✅ 可导出 | 555,217,228 (555.22M) | Vision Encoder, Text Encoder, Geometry Encoder, SAM组件 |
| ⚠️ 导出困难 | 9,404,161 (9.40M) | Segmentation Head, Scoring Module, Memory Attention |
| ❌ 无法导出 | 21,047,605 (21.05M) | Transformer Fusion |

## 说明

- **Stage1蒸馏**: 使用SAM1数据进行知识蒸馏得到的轻量化模型 (RepViT + MobileCLIP)
- **SAM3原始**: 直接从SAM3模型继承的组件，未经过蒸馏
- **ONNX导出状态**:
  - ✅ 可导出: 标准架构，可直接导出ONNX
  - ⚠️ 导出困难: 包含动态逻辑或复杂输入，需要特殊处理
  - ❌ 无法导出: 包含自定义Cross-Attention和条件分支，建议使用TorchScript
