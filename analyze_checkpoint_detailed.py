#!/usr/bin/env python3
"""Detailed checkpoint analysis with proper component breakdown."""

import torch
from collections import defaultdict

def analyze_checkpoint(checkpoint_path):
    """Analyze checkpoint and print component statistics."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Define component patterns for better grouping
    component_groups = {
        # Detector components
        'detector.backbone.vision_backbone': {
            'name': 'Vision Encoder (RepViT)',
            'source': 'Stage1蒸馏',
            'onnx': '✅ 可导出'
        },
        'detector.backbone.language_backbone': {
            'name': 'Text Encoder (MobileCLIP)',
            'source': 'Stage1蒸馏',
            'onnx': '✅ 可导出'
        },
        'detector.geometry_encoder': {
            'name': 'Geometry Encoder',
            'source': 'SAM3原始',
            'onnx': '✅ 可导出'
        },
        'detector.transformer': {
            'name': 'Transformer Fusion',
            'source': 'SAM3原始',
            'onnx': '❌ 不可导出'
        },
        'detector.segmentation_head': {
            'name': 'Segmentation Head',
            'source': 'SAM3原始',
            'onnx': '⚠️ 困难'
        },
        'detector.dot_prod_scoring': {
            'name': 'Scoring Module',
            'source': 'SAM3原始',
            'onnx': '⚠️ 困难'
        },
        # Tracker components
        'tracker.sam_prompt_encoder': {
            'name': 'SAM Prompt Encoder',
            'source': 'SAM3原始',
            'onnx': '✅ 可导出'
        },
        'tracker.sam_mask_decoder': {
            'name': 'SAM Mask Decoder',
            'source': 'SAM3原始',
            'onnx': '✅ 可导出'
        },
        'tracker.transformer': {
            'name': 'Memory Attention',
            'source': 'SAM3原始',
            'onnx': '⚠️ 困难'
        },
        'tracker.maskmem_backbone': {
            'name': 'Memory Encoder Backbone',
            'source': 'SAM3原始',
            'onnx': '✅ 可导出'
        },
        'tracker.obj_ptr': {
            'name': 'Object Pointer',
            'source': 'SAM3原始',
            'onnx': '✅ 可导出'
        },
    }

    # Count parameters for each component group
    component_stats = defaultdict(lambda: {'params': 0, 'keys': 0})
    unmatched_stats = defaultdict(lambda: {'params': 0, 'keys': 0})

    for key, value in state_dict.items():
        param_count = value.numel()
        matched = False

        for prefix, info in component_groups.items():
            if key.startswith(prefix):
                component_stats[prefix]['params'] += param_count
                component_stats[prefix]['keys'] += 1
                matched = True
                break

        if not matched:
            # Group unmatched by first two levels
            parts = key.split('.')
            if len(parts) >= 2:
                group_key = f"{parts[0]}.{parts[1]}"
            else:
                group_key = parts[0]
            unmatched_stats[group_key]['params'] += param_count
            unmatched_stats[group_key]['keys'] += 1

    # Calculate totals
    total_params = sum(v['params'] for v in component_stats.values())
    total_params += sum(v['params'] for v in unmatched_stats.values())
    total_keys = len(state_dict)

    detector_params = sum(v['params'] for k, v in component_stats.items() if k.startswith('detector'))
    detector_params += sum(v['params'] for k, v in unmatched_stats.items() if k.startswith('detector'))

    tracker_params = sum(v['params'] for k, v in component_stats.items() if k.startswith('tracker'))
    tracker_params += sum(v['params'] for k, v in unmatched_stats.items() if k.startswith('tracker'))

    # Print markdown output
    print("\n# EfficientSAM3 权重文件组件统计\n")
    print(f"**权重文件**: `{checkpoint_path.split('/')[-1]}`\n")
    print(f"**总参数量**: {total_params:,} ({total_params/1e6:.2f}M)\n")
    print(f"**总权重键数**: {total_keys}\n")

    print("## 总体结构\n")
    print("| 模块 | 参数量 | 占比 |")
    print("|------|--------|------|")
    print(f"| Detector (概念分割) | {detector_params:,} ({detector_params/1e6:.2f}M) | {detector_params/total_params*100:.1f}% |")
    print(f"| Tracker (视频追踪) | {tracker_params:,} ({tracker_params/1e6:.2f}M) | {tracker_params/total_params*100:.1f}% |")
    print(f"| **总计** | **{total_params:,}** | **100%** |")

    print("\n## Detector 详细组件 (单帧概念分割)\n")
    print("| 组件 | 参数量 | 参数量(M) | 权重键数 | 来源 | ONNX导出 |")
    print("|------|--------|-----------|----------|------|----------|")

    detector_components = [(k, v) for k, v in component_stats.items() if k.startswith('detector')]
    detector_unmatched = [(k, v) for k, v in unmatched_stats.items() if k.startswith('detector')]

    for prefix in sorted([k for k in component_stats.keys() if k.startswith('detector')]):
        info = component_groups[prefix]
        stats = component_stats[prefix]
        print(f"| {info['name']} | {stats['params']:,} | {stats['params']/1e6:.2f}M | {stats['keys']} | {info['source']} | {info['onnx']} |")

    # Handle unmatched detector components
    for prefix in sorted([k for k in unmatched_stats.keys() if k.startswith('detector')]):
        stats = unmatched_stats[prefix]
        name = prefix.replace('detector.', '').replace('_', ' ').title()
        print(f"| {name} | {stats['params']:,} | {stats['params']/1e6:.2f}M | {stats['keys']} | 未知 | 未知 |")

    print("\n## Tracker 详细组件 (视频追踪)\n")
    print("| 组件 | 参数量 | 参数量(M) | 权重键数 | 来源 | ONNX导出 |")
    print("|------|--------|-----------|----------|------|----------|")

    for prefix in sorted([k for k in component_stats.keys() if k.startswith('tracker')]):
        info = component_groups[prefix]
        stats = component_stats[prefix]
        print(f"| {info['name']} | {stats['params']:,} | {stats['params']/1e6:.2f}M | {stats['keys']} | {info['source']} | {info['onnx']} |")

    # Handle unmatched tracker components
    for prefix in sorted([k for k in unmatched_stats.keys() if k.startswith('tracker')]):
        stats = unmatched_stats[prefix]
        name = prefix.replace('tracker.', '').replace('_', ' ').title()
        print(f"| {name} | {stats['params']:,} | {stats['params']/1e6:.2f}M | {stats['keys']} | SAM3原始 | 未知 |")

    print("\n## Stage1 蒸馏模型 vs SAM3原始组件\n")

    distilled_params = 0
    sam3_params = 0

    for prefix, info in component_groups.items():
        if prefix in component_stats:
            if info['source'] == 'Stage1蒸馏':
                distilled_params += component_stats[prefix]['params']
            else:
                sam3_params += component_stats[prefix]['params']

    # Add unmatched to SAM3
    for prefix, stats in unmatched_stats.items():
        sam3_params += stats['params']

    print("| 类型 | 参数量 | 占比 |")
    print("|------|--------|------|")
    print(f"| Stage1蒸馏模型 | {distilled_params:,} ({distilled_params/1e6:.2f}M) | {distilled_params/total_params*100:.1f}% |")
    print(f"| SAM3原始组件 | {sam3_params:,} ({sam3_params/1e6:.2f}M) | {sam3_params/total_params*100:.1f}% |")

    print("\n## ONNX导出状态总结\n")

    exportable_params = 0
    difficult_params = 0
    non_exportable_params = 0

    for prefix, info in component_groups.items():
        if prefix in component_stats:
            params = component_stats[prefix]['params']
            if info['onnx'].startswith('✅'):
                exportable_params += params
            elif info['onnx'].startswith('⚠️'):
                difficult_params += params
            elif info['onnx'].startswith('❌'):
                non_exportable_params += params

    # Add unmatched to unknown
    unknown_params = sum(v['params'] for v in unmatched_stats.values())

    print("| 状态 | 参数量 | 组件 |")
    print("|------|--------|------|")
    print(f"| ✅ 可导出 | {exportable_params:,} ({exportable_params/1e6:.2f}M) | Vision Encoder, Text Encoder, Geometry Encoder, SAM组件 |")
    print(f"| ⚠️ 导出困难 | {difficult_params:,} ({difficult_params/1e6:.2f}M) | Segmentation Head, Scoring Module, Memory Attention |")
    print(f"| ❌ 无法导出 | {non_exportable_params:,} ({non_exportable_params/1e6:.2f}M) | Transformer Fusion |")

    print("\n## 说明\n")
    print("- **Stage1蒸馏**: 使用SAM1数据进行知识蒸馏得到的轻量化模型 (RepViT + MobileCLIP)")
    print("- **SAM3原始**: 直接从SAM3模型继承的组件，未经过蒸馏")
    print("- **ONNX导出状态**:")
    print("  - ✅ 可导出: 标准架构，可直接导出ONNX")
    print("  - ⚠️ 导出困难: 包含动态逻辑或复杂输入，需要特殊处理")
    print("  - ❌ 无法导出: 包含自定义Cross-Attention和条件分支，建议使用TorchScript")

if __name__ == '__main__':
    checkpoint_path = '/Users/wanghao/efficientsam3/checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth'
    analyze_checkpoint(checkpoint_path)
