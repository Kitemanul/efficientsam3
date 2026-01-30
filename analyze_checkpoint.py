#!/usr/bin/env python3
"""Analyze checkpoint file structure and parameter counts."""

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

    print(f"\nTotal keys in state_dict: {len(state_dict)}")

    # Analyze by top-level component
    components = defaultdict(lambda: {'params': 0, 'keys': 0, 'subcomponents': defaultdict(lambda: {'params': 0, 'keys': 0})})

    for key, value in state_dict.items():
        parts = key.split('.')
        if len(parts) >= 1:
            top_level = parts[0]
            # Count parameters
            param_count = value.numel()
            components[top_level]['params'] += param_count
            components[top_level]['keys'] += 1

            # Track subcomponents (second level)
            if len(parts) >= 2:
                sub_level = parts[1]
                components[top_level]['subcomponents'][sub_level]['params'] += param_count
                components[top_level]['subcomponents'][sub_level]['keys'] += 1

    # Print results in markdown table format
    print("\n## 权重文件组件统计表 (Weight File Component Statistics)\n")
    print("### 顶层组件 (Top-Level Components)\n")
    print("| 组件 (Component) | 参数数量 (Parameters) | 参数量(M) | 权重键数 (Keys) | 来源 (Source) | ONNX导出 |")
    print("|-----------------|---------------------|-----------|----------------|---------------|----------|")

    total_params = 0
    for comp_name in sorted(components.keys()):
        comp_data = components[comp_name]
        params = comp_data['params']
        total_params += params
        params_m = params / 1e6
        keys = comp_data['keys']

        # Determine source and ONNX export status
        if comp_name == 'detector':
            source = "混合 (Mixed)"
            onnx = "部分 (Partial)"
        elif comp_name == 'tracker':
            source = "SAM3原始"
            onnx = "✅ 是"
        else:
            source = "未知"
            onnx = "未知"

        print(f"| {comp_name} | {params:,} | {params_m:.2f}M | {keys} | {source} | {onnx} |")

    print(f"| **总计 (Total)** | **{total_params:,}** | **{total_params/1e6:.2f}M** | **{len(state_dict)}** | - | - |")

    # Print detailed breakdown for each top-level component
    for comp_name in sorted(components.keys()):
        comp_data = components[comp_name]
        print(f"\n### {comp_name} 子组件详情 (Subcomponents)\n")
        print("| 子组件 (Subcomponent) | 参数数量 (Parameters) | 参数量(M) | 权重键数 (Keys) | 来源 (Source) | ONNX导出 |")
        print("|----------------------|---------------------|-----------|----------------|---------------|----------|")

        subcomps = comp_data['subcomponents']
        for sub_name in sorted(subcomps.keys()):
            sub_data = subcomps[sub_name]
            params = sub_data['params']
            params_m = params / 1e6
            keys = sub_data['keys']

            # Determine source and ONNX status for detector subcomponents
            if comp_name == 'detector':
                if sub_name == 'vision_encoder':
                    source = "Stage1蒸馏 (Distilled)"
                    onnx = "✅ 是"
                elif sub_name == 'text_encoder':
                    source = "Stage1蒸馏 (Distilled)"
                    onnx = "✅ 是"
                elif sub_name == 'transformer':
                    source = "SAM3原始 (Original)"
                    onnx = "❌ 否"
                elif sub_name == 'segmentation_head':
                    source = "SAM3原始 (Original)"
                    onnx = "⚠️ 困难"
                elif sub_name == 'dot_prod_scoring':
                    source = "SAM3原始 (Original)"
                    onnx = "⚠️ 困难"
                else:
                    source = "未知"
                    onnx = "未知"
            elif comp_name == 'tracker':
                if sub_name in ['prompt_encoder', 'mask_decoder']:
                    source = "SAM3原始 (Original)"
                    onnx = "✅ 是"
                elif sub_name == 'memory_attention':
                    source = "SAM3原始 (Original)"
                    onnx = "⚠️ 困难"
                elif sub_name == 'memory_encoder':
                    source = "SAM3原始 (Original)"
                    onnx = "✅ 是"
                else:
                    source = "SAM3原始 (Original)"
                    onnx = "未知"
            else:
                source = "未知"
                onnx = "未知"

            print(f"| {sub_name} | {params:,} | {params_m:.2f}M | {keys} | {source} | {onnx} |")

    # Print all keys for reference
    print("\n### 所有权重键列表 (All Weight Keys)\n")
    print("<details>")
    print("<summary>点击展开完整键列表 (Click to expand full key list)</summary>\n")
    print("```")
    for key in sorted(state_dict.keys()):
        shape = list(state_dict[key].shape)
        print(f"{key}: {shape}")
    print("```")
    print("</details>")

if __name__ == '__main__':
    checkpoint_path = '/Users/wanghao/efficientsam3/checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth'
    analyze_checkpoint(checkpoint_path)
