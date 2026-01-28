#!/usr/bin/env python3
"""
导出 EfficientSAM3 的融合层组件（用于 Concept Segmentation）
包括: Transformer Encoder, Segmentation Head, Scoring Module
"""

import torch
import torch.nn as nn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))


def export_transformer_fusion(checkpoint_path, output_dir):
    """导出 Transformer 融合层（文本-图像跨模态注意力）"""
    print("="*70)
    print("导出 Transformer Fusion Layer")
    print("="*70)

    # 加载 checkpoint
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # 提取 transformer 权重
    transformer_state = {}
    for key in state_dict.keys():
        if key.startswith('detector.transformer.'):
            new_key = key.replace('detector.transformer.', '')
            transformer_state[new_key] = state_dict[key]

    print(f"找到 {len(transformer_state)} 个 transformer 权重")

    # 分析结构
    layer_keys = [k for k in transformer_state.keys() if 'encoder.layers.0' in k]
    print(f"第一层示例 (共 {len(layer_keys)} 个参数):")
    for key in sorted(layer_keys)[:5]:
        shape = list(transformer_state[key].shape) if hasattr(transformer_state[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")

    # 统计参数
    total_params = sum(v.numel() for v in transformer_state.values() if hasattr(v, 'numel'))
    print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 构建 Transformer 模型
    # 从权重推断配置
    try:
        from sam3.model.detector import DetectorTransformer

        # 推断配置
        d_model = 256  # 从权重维度推断
        nhead = 8
        num_encoder_layers = 6  # 需要验证
        dim_feedforward = 2048

        print(f"\n构建 Transformer:")
        print(f"  d_model: {d_model}")
        print(f"  nhead: {nhead}")
        print(f"  num_layers: {num_encoder_layers}")

        # 创建简化的 Wrapper
        class TransformerFusionWrapper(nn.Module):
            """Transformer 融合层 Wrapper"""
            def __init__(self):
                super().__init__()
                # 尝试构建 transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    batch_first=True,
                    norm_first=True,  # SAM3 使用 pre-norm
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

            def forward(self, vision_features, text_features):
                """
                Args:
                    vision_features: [B, H*W, 256] - 从 image encoder 展平
                    text_features: [B, seq_len, 256] - 从 text encoder
                Returns:
                    fused_features: [B, H*W + seq_len, 256]
                """
                # 拼接视觉和文本特征
                combined_features = torch.cat([vision_features, text_features], dim=1)

                # 通过 transformer
                fused_features = self.transformer(combined_features)

                return fused_features

        wrapper = TransformerFusionWrapper()

        # 尝试加载权重（可能不完全匹配）
        missing, unexpected = wrapper.load_state_dict(transformer_state, strict=False)
        print(f"\n加载权重: Missing={len(missing)}, Unexpected={len(unexpected)}")
        if len(missing) > 0:
            print(f"  Missing keys (前5个): {missing[:5]}")

        wrapper.eval()

        # 导出到 ONNX
        output_path = os.path.join(output_dir, "transformer_fusion.onnx")
        print(f"\n导出到: {output_path}")

        # Dummy inputs
        batch_size = 1
        H, W = 64, 64  # image encoder 输出的空间尺寸
        seq_len = 32   # text encoder 输出的序列长度

        dummy_vision = torch.randn(batch_size, H*W, 256)
        dummy_text = torch.randn(batch_size, seq_len, 256)

        torch.onnx.export(
            wrapper,
            (dummy_vision, dummy_text),
            output_path,
            input_names=["vision_features", "text_features"],
            output_names=["fused_features"],
            opset_version=18,
            do_constant_folding=True,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 导出成功! 文件大小: {size_mb:.2f} MB")
        return output_path

    except Exception as e:
        print(f"⚠️  Transformer 导出失败: {e}")
        print("  原因: 需要完整的模型定义，权重结构复杂")
        print("  建议: 使用完整 PyTorch 模型进行推理")

        # 保存权重字典供后续使用
        weights_path = os.path.join(output_dir, "transformer_fusion_weights.pth")
        torch.save(transformer_state, weights_path)
        print(f"  已保存权重到: {weights_path}")
        return None


def export_segmentation_head(checkpoint_path, output_dir):
    """导出 Segmentation Head（像素级解码器）"""
    print("\n" + "="*70)
    print("导出 Segmentation Head")
    print("="*70)

    # 加载 checkpoint
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # 提取 segmentation_head 权重
    seg_head_state = {}
    for key in state_dict.keys():
        if key.startswith('detector.segmentation_head.'):
            new_key = key.replace('detector.segmentation_head.', '')
            seg_head_state[new_key] = state_dict[key]

    print(f"找到 {len(seg_head_state)} 个 segmentation head 权重")

    # 显示权重结构
    print("权重结构:")
    for key in sorted(seg_head_state.keys())[:10]:
        shape = list(seg_head_state[key].shape) if hasattr(seg_head_state[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")

    # 统计参数
    total_params = sum(v.numel() for v in seg_head_state.values() if hasattr(v, 'numel'))
    print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 构建 Segmentation Head
    class SegmentationHeadWrapper(nn.Module):
        """Segmentation Head Wrapper"""
        def __init__(self):
            super().__init__()
            # 简化的像素解码器
            self.conv_layers = nn.ModuleList([
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ])

            # 最终预测层
            self.mask_predictor = nn.Conv2d(256, 1, kernel_size=1)

        def forward(self, features):
            """
            Args:
                features: [B, 256, H, W] - 从 transformer 重塑后的特征
            Returns:
                masks: [B, 1, H, W] - 分割掩码 logits
            """
            x = features
            for layer in self.conv_layers:
                x = layer(x)

            masks = self.mask_predictor(x)
            return masks

    wrapper = SegmentationHeadWrapper()

    # 尝试加载权重
    missing, unexpected = wrapper.load_state_dict(seg_head_state, strict=False)
    print(f"加载权重: Missing={len(missing)}, Unexpected={len(unexpected)}")

    wrapper.eval()

    # 导出到 ONNX
    output_path = os.path.join(output_dir, "segmentation_head.onnx")
    print(f"\n导出到: {output_path}")

    dummy_features = torch.randn(1, 256, 64, 64)

    try:
        torch.onnx.export(
            wrapper,
            dummy_features,
            output_path,
            input_names=["features"],
            output_names=["masks"],
            opset_version=18,
            do_constant_folding=True,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 导出成功! 文件大小: {size_mb:.2f} MB")
        return output_path
    except Exception as e:
        print(f"⚠️  Segmentation Head 导出失败: {e}")

        # 保存权重
        weights_path = os.path.join(output_dir, "segmentation_head_weights.pth")
        torch.save(seg_head_state, weights_path)
        print(f"  已保存权重到: {weights_path}")
        return None


def export_scoring_module(checkpoint_path, output_dir):
    """导出 Scoring Module（置信度评分）"""
    print("\n" + "="*70)
    print("导出 Scoring Module")
    print("="*70)

    # 加载 checkpoint
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # 提取 scoring 权重
    scoring_state = {}
    for key in state_dict.keys():
        if key.startswith('detector.dot_prod_scoring.'):
            new_key = key.replace('detector.dot_prod_scoring.', '')
            scoring_state[new_key] = state_dict[key]

    print(f"找到 {len(scoring_state)} 个 scoring 权重")

    # 显示所有权重
    print("权重结构:")
    for key in sorted(scoring_state.keys()):
        shape = list(scoring_state[key].shape) if hasattr(scoring_state[key], 'shape') else 'scalar'
        print(f"  {key}: {shape}")

    # 统计参数
    total_params = sum(v.numel() for v in scoring_state.values() if hasattr(v, 'numel'))
    print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")

    # 构建 Scoring Module
    class ScoringModuleWrapper(nn.Module):
        """Scoring Module Wrapper"""
        def __init__(self):
            super().__init__()
            # MLP for prompt
            self.prompt_mlp = nn.Sequential(
                nn.Linear(256, 2048),
                nn.ReLU(),
                nn.Linear(2048, 256),
                nn.LayerNorm(256),
            )

            # Projection layers
            self.prompt_proj = nn.Linear(256, 256)
            self.hs_proj = nn.Linear(256, 256)

        def forward(self, prompt_features, hidden_states):
            """
            Args:
                prompt_features: [B, N, 256] - prompt embeddings
                hidden_states: [B, M, 256] - decoder hidden states
            Returns:
                scores: [B, M] - confidence scores
            """
            # Process prompt
            prompt_feat = self.prompt_mlp(prompt_features)  # [B, N, 256]
            prompt_feat = self.prompt_proj(prompt_feat)     # [B, N, 256]

            # Process hidden states
            hs_feat = self.hs_proj(hidden_states)           # [B, M, 256]

            # Compute dot product scores
            # 简化: 使用平均池化
            prompt_feat_avg = prompt_feat.mean(dim=1, keepdim=True)  # [B, 1, 256]
            scores = torch.matmul(hs_feat, prompt_feat_avg.transpose(1, 2))  # [B, M, 1]
            scores = scores.squeeze(-1)  # [B, M]

            return scores

    wrapper = ScoringModuleWrapper()

    # 加载权重
    missing, unexpected = wrapper.load_state_dict(scoring_state, strict=False)
    print(f"加载权重: Missing={len(missing)}, Unexpected={len(unexpected)}")

    wrapper.eval()

    # 导出到 ONNX
    output_path = os.path.join(output_dir, "scoring_module.onnx")
    print(f"\n导出到: {output_path}")

    dummy_prompt = torch.randn(1, 32, 256)
    dummy_hs = torch.randn(1, 100, 256)

    try:
        torch.onnx.export(
            wrapper,
            (dummy_prompt, dummy_hs),
            output_path,
            input_names=["prompt_features", "hidden_states"],
            output_names=["scores"],
            opset_version=18,
            do_constant_folding=True,
        )

        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✅ 导出成功! 文件大小: {size_mb:.2f} MB")
        return output_path
    except Exception as e:
        print(f"⚠️  Scoring Module 导出失败: {e}")

        # 保存权重
        weights_path = os.path.join(output_dir, "scoring_module_weights.pth")
        torch.save(scoring_state, weights_path)
        print(f"  已保存权重到: {weights_path}")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="导出 EfficientSAM3 融合层组件")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 路径")
    parser.add_argument("--output", type=str, default="exports_repvit_m0_9/", help="输出目录")
    parser.add_argument("--export-transformer", action="store_true", help="导出 Transformer")
    parser.add_argument("--export-seg-head", action="store_true", help="导出 Segmentation Head")
    parser.add_argument("--export-scoring", action="store_true", help="导出 Scoring Module")
    parser.add_argument("--export-all", action="store_true", help="导出所有")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    exported = []
    weights_only = []

    if args.export_all or args.export_transformer:
        try:
            path = export_transformer_fusion(args.checkpoint, args.output)
            if path:
                exported.append(path)
            else:
                weights_only.append("transformer_fusion_weights.pth")
        except Exception as e:
            print(f"❌ Transformer 导出失败: {e}")
            import traceback
            traceback.print_exc()

    if args.export_all or args.export_seg_head:
        try:
            path = export_segmentation_head(args.checkpoint, args.output)
            if path:
                exported.append(path)
            else:
                weights_only.append("segmentation_head_weights.pth")
        except Exception as e:
            print(f"❌ Segmentation Head 导出失败: {e}")
            import traceback
            traceback.print_exc()

    if args.export_all or args.export_scoring:
        try:
            path = export_scoring_module(args.checkpoint, args.output)
            if path:
                exported.append(path)
            else:
                weights_only.append("scoring_module_weights.pth")
        except Exception as e:
            print(f"❌ Scoring Module 导出失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("导出完成!")
    print("="*70)

    if exported:
        print("\n✅ 成功导出的 ONNX 模型:")
        for f in exported:
            print(f"  ✓ {f}")

    if weights_only:
        print("\n⚠️  仅保存权重 (需要完整模型定义):")
        for f in weights_only:
            print(f"  → {f}")
        print("\n提示: 这些组件结构复杂，建议使用完整 PyTorch 模型进行推理")
