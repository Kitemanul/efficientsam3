#!/usr/bin/env python3
"""
Export SAM3 Prompt Encoder and Mask Decoder from checkpoint on CPU.
Extracts from tracker.sam_prompt_encoder and tracker.sam_mask_decoder.
"""

import torch
import torch.nn as nn
import os
import sys

# Add SAM3 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))

def export_prompt_encoder(checkpoint_path, output_dir):
    """Export prompt encoder to ONNX (extracts from tracker)."""
    print("="*70)
    print("导出 Prompt Encoder (从 tracker)")
    print("="*70)

    # Load checkpoint
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Build prompt encoder from SAM
    from sam3.sam.prompt_encoder import PromptEncoder

    # SAM3 uses 1024x1024 images with 16x downsampling = 64x64 embeddings
    image_size = 1024
    embed_dim = 256
    image_embedding_size = 64

    prompt_encoder = PromptEncoder(
        embed_dim=embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16
    )

    # Load weights from tracker.sam_prompt_encoder
    prompt_encoder_state = {}
    for key in state_dict.keys():
        if key.startswith('tracker.sam_prompt_encoder.'):
            new_key = key.replace('tracker.sam_prompt_encoder.', '')
            prompt_encoder_state[new_key] = state_dict[key]

    missing, unexpected = prompt_encoder.load_state_dict(prompt_encoder_state, strict=False)
    print(f"加载权重: Missing={len(missing)}, Unexpected={len(unexpected)}")

    # Count params
    num_params = sum(p.numel() for p in prompt_encoder.parameters())
    print(f"参数量: {num_params:,} ({num_params/1e6:.2f}M)")

    prompt_encoder.eval()

    # Create ONNX-friendly wrapper
    class PromptEncoderWrapper(nn.Module):
        def __init__(self, pe):
            super().__init__()
            self.pe = pe

        def forward(self, point_coords, point_labels):
            """
            Args:
                point_coords: [B, N, 2] - normalized coordinates [0, 1]
                point_labels: [B, N] - labels (1=foreground, 0=background)
            Returns:
                sparse_embeddings: [B, N+1, 256] - point embeddings + 1 background token
                dense_embeddings: [B, 256, 64, 64] - dense position encoding
            """
            # Prepare points in SAM format
            points = (point_coords, point_labels)

            # Get embeddings
            sparse_embeddings, dense_embeddings = self.pe(
                points=points,
                boxes=None,
                masks=None,
            )

            return sparse_embeddings, dense_embeddings

    wrapper = PromptEncoderWrapper(prompt_encoder)
    wrapper.eval()

    # Export to ONNX
    output_path = os.path.join(output_dir, "prompt_encoder.onnx")
    print(f"导出到: {output_path}")

    # Dummy inputs
    dummy_coords = torch.rand(1, 2, 2)  # 2 points
    dummy_labels = torch.tensor([[1, 0]], dtype=torch.long)  # 1 positive, 1 negative

    torch.onnx.export(
        wrapper,
        (dummy_coords, dummy_labels),
        output_path,
        input_names=["point_coords", "point_labels"],
        output_names=["sparse_embeddings", "dense_embeddings"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={
            "point_coords": {1: "num_points"},
            "point_labels": {1: "num_points"},
            "sparse_embeddings": {1: "num_points_plus_one"},
        }
    )

    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ 导出成功! 文件大小: {size_mb:.2f} MB")

    return output_path


def export_mask_decoder(checkpoint_path, output_dir):
    """Export mask decoder to ONNX (extracts from tracker)."""
    print("\n" + "="*70)
    print("导出 Mask Decoder (从 tracker)")
    print("="*70)

    # Load checkpoint
    print(f"加载 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # Build mask decoder from SAM
    from sam3.sam.transformer import TwoWayTransformer
    from sam3.sam.mask_decoder import MaskDecoder

    # SAM3 configuration
    transformer_dim = 256

    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=transformer_dim,
        num_heads=8,
        mlp_dim=2048,
    )

    mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
        transformer=transformer,
        transformer_dim=transformer_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    # Load weights from tracker.sam_mask_decoder
    mask_decoder_state = {}
    for key in state_dict.keys():
        if key.startswith('tracker.sam_mask_decoder.'):
            new_key = key.replace('tracker.sam_mask_decoder.', '')
            mask_decoder_state[new_key] = state_dict[key]

    missing, unexpected = mask_decoder.load_state_dict(mask_decoder_state, strict=False)
    print(f"加载权重: Missing={len(missing)}, Unexpected={len(unexpected)}")

    # Count params
    num_params = sum(p.numel() for p in mask_decoder.parameters())
    print(f"参数量: {num_params:,} ({num_params/1e6:.2f}M)")

    mask_decoder.eval()

    # Create ONNX-friendly wrapper
    class MaskDecoderWrapper(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, image_embeddings, sparse_embeddings, dense_embeddings):
            """
            Args:
                image_embeddings: [B, 256, 64, 64] - from image encoder
                sparse_embeddings: [B, N, 256] - from prompt encoder
                dense_embeddings: [B, 256, 64, 64] - from prompt encoder
            Returns:
                masks: [B, 4, 256, 256] - 1 single mask + 3 multi-masks
                iou_predictions: [B, 4] - quality scores
            """
            # Decoder returns: masks, iou_pred, sam_tokens_out, object_score_logits
            low_res_masks, iou_predictions, sam_tokens, obj_scores = self.decoder(
                image_embeddings=image_embeddings,
                image_pe=dense_embeddings,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,  # For single image inference
            )

            # Upsample masks to 256x256
            masks = torch.nn.functional.interpolate(
                low_res_masks,
                size=(256, 256),
                mode='bilinear',
                align_corners=False
            )

            return masks, iou_predictions

    wrapper = MaskDecoderWrapper(mask_decoder)
    wrapper.eval()

    # Export to ONNX
    output_path = os.path.join(output_dir, "mask_decoder.onnx")
    print(f"导出到: {output_path}")

    # Dummy inputs
    dummy_image_embeddings = torch.randn(1, 256, 64, 64)
    dummy_sparse_embeddings = torch.randn(1, 3, 256)  # 2 points + 1 background
    dummy_dense_embeddings = torch.randn(1, 256, 64, 64)

    torch.onnx.export(
        wrapper,
        (dummy_image_embeddings, dummy_sparse_embeddings, dummy_dense_embeddings),
        output_path,
        input_names=["image_embeddings", "sparse_embeddings", "dense_embeddings"],
        output_names=["masks", "iou_predictions"],
        opset_version=18,  # Use opset 18 to avoid conversion issues
        do_constant_folding=True,
        # Remove dynamic_axes due to conflicts with torch.export
    )

    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✅ 导出成功! 文件大小: {size_mb:.2f} MB")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="导出 Prompt Encoder 和 Mask Decoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint 路径")
    parser.add_argument("--output", type=str, default="exports_repvit_m0_9/", help="输出目录")
    parser.add_argument("--export-prompt-encoder", action="store_true", help="导出 Prompt Encoder")
    parser.add_argument("--export-mask-decoder", action="store_true", help="导出 Mask Decoder")
    parser.add_argument("--export-all", action="store_true", help="导出所有")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    exported = []

    if args.export_all or args.export_prompt_encoder:
        try:
            path = export_prompt_encoder(args.checkpoint, args.output)
            exported.append(path)
        except Exception as e:
            print(f"❌ Prompt Encoder 导出失败: {e}")
            import traceback
            traceback.print_exc()

    if args.export_all or args.export_mask_decoder:
        try:
            path = export_mask_decoder(args.checkpoint, args.output)
            exported.append(path)
        except Exception as e:
            print(f"❌ Mask Decoder 导出失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("导出完成!")
    print("="*70)
    for f in exported:
        print(f"  ✓ {f}")
