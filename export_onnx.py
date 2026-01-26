#!/usr/bin/env python3
"""
Export EfficientSAM3 Stage1 models to ONNX format.

Usage:
    # Export image encoder only
    python export_onnx.py \
        --checkpoint checkpoints/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \
        --backbone repvit_m1_1 \
        --output exports/

    # Export with specific image size
    python export_onnx.py \
        --checkpoint checkpoints/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \
        --backbone repvit_m1_1 \
        --img-size 1024 \
        --output exports/

    # Export and simplify
    python export_onnx.py \
        --checkpoint checkpoints/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \
        --backbone repvit_m1_1 \
        --simplify \
        --output exports/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn

# Add sam3 to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))


def get_backbone_config(backbone_name: str, img_size: int = 1024):
    """Get configuration for different backbone types."""
    configs = {
        # RepViT variants
        "repvit_m0_9": {"embed_dim": 256, "embed_size": 64},
        "repvit_m1_1": {"embed_dim": 256, "embed_size": 64},
        "repvit_m2_3": {"embed_dim": 256, "embed_size": 64},
        # TinyViT variants
        "tiny_vit_5m": {"embed_dim": 256, "embed_size": 64},
        "tiny_vit_11m": {"embed_dim": 256, "embed_size": 64},
        "tiny_vit_21m": {"embed_dim": 256, "embed_size": 64},
        # EfficientViT variants
        "efficientvit_b0": {"embed_dim": 256, "embed_size": 64},
        "efficientvit_b1": {"embed_dim": 256, "embed_size": 64},
        "efficientvit_b2": {"embed_dim": 256, "embed_size": 64},
    }

    if backbone_name not in configs:
        raise ValueError(f"Unknown backbone: {backbone_name}. Available: {list(configs.keys())}")

    config = configs[backbone_name]
    config["img_size"] = img_size
    config["backbone"] = backbone_name
    return config


def build_image_encoder(backbone_name: str, img_size: int = 1024):
    """Build image encoder model."""
    from sam3.backbones.repvit import (
        _make_divisible,
        repvit_m0_9,
        repvit_m1_1,
        repvit_m2_3,
    )
    from sam3.backbones.tiny_vit import (
        tiny_vit_5m_224,
        tiny_vit_11m_224,
        tiny_vit_21m_224,
    )
    from sam3.backbones.efficientvit import (
        efficientvit_backbone_b0,
        efficientvit_backbone_b1,
        efficientvit_backbone_b2,
    )

    config = get_backbone_config(backbone_name, img_size)

    # Build backbone
    if backbone_name.startswith("repvit"):
        fn = {
            "repvit_m0_9": repvit_m0_9,
            "repvit_m1_1": repvit_m1_1,
            "repvit_m2_3": repvit_m2_3,
        }[backbone_name]
        backbone_model = fn(pretrained=False, num_classes=0, distillation=False)
        out_channels = _make_divisible(backbone_model.cfgs[-1][2], 8)

        class RepViTAdapter(nn.Module):
            def __init__(self, model, out_channels):
                super().__init__()
                self.model = model
                self.out_channels = out_channels

            def forward(self, x):
                for layer in self.model.features:
                    x = layer(x)
                return x

        backbone = RepViTAdapter(backbone_model, out_channels)

    elif backbone_name.startswith("tiny_vit"):
        fn = {
            "tiny_vit_5m": tiny_vit_5m_224,
            "tiny_vit_11m": tiny_vit_11m_224,
            "tiny_vit_21m": tiny_vit_21m_224,
        }[backbone_name]
        backbone_model = fn(pretrained=False, img_size=img_size)

        class TinyViTAdapter(nn.Module):
            def __init__(self, model, img_size):
                super().__init__()
                self.model = model
                self.model.head = nn.Identity()
                self.final_hw = self._compute_resolution(img_size)
                self.out_channels = self.model.norm_head.normalized_shape[0]
                self.model.norm_head = nn.Identity()

            def forward(self, x):
                x = self.model.patch_embed(x)
                x = self.model.layers[0](x)
                for i in range(1, len(self.model.layers)):
                    x = self.model.layers[i](x)
                B, N, C = x.shape
                H, W = self.final_hw
                x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                return x

            def _compute_resolution(self, img_size):
                H, W = self.model.patches_resolution
                for _ in range(self.model.num_layers - 1):
                    H = (H - 1) // 2 + 1
                    W = (W - 1) // 2 + 1
                return (H, W)

        backbone = TinyViTAdapter(backbone_model, img_size)
        out_channels = backbone.out_channels

    elif backbone_name.startswith("efficientvit"):
        fn = {
            "efficientvit_b0": efficientvit_backbone_b0,
            "efficientvit_b1": efficientvit_backbone_b1,
            "efficientvit_b2": efficientvit_backbone_b2,
        }[backbone_name]
        backbone_model = fn()

        class EfficientViTAdapter(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.out_channels = self.model.width_list[-1]

            def forward(self, x):
                out = self.model(x)
                return out["stage_final"]

        backbone = EfficientViTAdapter(backbone_model)
        out_channels = backbone.out_channels
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Build full encoder with head
    class ImageEncoder(nn.Module):
        def __init__(self, backbone, in_channels, embed_dim=256, embed_size=64, img_size=1024):
            super().__init__()
            self.backbone = backbone
            self.embed_size = embed_size
            self.img_size = img_size
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )

        def forward(self, x):
            feats = self.backbone(x)
            feats = self.head(feats)
            if feats.shape[-1] != self.embed_size or feats.shape[-2] != self.embed_size:
                feats = nn.functional.interpolate(
                    feats,
                    size=(self.embed_size, self.embed_size),
                    mode="bilinear",
                    align_corners=False,
                )
            return feats

    encoder = ImageEncoder(
        backbone=backbone,
        in_channels=out_channels,
        embed_dim=config["embed_dim"],
        embed_size=config["embed_size"],
        img_size=img_size,
    )

    return encoder, config


def load_checkpoint(model: nn.Module, checkpoint_path: str, backbone_name: str):
    """Load checkpoint weights into model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in state_dict:
        state_dict = state_dict["model"]

    # Try to find image encoder weights
    encoder_state = {}

    # Pattern 1: Direct student encoder weights
    for key, value in state_dict.items():
        if key.startswith("backbone.") or key.startswith("head."):
            encoder_state[key] = value

    # Pattern 2: Nested in detector.backbone.vision_backbone
    if not encoder_state:
        prefix = "detector.backbone.vision_backbone.student_encoder."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                encoder_state[new_key] = value

    # Pattern 3: Try loading directly
    if not encoder_state:
        encoder_state = state_dict

    # Load with flexible matching
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)

    if missing:
        print(f"  Missing keys: {len(missing)}")
        if len(missing) <= 10:
            for k in missing:
                print(f"    - {k}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")
        if len(unexpected) <= 10:
            for k in unexpected:
                print(f"    - {k}")

    print("  Checkpoint loaded successfully")
    return model


def export_onnx(
    model: nn.Module,
    output_path: str,
    img_size: int = 1024,
    opset_version: int = 11,
    simplify: bool = False,
    dynamic_batch: bool = False,
):
    """Export model to ONNX format."""
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size)

    # Dynamic axes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        }

    print(f"Exporting to ONNX: {output_path}")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Opset version: {opset_version}")

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["image"],
        output_names=["embedding"],
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    print(f"  Exported successfully")

    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify

            print("  Simplifying ONNX model...")
            model_onnx = onnx.load(output_path)
            model_simp, check = onnx_simplify(model_onnx)

            if check:
                onnx.save(model_simp, output_path)
                print("  Simplified successfully")
            else:
                print("  Warning: Simplification check failed, keeping original")
        except ImportError:
            print("  Warning: onnxsim not installed, skipping simplification")
            print("  Install with: pip install onnxsim")

    # Print model info
    try:
        import onnx
        model_onnx = onnx.load(output_path)

        # Get operators
        ops = set([node.op_type for node in model_onnx.graph.node])
        print(f"\n  Operators used ({len(ops)}):")
        for op in sorted(ops):
            print(f"    - {op}")

        # Check for potentially unsupported ops
        potential_issues = {"LayerNormalization", "Gelu", "Attention", "ScaledDotProductAttention"}
        issues = ops & potential_issues
        if issues:
            print(f"\n  ⚠️  Potentially NPU-unfriendly ops: {issues}")
            print("     Consider replacing these for NPU deployment")

        # File size
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n  Output file size: {file_size:.2f} MB")

    except ImportError:
        pass

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Export EfficientSAM3 Stage1 models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export RepViT-M1.1 image encoder
  python export_onnx.py \\
      --checkpoint checkpoints/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \\
      --backbone repvit_m1_1 \\
      --output exports/

  # Export with simplification
  python export_onnx.py \\
      --checkpoint checkpoints/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \\
      --backbone repvit_m1_1 \\
      --simplify \\
      --output exports/

Available backbones:
  RepViT:      repvit_m0_9, repvit_m1_1, repvit_m2_3
  TinyViT:     tiny_vit_5m, tiny_vit_11m, tiny_vit_21m
  EfficientViT: efficientvit_b0, efficientvit_b1, efficientvit_b2
        """
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to EfficientSAM3 checkpoint (.pth/.pt)",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=[
            "repvit_m0_9", "repvit_m1_1", "repvit_m2_3",
            "tiny_vit_5m", "tiny_vit_11m", "tiny_vit_21m",
            "efficientvit_b0", "efficientvit_b1", "efficientvit_b2",
        ],
        help="Backbone architecture",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1024,
        help="Input image size (default: 1024)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/",
        help="Output directory or file path",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset version (default: 11)",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify ONNX model using onnxsim",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Enable dynamic batch size",
    )

    args = parser.parse_args()

    # Determine output path
    output_path = args.output
    if os.path.isdir(output_path) or output_path.endswith("/"):
        os.makedirs(output_path, exist_ok=True)
        output_path = os.path.join(
            output_path,
            f"efficientsam3_{args.backbone}_img{args.img_size}.onnx"
        )

    # Build model
    print(f"\nBuilding image encoder: {args.backbone}")
    model, config = build_image_encoder(args.backbone, args.img_size)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, args.backbone)

    # Export
    export_onnx(
        model=model,
        output_path=output_path,
        img_size=args.img_size,
        opset_version=args.opset,
        simplify=args.simplify,
        dynamic_batch=args.dynamic_batch,
    )

    print(f"\nDone! ONNX model saved to: {output_path}")


if __name__ == "__main__":
    main()
