#!/usr/bin/env python3
"""
Export EfficientSAM3 models to ONNX format.

Supports exporting:
- Image Encoder (RepViT, TinyViT, EfficientViT)
- Text Encoder (MobileCLIP S0, S1, MobileCLIP2-L)
- Mask Decoder (from SAM3)

Usage:
    # Download model first
    huggingface-cli download Simon7108528/EfficientSAM3 \
        stage1_all_converted/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \
        --local-dir checkpoints

    # Export all components
    python export_efficientsam3_onnx.py \
        --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \
        --image-backbone repvit_m1_1 \
        --text-backbone MobileCLIP-S1 \
        --output exports/ \
        --export-all

    # Export only image encoder
    python export_efficientsam3_onnx.py \
        --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m1_1_mobileclip_s1.pth \
        --image-backbone repvit_m1_1 \
        --output exports/ \
        --export-image-encoder
"""

from __future__ import annotations

import argparse
import os
import sys

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================================
# Image Encoder
# ============================================================================

def build_image_encoder(backbone_name: str, img_size: int = 1024):
    """Build image encoder with specified backbone."""
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

    # RepViT
    if backbone_name.startswith("repvit"):
        fn = {
            "repvit_m0_9": repvit_m0_9,
            "repvit_m1_1": repvit_m1_1,
            "repvit_m2_3": repvit_m2_3,
        }[backbone_name]
        backbone_model = fn(pretrained=False, num_classes=0, distillation=False)
        out_channels = _make_divisible(backbone_model.cfgs[-1][2], 8)

        class RepViTBackbone(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                for layer in self.model.features:
                    x = layer(x)
                return x

        backbone = RepViTBackbone(backbone_model)

    # TinyViT
    elif backbone_name.startswith("tiny_vit"):
        fn = {
            "tiny_vit_5m": tiny_vit_5m_224,
            "tiny_vit_11m": tiny_vit_11m_224,
            "tiny_vit_21m": tiny_vit_21m_224,
        }[backbone_name]
        backbone_model = fn(pretrained=False, img_size=img_size)

        class TinyViTBackbone(nn.Module):
            def __init__(self, model, img_size):
                super().__init__()
                self.model = model
                self.model.head = nn.Identity()
                self.final_hw = self._compute_resolution()
                self.out_channels = self.model.norm_head.normalized_shape[0]
                self.model.norm_head = nn.Identity()

            def forward(self, x):
                x = self.model.patch_embed(x)
                for layer in self.model.layers:
                    x = layer(x)
                B, N, C = x.shape
                H, W = self.final_hw
                x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                return x

            def _compute_resolution(self):
                H, W = self.model.patches_resolution
                for _ in range(self.model.num_layers - 1):
                    H = (H - 1) // 2 + 1
                    W = (W - 1) // 2 + 1
                return (H, W)

        backbone = TinyViTBackbone(backbone_model, img_size)
        out_channels = backbone.out_channels

    # EfficientViT
    elif backbone_name.startswith("efficientvit"):
        fn = {
            "efficientvit_b0": efficientvit_backbone_b0,
            "efficientvit_b1": efficientvit_backbone_b1,
            "efficientvit_b2": efficientvit_backbone_b2,
        }[backbone_name]
        backbone_model = fn()

        class EfficientViTBackbone(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
                self.out_channels = self.model.width_list[-1]

            def forward(self, x):
                out = self.model(x)
                return out["stage_final"]

        backbone = EfficientViTBackbone(backbone_model)
        out_channels = backbone.out_channels
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    # Full encoder with head
    class ImageEncoder(nn.Module):
        def __init__(self, backbone, in_channels, embed_dim=256, embed_size=64):
            super().__init__()
            self.backbone = backbone
            self.embed_size = embed_size
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
                feats = F.interpolate(
                    feats,
                    size=(self.embed_size, self.embed_size),
                    mode="bilinear",
                    align_corners=False,
                )
            return feats

    return ImageEncoder(backbone, out_channels), out_channels


# ============================================================================
# Text Encoder
# ============================================================================

def build_text_encoder(backbone_name: str, output_dim: int = 256):
    """Build text encoder with specified backbone."""

    # Config for different MobileCLIP variants
    configs = {
        "MobileCLIP-S0": {
            "dim": 512,
            "n_transformer_layers": 4,
            "n_heads_per_layer": 8,
            "model_name": "mct",
        },
        "MobileCLIP-S1": {
            "dim": 512,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 8,
            "model_name": "base",
        },
        "MobileCLIP2-L": {
            "dim": 768,
            "n_transformer_layers": 12,
            "n_heads_per_layer": 12,
            "model_name": "base",
        },
    }

    if backbone_name not in configs:
        raise ValueError(f"Unknown text backbone: {backbone_name}. Available: {list(configs.keys())}")

    cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "ffn_multiplier_per_layer": 4.0,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
        **configs[backbone_name]
    }

    from sam3.model.text_encoder_student import TextStudentEncoder

    model = TextStudentEncoder(
        cfg=cfg,
        context_length=32,
        output_dim=output_dim,
    )

    return model, cfg


# ============================================================================
# Mask Decoder Wrapper
# ============================================================================

class MaskDecoderWrapper(nn.Module):
    """Wrapper for SAM3 mask decoder for ONNX export."""

    def __init__(self, mask_decoder):
        super().__init__()
        self.decoder = mask_decoder

    def forward(self, image_embedding, point_coords, point_labels):
        """
        Args:
            image_embedding: [B, 256, 64, 64]
            point_coords: [B, N, 2] - point coordinates
            point_labels: [B, N] - point labels (1=positive, 0=negative)
        Returns:
            masks: [B, num_masks, H, W]
            scores: [B, num_masks]
        """
        # This is a simplified wrapper - actual implementation depends on SAM3 decoder
        pass


# ============================================================================
# Weight Loading
# ============================================================================

def load_weights(checkpoint_path: str):
    """Load checkpoint and return state dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    if "model" in state_dict:
        state_dict = state_dict["model"]

    return state_dict


def load_image_encoder_weights(model: nn.Module, state_dict: dict):
    """Load image encoder weights from combined checkpoint."""
    encoder_state = {}

    # Try different prefixes
    prefixes = [
        "detector.backbone.vision_backbone.student_encoder.",
        "backbone.vision_backbone.student_encoder.",
        "image_encoder.",
        "student_encoder.",
        "",
    ]

    for prefix in prefixes:
        for key, value in state_dict.items():
            if key.startswith(prefix) and ("backbone" in key or "head" in key):
                new_key = key[len(prefix):] if prefix else key
                # Clean up the key
                if new_key.startswith("backbone.") or new_key.startswith("head."):
                    encoder_state[new_key] = value

    if not encoder_state:
        # Try direct loading
        encoder_state = {k: v for k, v in state_dict.items()
                        if "backbone" in k or "head" in k}

    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(f"  Image encoder - Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    return model


def load_text_encoder_weights(model: nn.Module, state_dict: dict):
    """Load text encoder weights from combined checkpoint."""
    encoder_state = {}

    # Try different prefixes
    prefixes = [
        "detector.backbone.language_backbone.student_encoder.",
        "backbone.language_backbone.student_encoder.",
        "text_encoder.",
        "student_text_encoder.",
    ]

    for prefix in prefixes:
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                encoder_state[new_key] = value

    if encoder_state:
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"  Text encoder - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("  Warning: No text encoder weights found in checkpoint")

    return model


# ============================================================================
# ONNX Export
# ============================================================================

def export_to_onnx(
    model: nn.Module,
    output_path: str,
    dummy_input: tuple,
    input_names: list,
    output_names: list,
    opset_version: int = 11,
    dynamic_axes: dict = None,
):
    """Export model to ONNX."""
    model.eval()

    print(f"Exporting to: {output_path}")

    torch.onnx.export(
        model,
        dummy_input if len(dummy_input) > 1 else dummy_input[0],
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )

    # Analyze exported model
    try:
        import onnx
        onnx_model = onnx.load(output_path)

        ops = set([node.op_type for node in onnx_model.graph.node])
        print(f"  Operators: {sorted(ops)}")

        # Check for NPU-unfriendly ops
        issues = ops & {"LayerNormalization", "Gelu", "Attention"}
        if issues:
            print(f"  ⚠️  NPU-unfriendly ops: {issues}")

        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File size: {file_size:.2f} MB")

    except ImportError:
        pass

    return output_path


def export_image_encoder(
    checkpoint_path: str,
    backbone_name: str,
    output_dir: str,
    img_size: int = 1024,
    opset_version: int = 11,
):
    """Export image encoder to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting Image Encoder")
    print(f"{'='*60}")
    print(f"  Backbone: {backbone_name}")
    print(f"  Image size: {img_size}")

    # Build model
    model, _ = build_image_encoder(backbone_name, img_size)

    # Load weights
    state_dict = load_weights(checkpoint_path)
    model = load_image_encoder_weights(model, state_dict)

    # Count params
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")

    # Export
    output_path = os.path.join(output_dir, f"image_encoder_{backbone_name}.onnx")
    dummy_input = (torch.randn(1, 3, img_size, img_size),)

    export_to_onnx(
        model=model,
        output_path=output_path,
        dummy_input=dummy_input,
        input_names=["image"],
        output_names=["image_embedding"],
        opset_version=opset_version,
    )

    return output_path


def export_text_encoder(
    checkpoint_path: str,
    backbone_name: str,
    output_dir: str,
    opset_version: int = 11,
):
    """Export text encoder to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting Text Encoder")
    print(f"{'='*60}")
    print(f"  Backbone: {backbone_name}")

    # Build model
    model, cfg = build_text_encoder(backbone_name)

    # Load weights
    state_dict = load_weights(checkpoint_path)
    model = load_text_encoder_weights(model, state_dict)

    # Count params
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")

    # Create wrapper that accepts token IDs directly (skip tokenizer)
    class TextEncoderWrapper(nn.Module):
        """Wrapper that accepts pre-tokenized input."""
        def __init__(self, text_encoder):
            super().__init__()
            self.encoder = text_encoder.encoder  # MobileCLIPTextTransformer
            self.projector = text_encoder.projector

        def forward(self, input_ids):
            """
            Args:
                input_ids: [B, seq_len] - token IDs (int64)
            Returns:
                text_features: [B, seq_len, output_dim] - text embeddings
            """
            # Get embeddings from token IDs
            input_embeds = self.encoder.forward_embedding(input_ids)  # [B, Seq, Dim]

            # Pass through transformer
            text_memory = self.encoder(
                input_embeds,
                return_all_tokens=True,
                input_is_embeddings=True
            )  # [B, Seq, Dim]

            # Project to output dimension
            text_features = self.projector(text_memory)  # [B, Seq, OutputDim]

            return text_features

    wrapper = TextEncoderWrapper(model)
    wrapper.eval()

    # Export
    output_path = os.path.join(output_dir, f"text_encoder_{backbone_name.lower().replace('-', '_')}.onnx")

    # Text encoder input: token IDs [B, seq_len]
    seq_len = 32  # context_length used in training
    dummy_input = (torch.randint(0, 49408, (1, seq_len), dtype=torch.long),)

    print(f"Exporting to: {output_path}")
    export_to_onnx(
        model=wrapper,
        output_path=output_path,
        dummy_input=dummy_input,
        input_names=["input_ids"],
        output_names=["text_features"],
        opset_version=opset_version,
    )

    return output_path


# ============================================================================
# Mask Decoder
# ============================================================================

class MaskDecoderONNX(nn.Module):
    """Wrapper for SAM3 Mask Decoder for ONNX export."""

    def __init__(self, sam3_model):
        super().__init__()
        # Extract decoder components from SAM3 model
        self.decoder = sam3_model.decoder

    def forward(self, image_embedding, point_coords, point_labels):
        """
        Simplified forward for ONNX export.

        Args:
            image_embedding: [B, 256, 64, 64] - from image encoder
            point_coords: [B, N, 2] - point coordinates (normalized 0-1)
            point_labels: [B, N] - point labels (1=positive, 0=negative)

        Returns:
            masks: [B, 1, 256, 256] - predicted masks
            scores: [B, 1] - mask quality scores
        """
        # This is a simplified version - actual implementation depends on SAM3 decoder structure
        # The decoder typically takes image embeddings and prompt embeddings
        pass


def export_mask_decoder(
    checkpoint_path: str,
    output_dir: str,
    opset_version: int = 11,
):
    """Export SAM3 mask decoder to ONNX."""
    print(f"\n{'='*60}")
    print("Exporting Mask Decoder")
    print(f"{'='*60}")

    # Load full model to extract decoder
    print("  Loading SAM3 model...")

    try:
        from sam3.model_builder import build_sam3_image_model

        # Build SAM3 model and load weights
        sam3 = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            eval_mode=True,
            device="cpu",
        )

        # Extract mask decoder
        # SAM3 decoder structure may vary, try different paths
        decoder = None

        # Try to find decoder in model
        if hasattr(sam3, 'mask_decoder'):
            decoder = sam3.mask_decoder
        elif hasattr(sam3, 'decoder'):
            decoder = sam3.decoder
        elif hasattr(sam3, 'sam_mask_decoder'):
            decoder = sam3.sam_mask_decoder

        if decoder is None:
            print("  Warning: Could not find mask decoder in model")
            print("  Attempting to extract from detector...")

            if hasattr(sam3, 'detector') and hasattr(sam3.detector, 'mask_decoder'):
                decoder = sam3.detector.mask_decoder

        if decoder is None:
            raise RuntimeError("Could not locate mask decoder in checkpoint")

        decoder.eval()

        # Count params
        num_params = sum(p.numel() for p in decoder.parameters())
        print(f"  Parameters: {num_params / 1e6:.2f}M")

        # Create wrapper for ONNX export
        class DecoderWrapper(nn.Module):
            def __init__(self, decoder):
                super().__init__()
                self.decoder = decoder

            def forward(self, image_embedding, sparse_embedding, dense_embedding):
                """
                Args:
                    image_embedding: [B, 256, 64, 64]
                    sparse_embedding: [B, N, 256] - point/box prompt embeddings
                    dense_embedding: [B, 256, 64, 64] - dense prompt embedding
                """
                masks, scores = self.decoder(
                    image_embeddings=image_embedding,
                    sparse_prompt_embeddings=sparse_embedding,
                    dense_prompt_embeddings=dense_embedding,
                    multimask_output=True,
                )
                return masks, scores

        wrapper = DecoderWrapper(decoder)
        wrapper.eval()

        # Export
        output_path = os.path.join(output_dir, "mask_decoder.onnx")

        # Dummy inputs matching decoder interface
        batch_size = 1
        dummy_image_embedding = torch.randn(batch_size, 256, 64, 64)
        dummy_sparse_embedding = torch.randn(batch_size, 2, 256)  # 2 prompt tokens
        dummy_dense_embedding = torch.randn(batch_size, 256, 64, 64)

        dummy_input = (dummy_image_embedding, dummy_sparse_embedding, dummy_dense_embedding)

        export_to_onnx(
            model=wrapper,
            output_path=output_path,
            dummy_input=dummy_input,
            input_names=["image_embedding", "sparse_embedding", "dense_embedding"],
            output_names=["masks", "scores"],
            opset_version=opset_version,
        )

        return output_path

    except ImportError as e:
        print(f"  Error: Could not import SAM3 modules: {e}")
        print("  Make sure SAM3 is properly installed")
        return None
    except Exception as e:
        print(f"  Error exporting decoder: {e}")
        print("  The decoder export may require adjustments based on SAM3 version")
        return None


def export_prompt_encoder(
    checkpoint_path: str,
    output_dir: str,
    opset_version: int = 11,
):
    """Export SAM3 prompt encoder to ONNX (converts points/boxes to embeddings)."""
    print(f"\n{'='*60}")
    print("Exporting Prompt Encoder")
    print(f"{'='*60}")

    try:
        from sam3.model_builder import build_sam3_image_model

        sam3 = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
            eval_mode=True,
            device="cpu",
        )

        # Find prompt encoder
        prompt_encoder = None
        if hasattr(sam3, 'prompt_encoder'):
            prompt_encoder = sam3.prompt_encoder
        elif hasattr(sam3, 'detector') and hasattr(sam3.detector, 'prompt_encoder'):
            prompt_encoder = sam3.detector.prompt_encoder

        if prompt_encoder is None:
            print("  Warning: Could not find prompt encoder")
            return None

        prompt_encoder.eval()

        num_params = sum(p.numel() for p in prompt_encoder.parameters())
        print(f"  Parameters: {num_params / 1e6:.2f}M")

        # Wrapper for point prompts
        class PromptEncoderWrapper(nn.Module):
            def __init__(self, prompt_encoder):
                super().__init__()
                self.pe = prompt_encoder

            def forward(self, point_coords, point_labels):
                """
                Args:
                    point_coords: [B, N, 2] - normalized coordinates
                    point_labels: [B, N] - labels (1=pos, 0=neg)
                Returns:
                    sparse_embeddings: [B, N, 256]
                    dense_embeddings: [B, 256, 64, 64]
                """
                sparse, dense = self.pe(
                    points=(point_coords, point_labels),
                    boxes=None,
                    masks=None,
                )
                return sparse, dense

        wrapper = PromptEncoderWrapper(prompt_encoder)
        wrapper.eval()

        output_path = os.path.join(output_dir, "prompt_encoder.onnx")

        dummy_coords = torch.randn(1, 1, 2)
        dummy_labels = torch.ones(1, 1, dtype=torch.long)

        export_to_onnx(
            model=wrapper,
            output_path=output_path,
            dummy_input=(dummy_coords, dummy_labels),
            input_names=["point_coords", "point_labels"],
            output_names=["sparse_embeddings", "dense_embeddings"],
            opset_version=opset_version,
        )

        return output_path

    except Exception as e:
        print(f"  Error exporting prompt encoder: {e}")
        return None


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export EfficientSAM3 models to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export image encoder only
  python export_efficientsam3_onnx.py \\
      --checkpoint efficient_sam3_repvit-m1_1_mobileclip_s1.pth \\
      --image-backbone repvit_m1_1 \\
      --output exports/ \\
      --export-image-encoder

  # Export text encoder only
  python export_efficientsam3_onnx.py \\
      --checkpoint efficient_sam3_repvit-m1_1_mobileclip_s1.pth \\
      --text-backbone MobileCLIP-S1 \\
      --output exports/ \\
      --export-text-encoder

  # Export all (image encoder + text encoder + decoder)
  python export_efficientsam3_onnx.py \\
      --checkpoint efficient_sam3_repvit-m1_1_mobileclip_s1.pth \\
      --image-backbone repvit_m1_1 \\
      --text-backbone MobileCLIP-S1 \\
      --output exports/ \\
      --export-all

  # Export decoder only
  python export_efficientsam3_onnx.py \\
      --checkpoint efficient_sam3_repvit-m1_1_mobileclip_s1.pth \\
      --output exports/ \\
      --export-decoder

Available backbones:
  Image: repvit_m0_9, repvit_m1_1, repvit_m2_3,
         tiny_vit_5m, tiny_vit_11m, tiny_vit_21m,
         efficientvit_b0, efficientvit_b1, efficientvit_b2
  Text:  MobileCLIP-S0, MobileCLIP-S1, MobileCLIP2-L
        """
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to EfficientSAM3 checkpoint")
    parser.add_argument("--image-backbone", type=str, default="repvit_m1_1",
                        help="Image encoder backbone")
    parser.add_argument("--text-backbone", type=str, default="MobileCLIP-S1",
                        help="Text encoder backbone")
    parser.add_argument("--output", type=str, default="exports/",
                        help="Output directory")
    parser.add_argument("--img-size", type=int, default=1024,
                        help="Input image size")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset version")

    # Export options
    parser.add_argument("--export-image-encoder", action="store_true",
                        help="Export image encoder")
    parser.add_argument("--export-text-encoder", action="store_true",
                        help="Export text encoder")
    parser.add_argument("--export-decoder", action="store_true",
                        help="Export mask decoder")
    parser.add_argument("--export-prompt-encoder", action="store_true",
                        help="Export prompt encoder")
    parser.add_argument("--export-all", action="store_true",
                        help="Export all components (image enc, text enc, decoder)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Determine what to export
    export_image = args.export_image_encoder or args.export_all
    export_text = args.export_text_encoder or args.export_all
    export_decoder = args.export_decoder or args.export_all
    export_prompt = args.export_prompt_encoder

    if not any([export_image, export_text, export_decoder, export_prompt]):
        print("No export option specified.")
        print("Use: --export-image-encoder, --export-text-encoder, --export-decoder, --export-prompt-encoder, or --export-all")
        return

    exported_files = []

    # Export image encoder
    if export_image:
        path = export_image_encoder(
            checkpoint_path=args.checkpoint,
            backbone_name=args.image_backbone,
            output_dir=args.output,
            img_size=args.img_size,
            opset_version=args.opset,
        )
        if path:
            exported_files.append(path)

    # Export text encoder
    if export_text:
        path = export_text_encoder(
            checkpoint_path=args.checkpoint,
            backbone_name=args.text_backbone,
            output_dir=args.output,
            opset_version=args.opset,
        )
        if path:
            exported_files.append(path)

    # Export mask decoder
    if export_decoder:
        path = export_mask_decoder(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            opset_version=args.opset,
        )
        if path:
            exported_files.append(path)

    # Export prompt encoder
    if export_prompt:
        path = export_prompt_encoder(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            opset_version=args.opset,
        )
        if path:
            exported_files.append(path)

    # Summary
    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    for f in exported_files:
        print(f"  ✓ {f}")

    if not exported_files:
        print("  No files were exported successfully")
        return

    print(f"\nExported {len(exported_files)} file(s):")
    print(f"  - Image Encoder: For NPU (run optimize_onnx_for_npu.py first)")
    print(f"  - Text Encoder: For CPU (has Attention ops)")
    print(f"  - Mask Decoder: For CPU (has Attention ops)")
    print(f"\nNext steps for NPU deployment:")
    print(f"  1. Run optimize_onnx_for_npu.py on image_encoder to replace GELU")
    print(f"  2. Use your NPU compiler to compile the optimized image encoder")
    print(f"  3. Run text encoder and decoder on CPU")


if __name__ == "__main__":
    main()
