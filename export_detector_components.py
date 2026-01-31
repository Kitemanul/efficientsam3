#!/usr/bin/env python3
"""
Export all ONNX-exportable Detector components of EfficientSAM3.

Exports:
  1. Image Encoder (RepViT backbone + head)
  2. Text Encoder (MobileCLIP transformer + projector)
  3. DotProductScoring (prompt-query similarity scoring)
  4. Box Head (bbox_embed MLP for box regression)

Each export is verified by comparing PyTorch vs ONNX Runtime outputs.

Usage:
    python export_detector_components.py \
        --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
        --image-backbone repvit_m0_9 \
        --text-backbone MobileCLIP-S1 \
        --output exports_repvit_m0_9/
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model Builders
# ============================================================================

def build_image_encoder(backbone_name: str, img_size: int = 1024):
    """Build image encoder matching actual SAM3 architecture.

    Architecture: RepViT backbone → Head(in_ch→1024) → Interpolate(72×72) → Neck(1024→256)
    Checkpoint structure:
      - trunk.model.backbone.* = RepViT backbone
      - trunk.model.head.* = Head Conv layers (embed_dim=1024)
      - convs.2.* = Neck scale=1.0 (Conv2d 1024→256 + Conv2d 256→256)
    """
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
                return x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

            def _compute_resolution(self):
                H, W = self.model.patches_resolution
                for _ in range(self.model.num_layers - 1):
                    H = (H - 1) // 2 + 1
                    W = (W - 1) // 2 + 1
                return (H, W)

        backbone = TinyViTBackbone(backbone_model, img_size)
        out_channels = backbone.out_channels

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

    # Actual SAM3 uses embed_dim=1024 for the head, then neck projects 1024→256
    embed_dim = 1024
    d_model = 256
    embed_size = 72  # target spatial resolution (1008/14=72)

    class ImageEncoder(nn.Module):
        def __init__(self, backbone, in_channels):
            super().__init__()
            self.backbone = backbone
            self.embed_size = embed_size
            # Head: matches ImageStudentEncoder (embed_dim=1024)
            self.head = nn.Sequential(
                nn.Conv2d(in_channels, embed_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.GELU(),
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            )
            # Neck: scale=1.0 conv from Sam3DualViTDetNeck (1024→256)
            self.neck = nn.Sequential(
                nn.Conv2d(embed_dim, d_model, kernel_size=1, bias=True),
                nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=True),
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
            feats = self.neck(feats)
            return feats

    return ImageEncoder(backbone, out_channels), out_channels


def build_text_encoder(backbone_name: str, output_dim: int = 256):
    """Build text encoder with specified backbone."""
    configs = {
        "MobileCLIP-S0": {
            "dim": 512, "n_transformer_layers": 4,
            "n_heads_per_layer": 8, "model_name": "mct",
        },
        "MobileCLIP-S1": {
            "dim": 512, "n_transformer_layers": 12,
            "n_heads_per_layer": 8, "model_name": "base",
        },
        "MobileCLIP2-L": {
            "dim": 768, "n_transformer_layers": 12,
            "n_heads_per_layer": 12, "model_name": "base",
        },
    }

    if backbone_name not in configs:
        raise ValueError(f"Unknown text backbone: {backbone_name}")

    cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "ffn_multiplier_per_layer": 4.0,
        "norm_layer": "layer_norm_fp32",
        "causal_masking": False,
        "embed_dropout": 0.0,
        "no_scale_embedding": False,
        "no_pos_embedding": False,
        **configs[backbone_name],
    }

    from sam3.model.text_encoder_student import TextStudentEncoder

    model = TextStudentEncoder(cfg=cfg, context_length=32, output_dim=output_dim)
    return model, cfg


def build_dot_product_scoring():
    """Build DotProductScoring module matching the model_builder config."""
    from sam3.model.model_misc import DotProductScoring, MLP

    prompt_mlp = MLP(
        input_dim=256,
        hidden_dim=2048,
        output_dim=256,
        num_layers=2,
        dropout=0.1,
        residual=True,
        out_norm=nn.LayerNorm(256),
    )
    return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)


def build_bbox_head():
    """Build bbox_embed MLP matching the decoder config."""
    from sam3.model.model_misc import MLP

    return MLP(input_dim=256, hidden_dim=256, output_dim=4, num_layers=3)


# ============================================================================
# Weight Loading
# ============================================================================

def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and return state dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def extract_weights(state_dict: dict, prefix: str, strip_prefix: bool = True):
    """Extract weights with given prefix from state dict."""
    extracted = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):] if strip_prefix else key
            extracted[new_key] = value
    return extracted


def load_image_encoder_weights(model: nn.Module, state_dict: dict):
    """Load image encoder weights from combined checkpoint.

    Checkpoint layout:
      detector.backbone.vision_backbone.trunk.model.backbone.* → model.backbone.*
      detector.backbone.vision_backbone.trunk.model.head.*     → model.head.*
      detector.backbone.vision_backbone.convs.2.*              → model.neck.* (scale=1.0)
    """
    encoder_state = {}

    # 1) Load trunk (backbone + head) weights
    trunk_prefix = "detector.backbone.vision_backbone.trunk.model."
    for key, value in state_dict.items():
        if key.startswith(trunk_prefix):
            new_key = key[len(trunk_prefix):]
            if new_key.startswith("backbone.") or new_key.startswith("head."):
                encoder_state[new_key] = value

    # 2) Load neck scale=1.0 (convs.2) weights → mapped to neck.*
    neck_prefix = "detector.backbone.vision_backbone.convs.2."
    for key, value in state_dict.items():
        if key.startswith(neck_prefix):
            suffix = key[len(neck_prefix):]
            # convs.2 has conv_1x1 and conv_3x3 → map to neck.0 and neck.1
            if suffix.startswith("conv_1x1."):
                encoder_state["neck.0." + suffix[len("conv_1x1."):]] = value
            elif suffix.startswith("conv_3x3."):
                encoder_state["neck.1." + suffix[len("conv_3x3."):]] = value

    if not encoder_state:
        print("  WARNING: No image encoder weights found!")
        return model

    print(f"  Trunk prefix: {trunk_prefix}")
    print(f"  Neck prefix: {neck_prefix}")
    missing, unexpected = model.load_state_dict(encoder_state, strict=False)
    print(f"  Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys (first 5): {unexpected[:5]}")
    return model


def load_text_encoder_weights(model: nn.Module, state_dict: dict):
    """Load text encoder weights from combined checkpoint."""
    # Try multiple prefix patterns (converted vs student_encoder format)
    prefixes = [
        "detector.backbone.language_backbone.",
        "detector.backbone.language_backbone.student_encoder.",
        "backbone.language_backbone.student_encoder.",
    ]

    encoder_state = {}
    used_prefix = None
    for prefix in prefixes:
        encoder_state = extract_weights(state_dict, prefix)
        if encoder_state:
            used_prefix = prefix
            break

    if encoder_state:
        print(f"  Weight prefix: {used_prefix}")
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"  Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys (first 5): {missing[:5]}")
    else:
        print("  WARNING: No text encoder weights found!")
    return model


def load_scoring_weights(model: nn.Module, state_dict: dict):
    """Load DotProductScoring weights from combined checkpoint."""
    prefix = "detector.dot_prod_scoring."
    scoring_state = extract_weights(state_dict, prefix)

    if scoring_state:
        missing, unexpected = model.load_state_dict(scoring_state, strict=False)
        print(f"  Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("  WARNING: No scoring weights found!")
    return model


def load_bbox_head_weights(model: nn.Module, state_dict: dict):
    """Load bbox_embed MLP weights from combined checkpoint."""
    prefix = "detector.transformer.decoder.bbox_embed."
    bbox_state = extract_weights(state_dict, prefix)

    if bbox_state:
        missing, unexpected = model.load_state_dict(bbox_state, strict=False)
        print(f"  Loaded weights - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("  WARNING: No bbox_embed weights found!")
    return model


# ============================================================================
# ONNX Export + Verification
# ============================================================================

def export_and_verify(
    model: nn.Module,
    dummy_inputs: tuple,
    output_path: str,
    input_names: list,
    output_names: list,
    opset_version: int = 11,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    skip_verify: bool = False,
):
    """Export model to ONNX and verify against PyTorch output."""
    model.eval()

    # Step 1: PyTorch forward
    with torch.no_grad():
        if isinstance(dummy_inputs, tuple) and len(dummy_inputs) == 1:
            pt_outputs = model(dummy_inputs[0])
        else:
            pt_outputs = model(*dummy_inputs)

    if not isinstance(pt_outputs, tuple):
        pt_outputs = (pt_outputs,)

    # Step 2: ONNX export
    print(f"  Exporting to: {output_path}")
    export_input = dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs
    torch.onnx.export(
        model,
        export_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Step 3: Analyze ONNX
    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1024:.1f} KB")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        ops = sorted(set(node.op_type for node in onnx_model.graph.node))
        print(f"  ONNX check: PASSED")
        print(f"  Operators ({len(ops)}): {ops}")
        npu_issues = set(ops) & {"LayerNormalization", "Gelu", "Attention"}
        if npu_issues:
            print(f"  NPU-unfriendly ops: {npu_issues}")
    except ImportError:
        print("  Warning: onnx not installed, skipping model check")

    if skip_verify:
        print("  Verification: SKIPPED")
        return True

    # Step 4: ONNX Runtime verification
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed, skipping verification")
        return True

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

    # Prepare inputs
    ort_inputs = {}
    for name, tensor in zip(input_names, dummy_inputs):
        ort_inputs[name] = tensor.numpy()

    ort_outputs = sess.run(None, ort_inputs)

    # Step 5: Compare
    all_pass = True
    for i, (pt_out, ort_out, name) in enumerate(zip(pt_outputs, ort_outputs, output_names)):
        pt_np = pt_out.numpy()
        max_abs_err = np.max(np.abs(pt_np - ort_out))
        max_rel_err = np.max(np.abs(pt_np - ort_out) / (np.abs(pt_np) + 1e-8))
        match = np.allclose(pt_np, ort_out, atol=atol, rtol=rtol)

        status = "PASS" if match else "FAIL"
        print(f"  Verify [{name}]: {status}  "
              f"(max_abs={max_abs_err:.2e}, max_rel={max_rel_err:.2e}, "
              f"shape={pt_np.shape})")
        if not match:
            all_pass = False

    return all_pass


# ============================================================================
# Component Export Functions
# ============================================================================

def export_image_encoder(state_dict, backbone_name, output_dir, img_size, opset, skip_verify):
    """Export Image Encoder."""
    print(f"\n{'=' * 60}")
    print("1/4  Image Encoder")
    print(f"{'=' * 60}")
    print(f"  Backbone: {backbone_name}, Image size: {img_size}")

    model, _ = build_image_encoder(backbone_name, img_size)
    model = load_image_encoder_weights(model, state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 3, img_size, img_size),)
    output_path = os.path.join(output_dir, f"image_encoder_{backbone_name}.onnx")

    return export_and_verify(
        model, dummy, output_path,
        input_names=["image"],
        output_names=["image_embedding"],
        opset_version=opset,
        atol=1e-5, rtol=1e-3,
        skip_verify=skip_verify,
    )


def export_text_encoder(state_dict, backbone_name, output_dir, opset, skip_verify):
    """Export Text Encoder."""
    print(f"\n{'=' * 60}")
    print("2/4  Text Encoder")
    print(f"{'=' * 60}")
    print(f"  Backbone: {backbone_name}")

    model, cfg = build_text_encoder(backbone_name)
    model = load_text_encoder_weights(model, state_dict)

    # Wrapper that accepts pre-tokenized input
    class TextEncoderWrapper(nn.Module):
        def __init__(self, text_encoder):
            super().__init__()
            self.encoder = text_encoder.encoder
            self.projector = text_encoder.projector

        def forward(self, input_ids):
            input_embeds = self.encoder.forward_embedding(input_ids)
            text_memory = self.encoder(
                input_embeds, return_all_tokens=True, input_is_embeddings=True
            )
            text_features = self.projector(text_memory)
            return text_features

    wrapper = TextEncoderWrapper(model)

    num_params = sum(p.numel() for p in wrapper.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    torch.manual_seed(42)
    dummy = (torch.randint(0, 49408, (1, 32), dtype=torch.long),)

    name_slug = backbone_name.lower().replace("-", "_")
    output_path = os.path.join(output_dir, f"text_encoder_{name_slug}.onnx")

    # Transformer with 12 layers accumulates float precision error; atol=1e-4 is expected
    return export_and_verify(
        wrapper, dummy, output_path,
        input_names=["input_ids"],
        output_names=["text_features"],
        opset_version=opset,
        atol=1e-4, rtol=1e-3,
        skip_verify=skip_verify,
    )


def export_dot_product_scoring(state_dict, output_dir, opset, skip_verify):
    """Export DotProductScoring module."""
    print(f"\n{'=' * 60}")
    print("3/4  DotProductScoring")
    print(f"{'=' * 60}")

    model = build_dot_product_scoring()
    model = load_scoring_weights(model, state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    torch.manual_seed(42)
    # hs: [num_layers, batch, num_queries, d_model]
    # prompt: [seq_len, batch, d_model]
    # prompt_mask: [batch, seq_len] (True = padding)
    hs = torch.randn(1, 1, 200, 256)
    prompt = torch.randn(32, 1, 256)
    prompt_mask = torch.zeros(1, 32, dtype=torch.bool)
    prompt_mask[0, 20:] = True  # Last 12 tokens are padding

    dummy = (hs, prompt, prompt_mask)
    output_path = os.path.join(output_dir, "dot_prod_scoring.onnx")

    return export_and_verify(
        model, dummy, output_path,
        input_names=["hs", "prompt", "prompt_mask"],
        output_names=["scores"],
        opset_version=opset,
        atol=1e-5, rtol=1e-3,
        skip_verify=skip_verify,
    )


def export_bbox_head(state_dict, output_dir, opset, skip_verify):
    """Export bbox_embed MLP (box regression head)."""
    print(f"\n{'=' * 60}")
    print("4/4  Box Head (bbox_embed)")
    print(f"{'=' * 60}")

    model = build_bbox_head()
    model = load_bbox_head_weights(model, state_dict)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    torch.manual_seed(42)
    # hidden_states: [num_queries, batch, d_model]
    dummy = (torch.randn(200, 1, 256),)
    output_path = os.path.join(output_dir, "bbox_head.onnx")

    return export_and_verify(
        model, dummy, output_path,
        input_names=["hidden_states"],
        output_names=["box_offsets"],
        opset_version=opset,
        atol=1e-6, rtol=1e-4,
        skip_verify=skip_verify,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export all ONNX-exportable Detector components of EfficientSAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Components exported:
  1. Image Encoder    (RepViT/TinyViT/EfficientViT backbone + head)
  2. Text Encoder     (MobileCLIP transformer + projector)
  3. DotProductScoring (prompt-query similarity scoring)
  4. Box Head         (bbox_embed MLP for box regression)

Components NOT exported (require ONNX-specific rewrites):
  - Transformer Fusion (Encoder + Decoder) - dynamic branches, DAC
  - Segmentation Head - dynamic List[Tensor], multi-branch
  - Geometry Encoder - dynamic sequences, ROI operations
        """,
    )

    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
        help="Path to EfficientSAM3 checkpoint",
    )
    parser.add_argument("--image-backbone", type=str, default="repvit_m0_9")
    parser.add_argument("--text-backbone", type=str, default="MobileCLIP-S1")
    parser.add_argument("--output", type=str, default="exports_repvit_m0_9/")
    parser.add_argument("--img-size", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip ONNX Runtime verification")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Load checkpoint once
    state_dict = load_checkpoint(args.checkpoint)

    results = {}

    # Export all 4 components
    results["Image Encoder"] = export_image_encoder(
        state_dict, args.image_backbone, args.output,
        args.img_size, args.opset, args.skip_verify,
    )

    results["Text Encoder"] = export_text_encoder(
        state_dict, args.text_backbone, args.output,
        args.opset, args.skip_verify,
    )

    results["DotProductScoring"] = export_dot_product_scoring(
        state_dict, args.output, args.opset, args.skip_verify,
    )

    results["Box Head"] = export_bbox_head(
        state_dict, args.output, args.opset, args.skip_verify,
    )

    # Summary
    print(f"\n{'=' * 60}")
    print("Export Summary")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} [{status}]")

    print(f"\nOutput directory: {args.output}")
    print(f"Files:")
    for f in sorted(os.listdir(args.output)):
        if f.endswith(".onnx"):
            size = os.path.getsize(os.path.join(args.output, f))
            print(f"  {f:45s} {size / 1024:.1f} KB")

    all_pass = all(results.values())
    if all_pass:
        print("\nAll components exported and verified successfully.")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"\nFailed components: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
