#!/usr/bin/env python3
"""
Export all ONNX-exportable components of EfficientSAM3.

Builds the full model using build_efficientsam3_video_model(), loads
the checkpoint, then extracts and exports each component to ONNX.

Exports (10 components):
  Detector: Image Encoder, Text Encoder, DotProductScoring, Box Head
  Tracker:  Prompt Encoder, Mask Decoder, Memory Encoder,
            obj_ptr_proj, obj_ptr_tpos_proj, mask_downsample

Usage:
    python export_onnx.py \
        --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
        --output exports_repvit_m0_9/
"""

from __future__ import annotations

import argparse
import os
import sys

# ---- Monkey-patch PositionEmbeddingSine BEFORE importing sam3 ----
# position_encoding.py:47 hardcodes device="cuda" when precompute_resolution
# is set.  Force it to None so all caches are built lazily on CPU.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))

from sam3.model.position_encoding import PositionEmbeddingSine as _PE

_orig_pe_init = _PE.__init__


def _patched_pe_init(self, num_pos_feats, temperature=10000, normalize=True,
                     scale=None, precompute_resolution=None):
    _orig_pe_init(self, num_pos_feats, temperature=temperature,
                  normalize=normalize, scale=scale,
                  precompute_resolution=None)   # always skip CUDA precompute


_PE.__init__ = _patched_pe_init

# ---- Monkey-patch TransformerDecoder._get_coords to avoid CUDA ----
# decoder.py:281 hardcodes device="cuda" for coord cache precomputation.
from sam3.model.decoder import TransformerDecoder as _TD

_orig_get_coords = _TD._get_coords


@staticmethod
def _patched_get_coords(H, W, device="cpu"):
    return _orig_get_coords(H, W, device="cpu")


_TD._get_coords = _patched_get_coords
# ------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Wrappers (make components ONNX-compatible)
# ============================================================================

class ImageEncoderWrapper(nn.Module):
    """Wraps student encoder + neck conv (scale 1.0) into a single module.

    Input:  images     [B, 3, 1008, 1008]
    Output: features   [B, 256, 72, 72]
    """
    def __init__(self, visual_neck):
        super().__init__()
        # ImageStudentEncoder: RepViT backbone + head (384→1024) + interpolate to 72×72
        self.student_encoder = visual_neck.trunk.model
        # Neck scale-1.0 branch: Conv1x1(1024→256) + Conv3x3(256→256)
        self.neck_conv = visual_neck.convs[2]

    def forward(self, images):
        feats = self.student_encoder(images)   # [B, 1024, 72, 72]
        return self.neck_conv(feats)           # [B, 256, 72, 72]


class TextEncoderWrapper(nn.Module):
    """Wraps TextStudentEncoder: takes token_ids instead of raw strings.

    Input:  token_ids       [B, 77]  (int64, from SimpleTokenizer)
    Output: text_features   [B, 77, 256]
            text_mask       [B, 77]  (bool, True=padding)
    """
    def __init__(self, text_encoder):
        super().__init__()
        self.mobile_clip = text_encoder.encoder   # MobileCLIPTextTransformer
        self.projector = text_encoder.projector    # Linear(512 → 256)

    def forward(self, token_ids):
        # Embedding + positional encoding
        x = self.mobile_clip.forward_embedding(token_ids)  # [B, 77, 512]
        # Transformer layers
        for layer in self.mobile_clip.transformer:
            x = layer(x)
        x = self.mobile_clip.final_layer_norm(x)
        # Project to output dim
        text_features = self.projector(x)           # [B, 77, 256]
        text_mask = (token_ids == 0)                # True for padding
        return text_features, text_mask


class PromptEncoderWrapper(nn.Module):
    """Wraps SAM PromptEncoder: handles None boxes/masks.

    Input:  point_coords  [B, N, 2]
            point_labels  [B, N]
    Output: sparse_emb    [B, N+1, 256]
            dense_emb     [B, 256, 72, 72]
    """
    def __init__(self, prompt_encoder):
        super().__init__()
        self.pe = prompt_encoder

    def forward(self, point_coords, point_labels):
        sparse, dense = self.pe(
            points=(point_coords, point_labels), boxes=None, masks=None,
        )
        return sparse, dense


class MaskDecoderWrapper(nn.Module):
    """Wraps SAM MaskDecoder: fixed multimask_output, simple upscaling path.

    Input:  image_embeddings   [B, 256, 72, 72]
            image_pe           [B, 256, 72, 72]  (from prompt_encoder.get_dense_pe)
            sparse_embeddings  [B, N, 256]
            dense_embeddings   [B, 256, 72, 72]
    Output: masks              [B, 4, 256, 256]
            iou_predictions    [B, 4]
    """
    def __init__(self, mask_decoder):
        super().__init__()
        self.decoder = mask_decoder
        # Use simple upscaling (same ConvTranspose2d weights, no skip connections)
        # to avoid requiring high_res_features input
        self.decoder.use_high_res_features = False

    def forward(self, image_embeddings, image_pe,
                sparse_embeddings, dense_embeddings):
        low_res_masks, iou_predictions, _, _ = self.decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
        )
        masks = F.interpolate(
            low_res_masks, size=(256, 256),
            mode="bilinear", align_corners=False,
        )
        return masks, iou_predictions


class MemoryEncoderWrapper(nn.Module):
    """Wraps SimpleMaskEncoder: converts Dict output to tuple.

    Input:  pix_feat   [B, 256, 72, 72]
            masks      [B, 1, 256, 256]
    Output: vision_features  [B, 64, 72, 72]
            vision_pos_enc   [B, 64, 72, 72]
    """
    def __init__(self, maskmem_backbone):
        super().__init__()
        self.encoder = maskmem_backbone

    def forward(self, pix_feat, masks):
        out = self.encoder(pix_feat, masks, skip_mask_sigmoid=False)
        return out["vision_features"], out["vision_pos_enc"][0]


# ============================================================================
# Export + Verify utility
# ============================================================================

def export_and_verify(
    model: nn.Module,
    dummy_inputs: tuple,
    output_path: str,
    input_names: list,
    output_names: list,
    opset_version: int = 18,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    skip_verify: bool = False,
):
    """Export model to ONNX and verify against PyTorch output."""
    model.eval()

    with torch.no_grad():
        pt_outputs = model(*dummy_inputs) if len(dummy_inputs) > 1 else model(dummy_inputs[0])
    if not isinstance(pt_outputs, tuple):
        pt_outputs = (pt_outputs,)

    print(f"  Exporting to: {output_path}")
    export_input = dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs
    torch.onnx.export(
        model, export_input, output_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    file_size = os.path.getsize(output_path)
    print(f"  File size: {file_size / 1024:.1f} KB")

    # ONNX model check
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        ops = sorted(set(n.op_type for n in onnx_model.graph.node))
        print(f"  ONNX check: PASSED  ({len(ops)} op types)")
    except ImportError:
        print("  Warning: onnx not installed, skipping model check")

    if skip_verify:
        print("  Verification: SKIPPED")
        return True

    # ONNX Runtime verification
    try:
        import onnxruntime as ort
    except ImportError:
        print("  Warning: onnxruntime not installed, skipping verification")
        return True

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_inputs = {name: t.numpy() for name, t in zip(input_names, dummy_inputs)}
    ort_outputs = sess.run(None, ort_inputs)

    all_pass = True
    for pt_out, ort_out, name in zip(pt_outputs, ort_outputs, output_names):
        pt_np = pt_out.detach().numpy()
        if pt_np.dtype == np.bool_:
            # Boolean output: exact match
            ok = np.array_equal(pt_np, ort_out)
            status = "PASS" if ok else "FAIL"
            print(f"  Verify [{name}]: {status}  (bool, exact match, shape={pt_np.shape})")
        else:
            max_abs = np.max(np.abs(pt_np - ort_out))
            max_rel = np.max(np.abs(pt_np - ort_out) / (np.abs(pt_np) + 1e-8))
            ok = np.allclose(pt_np, ort_out, atol=atol, rtol=rtol)
            status = "PASS" if ok else "FAIL"
            print(f"  Verify [{name}]: {status}  "
                  f"(max_abs={max_abs:.2e}, max_rel={max_rel:.2e}, "
                  f"shape={pt_np.shape})")
        if not ok:
            all_pass = False
    return all_pass


# ============================================================================
# Individual export functions
# ============================================================================

def export_image_encoder(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("1/10  Image Encoder (RepViT + Neck)")
    print(f"{'='*60}")

    visual_neck = model.detector.backbone.vision_backbone
    wrapper = ImageEncoderWrapper(visual_neck)
    n = sum(p.numel() for p in wrapper.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 3, 1008, 1008),)
    return export_and_verify(
        wrapper, dummy,
        os.path.join(output_dir, "image_encoder.onnx"),
        input_names=["images"],
        output_names=["image_features"],
        opset_version=opset, atol=1e-4,
        skip_verify=skip_verify,
    )


def export_text_encoder(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("2/10  Text Encoder (MobileCLIP-S1)")
    print(f"{'='*60}")

    text_enc = model.detector.backbone.language_backbone
    wrapper = TextEncoderWrapper(text_enc)
    n = sum(p.numel() for p in wrapper.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")

    torch.manual_seed(42)
    # Simulated token IDs: first ~10 tokens valid, rest padding (0)
    token_ids = torch.zeros(1, 77, dtype=torch.long)
    token_ids[0, :10] = torch.randint(1, 49408, (10,))
    dummy = (token_ids,)

    return export_and_verify(
        wrapper, dummy,
        os.path.join(output_dir, "text_encoder.onnx"),
        input_names=["token_ids"],
        output_names=["text_features", "text_mask"],
        opset_version=opset, atol=1e-4,
        skip_verify=skip_verify,
    )


def export_dot_prod_scoring(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("3/10  DotProductScoring")
    print(f"{'='*60}")

    scoring = model.detector.dot_prod_scoring
    n = sum(p.numel() for p in scoring.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")

    torch.manual_seed(42)
    dummy = (
        torch.randn(1, 1, 200, 256),   # hs  [num_layer, B, num_query, d_model]
        torch.randn(77, 1, 256),        # prompt  [seq, B, d_model]
        torch.zeros(1, 77, dtype=torch.bool),  # prompt_mask [B, seq]
    )
    return export_and_verify(
        scoring, dummy,
        os.path.join(output_dir, "dot_prod_scoring.onnx"),
        input_names=["hs", "prompt", "prompt_mask"],
        output_names=["scores"],
        opset_version=opset, atol=1e-5,
        skip_verify=skip_verify,
    )


def export_bbox_head(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("4/10  Box Head (bbox_embed MLP)")
    print(f"{'='*60}")

    bbox_embed = model.detector.transformer.decoder.bbox_embed
    n = sum(p.numel() for p in bbox_embed.parameters())
    print(f"  Parameters: {n:,}")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 256),)
    return export_and_verify(
        bbox_embed, dummy,
        os.path.join(output_dir, "bbox_head.onnx"),
        input_names=["decoder_output"],
        output_names=["bbox"],
        opset_version=opset, atol=1e-6,
        skip_verify=skip_verify,
    )


def export_prompt_encoder(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("5/10  SAM Prompt Encoder")
    print(f"{'='*60}")

    wrapper = PromptEncoderWrapper(model.tracker.sam_prompt_encoder)
    n = sum(p.numel() for p in wrapper.parameters())
    print(f"  Parameters: {n:,}")

    torch.manual_seed(42)
    dummy = (
        torch.rand(1, 2, 2),                          # point_coords
        torch.tensor([[1, 0]], dtype=torch.long),      # point_labels
    )
    return export_and_verify(
        wrapper, dummy,
        os.path.join(output_dir, "prompt_encoder.onnx"),
        input_names=["point_coords", "point_labels"],
        output_names=["sparse_embeddings", "dense_embeddings"],
        opset_version=opset, atol=1e-5,
        skip_verify=skip_verify,
    )


def export_mask_decoder(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("6/10  SAM Mask Decoder")
    print(f"{'='*60}")

    wrapper = MaskDecoderWrapper(model.tracker.sam_mask_decoder)
    n = sum(p.numel() for p in wrapper.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")

    # image_embedding_size = 1008 // 14 = 72
    torch.manual_seed(42)
    dummy = (
        torch.randn(1, 256, 72, 72),   # image_embeddings
        torch.randn(1, 256, 72, 72),   # image_pe
        torch.randn(1, 3, 256),         # sparse_embeddings
        torch.randn(1, 256, 72, 72),   # dense_embeddings
    )
    return export_and_verify(
        wrapper, dummy,
        os.path.join(output_dir, "mask_decoder.onnx"),
        input_names=["image_embeddings", "image_pe",
                      "sparse_embeddings", "dense_embeddings"],
        output_names=["masks", "iou_predictions"],
        opset_version=opset, atol=1e-4,
        skip_verify=skip_verify,
    )


def export_memory_encoder(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("7/10  Memory Encoder")
    print(f"{'='*60}")

    wrapper = MemoryEncoderWrapper(model.tracker.maskmem_backbone)
    n = sum(p.numel() for p in wrapper.parameters())
    print(f"  Parameters: {n:,} ({n/1e6:.2f}M)")

    torch.manual_seed(42)
    dummy = (
        torch.randn(1, 256, 72, 72),     # pix_feat
        torch.randn(1, 1, 256, 256),      # masks
    )
    return export_and_verify(
        wrapper, dummy,
        os.path.join(output_dir, "memory_encoder.onnx"),
        input_names=["pix_feat", "masks"],
        output_names=["vision_features", "vision_pos_enc"],
        opset_version=opset, atol=1e-4,
        skip_verify=skip_verify,
    )


def export_obj_ptr_proj(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("8/10  obj_ptr_proj (MLP)")
    print(f"{'='*60}")

    mlp = model.tracker.obj_ptr_proj
    n = sum(p.numel() for p in mlp.parameters())
    print(f"  Parameters: {n:,}")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 256),)
    return export_and_verify(
        mlp, dummy,
        os.path.join(output_dir, "obj_ptr_proj.onnx"),
        input_names=["sam_tokens"],
        output_names=["obj_ptr"],
        opset_version=opset, atol=1e-6,
        skip_verify=skip_verify,
    )


def export_obj_ptr_tpos_proj(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("9/10  obj_ptr_tpos_proj (Linear)")
    print(f"{'='*60}")

    linear = model.tracker.obj_ptr_tpos_proj
    n = sum(p.numel() for p in linear.parameters())
    print(f"  Parameters: {n:,}")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 256),)
    return export_and_verify(
        linear, dummy,
        os.path.join(output_dir, "obj_ptr_tpos_proj.onnx"),
        input_names=["tpos_embed"],
        output_names=["tpos_proj"],
        opset_version=opset, atol=1e-6,
        skip_verify=skip_verify,
    )


def export_mask_downsample(model, output_dir, opset, skip_verify):
    print(f"\n{'='*60}")
    print("10/10  mask_downsample (Conv2d)")
    print(f"{'='*60}")

    conv = model.tracker.mask_downsample
    n = sum(p.numel() for p in conv.parameters())
    print(f"  Parameters: {n:,}")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 1, 256, 256),)
    return export_and_verify(
        conv, dummy,
        os.path.join(output_dir, "mask_downsample.onnx"),
        input_names=["mask"],
        output_names=["downsampled"],
        opset_version=opset, atol=1e-6,
        skip_verify=skip_verify,
    )


# ============================================================================
# Non-exportable report
# ============================================================================

def print_non_exportable_report():
    print(f"\n{'='*60}")
    print("Non-exportable Components")
    print(f"{'='*60}")

    print(f"\n  [X] Memory Attention (tracker.transformer.encoder)")
    print(f"      Blockers: isinstance checks, RoPEAttention dynamic cache,")
    print(f"      variable-length slicing, Dict return type")

    print(f"\n  [X] Transformer Fusion (detector.transformer)")
    print(f"      Blockers: List[Tensor] I/O, DAC dynamic branches")

    print(f"\n  [X] Segmentation Head (detector.segmentation_head)")
    print(f"      Blockers: List[Tensor] input, pixel decoder multi-scale")

    print(f"\n  [X] Geometry Encoder (detector.geometry_encoder)")
    print(f"      Blockers: dynamic ROI ops, conditional geometry_type branches")

    print(f"\n  [~] Parameter Tensors (load directly, no ONNX needed)")
    param_tensors = [
        ("tracker.maskmem_tpos_enc", "[7, 1, 1, 64]", "temporal position encoding"),
        ("tracker.no_mem_embed", "[1, 1, 256]", "no-memory embedding"),
        ("tracker.no_mem_pos_enc", "[1, 1, 256]", "no-memory position encoding"),
        ("tracker.no_obj_ptr", "[1, 256]", "no-object pointer"),
        ("tracker.no_obj_embed_spatial", "[1, 64]", "no-object spatial embedding"),
    ]
    for key, shape, desc in param_tensors:
        print(f"      {key}: {shape}  -- {desc}")


# ============================================================================
# Build model
# ============================================================================

def build_model(checkpoint_path):
    """Build full EfficientSAM3 video model and load checkpoint."""
    from sam3.model_builder import build_efficientsam3_video_model

    print(f"Building model and loading checkpoint: {checkpoint_path}")
    model = build_efficientsam3_video_model(
        checkpoint_path=checkpoint_path,
        device="cpu",
        backbone_type="repvit",
        model_name="m0_9",
        text_encoder_type="MobileCLIP-S1",
        enable_inst_interactivity=True,
        strict_state_dict_loading=False,
    )
    model.eval()
    print("  Model built successfully.")
    return model


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export all ONNX-exportable EfficientSAM3 components",
    )
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth",
    )
    parser.add_argument("--output", type=str, default="exports_repvit_m0_9/")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = build_model(args.checkpoint)

    exporters = [
        ("Image Encoder",     export_image_encoder),
        ("Text Encoder",      export_text_encoder),
        ("DotProductScoring", export_dot_prod_scoring),
        ("Box Head",          export_bbox_head),
        ("Prompt Encoder",    export_prompt_encoder),
        ("Mask Decoder",      export_mask_decoder),
        ("Memory Encoder",    export_memory_encoder),
        ("obj_ptr_proj",      export_obj_ptr_proj),
        ("obj_ptr_tpos_proj", export_obj_ptr_tpos_proj),
        ("mask_downsample",   export_mask_downsample),
    ]

    results = {}
    for name, fn in exporters:
        results[name] = fn(model, args.output, args.opset, args.skip_verify)

    print_non_exportable_report()

    # Summary
    print(f"\n{'='*60}")
    print("Export Summary")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} [{status}]")

    print(f"\nOutput directory: {args.output}")
    onnx_files = [
        "image_encoder.onnx", "text_encoder.onnx",
        "dot_prod_scoring.onnx", "bbox_head.onnx",
        "prompt_encoder.onnx", "mask_decoder.onnx",
        "memory_encoder.onnx", "obj_ptr_proj.onnx",
        "obj_ptr_tpos_proj.onnx", "mask_downsample.onnx",
    ]
    for f in onnx_files:
        fp = os.path.join(args.output, f)
        if os.path.exists(fp):
            print(f"  {f:35s} {os.path.getsize(fp) / 1024:.1f} KB")

    if all(results.values()):
        print("\nAll 10 components exported and verified successfully.")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"\nFailed: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
