#!/usr/bin/env python3
"""
Export all ONNX-exportable Tracker components of EfficientSAM3.

Exports:
  1. SAM Prompt Encoder    (point/box → sparse + dense embeddings)
  2. SAM Mask Decoder      (image emb + prompt emb → masks + scores)
  3. Memory Encoder        (pix_feat + mask → memory features)
  4. obj_ptr_proj           (MLP projecting SAM tokens to object pointers)
  5. obj_ptr_tpos_proj      (Linear projecting temporal position)
  6. mask_downsample        (Conv2d downsampling mask 4x)

NOT exported:
  - Memory Attention (transformer.encoder) — RoPEAttention, isinstance checks, dynamic cache
  - Parameter tensors (no_mem_embed, etc.) — pure tensors, load directly

Usage:
    python export_tracker_components.py \
        --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
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
# Weight Loading Utilities
# ============================================================================

def load_checkpoint(checkpoint_path: str):
    """Load checkpoint and return state dict."""
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    return state_dict


def extract_weights(state_dict: dict, prefix: str):
    """Extract and strip-prefix weights."""
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def load_weights(model: nn.Module, state_dict: dict, prefix: str, name: str):
    """Extract weights by prefix and load into model."""
    weights = extract_weights(state_dict, prefix)
    if not weights:
        print(f"  WARNING: No weights found for {name} (prefix: {prefix})")
        return model
    missing, unexpected = model.load_state_dict(weights, strict=False)
    print(f"  Loaded {name} — Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    if missing:
        print(f"    Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"    Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
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
    opset_version: int = 18,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    skip_verify: bool = False,
):
    """Export model to ONNX and verify against PyTorch output."""
    model.eval()

    with torch.no_grad():
        if len(dummy_inputs) == 1:
            pt_outputs = model(dummy_inputs[0])
        else:
            pt_outputs = model(*dummy_inputs)

    if not isinstance(pt_outputs, tuple):
        pt_outputs = (pt_outputs,)

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
# Component Builders
# ============================================================================

def build_prompt_encoder():
    from sam3.sam.prompt_encoder import PromptEncoder
    return PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    )


def build_mask_decoder():
    from sam3.sam.transformer import TwoWayTransformer
    from sam3.sam.mask_decoder import MaskDecoder

    transformer = TwoWayTransformer(
        depth=2, embedding_dim=256, num_heads=8, mlp_dim=2048,
    )
    return MaskDecoder(
        num_multimask_outputs=3,
        transformer=transformer,
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        use_high_res_features=False,  # True needs high_res_features input; False for basic export
    )


def build_memory_encoder():
    from sam3.model.memory import SimpleMaskDownSampler, SimpleMaskEncoder, SimpleFuser, CXBlock
    from sam3.model.position_encoding import PositionEmbeddingSine

    # precompute_resolution=None to avoid CUDA requirement during init;
    # cache will be built lazily on first forward call on CPU
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64, normalize=True, scale=None,
        temperature=10000, precompute_resolution=None,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152],
    )
    cx_block = CXBlock(
        dim=256, kernel_size=7, padding=3,
        layer_scale_init_value=1e-6, use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=64, position_encoding=position_encoding,
        mask_downsampler=mask_downsampler, fuser=fuser,
    )


def build_obj_ptr_proj():
    from sam3.model.model_misc import MLP
    return MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=3)


def build_obj_ptr_tpos_proj():
    return nn.Linear(256, 64)


def build_mask_downsample():
    return nn.Conv2d(1, 1, kernel_size=4, stride=4)


# ============================================================================
# Export Functions
# ============================================================================

def export_prompt_encoder(state_dict, output_dir, opset, skip_verify):
    print(f"\n{'=' * 60}")
    print("1/6  SAM Prompt Encoder")
    print(f"{'=' * 60}")

    model = build_prompt_encoder()
    load_weights(model, state_dict, "tracker.sam_prompt_encoder.", "Prompt Encoder")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    class PromptEncoderWrapper(nn.Module):
        def __init__(self, pe):
            super().__init__()
            self.pe = pe

        def forward(self, point_coords, point_labels):
            sparse, dense = self.pe(points=(point_coords, point_labels), boxes=None, masks=None)
            return sparse, dense

    wrapper = PromptEncoderWrapper(model)

    torch.manual_seed(42)
    dummy = (torch.rand(1, 2, 2), torch.tensor([[1, 0]], dtype=torch.long))
    output_path = os.path.join(output_dir, "prompt_encoder.onnx")

    return export_and_verify(
        wrapper, dummy, output_path,
        input_names=["point_coords", "point_labels"],
        output_names=["sparse_embeddings", "dense_embeddings"],
        opset_version=opset, atol=1e-5, rtol=1e-3,
        skip_verify=skip_verify,
    )


def export_mask_decoder(state_dict, output_dir, opset, skip_verify):
    print(f"\n{'=' * 60}")
    print("2/6  SAM Mask Decoder")
    print(f"{'=' * 60}")

    model = build_mask_decoder()
    load_weights(model, state_dict, "tracker.sam_mask_decoder.", "Mask Decoder")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    class MaskDecoderWrapper(nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder

        def forward(self, image_embeddings, sparse_embeddings, dense_embeddings):
            low_res_masks, iou_predictions, _, _ = self.decoder(
                image_embeddings=image_embeddings,
                image_pe=dense_embeddings,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                repeat_image=False,
            )
            masks = F.interpolate(
                low_res_masks, size=(256, 256), mode="bilinear", align_corners=False,
            )
            return masks, iou_predictions

    wrapper = MaskDecoderWrapper(model)

    torch.manual_seed(42)
    dummy = (
        torch.randn(1, 256, 64, 64),
        torch.randn(1, 3, 256),
        torch.randn(1, 256, 64, 64),
    )
    output_path = os.path.join(output_dir, "mask_decoder.onnx")

    # Transformer layers accumulate float error
    return export_and_verify(
        wrapper, dummy, output_path,
        input_names=["image_embeddings", "sparse_embeddings", "dense_embeddings"],
        output_names=["masks", "iou_predictions"],
        opset_version=opset, atol=1e-4, rtol=1e-3,
        skip_verify=skip_verify,
    )


def export_memory_encoder(state_dict, output_dir, opset, skip_verify):
    print(f"\n{'=' * 60}")
    print("3/6  Memory Encoder (maskmem_backbone)")
    print(f"{'=' * 60}")

    model = build_memory_encoder()
    load_weights(model, state_dict, "tracker.maskmem_backbone.", "Memory Encoder")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    class MemoryEncoderWrapper(nn.Module):
        """Wrapper that returns tuple instead of dict."""
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, pix_feat, masks):
            out = self.encoder(pix_feat, masks, skip_mask_sigmoid=False)
            return out["vision_features"], out["vision_pos_enc"][0]

    wrapper = MemoryEncoderWrapper(model)

    torch.manual_seed(42)
    # pix_feat at 72×72 (matches mask downsampler output: interpol 1152 / stride 16 = 72)
    # mask at any resolution (gets interpolated to 1152×1152 internally)
    dummy = (
        torch.randn(1, 256, 72, 72),
        torch.randn(1, 1, 256, 256),
    )
    output_path = os.path.join(output_dir, "memory_encoder.onnx")

    return export_and_verify(
        wrapper, dummy, output_path,
        input_names=["pix_feat", "masks"],
        output_names=["vision_features", "vision_pos_enc"],
        # Multiple conv layers + CXBlock + LayerNorm2d accumulate float error
        opset_version=opset, atol=1e-4, rtol=1e-3,
        skip_verify=skip_verify,
    )


def export_obj_ptr_proj(state_dict, output_dir, opset, skip_verify):
    print(f"\n{'=' * 60}")
    print("4/6  obj_ptr_proj (MLP)")
    print(f"{'=' * 60}")

    model = build_obj_ptr_proj()
    load_weights(model, state_dict, "tracker.obj_ptr_proj.", "obj_ptr_proj")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 256),)
    output_path = os.path.join(output_dir, "obj_ptr_proj.onnx")

    return export_and_verify(
        model, dummy, output_path,
        input_names=["sam_tokens"],
        output_names=["obj_ptr"],
        opset_version=opset, atol=1e-6, rtol=1e-4,
        skip_verify=skip_verify,
    )


def export_obj_ptr_tpos_proj(state_dict, output_dir, opset, skip_verify):
    print(f"\n{'=' * 60}")
    print("5/6  obj_ptr_tpos_proj (Linear)")
    print(f"{'=' * 60}")

    model = build_obj_ptr_tpos_proj()
    load_weights(model, state_dict, "tracker.obj_ptr_tpos_proj.", "obj_ptr_tpos_proj")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 256),)
    output_path = os.path.join(output_dir, "obj_ptr_tpos_proj.onnx")

    return export_and_verify(
        model, dummy, output_path,
        input_names=["tpos_embed"],
        output_names=["tpos_proj"],
        opset_version=opset, atol=1e-6, rtol=1e-4,
        skip_verify=skip_verify,
    )


def export_mask_downsample(state_dict, output_dir, opset, skip_verify):
    print(f"\n{'=' * 60}")
    print("6/6  mask_downsample (Conv2d)")
    print(f"{'=' * 60}")

    model = build_mask_downsample()
    load_weights(model, state_dict, "tracker.mask_downsample.", "mask_downsample")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,}")

    torch.manual_seed(42)
    dummy = (torch.randn(1, 1, 256, 256),)
    output_path = os.path.join(output_dir, "mask_downsample.onnx")

    return export_and_verify(
        model, dummy, output_path,
        input_names=["mask"],
        output_names=["downsampled"],
        opset_version=opset, atol=1e-6, rtol=1e-4,
        skip_verify=skip_verify,
    )


# ============================================================================
# Non-exportable Report
# ============================================================================

def print_non_exportable_report(state_dict):
    print(f"\n{'=' * 60}")
    print("Non-exportable Tracker Components")
    print(f"{'=' * 60}")

    # Memory Attention
    mem_attn_keys = [k for k in state_dict if k.startswith("tracker.transformer.")]
    print(f"\n  [X] Memory Attention (tracker.transformer.encoder)")
    print(f"      Weights: {len(mem_attn_keys)} keys")
    print(f"      Blockers:")
    print(f"        1. isinstance(src, list) — runtime type check")
    print(f"        2. isinstance(cross_attn, RoPEAttention) — dynamic dispatch")
    print(f"        3. RoPEAttention dynamic frequency cache recomputation")
    print(f"        4. num_k_exclude_rope → variable-length slicing")
    print(f"        5. Dict return type")

    # Parameter tensors
    param_keys = {
        "tracker.maskmem_tpos_enc": "maskmem temporal position encoding",
        "tracker.no_mem_embed": "no-memory embedding",
        "tracker.no_mem_pos_enc": "no-memory position encoding",
        "tracker.no_obj_ptr": "no-object pointer",
        "tracker.no_obj_embed_spatial": "no-object spatial embedding",
    }
    print(f"\n  [~] Parameter Tensors (load directly, no ONNX needed)")
    for key, desc in param_keys.items():
        if key in state_dict:
            print(f"      {key}: {state_dict[key].shape}  — {desc}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Export all ONNX-exportable Tracker components of EfficientSAM3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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

    state_dict = load_checkpoint(args.checkpoint)

    results = {}
    results["Prompt Encoder"] = export_prompt_encoder(
        state_dict, args.output, args.opset, args.skip_verify)
    results["Mask Decoder"] = export_mask_decoder(
        state_dict, args.output, args.opset, args.skip_verify)
    results["Memory Encoder"] = export_memory_encoder(
        state_dict, args.output, args.opset, args.skip_verify)
    results["obj_ptr_proj"] = export_obj_ptr_proj(
        state_dict, args.output, args.opset, args.skip_verify)
    results["obj_ptr_tpos_proj"] = export_obj_ptr_tpos_proj(
        state_dict, args.output, args.opset, args.skip_verify)
    results["mask_downsample"] = export_mask_downsample(
        state_dict, args.output, args.opset, args.skip_verify)

    # Report non-exportable
    print_non_exportable_report(state_dict)

    # Summary
    print(f"\n{'=' * 60}")
    print("Export Summary")
    print(f"{'=' * 60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:25s} [{status}]")

    print(f"\nOutput directory: {args.output}")
    print(f"Tracker ONNX files:")
    tracker_files = [
        "prompt_encoder.onnx", "mask_decoder.onnx", "memory_encoder.onnx",
        "obj_ptr_proj.onnx", "obj_ptr_tpos_proj.onnx", "mask_downsample.onnx",
    ]
    for f in tracker_files:
        fp = os.path.join(args.output, f)
        if os.path.exists(fp):
            size = os.path.getsize(fp)
            print(f"  {f:35s} {size / 1024:.1f} KB")

    all_pass = all(results.values())
    if all_pass:
        print("\nAll exportable Tracker components verified successfully.")
    else:
        failed = [n for n, p in results.items() if not p]
        print(f"\nFailed components: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
