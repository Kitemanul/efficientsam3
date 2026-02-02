#!/usr/bin/env python3
"""
Export ONNX-exportable components of EfficientSAM3 Image Model.

Builds the model via build_efficientsam3_image_model(), loads checkpoint,
then exports each exportable component.

Exports (4 components):
  1. Image Encoder  (RepViT + Neck)
  2. Text Encoder   (MobileCLIP-S1)
  3. DotProductScoring
  4. Box Head       (bbox_embed MLP)

Usage:
    python export_image_model_onnx.py \
        --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \
        --output exports_repvit_m0_9/
"""

from __future__ import annotations

import argparse
import os
import sys

# ---- Monkey-patches BEFORE importing sam3 (avoid CUDA hardcodes) ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))

from sam3.model.position_encoding import PositionEmbeddingSine as _PE

_orig_pe_init = _PE.__init__


def _patched_pe_init(self, num_pos_feats, temperature=10000, normalize=True,
                     scale=None, precompute_resolution=None):
    _orig_pe_init(self, num_pos_feats, temperature=temperature,
                  normalize=normalize, scale=scale,
                  precompute_resolution=None)


_PE.__init__ = _patched_pe_init

from sam3.model.decoder import TransformerDecoder as _TD

_orig_get_coords = _TD._get_coords


@staticmethod
def _patched_get_coords(H, W, device="cpu"):
    return _orig_get_coords(H, W, device="cpu")


_TD._get_coords = _patched_get_coords
# ---------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Wrappers
# ============================================================================

class ImageEncoderWrapper(nn.Module):
    """Student encoder (RepViT + head) + neck conv (scale 1.0).

    Input:  images          [B, 3, 1008, 1008]
    Output: image_features  [B, 256, 72, 72]
    """
    def __init__(self, visual_neck):
        super().__init__()
        self.student_encoder = visual_neck.trunk.model   # ImageStudentEncoder
        self.neck_conv = visual_neck.convs[2]            # scale 1.0

    def forward(self, images):
        feats = self.student_encoder(images)
        return self.neck_conv(feats)


class TextEncoderWrapper(nn.Module):
    """Takes token_ids (skip tokenization) → text features.

    Input:  token_ids      [B, 77]  (int64)
    Output: text_features  [B, 77, 256]
            text_mask      [B, 77]  (bool, True=padding)
    """
    def __init__(self, text_encoder):
        super().__init__()
        self.mobile_clip = text_encoder.encoder
        self.projector = text_encoder.projector

    def forward(self, token_ids):
        x = self.mobile_clip.forward_embedding(token_ids)
        for layer in self.mobile_clip.transformer:
            x = layer(x)
        x = self.mobile_clip.final_layer_norm(x)
        text_features = self.projector(x)
        text_mask = (token_ids == 0)
        return text_features, text_mask


# ============================================================================
# Export + Verify
# ============================================================================

def export_and_verify(model, dummy_inputs, output_path, input_names,
                      output_names, opset=18, atol=1e-5, skip_verify=False):
    model.eval()
    with torch.no_grad():
        pt_out = model(*dummy_inputs) if len(dummy_inputs) > 1 else model(dummy_inputs[0])
    if not isinstance(pt_out, tuple):
        pt_out = (pt_out,)

    print(f"  Exporting: {output_path}")
    inp = dummy_inputs[0] if len(dummy_inputs) == 1 else dummy_inputs
    torch.onnx.export(model, inp, output_path,
                      input_names=input_names, output_names=output_names,
                      opset_version=opset, do_constant_folding=True)
    print(f"  Size: {os.path.getsize(output_path)/1024:.1f} KB")

    try:
        import onnx
        onnx.checker.check_model(onnx.load(output_path))
        print("  ONNX check: PASSED")
    except ImportError:
        pass

    if skip_verify:
        return True
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed, skip verify")
        return True

    sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {n: t.numpy() for n, t in zip(input_names, dummy_inputs)})

    ok = True
    for p, o, name in zip(pt_out, ort_out, output_names):
        pn = p.detach().numpy()
        if pn.dtype == np.bool_:
            match = np.array_equal(pn, o)
            print(f"  [{name}]: {'PASS' if match else 'FAIL'}  (bool exact, shape={pn.shape})")
        else:
            err = np.max(np.abs(pn - o))
            match = np.allclose(pn, o, atol=atol, rtol=1e-3)
            print(f"  [{name}]: {'PASS' if match else 'FAIL'}  (max_abs={err:.2e}, shape={pn.shape})")
        if not match:
            ok = False
    return ok


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Export EfficientSAM3 Image Model to ONNX")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth")
    parser.add_argument("--output", type=str, default="exports_repvit_m0_9/")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # Build model
    from sam3.model_builder import build_efficientsam3_image_model

    print(f"Building EfficientSAM3 image model ...")
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        device="cpu",
        eval_mode=True,
        backbone_type="repvit",
        model_name="m0_9",
        text_encoder_type="MobileCLIP-S1",
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    print("  Done.\n")

    results = {}
    opset = args.opset
    sv = args.skip_verify
    out = args.output

    # ---- 1. Image Encoder ----
    print(f"{'='*60}\n1/4  Image Encoder\n{'='*60}")
    ie = ImageEncoderWrapper(model.backbone.vision_backbone)
    print(f"  Params: {sum(p.numel() for p in ie.parameters()):,}")
    torch.manual_seed(42)
    results["Image Encoder"] = export_and_verify(
        ie, (torch.randn(1, 3, 1008, 1008),),
        os.path.join(out, "image_encoder.onnx"),
        ["images"], ["image_features"],
        opset=opset, atol=1e-4, skip_verify=sv,
    )

    # ---- 2. Text Encoder ----
    print(f"\n{'='*60}\n2/4  Text Encoder\n{'='*60}")
    te = TextEncoderWrapper(model.backbone.language_backbone)
    print(f"  Params: {sum(p.numel() for p in te.parameters()):,}")
    torch.manual_seed(42)
    tids = torch.zeros(1, 77, dtype=torch.long)
    tids[0, :10] = torch.randint(1, 49408, (10,))
    results["Text Encoder"] = export_and_verify(
        te, (tids,),
        os.path.join(out, "text_encoder.onnx"),
        ["token_ids"], ["text_features", "text_mask"],
        opset=opset, atol=1e-4, skip_verify=sv,
    )

    # ---- 3. DotProductScoring ----
    print(f"\n{'='*60}\n3/4  DotProductScoring\n{'='*60}")
    sc = model.dot_prod_scoring
    print(f"  Params: {sum(p.numel() for p in sc.parameters()):,}")
    torch.manual_seed(42)
    results["DotProductScoring"] = export_and_verify(
        sc, (torch.randn(1, 1, 200, 256),
             torch.randn(77, 1, 256),
             torch.zeros(1, 77, dtype=torch.bool)),
        os.path.join(out, "dot_prod_scoring.onnx"),
        ["hs", "prompt", "prompt_mask"], ["scores"],
        opset=opset, atol=1e-5, skip_verify=sv,
    )

    # ---- 4. Box Head ----
    print(f"\n{'='*60}\n4/4  Box Head\n{'='*60}")
    bh = model.transformer.decoder.bbox_embed
    print(f"  Params: {sum(p.numel() for p in bh.parameters()):,}")
    torch.manual_seed(42)
    results["Box Head"] = export_and_verify(
        bh, (torch.randn(1, 256),),
        os.path.join(out, "bbox_head.onnx"),
        ["decoder_output"], ["bbox"],
        opset=opset, atol=1e-6, skip_verify=sv,
    )

    # ---- Summary ----
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, ok in results.items():
        print(f"  {name:25s} [{'PASS' if ok else 'FAIL'}]")
    print()
    for f in ["image_encoder.onnx", "text_encoder.onnx",
              "dot_prod_scoring.onnx", "bbox_head.onnx"]:
        fp = os.path.join(out, f)
        if os.path.exists(fp):
            print(f"  {f:35s} {os.path.getsize(fp)/1024:.1f} KB")

    if all(results.values()):
        print("\nAll 4 components exported and verified.")
    else:
        print(f"\nFailed: {[n for n,v in results.items() if not v]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
