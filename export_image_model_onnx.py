#!/usr/bin/env python3
"""
Export EfficientSAM3 Image Model to ONNX — full text-grounding pipeline.

Exports 3 components:
  1. Image Encoder  (RepViT + Neck → 3 scales + position encoding)
  2. Text Encoder   (MobileCLIP-S1 → text features + mask)
  3. Decoder        (Geometry CLS + Encoder Fusion + Decoder + Scoring + Segmentation)

End-to-end inference flow:
  feat_4x, feat_2x, feat_1x, pos_1x = image_encoder(images)
  text_features, text_mask           = text_encoder(token_ids)
  scores, boxes_xyxy, mask_logits    = decoder(feat_4x, feat_2x, feat_1x, pos_1x,
                                                 text_features, text_mask)

Usage:
    python export_image_model_onnx.py \\
        --checkpoint checkpoints/stage1_all_converted/efficient_sam3_repvit-m0_9_mobileclip_s1.pth \\
        --output exports_repvit_m0_9/
"""

from __future__ import annotations

import argparse
import math
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

# Monkey-patch _get_rpb_matrix to always use the precomputed coord cache.
# The original code has `if self.compilable_stored_size == (H, W)` which is
# a data-dependent guard that torch.export cannot handle with symbolic shapes.
import numpy as _np_rpb
from sam3.model.box_ops import box_cxcywh_to_xyxy as _box_cxcywh_to_xyxy
from sam3.model.act_ckpt_utils import activation_ckpt_wrapper as _act_ckpt

def _patched_get_rpb_matrix(self, reference_boxes, feat_size):
    H, W = feat_size
    if self.compilable_cord_cache is None:
        self.compilable_cord_cache = self._get_coords(H, W, reference_boxes.device)
        self.compilable_stored_size = (H, W)
    coords_h, coords_w = self.compilable_cord_cache

    boxes_xyxy = _box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)
    bs, num_queries, _ = boxes_xyxy.shape

    deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
    deltas_y = deltas_y.view(bs, num_queries, -1, 2)
    deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
    deltas_x = deltas_x.view(bs, num_queries, -1, 2)

    if self.boxRPB in ["log", "both"]:
        deltas_x_log = deltas_x * 8
        deltas_x_log = _torch_sign_log2(deltas_x_log)
        deltas_y_log = deltas_y * 8
        deltas_y_log = _torch_sign_log2(deltas_y_log)
        if self.boxRPB == "log":
            deltas_x, deltas_y = deltas_x_log, deltas_y_log
        else:
            deltas_x = _torch.cat([deltas_x, deltas_x_log], dim=-1)
            deltas_y = _torch.cat([deltas_y, deltas_y_log], dim=-1)

    deltas_x = _act_ckpt(self.boxRPB_embed_x)(x=deltas_x, act_ckpt_enable=False)
    deltas_y = _act_ckpt(self.boxRPB_embed_y)(x=deltas_y, act_ckpt_enable=False)

    B = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)
    B = B.flatten(2, 3).permute(0, 3, 1, 2).contiguous()
    return B

import torch as _torch
def _torch_sign_log2(x):
    return _torch.sign(x) * _torch.log2(_torch.abs(x) + 1.0) / _np_rpb.log2(8)

_TD._get_rpb_matrix = _patched_get_rpb_matrix
# ---------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn

from sam3.model.model_misc import inverse_sigmoid
from sam3.model.box_ops import box_cxcywh_to_xyxy


# ============================================================================
# Wrappers
# ============================================================================

class ImageEncoderWrapper(nn.Module):
    """Student encoder (RepViT) + all 3 neck scales + position encoding.

    Input:  images   [1, 3, 1008, 1008]
    Output: feat_4x  [1, 256, 288, 288]   (4x upsampled)
            feat_2x  [1, 256, 144, 144]   (2x upsampled)
            feat_1x  [1, 256, 72, 72]     (1x)
            pos_1x   [1, 256, 72, 72]     (sinusoidal position encoding for 1x)
    """
    def __init__(self, visual_neck):
        super().__init__()
        self.student_encoder = visual_neck.trunk.model
        self.neck_conv_4x = visual_neck.convs[0]
        self.neck_conv_2x = visual_neck.convs[1]
        self.neck_conv_1x = visual_neck.convs[2]
        self.position_encoding = visual_neck.position_encoding

    def forward(self, images):
        feats = self.student_encoder(images)           # [1, 1024, 72, 72]
        feat_4x = self.neck_conv_4x(feats)             # [1, 256, 288, 288]
        feat_2x = self.neck_conv_2x(feats)             # [1, 256, 144, 144]
        feat_1x = self.neck_conv_1x(feats)             # [1, 256, 72, 72]
        pos_1x = self.position_encoding(feat_1x).to(feat_1x.dtype)
        return feat_4x, feat_2x, feat_1x, pos_1x


class TextEncoderWrapper(nn.Module):
    """MobileCLIP text encoder: token_ids → text features + padding mask.

    Input:  token_ids      [1, 77]  (int64, from SimpleTokenizer)
    Output: text_features  [1, 77, 256]
            text_mask      [1, 77]  (bool, True=padding)
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


class DecoderWrapper(nn.Module):
    """Full text-grounding decoder pipeline in a single ONNX-exportable module.

    Wraps: Geometry CLS → Encoder Fusion → Decoder → Scoring → Boxes → Masks.

    Inputs:
        feat_4x         [1, 256, 288, 288]
        feat_2x         [1, 256, 144, 144]
        feat_1x         [1, 256, 72, 72]
        pos_1x          [1, 256, 72, 72]
        text_features   [1, 77, 256]
        text_mask        [1, 77]  (bool)

    Outputs:
        scores          [1, 200]           final detection scores (sigmoid, with presence)
        boxes_xyxy      [1, 200, 4]        normalized boxes in xyxy format
        mask_logits     [1, 200, 288, 288] mask logits (apply sigmoid for probabilities)
    """
    def __init__(self, model):
        super().__init__()
        # Geometry encoder — only CLS token path
        geo = model.geometry_encoder
        self.geo_cls_embed = geo.cls_embed
        self.geo_final_proj = geo.final_proj
        self.geo_norm = geo.norm
        self.geo_encode_layers = geo.encode
        self.geo_encode_norm = geo.encode_norm

        # Encoder fusion
        self.encoder = model.transformer.encoder

        # Decoder
        self.decoder = model.transformer.decoder

        # Scoring
        self.dot_prod_scoring = model.dot_prod_scoring

        # Segmentation head
        self.segmentation_head = model.segmentation_head

        # Config flags from the model
        self.supervise_joint_box_scores = model.supervise_joint_box_scores

    def forward(self, feat_4x, feat_2x, feat_1x, pos_1x, text_features, text_mask):
        # ---- 1. Text features to seq-first ----
        txt_feats = text_features.permute(1, 0, 2)   # [77, 1, 256]
        txt_masks = text_mask                          # [1, 77]

        # ---- 2. Geometry Encoder CLS token ----
        # Convert image features to seq-first for geo encoder cross-attention
        img_feat_seq = feat_1x.flatten(2).permute(2, 0, 1)   # [5184, 1, 256]
        img_pos_seq = pos_1x.flatten(2).permute(2, 0, 1)     # [5184, 1, 256]

        bs = feat_1x.shape[0]  # 1
        cls = self.geo_cls_embed.weight.view(1, 1, -1).expand(1, bs, -1)
        cls_mask = torch.zeros(bs, 1, dtype=torch.bool, device=feat_1x.device)

        # Project and normalize CLS
        cls = self.geo_norm(self.geo_final_proj(cls))

        # 3-layer transformer with cross-attention to image features
        for lay in self.geo_encode_layers:
            cls = lay(
                tgt=cls,
                memory=img_feat_seq,
                tgt_key_padding_mask=cls_mask,
                pos=img_pos_seq,
            )
        cls = self.geo_encode_norm(cls)   # [1, 1, 256]

        # ---- 3. Build prompt = text + geo CLS ----
        prompt = torch.cat([txt_feats, cls], dim=0)        # [78, 1, 256]
        prompt_mask = torch.cat([txt_masks, cls_mask], dim=1)  # [1, 78]

        # ---- 4. Encoder Fusion ----
        # Encoder expects seq-first [HW, B, C] when feat_sizes is provided.
        # It reshapes them back to [B, C, H, W] internally.
        H, W = feat_1x.shape[2], feat_1x.shape[3]
        src_seq = feat_1x.flatten(2).permute(2, 0, 1)    # [5184, 1, 256]
        pos_seq = pos_1x.flatten(2).permute(2, 0, 1)     # [5184, 1, 256]
        prompt_pos = torch.zeros_like(prompt)
        memory_dict = self.encoder(
            src=[src_seq],
            src_pos=[pos_seq],
            prompt=prompt,
            prompt_pos=prompt_pos,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=[(H, W)],
        )
        memory = memory_dict["memory"]               # [5184, 1, 256] seq-first
        pos_embed = memory_dict["pos_embed"]          # [5184, 1, 256]
        padding_mask = memory_dict["padding_mask"]    # None
        level_start_index = memory_dict["level_start_index"]
        spatial_shapes = memory_dict["spatial_shapes"]
        valid_ratios = memory_dict["valid_ratios"]
        prompt_after_enc = memory_dict.get("memory_text", prompt)

        # ---- 5. Decoder ----
        query_embed = self.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)   # [200, 1, 256]

        hs, reference_boxes, dec_presence_out, _ = self.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=padding_mask,
            pos=pos_embed,
            reference_boxes=None,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            memory_text=prompt_after_enc,
            text_attention_mask=prompt_mask,
            apply_dac=False,
        )
        # hs: [6, 200, 1, 256] → [6, 1, 200, 256]
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)
        if dec_presence_out is not None:
            dec_presence_out = dec_presence_out.transpose(1, 2)  # [6, 1, 1]

        # ---- 6. Scoring (DotProductScoring on all layers) ----
        # hs: [6, 1, 200, 256], prompt_after_enc: [78, 1, 256], prompt_mask: [1, 78]
        outputs_class = self.dot_prod_scoring(hs, prompt_after_enc, prompt_mask)
        # outputs_class: [6, 1, 200, 1]

        # ---- 7. Joint presence scoring (applied during both train & eval) ----
        if self.supervise_joint_box_scores and dec_presence_out is not None:
            prob_presence = dec_presence_out.sigmoid()  # [6, 1, 1]
            outputs_class = inverse_sigmoid(
                outputs_class.sigmoid() * prob_presence.unsqueeze(2)
            ).clamp(min=-10.0, max=10.0)

        # ---- 8. Boxes ----
        anchor_box_offsets = self.decoder.bbox_embed(hs)  # [6, 1, 200, 4]
        ref_inv_sig = inverse_sigmoid(reference_boxes)
        outputs_coord = (ref_inv_sig + anchor_box_offsets).sigmoid()
        outputs_boxes_xyxy = box_cxcywh_to_xyxy(outputs_coord)

        # ---- 9. Segmentation Head ----
        seg_out = self.segmentation_head(
            backbone_feats=[feat_4x, feat_2x, feat_1x],
            obj_queries=hs,
            image_ids=torch.zeros(1, dtype=torch.long, device=feat_1x.device),
            encoder_hidden_states=memory,
            prompt=prompt_after_enc,
            prompt_mask=prompt_mask,
        )
        mask_logits = seg_out["pred_masks"]  # [1, 200, 288, 288]

        # ---- 10. Extract last decoder layer results ----
        # Scores: apply sigmoid to get final detection probabilities
        scores = outputs_class[-1, :, :, 0].sigmoid()   # [1, 200]
        # Also multiply by presence score (matching processor behavior)
        if dec_presence_out is not None:
            presence = dec_presence_out[-1].sigmoid()     # [1, 1]
            scores = scores * presence

        boxes_xyxy = outputs_boxes_xyxy[-1]              # [1, 200, 4]

        return scores, boxes_xyxy, mask_logits


# ============================================================================
# Export + Verify
# ============================================================================

def export_and_verify(model, dummy_inputs, output_path, input_names,
                      output_names, opset=18, atol=1e-5, skip_verify=False,
                      use_dynamo=True):
    model.eval()
    with torch.no_grad():
        pt_out = model(*dummy_inputs)
    if not isinstance(pt_out, tuple):
        pt_out = (pt_out,)

    print(f"  Exporting: {output_path}  (dynamo={use_dynamo})")
    torch.onnx.export(model, dummy_inputs, output_path,
                      input_names=input_names, output_names=output_names,
                      opset_version=opset, do_constant_folding=True,
                      dynamo=use_dynamo)
    print(f"  Size: {os.path.getsize(output_path)/1024/1024:.1f} MB")

    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        ops = sorted(set(n.op_type for n in onnx_model.graph.node))
        print(f"  ONNX check: PASSED  ({len(ops)} op types)")
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
    print(f"{'='*60}\n1/3  Image Encoder (RepViT + Neck 3-scale + PosEnc)\n{'='*60}")
    ie = ImageEncoderWrapper(model.backbone.vision_backbone)
    print(f"  Params: {sum(p.numel() for p in ie.parameters()):,}")
    torch.manual_seed(42)
    results["Image Encoder"] = export_and_verify(
        ie, (torch.randn(1, 3, 1008, 1008),),
        os.path.join(out, "image_encoder.onnx"),
        ["images"], ["feat_4x", "feat_2x", "feat_1x", "pos_1x"],
        opset=opset, atol=1e-4, skip_verify=sv,
    )

    # ---- 2. Text Encoder ----
    print(f"\n{'='*60}\n2/3  Text Encoder (MobileCLIP-S1)\n{'='*60}")
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

    # ---- 3. Decoder (full pipeline) ----
    print(f"\n{'='*60}\n3/3  Decoder (GeoEnc + EncoderFusion + Decoder + Seg)\n{'='*60}")
    dec = DecoderWrapper(model)
    print(f"  Params: {sum(p.numel() for p in dec.parameters()):,}")

    # Generate realistic dummy inputs by running image & text encoders
    torch.manual_seed(42)
    ie.eval()
    te.eval()
    with torch.no_grad():
        dummy_img = torch.randn(1, 3, 1008, 1008)
        d_feat_4x, d_feat_2x, d_feat_1x, d_pos_1x = ie(dummy_img)
        d_text_feat, d_text_mask = te(tids)

    results["Decoder"] = export_and_verify(
        dec,
        (d_feat_4x, d_feat_2x, d_feat_1x, d_pos_1x, d_text_feat, d_text_mask),
        os.path.join(out, "decoder.onnx"),
        ["feat_4x", "feat_2x", "feat_1x", "pos_1x", "text_features", "text_mask"],
        ["scores", "boxes_xyxy", "mask_logits"],
        opset=opset, atol=1e-3, skip_verify=sv,
        use_dynamo=False,  # legacy tracer for complex decoder
    )

    # ---- Summary ----
    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for name, ok in results.items():
        print(f"  {name:25s} [{'PASS' if ok else 'FAIL'}]")
    print()
    for f in ["image_encoder.onnx", "text_encoder.onnx", "decoder.onnx"]:
        fp = os.path.join(out, f)
        if os.path.exists(fp):
            sz = os.path.getsize(fp)
            print(f"  {f:35s} {sz/1024/1024:.1f} MB")

    if all(results.values()):
        print("\nAll 3 components exported and verified.")
    else:
        print(f"\nFailed: {[n for n,v in results.items() if not v]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
