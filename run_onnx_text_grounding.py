#!/usr/bin/env python3
"""
Run EfficientSAM3 text-grounding inference using ONNX Runtime.

Uses the 3 exported ONNX models (image_encoder, text_encoder, decoder)
to perform text-grounded detection and segmentation, then visualizes
results with boxes, masks, and scores after NMS + top-k filtering.

Usage:
    python run_onnx_text_grounding.py \
        --image assets/test.jpg \
        --prompt "person" \
        --onnx-dir exports_repvit_m0_9/ \
        --output result.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

# ---- Tokenizer ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
from sam3.model.tokenizer_ve import SimpleTokenizer


# ============================================================================
# Image preprocessing (matches Sam3Processor)
# ============================================================================

def preprocess_image(image: Image.Image, resolution: int = 1008) -> np.ndarray:
    """Preprocess PIL image to model input format.

    Matches the Sam3Processor transform pipeline:
      1. Resize to (resolution, resolution)
      2. Convert to float32 [0, 1]
      3. Normalize with mean=0.5, std=0.5 → [-1, 1]

    Returns:
        np.ndarray of shape [1, 3, resolution, resolution], dtype float32
    """
    img = image.resize((resolution, resolution), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0       # [H, W, 3] in [0, 1]
    arr = (arr - 0.5) / 0.5                              # [-1, 1]
    arr = arr.transpose(2, 0, 1)                          # [3, H, W]
    return arr[np.newaxis]                                # [1, 3, H, W]


# ============================================================================
# Text tokenization
# ============================================================================

def tokenize_text(text: str, bpe_path: str, context_length: int = 77) -> np.ndarray:
    """Tokenize text prompt using SimpleTokenizer.

    Returns:
        np.ndarray of shape [1, 77], dtype int64
    """
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    tokens = tokenizer([text], context_length=context_length)
    return tokens.numpy().astype(np.int64)


# ============================================================================
# NMS
# ============================================================================

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Non-Maximum Suppression on xyxy boxes.

    Args:
        boxes:  [N, 4] in xyxy format (normalized 0~1)
        scores: [N] confidence scores
        iou_threshold: IoU threshold for suppression

    Returns:
        keep: indices of kept boxes
    """
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int64)


# ============================================================================
# Visualization
# ============================================================================

PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def draw_results(
    image: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Draw boxes, masks, and scores on image.

    Args:
        image:  [H, W, 3] uint8 BGR
        boxes:  [K, 4] xyxy in pixel coordinates
        masks:  [K, H, W] binary masks (resized to image size)
        scores: [K] confidence scores

    Returns:
        [H, W, 3] uint8 BGR image with overlay
    """
    vis = image.copy()

    for i in range(len(scores)):
        color = PALETTE[i % len(PALETTE)]
        mask = masks[i]
        score = scores[i]
        box = boxes[i].astype(int)

        # Draw mask overlay
        colored = np.zeros_like(vis, dtype=np.uint8)
        colored[:] = color
        vis = np.where(mask[..., None],
                       (vis * (1 - alpha) + colored * alpha).astype(np.uint8),
                       vis)

        # Draw mask contour
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, color, 2)

        # Draw box
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)

        # Draw label
        label = f"{score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (box[0], box[1] - th - 6), (box[0] + tw + 4, box[1]), color, -1)
        cv2.putText(vis, label, (box[0] + 2, box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EfficientSAM3 ONNX text-grounding inference"
    )
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt (e.g. 'person')")
    parser.add_argument("--onnx-dir", type=str, default="exports_repvit_m0_9/",
                        help="Directory containing the 3 ONNX models")
    parser.add_argument("--output", type=str, default="result.png",
                        help="Output visualization path")
    parser.add_argument("--resolution", type=int, default=1008,
                        help="Input resolution (must match exported model)")
    parser.add_argument("--score-threshold", type=float, default=0.1,
                        help="Minimum confidence score")
    parser.add_argument("--nms-threshold", type=float, default=0.5,
                        help="NMS IoU threshold")
    parser.add_argument("--top-k", type=int, default=6,
                        help="Maximum number of detections to display")
    parser.add_argument("--bpe-path", type=str, default=None,
                        help="Path to BPE vocab file (default: auto-detect)")
    args = parser.parse_args()

    # ---- Resolve BPE path ----
    bpe_path = args.bpe_path
    if bpe_path is None:
        bpe_path = os.path.join(
            os.path.dirname(__file__), "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz"
        )
    if not os.path.exists(bpe_path):
        print(f"ERROR: BPE vocab not found at {bpe_path}")
        print("Please specify --bpe-path")
        sys.exit(1)

    # ---- Load ONNX models ----
    print("Loading ONNX models ...")
    providers = ["CPUExecutionProvider"]
    img_sess = ort.InferenceSession(
        os.path.join(args.onnx_dir, "image_encoder.onnx"), providers=providers
    )
    txt_sess = ort.InferenceSession(
        os.path.join(args.onnx_dir, "text_encoder.onnx"), providers=providers
    )
    dec_sess = ort.InferenceSession(
        os.path.join(args.onnx_dir, "detector.onnx"), providers=providers
    )
    print("  Done.")

    # ---- Load and preprocess image ----
    pil_image = Image.open(args.image).convert("RGB")
    orig_w, orig_h = pil_image.size
    image_input = preprocess_image(pil_image, resolution=args.resolution)
    print(f"Image: {args.image} ({orig_w}x{orig_h})")

    # ---- Tokenize text ----
    token_ids = tokenize_text(args.prompt, bpe_path)
    print(f"Prompt: '{args.prompt}' → {(token_ids != 0).sum()} tokens")

    # ---- Run image encoder ----
    print("Running image encoder ...")
    feat_4x, feat_2x, feat_1x, pos_1x = img_sess.run(None, {"images": image_input})

    # ---- Run text encoder ----
    print("Running text encoder ...")
    text_features, text_mask = txt_sess.run(None, {"token_ids": token_ids})

    # ---- Run decoder ----
    print("Running decoder ...")
    scores, boxes_xyxy, mask_logits = dec_sess.run(None, {
        "feat_4x": feat_4x,
        "feat_2x": feat_2x,
        "feat_1x": feat_1x,
        "pos_1x": pos_1x,
        "text_features": text_features,
        "text_mask": text_mask
    })
    # scores: [1, 200], boxes_xyxy: [1, 200, 4], mask_logits: [1, 200, H, H]
    scores = scores[0]          # [200]
    boxes_xyxy = boxes_xyxy[0]  # [200, 4]
    mask_logits = mask_logits[0]  # [200, mask_H, mask_W]

    print(f"Decoder output: {scores.shape[0]} queries, "
          f"mask size {mask_logits.shape[1]}x{mask_logits.shape[2]}")

    # ---- Post-processing ----
    # 1. Score threshold
    valid = scores > args.score_threshold
    if not valid.any():
        print(f"No detections above threshold {args.score_threshold}")
        # Save original image
        pil_image.save(args.output)
        print(f"Saved (no detections): {args.output}")
        return

    scores_valid = scores[valid]
    boxes_valid = boxes_xyxy[valid]
    masks_valid = mask_logits[valid]

    print(f"After score filter (>{args.score_threshold}): {len(scores_valid)} detections")

    # 2. NMS
    keep = nms(boxes_valid, scores_valid, iou_threshold=args.nms_threshold)
    scores_nms = scores_valid[keep]
    boxes_nms = boxes_valid[keep]
    masks_nms = masks_valid[keep]

    print(f"After NMS (IoU>{args.nms_threshold}): {len(scores_nms)} detections")

    # 3. Top-K
    if len(scores_nms) > args.top_k:
        topk_idx = scores_nms.argsort()[::-1][:args.top_k]
        scores_nms = scores_nms[topk_idx]
        boxes_nms = boxes_nms[topk_idx]
        masks_nms = masks_nms[topk_idx]

    print(f"Top-{args.top_k}: {len(scores_nms)} detections")

    # 4. Convert boxes from normalized [0,1] to pixel coordinates
    boxes_pixel = boxes_nms.copy()
    boxes_pixel[:, [0, 2]] *= orig_w
    boxes_pixel[:, [1, 3]] *= orig_h

    # 5. Sigmoid mask logits and resize to original image size
    masks_prob = 1.0 / (1.0 + np.exp(-masks_nms))  # sigmoid
    masks_resized = np.zeros((len(masks_prob), orig_h, orig_w), dtype=bool)
    for i in range(len(masks_prob)):
        resized = cv2.resize(
            masks_prob[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        masks_resized[i] = resized > 0.5

    # ---- Print results ----
    print(f"\n{'='*50}")
    print(f"Prompt: '{args.prompt}'")
    print(f"{'='*50}")
    for i in range(len(scores_nms)):
        bx = boxes_pixel[i]
        print(f"  [{i}] score={scores_nms[i]:.3f}  "
              f"box=({bx[0]:.0f}, {bx[1]:.0f}, {bx[2]:.0f}, {bx[3]:.0f})")

    # ---- Visualization ----
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    vis = draw_results(img_bgr, boxes_pixel, masks_resized, scores_nms)
    cv2.imwrite(args.output, vis)
    print(f"\nSaved visualization to: {args.output}")


if __name__ == "__main__":
    main()
