#!/usr/bin/env python3
"""Test ONNX exported models with point, box, and text prompts."""

import argparse
import os
import sys
import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sam3"))
from sam3.model.tokenizer_ve import SimpleTokenizer


def preprocess_image(image_path, target_size=1008):
    """Resize and normalize image to [1, 3, H, W] float32."""
    img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = img.size
    img = img.resize((target_size, target_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis], orig_w, orig_h


def tokenize_text(text, bpe_path):
    """Tokenize text to [1, 77] int64."""
    tokenizer = SimpleTokenizer(bpe_path=bpe_path)
    tokens = tokenizer(text, context_length=77)
    return tokens.numpy().astype(np.int64)


def nms(boxes, scores, iou_threshold=0.5):
    """NMS on xyxy boxes."""
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
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return np.array(keep, dtype=np.int64)


PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
]


def save_result(image_path, scores, boxes_xyxy, mask_logits, output_path,
                score_threshold, nms_threshold, top_k=10, alpha=0.5):
    """Filter by threshold + NMS, then save visualization."""
    pil_image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_image.size

    valid = scores > score_threshold
    if not valid.any():
        print(f"  No detections above threshold {score_threshold}")
        pil_image.save(output_path)
        print(f"  Saved (no detections): {output_path}")
        return

    scores_v = scores[valid]
    boxes_v = boxes_xyxy[valid]
    masks_v = mask_logits[valid]

    keep = nms(boxes_v, scores_v, iou_threshold=nms_threshold)
    scores_k = scores_v[keep]
    boxes_k = boxes_v[keep]
    masks_k = masks_v[keep]

    # Top-K
    if len(scores_k) > top_k:
        topk_idx = scores_k.argsort()[::-1][:top_k]
        scores_k = scores_k[topk_idx]
        boxes_k = boxes_k[topk_idx]
        masks_k = masks_k[topk_idx]

    print(f"  After filter (>{score_threshold}) + NMS + top-{top_k}: {len(scores_k)} detections")

    # Convert boxes to pixel coords
    boxes_pixel = boxes_k.copy()
    boxes_pixel[:, [0, 2]] *= orig_w
    boxes_pixel[:, [1, 3]] *= orig_h

    # Sigmoid + resize masks
    masks_prob = 1.0 / (1.0 + np.exp(-masks_k))
    masks_resized = np.zeros((len(masks_prob), orig_h, orig_w), dtype=bool)
    for i in range(len(masks_prob)):
        resized = cv2.resize(masks_prob[i], (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        masks_resized[i] = resized > 0.5

    # Draw
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    vis = img_bgr.copy()
    for i in range(len(scores_k)):
        color = PALETTE[i % len(PALETTE)]
        mask = masks_resized[i]
        box = boxes_pixel[i].astype(int)

        colored = np.zeros_like(vis, dtype=np.uint8)
        colored[:] = color
        vis = np.where(mask[..., None],
                       (vis * (1 - alpha) + colored * alpha).astype(np.uint8), vis)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 2)
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, 2)
        label = f"{scores_k[i]:.3f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (box[0], box[1] - th - 6), (box[0] + tw + 4, box[1]), color, -1)
        cv2.putText(vis, label, (box[0] + 2, box[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        print(f"  [{i}] score={scores_k[i]:.4f} box=({box[0]}, {box[1]}, {box[2]}, {box[3]})")

    cv2.imwrite(output_path, vis)
    print(f"  Saved: {output_path}")


def run_decoder(dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
                text_features, text_mask, point_coords, point_labels,
                box_coords, box_labels):
    """Run decoder. Returns (scores[200], boxes_xyxy[200,4], mask_logits[200,288,288])."""
    scores, boxes_xyxy, mask_logits = dec_sess.run(None, {
        "feat_4x": feat_4x, "feat_2x": feat_2x,
        "feat_1x": feat_1x, "pos_1x": pos_1x,
        "text_features": text_features, "text_mask": text_mask,
        "point_coords": point_coords, "point_labels": point_labels,
        "box_coords": box_coords, "box_labels": box_labels,
    })
    print(f"  scores top5: {np.sort(scores[0])[::-1][:5]}")
    return scores[0], boxes_xyxy[0], mask_logits[0]


def test_point(dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
               text_features, text_mask, image_path, output_dir,
               orig_w, orig_h, point_xy, score_threshold, nms_threshold, top_k):
    """Test point prompt."""
    print("\n" + "=" * 60)
    print("Point prompt")
    print("=" * 60)

    px, py = point_xy
    nx, ny = px / orig_w, py / orig_h
    print(f"  Point pixel: ({px:.0f}, {py:.0f}) -> normalized: ({nx:.3f}, {ny:.3f})")

    point_coords = np.array([[nx, ny]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int64)
    box_coords = np.array([[0.5, 0.5, 0.01, 0.01]], dtype=np.float32)
    box_labels = np.array([0], dtype=np.int64)

    scores, boxes_xyxy, mask_logits = run_decoder(
        dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
        text_features, text_mask, point_coords, point_labels,
        box_coords, box_labels)
    save_result(image_path, scores, boxes_xyxy, mask_logits,
                os.path.join(output_dir, "point_prompt_mask.png"),
                score_threshold, nms_threshold, top_k)


def test_box(dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
             text_features, text_mask, image_path, output_dir,
             orig_w, orig_h, box_xyxy, score_threshold, nms_threshold, top_k):
    """Test box prompt."""
    print("\n" + "=" * 60)
    print("Box prompt")
    print("=" * 60)

    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0 / orig_w
    cy = (y1 + y2) / 2.0 / orig_h
    w = (x2 - x1) / orig_w
    h = (y2 - y1) / orig_h
    print(f"  Box pixel (xyxy): ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
    print(f"  Box normalized (cxcywh): ({cx:.3f}, {cy:.3f}, {w:.3f}, {h:.3f})")

    box_coords = np.array([[cx, cy, w, h]], dtype=np.float32)
    box_labels = np.array([1], dtype=np.int64)
    point_coords = np.array([[0.5, 0.5]], dtype=np.float32)
    point_labels = np.array([0], dtype=np.int64)

    scores, boxes_xyxy_out, mask_logits = run_decoder(
        dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
        text_features, text_mask, point_coords, point_labels,
        box_coords, box_labels)
    save_result(image_path, scores, boxes_xyxy_out, mask_logits,
                os.path.join(output_dir, "box_prompt_mask.png"),
                score_threshold, nms_threshold, top_k)


def test_text(dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
              text_features, text_mask, image_path, output_dir,
              score_threshold, nms_threshold, top_k):
    """Test text-only prompt (dummy geometry)."""
    print("\n" + "=" * 60)
    print("Text-only prompt (dummy geometry)")
    print("=" * 60)

    point_coords = np.zeros((1, 2), dtype=np.float32)
    point_labels = np.zeros(1, dtype=np.int64)
    box_coords = np.zeros((1, 4), dtype=np.float32)
    box_labels = np.zeros(1, dtype=np.int64)

    scores, boxes_xyxy, mask_logits = run_decoder(
        dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
        text_features, text_mask, point_coords, point_labels,
        box_coords, box_labels)
    save_result(image_path, scores, boxes_xyxy, mask_logits,
                os.path.join(output_dir, "text_only_mask.png"),
                score_threshold, nms_threshold, top_k)


def parse_float_list(s):
    """Parse comma-separated floats."""
    return [float(x) for x in s.split(",")]


def main():
    parser = argparse.ArgumentParser(description="Test ONNX models with different prompts")
    parser.add_argument("--prompt", choices=["point", "box", "text", "all"], default="all",
                        help="Prompt type to test (default: all)")
    parser.add_argument("--image", default="sam3/assets/images/truck.jpg")
    parser.add_argument("--text", default="truck", help="Text prompt")
    parser.add_argument("--point", type=parse_float_list, default=None,
                        help="Point in pixel coords: x,y (e.g. 500,300)")
    parser.add_argument("--box", type=parse_float_list, default=None,
                        help="Box in pixel coords: x1,y1,x2,y2 (e.g. 50,100,950,480)")
    parser.add_argument("--onnx-dir", default="exports_repvit_m0_9")
    parser.add_argument("--output-dir", default="test_onnx_output")
    parser.add_argument("--score-threshold", type=float, default=0.0015)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bpe_path = os.path.join("sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")

    print("Loading ONNX models...")
    ie_sess = ort.InferenceSession(os.path.join(args.onnx_dir, "image_encoder.onnx"),
                                    providers=["CPUExecutionProvider"])
    te_sess = ort.InferenceSession(os.path.join(args.onnx_dir, "text_encoder.onnx"),
                                    providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(os.path.join(args.onnx_dir, "decoder.onnx"),
                                     providers=["CPUExecutionProvider"])

    print(f"\nPreprocessing: {args.image}")
    image, orig_w, orig_h = preprocess_image(args.image)
    print(f"  Original size: {orig_w}x{orig_h}")

    print("\nRunning Image Encoder...")
    feat_4x, feat_2x, feat_1x, pos_1x = ie_sess.run(None, {"images": image})
    print(f"  feat_1x: {feat_1x.shape}, feat_4x: {feat_4x.shape}")

    print(f"\nRunning Text Encoder (prompt='{args.text}')...")
    token_ids = tokenize_text(args.text, bpe_path)
    text_features, text_mask = te_sess.run(None, {"token_ids": token_ids})
    print(f"  text_features: {text_features.shape}")

    common = (dec_sess, feat_4x, feat_2x, feat_1x, pos_1x,
              text_features, text_mask, args.image, args.output_dir)

    if args.prompt in ("point", "all"):
        point_xy = args.point if args.point else [orig_w * 0.5, orig_h * 0.55]
        test_point(*common, orig_w, orig_h, point_xy,
                   args.score_threshold, args.nms_threshold, args.top_k)

    if args.prompt in ("box", "all"):
        box_xyxy = args.box if args.box else [
            orig_w * 0.05, orig_h * 0.20, orig_w * 0.95, orig_h * 0.85]
        test_box(*common, orig_w, orig_h, box_xyxy,
                 args.score_threshold, args.nms_threshold, args.top_k)

    if args.prompt in ("text", "all"):
        test_text(*common, args.score_threshold, args.nms_threshold, args.top_k)

    print(f"\n{'=' * 60}")
    print(f"Done. Check {args.output_dir}/ for visualizations.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
