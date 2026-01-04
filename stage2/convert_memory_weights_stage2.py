#!/usr/bin/env python3
"""
Merge Stage 2 trained memory weights into the full SAM3 checkpoint.

Usage:
python stage2/convert_memory_weights_stage2.py \
    --trained_checkpoint output/stage2_checkpoints/sav_smoke/efficient_sam3_stage2_final.pt \
    --sam3_checkpoint sam3_checkpoints/sam3.pt \
    --output_path output/stage2_checkpoints/sav_smoke/efficient_sam3_stage2_full.pth

The merge copies all matching parameter keys from the Stage 2 checkpoint
into the original SAM3 checkpoint, so the resulting file is drop-in for
full SAM3 inference with the new efficient memory modules.
"""

import argparse
import os
import torch


def load_state(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt, ckpt["model"]
    return ckpt, ckpt


def map_key(trained_key: str) -> str:
    """Map EfficientSAM3 training key names to SAM3 checkpoint key names."""
    if trained_key.startswith("backbone."):
        return f"detector.{trained_key}"
    return f"tracker.{trained_key}"


def main(args):
    assert os.path.exists(args.trained_checkpoint), f"Missing trained checkpoint {args.trained_checkpoint}"
    assert os.path.exists(args.sam3_checkpoint), f"Missing sam3 checkpoint {args.sam3_checkpoint}"
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    base_ckpt, base_state = load_state(args.sam3_checkpoint)
    trained_ckpt, trained_state = load_state(args.trained_checkpoint)

    merged_state = base_state.copy()
    updated = []
    skipped = []

    for k, v in trained_state.items():
        target = map_key(k)
        if target in merged_state and merged_state[target].shape == v.shape:
            merged_state[target] = v
            updated.append(target)
        else:
            skipped.append((k, target))

    print(f"Total keys in base: {len(base_state)} | trained: {len(trained_state)}")
    print(f"Updated {len(updated)} keys from trained checkpoint")
    if skipped:
        missing = sum(1 for _, tgt in skipped if tgt not in merged_state)
        shape_mismatch = len(skipped) - missing
        print(f"Skipped {len(skipped)} keys (missing={missing}, shape_mismatch={shape_mismatch})")

    # Save merged
    out = {}
    if isinstance(base_ckpt, dict):
        out.update({k: v for k, v in base_ckpt.items() if k != "model"})
    out["model"] = merged_state
    torch.save(out, args.output_path)
    print(f"Saved merged checkpoint to {args.output_path}")

    # Optional: quick diff on overlapping keys to confirm memory modules changed
    changed = []
    for k in updated:
        if not torch.equal(base_state[k], merged_state[k]):
            changed.append(k)
    print(f"Changed tensors (value diff) count: {len(changed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge Stage 2 memory weights into SAM3 checkpoint")
    parser.add_argument("--trained_checkpoint", type=str, required=True, help="Path to Stage 2 trained checkpoint")
    parser.add_argument("--sam3_checkpoint", type=str, required=True, help="Path to original SAM3 checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Where to save merged checkpoint")
    args = parser.parse_args()
    main(args)

