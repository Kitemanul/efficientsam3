#!/usr/bin/env python
# --------------------------------------------------------
# Stage 2: Merge Text Encoder into Trained Model
# --------------------------------------------------------

"""
Merge the text encoder from the original SAM3 checkpoint into the trained Stage 2 model.
Since Stage 2 training only involves the vision backbone and memory modules, 
we need to restore the text encoder for full SAM3 functionality (e.g., text prompts).

Usage:
    python stage2/convert_memory_weights_stage2.py \
        --trained_checkpoint output/stage2_checkpoints/model_final.pth \
        --sam3_checkpoint sam3_checkpoints/sam3.pt \
        --output_path output/efficient_sam3_stage2_full.pth
"""

import argparse
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Merge Text Encoder into Trained Model")
    parser.add_argument("--trained_checkpoint", type=str, required=True, help="Path to the trained Stage 2 checkpoint")
    parser.add_argument("--sam3_checkpoint", type=str, required=True, help="Path to the original SAM3 checkpoint")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the merged checkpoint")
    args = parser.parse_args()

    logger.info(f"Loading trained checkpoint from {args.trained_checkpoint}...")
    trained_state_dict = torch.load(args.trained_checkpoint, map_location="cpu")
    if "model" in trained_state_dict:
        trained_state_dict = trained_state_dict["model"]

    logger.info(f"Loading SAM3 checkpoint from {args.sam3_checkpoint}...")
    sam3_state_dict = torch.load(args.sam3_checkpoint, map_location="cpu")
    if "model" in sam3_state_dict:
        sam3_state_dict = sam3_state_dict["model"]

    # Identify text encoder keys in SAM3
    # Typically under 'backbone.language_backbone' or similar depending on how SAM3VLBackbone is used
    # In EfficientSAM3Stage2, self.backbone is the SAM3VLBackbone.
    # So keys should be 'backbone.language_backbone.*'
    
    text_keys = [k for k in sam3_state_dict.keys() if "language_backbone" in k or "text_encoder" in k]
    logger.info(f"Found {len(text_keys)} text encoder keys in SAM3 checkpoint.")

    # Merge
    merged_state_dict = trained_state_dict.copy()
    for k in text_keys:
        if k not in merged_state_dict:
            merged_state_dict[k] = sam3_state_dict[k]
        else:
            # If it exists (e.g. maybe initialized but not trained), we overwrite it with the original weights
            # just to be safe, assuming we didn't train text encoder.
            merged_state_dict[k] = sam3_state_dict[k]

    logger.info(f"Saving merged checkpoint to {args.output_path}...")
    torch.save({"model": merged_state_dict}, args.output_path)
    logger.info("Done.")

if __name__ == "__main__":
    main()
