#!/usr/bin/env python
# --------------------------------------------------------
# Stage 2: Convert Trained Memory Weights to SAM3 Format
# --------------------------------------------------------

"""
Convert trained Stage 2 memory weights to SAM3 tracker format.

This script:
1. Loads trained Perceiver and EfficientMemoryAttention weights
2. Merges them with SAM3 tracker weights
3. Saves a complete checkpoint for inference

Usage:
    python convert_memory_weights_stage2.py \
        --stage2_weights output/stage2_memory/ckpt_best.pth \
        --sam3_weights checkpoints/sam3_tracker.pth \
        --output_path output/efficient_sam3_memory.pth
"""

import os
import sys
import argparse
from pathlib import Path
from collections import OrderedDict

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def convert_perceiver_weights(
    stage2_state: dict,
    prefix_mapping: dict = None,
) -> dict:
    """
    Convert Perceiver weights to SAM3 format.
    
    Args:
        stage2_state: Stage 2 model state dict
        prefix_mapping: Optional mapping from stage2 keys to sam3 keys
        
    Returns:
        Converted state dict
    """
    converted = {}
    
    # Default mapping (Stage 2 modules live under `tracker.*` in SAM3 checkpoints)
    if prefix_mapping is None:
        prefix_mapping = {
            'spatial_perceiver.': 'tracker.spatial_perceiver.',
        }
    
    for key, value in stage2_state.items():
        if key.startswith('spatial_perceiver.'):
            new_key = key.replace(
                'spatial_perceiver.',
                prefix_mapping.get('spatial_perceiver.', 'memory_encoder.perceiver.')
            )
            converted[new_key] = value
    
    return converted


def convert_memory_attention_weights(
    stage2_state: dict,
    prefix_mapping: dict = None,
) -> dict:
    """
    Convert EfficientMemoryAttention weights to SAM3 format.
    
    Args:
        stage2_state: Stage 2 model state dict
        prefix_mapping: Optional mapping
        
    Returns:
        Converted state dict
    """
    converted = {}
    
    # Default mapping (Stage 2 memory attention is stored under `tracker.memory_attention.*`)
    if prefix_mapping is None:
        prefix_mapping = {
            'memory_attention.': 'tracker.memory_attention.',
        }
    
    for key, value in stage2_state.items():
        if key.startswith('memory_attention.'):
            new_key = key.replace(
                'memory_attention.',
                prefix_mapping.get('memory_attention.', 'memory_attention.')
            )
            converted[new_key] = value
    
    return converted


def convert_temporal_encoding_weights(stage2_state: dict) -> dict:
    """
    Convert temporal position encoding weights.
    
    Args:
        stage2_state: Stage 2 model state dict
        
    Returns:
        Converted state dict
    """
    converted = {}
    
    temporal_keys = ['maskmem_tpos_enc', 'no_mem_embed', 'no_mem_pos_enc']
    
    for key, value in stage2_state.items():
        if key in temporal_keys:
            converted[f"tracker.{key}"] = value
    
    return converted


def convert_memory_projection_weights(stage2_state: dict) -> dict:
    """Convert memory projection weights (64 -> 256) to tracker prefix."""
    converted = {}
    for key, value in stage2_state.items():
        if key.startswith("mem_proj.") or key.startswith("mem_pos_proj."):
            converted[f"tracker.{key}"] = value
    return converted


def merge_weights(
    sam3_state: dict,
    stage2_converted: dict,
    replace_existing: bool = True,
) -> dict:
    """
    Merge Stage 2 weights with SAM3 weights.
    
    Args:
        sam3_state: SAM3 tracker state dict
        stage2_converted: Converted Stage 2 weights
        replace_existing: Whether to replace existing keys
        
    Returns:
        Merged state dict
    """
    merged = OrderedDict(sam3_state)
    
    for key, value in stage2_converted.items():
        if key in merged:
            if replace_existing:
                print(f"Replacing: {key}")
                merged[key] = value
            else:
                print(f"Skipping (exists): {key}")
        else:
            print(f"Adding new: {key}")
            merged[key] = value
    
    return merged


def verify_checkpoint(state_dict: dict, expected_keys: list = None):
    """
    Verify checkpoint has expected keys.
    
    Args:
        state_dict: State dict to verify
        expected_keys: List of expected key prefixes
    """
    if expected_keys is None:
        expected_keys = [
            'tracker.spatial_perceiver',
            'tracker.memory_attention',
            'tracker.mem_proj',
            'tracker.mem_pos_proj',
            'tracker.maskmem_tpos_enc',
        ]
    
    found_keys = {prefix: False for prefix in expected_keys}
    
    for key in state_dict.keys():
        for prefix in expected_keys:
            if key.startswith(prefix) or key == prefix:
                found_keys[prefix] = True
    
    print("\nCheckpoint verification:")
    for prefix, found in found_keys.items():
        status = "✓" if found else "✗"
        print(f"  {status} {prefix}")
    
    return all(found_keys.values())


def convert_weights(
    stage2_weights_path: str,
    sam3_weights_path: str = None,
    output_path: str = None,
    verify: bool = True,
) -> dict:
    """
    Main conversion function.
    
    Args:
        stage2_weights_path: Path to Stage 2 checkpoint
        sam3_weights_path: Path to SAM3 weights (optional)
        output_path: Path to save converted weights
        verify: Whether to verify output
        
    Returns:
        Converted state dict
    """
    print(f"Loading Stage 2 weights from: {stage2_weights_path}")

    # Load Stage 2 checkpoint (may contain a YACS config object, so PyTorch 2.6+ weights_only=True can fail)
    try:
        stage2_ckpt = torch.load(stage2_weights_path, map_location='cpu')
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" in msg or "WeightsUnpickler" in msg:
            print(
                "Warning: Stage 2 checkpoint contains non-tensor objects; reloading with weights_only=False.\n"
                "Only do this if you trust the checkpoint source."
            )
            stage2_ckpt = torch.load(stage2_weights_path, map_location='cpu', weights_only=False)
        else:
            raise
    
    if 'model' in stage2_ckpt:
        stage2_state = stage2_ckpt['model']
    else:
        stage2_state = stage2_ckpt
    
    print(f"Stage 2 keys: {len(stage2_state)}")
    
    # Convert each component
    perceiver_converted = convert_perceiver_weights(stage2_state)
    memory_attn_converted = convert_memory_attention_weights(stage2_state)
    proj_converted = convert_memory_projection_weights(stage2_state)
    temporal_converted = convert_temporal_encoding_weights(stage2_state)
    
    # Combine converted weights
    stage2_converted = {}
    stage2_converted.update(perceiver_converted)
    stage2_converted.update(memory_attn_converted)
    stage2_converted.update(proj_converted)
    stage2_converted.update(temporal_converted)
    
    print(f"\nConverted Stage 2 keys: {len(stage2_converted)}")
    
    # Merge with SAM3 if provided
    if sam3_weights_path:
        print(f"\nLoading SAM3 weights from: {sam3_weights_path}")
        
        # SAM3 weights are a pure tensor state dict, safe to load with default weights_only behavior.
        sam3_ckpt = torch.load(sam3_weights_path, map_location='cpu')
        
        if 'model' in sam3_ckpt:
            sam3_state = sam3_ckpt['model']
        else:
            sam3_state = sam3_ckpt
        
        print(f"SAM3 keys: {len(sam3_state)}")
        
        # Merge
        merged_state = merge_weights(sam3_state, stage2_converted)
        final_state = merged_state
        
        print(f"\nMerged keys: {len(final_state)}")
    else:
        final_state = stage2_converted
    
    # Verify
    if verify:
        verify_checkpoint(final_state)
    
    # Save
    if output_path:
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save a weights-only checkpoint (same format as `sam3.pt`) for easy loading.
        torch.save(final_state, output_path)
        print(f"\nSaved to: {output_path}")
    
    return final_state


def main():
    parser = argparse.ArgumentParser(
        description='Convert Stage 2 memory weights to SAM3 format'
    )
    parser.add_argument(
        '--stage2_weights',
        type=str,
        required=True,
        help='Path to Stage 2 checkpoint'
    )
    parser.add_argument(
        '--sam3_weights',
        type=str,
        default=None,
        help='Path to SAM3 weights (optional, for merging)'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        required=True,
        help='Path to save converted weights'
    )
    parser.add_argument(
        '--no_verify',
        action='store_true',
        help='Skip verification'
    )
    
    args = parser.parse_args()
    
    convert_weights(
        stage2_weights_path=args.stage2_weights,
        sam3_weights_path=args.sam3_weights,
        output_path=args.output_path,
        verify=not args.no_verify,
    )


if __name__ == '__main__':
    main()
