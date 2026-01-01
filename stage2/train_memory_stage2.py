#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# EfficientSAM3 Stage 2: Video Memory Training Script
#
# This script trains the memory modules (Perceiver, EfficientMemoryAttention)
# using SAM2-style training with video sequences.
#
# Usage:
#   python train_memory_stage2_v2.py --data_path data/sa-v/formatted --epochs 5

import os
import sys
import argparse
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from sam3.model.efficient_sam3_model_builder import (
    build_efficient_sam3_train,
    save_efficient_sam3_checkpoint,
)
from stage2.data.sav_dataset import SAVVideoDataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Loss Functions
# ============================================================================

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Compute Dice loss for binary segmentation."""
    pred_sigmoid = torch.sigmoid(pred)
    pred_flat = pred_sigmoid.flatten(1)
    target_flat = target.flatten(1).float()
    
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def sigmoid_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute Sigmoid Focal Loss for binary classification."""
    pred_sigmoid = torch.sigmoid(pred)
    target_float = target.float()
    
    bce = F.binary_cross_entropy_with_logits(pred, target_float, reduction='none')
    p_t = pred_sigmoid * target_float + (1 - pred_sigmoid) * (1 - target_float)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = alpha * target_float + (1 - alpha) * (1 - target_float)
    
    focal_loss = alpha_t * focal_weight * bce
    return focal_loss.mean()


def compute_loss(outputs: List[Dict[str, Any]], gt_masks_per_frame: Dict[int, torch.Tensor]) -> torch.Tensor:
    """
    Compute multi-frame loss.
    
    Args:
        outputs: List of per-frame outputs with 'pred_masks_high_res'
        gt_masks_per_frame: Dict mapping frame_idx to GT masks
        
    Returns:
        Total loss
    """
    total_loss = torch.tensor(0.0, device=outputs[0]["pred_masks_high_res"].device)
    num_frames = len(outputs)
    
    for t, out in enumerate(outputs):
        pred_masks = out["pred_masks_high_res"]
        gt_mask = gt_masks_per_frame[t]
        
        # Resize if needed
        if pred_masks.shape[-2:] != gt_mask.shape[-2:]:
            gt_mask = F.interpolate(
                gt_mask.float(),
                size=pred_masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        frame_loss = dice_loss(pred_masks, gt_mask) + sigmoid_focal_loss(pred_masks, gt_mask)
        total_loss = total_loss + frame_loss
    
    return total_loss / num_frames


# ============================================================================
# Distributed Training Setup
# ============================================================================

def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
        dist.barrier()
        return rank, local_rank, world_size
    else:
        return 0, 0, 1


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# Data Collation
# ============================================================================

def collate_video_batch(batch):
    """
    Collate video batch for SAM2-style training.
    
    Returns:
        Dict with:
        - flat_img_batch: [B*T, C, H, W] flattened images
        - masks: Dict[int, Tensor] mapping frame_idx to masks [B, 1, H, W]
        - flat_obj_to_img_idx: Dict[int, Tensor] mapping frame_idx to img indices
        - num_frames: Number of frames
    """
    B = len(batch)
    T = batch[0]['frames'].shape[0]
    
    # Stack frames: [B, T, C, H, W]
    frames = torch.stack([b['frames'] for b in batch], dim=0)
    
    # Stack masks: [B, T, num_objects, H, W]
    masks = torch.stack([b['masks'] for b in batch], dim=0)
    
    # Flatten images for backbone: [B*T, C, H, W]
    flat_img_batch = frames.flatten(0, 1)
    
    # Create masks dict per frame: {t: [B, 1, H, W]} as bool
    masks_per_frame = {}
    for t in range(T):
        # Take first object only for simplicity, convert to bool
        mask_t = masks[:, t, 0:1]
        masks_per_frame[t] = (mask_t > 0.5).bool()
    
    # Create img index mapping
    flat_obj_to_img_idx = {}
    for t in range(T):
        # For frame t, the image indices in flat_img_batch are [0*T+t, 1*T+t, ...]
        flat_obj_to_img_idx[t] = torch.arange(B) * T + t
    
    return {
        'flat_img_batch': flat_img_batch,
        'masks': masks_per_frame,
        'flat_obj_to_img_idx': flat_obj_to_img_idx,
        'num_frames': T,
    }


# ============================================================================
# Training Loop
# ============================================================================

def train(args):
    """Main training function."""
    rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)
    
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    
    # Create output directory
    if is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Output directory: {args.output_dir}")
    
    # ========================================================================
    # Build Model
    # ========================================================================
    if is_main_process:
        logger.info("Building EfficientSam3Train model...")
    
    model = build_efficient_sam3_train(
        # Backbone
        backbone_type="sam3",
        sam3_checkpoint=args.sam3_checkpoint if os.path.exists(args.sam3_checkpoint) else None,
        # Model architecture
        d_model=256,
        num_maskmem=7,
        image_size=1008,
        backbone_stride=14,
        # Memory attention
        num_heads=8,
        num_layers=2,
        dim_feedforward=1024,
        # Perceiver
        use_perceiver=True,
        perceiver_num_latents=64,
        perceiver_depth=2,
        perceiver_num_heads=8,
        # Training settings
        freeze_image_encoder=True,
        freeze_sam_heads=True,
        prob_to_use_pt_input_for_train=1.0,
        num_init_cond_frames_for_train=1,
        # Device
        device=device,
        load_pretrained_sam_heads=True,
    )
    
    # Get trainable parameters
    trainable_params = model.get_trainable_parameters()
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_count:,}")
        logger.info(f"Trainable ratio: {100.0 * trainable_count / total_params:.2f}%")
        
        # Log trainable modules
        logger.info("Trainable modules:")
        logger.info(f"  - maskmem_backbone: {sum(p.numel() for p in model.maskmem_backbone.parameters()):,}")
        if hasattr(model, 'spatial_perceiver') and model.spatial_perceiver is not None:
            logger.info(f"  - spatial_perceiver: {sum(p.numel() for p in model.spatial_perceiver.parameters()):,}")
        if hasattr(model, 'memory_attention') and model.memory_attention is not None:
            logger.info(f"  - memory_attention: {sum(p.numel() for p in model.memory_attention.parameters()):,}")
        if hasattr(model, 'mem_proj'):
            logger.info(f"  - mem_proj: {model.mem_proj.weight.numel():,}")
        if hasattr(model, 'mem_pos_proj'):
            logger.info(f"  - mem_pos_proj: {model.mem_pos_proj.weight.numel():,}")
    
    # Wrap in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # ========================================================================
    # Optimizer & Scheduler
    # ========================================================================
    optimizer = optim.AdamW(
        [p for p in trainable_params if p.requires_grad],
        lr=args.lr,
        weight_decay=0.1,
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    
    # ========================================================================
    # Dataset & DataLoader
    # ========================================================================
    if is_main_process:
        logger.info(f"Loading dataset from {args.data_path}...")
        if args.subset_fraction < 1.0:
            logger.info(f"Using {args.subset_fraction*100:.1f}% of available videos")
    
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data path not found: {args.data_path}")
    
    dataset = SAVVideoDataset(
        data_path=args.data_path,
        frames_per_video=args.frames_per_video,
        img_size=1008,
        subset_fraction=args.subset_fraction,
    )
    
    if is_main_process:
        logger.info(f"Dataset size: {len(dataset)}")
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    ) if world_size > 1 else None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=collate_video_batch,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    if is_main_process:
        logger.info("Starting training...")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.lr}")
        logger.info(f"  Frames per video: {args.frames_per_video}")
    
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        if is_main_process:
            logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                batch['flat_img_batch'] = batch['flat_img_batch'].to(device)
                for t in batch['masks']:
                    batch['masks'][t] = batch['masks'][t].to(device)
                for t in batch['flat_obj_to_img_idx']:
                    batch['flat_obj_to_img_idx'][t] = batch['flat_obj_to_img_idx'][t].to(device)
                
                # Forward pass
                if world_size > 1:
                    outputs = model.module.forward(batch)
                else:
                    outputs = model.forward(batch)
                
                # Compute loss
                loss = compute_loss(outputs, batch['masks'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [p for p in trainable_params if p.requires_grad],
                    max_norm=1.0
                )
                
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Logging
                if is_main_process and batch_idx % args.log_interval == 0:
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        f"  Step {batch_idx}/{len(dataloader)} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Avg Loss: {avg_loss:.4f} | "
                        f"LR: {scheduler.get_last_lr()[0]:.2e}"
                    )
                
            except Exception as e:
                import traceback
                logger.error(f"Error in batch {batch_idx}: {e}")
                traceback.print_exc()
                continue
        
        # End of epoch
        scheduler.step()
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        
        if is_main_process:
            logger.info(f"Epoch {epoch + 1} completed | Average Loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint (if save_every > 0 and epoch is multiple)
            if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
                checkpoint_path = os.path.join(
                    args.output_dir,
                    f"efficient_sam3_stage2_epoch_{epoch + 1}.pt"
                )
                
                model_to_save = model.module if world_size > 1 else model
                save_efficient_sam3_checkpoint(
                    model=model_to_save,
                    checkpoint_path=checkpoint_path,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=avg_epoch_loss,
                )
    
    # Final checkpoint
    if is_main_process:
        final_path = os.path.join(args.output_dir, "efficient_sam3_stage2_final.pt")
        model_to_save = model.module if world_size > 1 else model
        save_efficient_sam3_checkpoint(
            model=model_to_save,
            checkpoint_path=final_path,
            optimizer=optimizer,
            epoch=args.epochs,
        )
        logger.info(f"Training completed! Final checkpoint saved to {final_path}")
    
    cleanup_distributed()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EfficientSAM3 Stage 2 Training")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to video dataset")
    parser.add_argument("--frames_per_video", type=int, default=8,
                        help="Number of frames per video clip")
    parser.add_argument("--subset_fraction", type=float, default=1.0,
                        help="Fraction of videos to use (0.0-1.0). Use 0.01 for 1%% of data.")
    
    # Model
    parser.add_argument("--sam3_checkpoint", type=str, default="sam3_checkpoints/sam3.pt",
                        help="Path to SAM3 checkpoint")
    
    # Training
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="output/stage2_checkpoints",
                        help="Output directory for checkpoints")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval (steps)")
    parser.add_argument("--save_every", type=int, default=1,
                        help="Save checkpoint every N epochs (0 = only save final)")
    
    args = parser.parse_args()
    train(args)
