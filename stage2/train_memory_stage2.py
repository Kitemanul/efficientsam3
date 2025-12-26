#!/usr/bin/env python
# --------------------------------------------------------
# Stage 2: Train Efficient Memory Bank
# --------------------------------------------------------

"""
Training script for Stage 2 memory distillation.

This script trains:
1. PerceiverResampler for memory compression
2. EfficientMemoryAttention for fusing memory with current frame

The training uses:
- Pre-computed trunk embeddings (from save_video_embeddings_stage2.py)
- Frozen SAM3 FPN + decoder for generating predictions
- Distillation loss on memory features + task losses

Usage:
    python train_memory_stage2.py \
        --config configs/efficient_memory.yaml \
        --output_dir output/stage2_memory
"""

import os
import sys
import argparse
import logging
import json
import math
from pathlib import Path
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stage2.config import get_config, update_config_from_file
from stage2.model import SAM3MemoryTeacher, SAM3MemoryStudent
from stage2.data.sav_dataset import build_loader


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, 1, H, W] - predicted logits
            targets: [B, 1, H, W] - binary ground truth
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, 1, H, W] - predicted logits
            targets: [B, 1, H, W] - binary ground truth
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            inputs.sum() + targets.sum() + self.smooth
        )
        
        return 1 - dice


class IoULoss(nn.Module):
    """IoU loss for segmentation."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: [B, 1, H, W] - predicted logits
            targets: [B, 1, H, W] - binary ground truth
        """
        inputs = torch.sigmoid(inputs)
        inputs = inputs.flatten()
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()
        total = inputs.sum() + targets.sum()
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1 - iou


class DistillationLoss(nn.Module):
    """
    Combined loss for memory distillation.
    
    Includes:
    1. Focal loss on mask predictions
    2. Dice loss on mask predictions
    3. IoU loss on mask predictions
    4. MSE loss on memory features (distillation)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Task losses
        self.focal_loss = FocalLoss(
            alpha=config.DISTILL.FOCAL_ALPHA,
            gamma=config.DISTILL.FOCAL_GAMMA,
        )
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
        
        # Loss weights
        self.focal_weight = config.DISTILL.FOCAL_WEIGHT
        self.dice_weight = config.DISTILL.DICE_WEIGHT
        self.iou_weight = config.DISTILL.IOU_WEIGHT
        self.mse_weight = config.DISTILL.MSE_WEIGHT
    
    def forward(
        self,
        pred_masks: torch.Tensor,
        target_masks: torch.Tensor,
        student_features: torch.Tensor = None,
        teacher_features: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined loss.
        
        Args:
            pred_masks: Predicted masks from student
            target_masks: Ground truth masks
            student_features: Student memory features
            teacher_features: Teacher memory features (for distillation)
            
        Returns:
            dict with individual and total losses
        """
        losses = {}
        
        # Task losses
        losses['focal'] = self.focal_weight * self.focal_loss(
            pred_masks, target_masks
        )
        losses['dice'] = self.dice_weight * self.dice_loss(
            pred_masks, target_masks
        )
        losses['iou'] = self.iou_weight * self.iou_loss(
            pred_masks, target_masks
        )
        
        # Distillation loss on memory features
        if student_features is not None and teacher_features is not None:
            losses['mse'] = self.mse_weight * F.mse_loss(
                student_features, teacher_features
            )
        else:
            losses['mse'] = torch.tensor(0.0, device=pred_masks.device)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    """
    Cosine learning rate schedule with linear warmup.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_lr_ratio,
            0.5 * (1.0 + math.cos(math.pi * progress))
        )
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class Trainer:
    """
    Trainer for Stage 2 memory distillation.
    """
    
    def __init__(self, config, output_dir: str, resume: str = None):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Setup logging
        self.setup_logging()
        
        # Build models
        self.build_models()
        
        # Build data loaders
        self.build_dataloaders()
        
        # Build optimizer and scheduler
        self.build_optimizer()
        
        # Build loss function
        self.loss_fn = DistillationLoss(config)
        
        # Gradient scaler for mixed precision
        self.scaler = GradScaler(enabled=config.TRAIN.AMP)
        
        # Tensorboard
        self.writer = SummaryWriter(self.output_dir / 'tensorboard')
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf')
        
        # Resume if specified
        if resume:
            self.resume_checkpoint(resume)
        
        logger.info(f"Training with device: {self.device}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def setup_logging(self):
        """Setup file logging."""
        log_file = self.output_dir / 'train.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    def build_models(self):
        """Build teacher and student models."""
        logger.info("Building models...")

        # Student model (trainable memory modules)
        self.student = SAM3MemoryStudent(self.config).to(self.device)

        # Teacher trunk extractor is only needed when training on raw frames
        if not self.config.DATA.USE_PRECOMPUTED:
            self.teacher = SAM3MemoryTeacher(
                checkpoint_path=self.config.MODEL.RESUME,
            ).to(self.device)
            self.teacher.eval()
        else:
            self.teacher = None

        # Frozen adapters to map trunk embeddings (1024-d) into:
        # - query space (256-d) for memory attention
        # - memory space (64-d) for Perceiver compression
        trunk_dim = getattr(self.config.DISTILL, "EMBED_DIM", 1024)
        self.query_adapter = nn.Conv2d(
            trunk_dim,
            self.config.MODEL.MEMORY_ATTENTION.D_MODEL,
            kernel_size=1,
            bias=False,
        ).to(self.device)
        self.mem_adapter = nn.Conv2d(
            trunk_dim,
            self.config.MODEL.MEMORY_ENCODER.OUT_DIM,
            kernel_size=1,
            bias=False,
        ).to(self.device)

        for p in self.query_adapter.parameters():
            p.requires_grad = False
        for p in self.mem_adapter.parameters():
            p.requires_grad = False
        
        # Log parameter counts
        param_counts = self.student.count_parameters()
        logger.info("Student model parameter counts:")
        for name, count in param_counts.items():
            logger.info(f"  {name}: {count:,}")
    
    def build_dataloaders(self):
        """Build training and validation data loaders."""
        logger.info("Building data loaders...")
        
        self.train_loader = build_loader(
            self.config,
            is_train=True,
            use_precomputed=self.config.DATA.USE_PRECOMPUTED,
        )
        
        self.val_loader = build_loader(
            self.config,
            is_train=False,
            use_precomputed=self.config.DATA.USE_PRECOMPUTED,
        )
        
        logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        logger.info(f"Val samples: {len(self.val_loader.dataset)}")
    
    def build_optimizer(self):
        """Build optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = self.student.get_trainable_params()
        
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.TRAIN.LR,
            weight_decay=self.config.TRAIN.WEIGHT_DECAY,
            betas=(0.9, 0.999),
        )
        
        # Learning rate scheduler
        total_steps = self.config.TRAIN.MAX_STEPS
        warmup_steps = self.config.TRAIN.WARMUP_STEPS
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            min_lr_ratio=self.config.TRAIN.MIN_LR / self.config.TRAIN.LR,
        )
        
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
    
    def save_checkpoint(self, name: str = 'latest', is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'model': self.student.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        # Save latest
        torch.save(checkpoint, self.output_dir / f'ckpt_{name}.pth')
        
        # Save best if applicable
        if is_best:
            torch.save(checkpoint, self.output_dir / 'ckpt_best.pth')
    
    def resume_checkpoint(self, path: str):
        """Resume from checkpoint."""
        logger.info(f"Resuming from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.student.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])
        
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        
        logger.info(f"Resumed at step {self.global_step}, epoch {self.epoch}")
    
    def train_step(self, batch: dict) -> dict:
        """
        Single training step.
        
        Args:
            batch: Dict with video frames, masks, etc.
            
        Returns:
            Dict with loss values
        """
        self.student.train()
        
        # Either use precomputed trunk embeddings or compute them from raw frames
        if 'embeddings' in batch:
            # Precomputed embeddings (cannot use Teacher Forcing with masks unless masks are also precomputed)
            # Fallback to original behavior (distilling trunk features only - suboptimal)
            teacher_features = batch['embeddings'].to(self.device)  # [B, T, 1024, 72, 72]
            teacher_conditioned = None
        else:
            frames = batch['frames'].to(self.device)  # [B, T, 3, 1008, 1008]
            masks = batch.get('masks')
            
            if masks is not None:
                masks = masks.to(self.device)
                # Handle multiple objects: select first one for now
                # masks: [B, T, N_obj, H, W] -> [B, T, 1, H, W]
                if masks.dim() == 5:
                    masks = masks[:, :, 0:1]
            
            if self.teacher is None:
                raise RuntimeError("Teacher model is required when training without precomputed embeddings.")
            
            with torch.no_grad():
                teacher_out = self.teacher(frames, masks)
                teacher_features = teacher_out['trunk_features']  # [B, T, 1024, 72, 72]
                teacher_conditioned = teacher_out.get('conditioned_features') # [B, T, C, H, W]

        B, T, C, H, W = teacher_features.shape
        
        # Process video frame by frame
        memory_list = []
        all_losses = []
        
        for t in range(T):
            trunk_t = teacher_features[:, t]  # [B, 1024, 72, 72]
            # Frozen projections into query/memory spaces
            query_feat = self.query_adapter(trunk_t).detach()
            mem_feat = self.mem_adapter(trunk_t).detach()
            
            # Prepare memory from previous frames
            if t == 0:
                memory_features = None
                memory_pos_enc = None
                is_first = True
            else:
                memory_features = torch.stack(memory_list[-self.config.MODEL.TRACKER.NUM_MASKMEM:], dim=1)
                is_first = False
            
            # Student forward
            with autocast(enabled=self.config.TRAIN.AMP):
                student_out = self.student(
                    current_features=query_feat,
                    current_pos_enc=None,
                    memory_features=memory_features,
                    memory_pos_enc=None,
                    current_mem_features=mem_feat,
                    current_mem_pos_enc=None,
                    is_first_frame=is_first,
                )
            
            # Store compressed memory for next iteration
            memory_list.append(mem_feat)
            
            # Compute loss (simplified - just MSE on conditioned features)
            if t > 0:  # Skip first frame (no memory conditioning)
                with autocast(enabled=self.config.TRAIN.AMP):
                    # Target: Teacher's conditioned features if available, else query_feat (fallback)
                    target = teacher_conditioned[:, t] if teacher_conditioned is not None else query_feat
                    
                    loss = F.mse_loss(
                        student_out['conditioned_features'],
                        target,
                    )
                all_losses.append(loss)
        
        # Average losses across frames
        if len(all_losses) > 0:
            total_loss = torch.stack(all_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        
        # Gradient clipping
        if self.config.TRAIN.GRAD_CLIP > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.student.get_trainable_params(),
                self.config.TRAIN.GRAD_CLIP,
            )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        
        return {
            'total': total_loss.item(),
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.student.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            if 'embeddings' in batch:
                teacher_features = batch['embeddings'].to(self.device)
            else:
                frames = batch['frames'].to(self.device)
                if self.teacher is None:
                    raise RuntimeError("Teacher model is required when validating without precomputed embeddings.")
                teacher_out = self.teacher(frames)
                teacher_features = teacher_out['trunk_features']

            B, T, C, H, W = teacher_features.shape
            
            # Process video
            memory_list = []
            batch_losses = []
            
            for t in range(T):
                trunk_t = teacher_features[:, t]
                query_feat = self.query_adapter(trunk_t)
                mem_feat = self.mem_adapter(trunk_t)
                
                if t == 0:
                    memory_features = None
                    memory_pos_enc = None
                    is_first = True
                else:
                    memory_features = torch.stack(memory_list[-self.config.MODEL.TRACKER.NUM_MASKMEM:], dim=1)
                    is_first = False
                
                student_out = self.student(
                    current_features=query_feat,
                    current_pos_enc=None,
                    memory_features=memory_features,
                    memory_pos_enc=None,
                    current_mem_features=mem_feat,
                    current_mem_pos_enc=None,
                    is_first_frame=is_first,
                )
                
                memory_list.append(mem_feat)
                
                if t > 0:
                    loss = F.mse_loss(
                        student_out['conditioned_features'],
                        query_feat,
                    )
                    batch_losses.append(loss)
            
            if len(batch_losses) > 0:
                total_loss += torch.stack(batch_losses).mean().item()
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        
        return {'val_loss': avg_loss}
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        max_steps = self.config.TRAIN.MAX_STEPS
        log_interval = self.config.TRAIN.LOG_INTERVAL
        save_interval = self.config.TRAIN.SAVE_INTERVAL
        val_interval = self.config.TRAIN.VAL_INTERVAL
        
        running_loss = 0.0
        num_steps = 0
        
        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break
                
                # Train step
                losses = self.train_step(batch)
                
                running_loss += losses['total']
                num_steps += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % log_interval == 0:
                    avg_loss = running_loss / num_steps
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    logger.info(
                        f"Step {self.global_step}/{max_steps} | "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.6f}"
                    )
                    
                    self.writer.add_scalar('train/loss', avg_loss, self.global_step)
                    self.writer.add_scalar('train/lr', lr, self.global_step)
                    
                    running_loss = 0.0
                    num_steps = 0
                
                # Validation
                if self.global_step % val_interval == 0:
                    val_metrics = self.validate()
                    
                    logger.info(
                        f"Validation at step {self.global_step} | "
                        f"Val Loss: {val_metrics['val_loss']:.4f}"
                    )
                    
                    self.writer.add_scalar(
                        'val/loss', val_metrics['val_loss'], self.global_step
                    )
                    
                    # Check if best
                    is_best = val_metrics['val_loss'] < self.best_metric
                    if is_best:
                        self.best_metric = val_metrics['val_loss']
                        logger.info(f"New best validation loss: {self.best_metric:.4f}")
                    
                    self.save_checkpoint('latest', is_best=is_best)
                
                # Periodic save
                if self.global_step % save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}')
            
            self.epoch += 1
        
        # Final save
        self.save_checkpoint('final')
        logger.info("Training complete!")
        
        # Close tensorboard
        self.writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Stage 2 memory distillation'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='',
        help='Path to config file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='output/stage2_memory',
        help='Output directory'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        'opts',
        nargs=argparse.REMAINDER,
        help='Additional config options'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get config
    config = get_config()
    if args.config:
        config.defrost()
        config = update_config_from_file(config, args.config)
    if args.opts:
        config.defrost()
        config.merge_from_list(args.opts)
    config.freeze()
    
    # Save config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config.yaml', 'w') as f:
        f.write(config.dump())
    
    # Create trainer and train
    trainer = Trainer(config, args.output_dir, args.resume)
    trainer.train()


if __name__ == '__main__':
    main()
