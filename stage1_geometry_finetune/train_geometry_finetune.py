"""Stage 1 Geometry Fine-tuning Training Script.

This script trains the student backbone using:
1. Embedding distillation (MSE loss on trunk outputs)
2. Mask distillation (BCE + Dice loss on predicted masks)

Usage:
    python train_geometry_finetune.py --cfg configs/repvit_m1_1.yaml \
        --data-path /path/to/sa1b \
        --pretrained /path/to/stage1_checkpoint.pth \
        --sam3-checkpoint /path/to/sam3.pt \
        --teacher-embed-path /path/to/teacher_embeddings
"""

import os
import sys
import time
import random
import argparse
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

# Add parent directory to path for imports
# Add stage1 first so its relative imports (from utils import ...) work correctly
sys.path.insert(0, str(Path(__file__).parent.parent / 'stage1'))
sys.path.insert(0, str(Path(__file__).parent.parent))

from stage1_geometry_finetune.config import get_config
from stage1_geometry_finetune.data import build_loader
from stage1_geometry_finetune.model import GeometryFinetuneModel, load_stage1_weights
from stage1_geometry_finetune.losses import GeometryFinetuningLoss, create_valid_mask

from stage1.logger import create_logger
from stage1.lr_scheduler import build_scheduler
from stage1.optimizer import build_optimizer
from stage1.my_meter import AverageMeter
from stage1.utils import save_checkpoint, load_checkpoint, NativeScalerWithGradNormCount


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from xyxy format to cxcywh format.
    
    Args:
        boxes: (B, N, 4) in xyxy format, normalized to [0, 1]
        
    Returns:
        boxes: (B, N, 4) in cxcywh format
    """
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return torch.stack([cx, cy, w, h], dim=-1)


@torch.no_grad()
def sample_refinement_points(
    student_mask_logits: torch.Tensor,
    teacher_mask_logits: torch.Tensor,
    valid_mask: torch.Tensor | None,
    num_points: int,
    threshold: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample refinement point prompts from teacher/student disagreement regions.

    Strategy (EdgeSAM-style):
    - False positive region (student=1, teacher=0) -> negative point (label=0)
    - False negative region (student=0, teacher=1) -> positive point (label=1)

    Args:
        student_mask_logits: (B, H, W) mask logits
        teacher_mask_logits: (B, H, W) mask logits
        valid_mask: Optional (B, 1, H, W) float mask for valid (non-padded) pixels
        num_points: number of points to sample per image
        threshold: logits threshold used to binarize masks

    Returns:
        points: (B, num_points, 2) normalized to [0,1] in (x,y) order
        labels: (B, num_points) long tensor in {0,1}
        point_mask: (B, num_points) bool tensor where True indicates padded/invalid points
    """
    assert student_mask_logits.ndim == 3 and teacher_mask_logits.ndim == 3
    assert student_mask_logits.shape == teacher_mask_logits.shape

    B, H, W = student_mask_logits.shape
    device = student_mask_logits.device

    if num_points <= 0:
        return (
            torch.zeros(B, 0, 2, device=device),
            torch.zeros(B, 0, dtype=torch.long, device=device),
            torch.zeros(B, 0, dtype=torch.bool, device=device),
        )

    # Binarize
    s = student_mask_logits > threshold
    t = teacher_mask_logits > threshold

    # Restrict to valid pixels (ignore padding)
    if valid_mask is not None:
        valid_bool = valid_mask.squeeze(1) > 0.5
        s = s & valid_bool
        t = t & valid_bool

    fp = s & (~t)  # student predicted 1 but teacher says 0 -> negative point
    fn = (~s) & t  # student predicted 0 but teacher says 1 -> positive point
    err = fp | fn

    points_out = torch.zeros(B, num_points, 2, device=device)
    labels_out = torch.zeros(B, num_points, dtype=torch.long, device=device)
    mask_out = torch.ones(B, num_points, dtype=torch.bool, device=device)  # True = masked

    for b in range(B):
        coords = err[b].nonzero(as_tuple=False)  # (K, 2) in (y, x)
        if coords.numel() == 0 or coords.shape[0] < num_points * 10:
            # Not enough disagreement pixels -> don't refine for this image
            continue

        sel = torch.randint(coords.shape[0], (num_points,), device=device)
        yx = coords[sel]
        y = yx[:, 0]
        x = yx[:, 1]

        # Label: 0 for fp, 1 for fn
        labels = fn[b, y, x].long()

        # Normalize to [0,1] in image coordinate space (same convention as dataset)
        pts = torch.stack([x.float() / float(W), y.float() / float(H)], dim=-1)

        points_out[b] = pts
        labels_out[b] = labels
        mask_out[b] = False

    return points_out, labels_out, mask_out


def _get_num_refine_points(config, device: torch.device) -> int:
    """Return how many refinement points to sample for this iteration."""
    max_n = int(getattr(config.DISTILL, "POINTS_PER_REFINE_ITER_MAX", 0) or 0)
    if max_n > 0:
        min_n = int(getattr(config.DISTILL, "POINTS_PER_REFINE_ITER_MIN", 1) or 1)
        min_n = max(0, min_n)
        max_n = max(min_n, max_n)
        if min_n == max_n:
            return min_n
        # randint is [low, high) so we use high=max_n+1
        return int(torch.randint(min_n, max_n + 1, (1,), device=device).item())
    return int(getattr(config.DISTILL, "POINTS_PER_REFINE_ITER", 0) or 0)


def parse_option():
    parser = argparse.ArgumentParser(
        "EfficientSAM3 Stage 2 Geometry Fine-tuning", add_help=False
    )
    
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE",
                        help='path to config file')
    parser.add_argument('--opts', nargs='+', default=None,
                        help="Modify config options by adding 'KEY VALUE' pairs")
    
    # Data
    parser.add_argument('--batch-size', type=int, help="batch size per GPU")
    parser.add_argument('--data-path', type=str, help='path to SA-1B dataset')
    
    # Model
    parser.add_argument('--pretrained', type=str, 
                        help='path to Stage 1 pretrained checkpoint')
    parser.add_argument('--resume', type=str, 
                        help='path to checkpoint to resume from')
    parser.add_argument('--sam3-checkpoint', type=str,
                        help='path to SAM3 checkpoint for frozen components')
    parser.add_argument('--teacher-embed-path', type=str,
                        help='path to saved teacher embeddings')
    
    # Training
    parser.add_argument('--accumulation-steps', type=int,
                        help='gradient accumulation steps')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='use gradient checkpointing')
    parser.add_argument('--disable-amp', action='store_true',
                        help='disable automatic mixed precision')
    parser.add_argument('--only-cpu', action='store_true',
                        help='use CPU only')
    
    # Output
    parser.add_argument('--output', type=str, default='output_geometry_finetune',
                        help='output directory')
    parser.add_argument('--tag', type=str, default='default',
                        help='tag for experiment')
    
    # Evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluate only')
    parser.add_argument('--throughput', action='store_true',
                        help='test throughput only')
    
    # Distributed
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')
    
    args = parser.parse_args()
    
    config = get_config(args)
    return args, config


def main(config):
    # Build data loaders
    logger.info("Building data loaders...")
    dataset_train, dataset_val, data_loader_train, data_loader_val = build_loader(config)
    logger.info(f"Training samples: {len(dataset_train)}")
    
    # Build model with SAM3 for DUAL-PATH distillation
    logger.info("Building model (DUAL-PATH: embedding + mask distillation)...")
    model = GeometryFinetuneModel(
        student_backbone_name=config.MODEL.BACKBONE,
        sam3_checkpoint_path=config.MODEL.SAM3_CHECKPOINT,
        embed_dim=config.DISTILL.EMBED_DIM,
        embed_size=config.DISTILL.EMBED_SIZE,
        img_size=config.DATA.IMG_SIZE,
        freeze_fpn=True,
    )
    
    # Load Stage 1 pretrained weights
    if config.MODEL.PRETRAINED:
        load_stage1_weights(model, config.MODEL.PRETRAINED, logger)
    
    model.cuda()
    
    # Distributed training
    model_without_ddp = model
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model, 
            device_ids=[config.LOCAL_RANK],
            find_unused_parameters=config.TRAIN.FIND_UNUSED_PARAMETERS,
        )
        model_without_ddp = model.module
    
    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {n_parameters / 1e6:.2f}M")
    
    # Build loss function
    criterion = GeometryFinetuningLoss(
        embedding_weight=config.DISTILL.EMBEDDING_LOSS_WEIGHT,
        mask_bce_weight=config.DISTILL.MASK_BCE_WEIGHT,
        mask_dice_weight=config.DISTILL.MASK_DICE_WEIGHT,
        mask_focal_weight=config.DISTILL.MASK_FOCAL_WEIGHT,
        iou_weight=config.DISTILL.IOU_LOSS_WEIGHT,
        temperature=config.DISTILL.TEMPERATURE,
    )
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(config, model)
    n_iter_per_epoch = max(1, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    lr_scheduler = build_scheduler(config, optimizer, n_iter_per_epoch)
    loss_scaler = NativeScalerWithGradNormCount()
    
    # Resume from checkpoint
    max_accuracy = 0.0
    if config.TRAIN.AUTO_RESUME:
        resume_file = os.path.join(config.OUTPUT, 'ckpt_epoch_latest.pth')
        if os.path.exists(resume_file):
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"Auto-resuming from {resume_file}")
    
    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger
        )
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        if dist.is_initialized():
            data_loader_train.sampler.set_epoch(epoch)
        dataset_train.set_epoch(epoch)
        
        train_one_epoch(
            config, model, criterion, data_loader_train,
            optimizer, lr_scheduler, loss_scaler, epoch, logger
        )
        
        # Save checkpoint
        if (epoch + 1) % config.SAVE_FREQ == 0 or epoch == config.TRAIN.EPOCHS - 1:
            save_checkpoint(
                config, epoch, model_without_ddp, max_accuracy,
                optimizer, lr_scheduler, loss_scaler, logger
            )
        
        # Validation
        if data_loader_val is not None and (epoch + 1) % config.SAVE_FREQ == 0:
            val_loss = validate(config, model, criterion, data_loader_val, logger)
            logger.info(f"Validation loss: {val_loss:.4f}")
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(config, model, criterion, data_loader, optimizer, 
                    lr_scheduler, loss_scaler, epoch, logger):
    """
    Train for one epoch with DUAL-PATH distillation:
    
    Loss 1 (Embedding): MSE(student_embedding, teacher_embedding)
    Loss 2 (Mask): BCE+Dice(student_mask, teacher_mask)
    
    Both masks are generated by running embeddings through frozen SAM3 components.
    """
    model.train()
    
    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    meters = defaultdict(AverageMeter)
    
    # Check if mask distillation is enabled
    use_mask_loss = (
        config.DISTILL.MASK_BCE_WEIGHT > 0
        or config.DISTILL.MASK_DICE_WEIGHT > 0
        or config.DISTILL.MASK_FOCAL_WEIGHT > 0
        or config.DISTILL.IOU_LOSS_WEIGHT > 0
    )
    
    start = time.time()
    end = time.time()
    
    for idx, batch in enumerate(data_loader):
        # Move data to GPU
        images = batch['images'].cuda(non_blocking=True)
        teacher_embeddings = batch['teacher_embeddings'].cuda(non_blocking=True)
        img_sizes = batch['img_sizes']
        
        # Prompts for mask prediction
        boxes = batch.get('boxes', None)
        points = batch.get('points', None)
        point_labels = batch.get('point_labels', None)
        prompt_mask = batch.get('prompt_mask', None)

        if boxes is not None:
            boxes = boxes.cuda(non_blocking=True)
        if points is not None:
            points = points.cuda(non_blocking=True)
        if point_labels is not None:
            point_labels = point_labels.cuda(non_blocking=True)
        if prompt_mask is not None:
            prompt_mask = prompt_mask.cuda(non_blocking=True)

        # Respect config flags (dataset/collate always provides tensors, but we may want
        # boxes-only or points-only training).
        if not config.DISTILL.USE_BOX_PROMPTS:
            boxes = None
        if not config.DISTILL.USE_POINT_PROMPTS:
            points = None
            point_labels = None
        if boxes is None and points is None:
            prompt_mask = None
        
        batch_size = images.shape[0]
        
        # Create valid mask for embeddings (handles padding)
        valid_mask = create_valid_mask(
            batch_size=batch_size,
            embed_size=config.DISTILL.EMBED_SIZE,
            img_sizes=img_sizes,
            img_size=config.DATA.IMG_SIZE,
            device=images.device,
        )
        
        with autocast(enabled=config.AMP_ENABLE):
            # Get model reference (handle DDP)
            model_ref = model.module if dist.is_initialized() else model
            
            # Forward student trunk → student embeddings
            student_embeddings = model_ref.forward_student(images)

            # Always compute embedding loss once.
            losses_embed = criterion(
                student_embedding=student_embeddings,
                teacher_embedding=teacher_embeddings,
                valid_mask=valid_mask,
            )
            embed_total = losses_embed["total_loss"]

            # Optional: Mask distillation (with optional iterative refinement)
            losses_mask_sum = {}
            mask_total = torch.zeros_like(embed_total)

            if use_mask_loss and (boxes is not None or points is not None):
                # Prompt mixing: if enabled and both boxes+points are present, randomly pick box-only vs point-only.
                if (
                    config.DISTILL.PROMPT_MIX
                    and boxes is not None
                    and points is not None
                    and config.DISTILL.USE_BOX_PROMPTS
                    and config.DISTILL.USE_POINT_PROMPTS
                ):
                    if torch.rand(1, device=images.device).item() < float(config.DISTILL.PROMPT_MIX_PROB_BOX):
                        # box-only
                        points = None
                        point_labels = None
                    else:
                        # point-only
                        boxes = None
                    if boxes is None and points is None:
                        prompt_mask = None

                # Convert boxes from xyxy normalized to cxcywh normalized
                boxes_cxcywh = xyxy_to_cxcywh(boxes) if boxes is not None else None

                # Build separate masks (needed for refinement where point count can change).
                box_mask = prompt_mask if boxes is not None else None
                point_mask = prompt_mask if points is not None else None

                # If starting from box-only, allow refinement to add points (EdgeSAM-style ITER_ON_BOX).
                if (
                    points is None
                    and config.DISTILL.DECODE_ITERS > 1
                    and _get_num_refine_points(config, images.device) > 0
                    and boxes is not None
                    and config.DISTILL.ITER_ON_BOX
                ):
                    points = torch.zeros(batch_size, 0, 2, device=images.device)
                    point_labels = torch.zeros(batch_size, 0, dtype=torch.long, device=images.device)
                    point_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=images.device)

                decode_iters = max(1, int(config.DISTILL.DECODE_ITERS))

                for iter_i in range(decode_iters):
                    student_iou = None
                    teacher_iou = None
                    student_masks = None
                    teacher_masks = None

                    # Teacher masks (no grad)
                    with torch.no_grad():
                        teacher_out = model_ref.forward_mask_prediction(
                            trunk_output=teacher_embeddings,
                            boxes=boxes_cxcywh,
                            points=points,
                            point_labels=point_labels,
                            box_mask=box_mask,
                            point_mask=point_mask,
                        )
                        teacher_masks = teacher_out.get("pred_masks")

                    # Student masks (grad flows into trunk output only)
                    student_out = model_ref.forward_mask_prediction(
                        trunk_output=student_embeddings,
                        boxes=boxes_cxcywh,
                        points=points,
                        point_labels=point_labels,
                        box_mask=box_mask,
                        point_mask=point_mask,
                    )
                    student_masks = student_out.get("pred_masks")

                    # Try to find IoU-like predictions if they exist in the model outputs.
                    for key in ("pred_ious", "pred_iou", "iou_pred", "iou_preds"):
                        if teacher_iou is None and key in teacher_out:
                            teacher_iou = teacher_out[key]
                        if student_iou is None and key in student_out:
                            student_iou = student_out[key]

                    # Optionally distill only the teacher's best mask (closer to interactive use).
                    if config.DISTILL.SELECT_BEST_MASK:
                        teacher_logits = teacher_out.get("pred_logits")
                        if teacher_logits is not None:
                            teacher_best_idx = teacher_logits.squeeze(-1).argmax(dim=1)
                        else:
                            teacher_best_idx = torch.zeros(batch_size, dtype=torch.long, device=images.device)
                        batch_idx = torch.arange(batch_size, device=images.device)
                        # Index both teacher and student with the TEACHER idx for stable supervision.
                        teacher_masks_used = teacher_masks[batch_idx, teacher_best_idx].unsqueeze(1)
                        student_masks_used = student_masks[batch_idx, teacher_best_idx].unsqueeze(1)
                    else:
                        teacher_masks_used = teacher_masks
                        student_masks_used = student_masks

                    # Ignore padded image regions in mask distillation losses.
                    mask_h, mask_w = student_masks_used.shape[-2:]
                    valid_img = torch.zeros(
                        batch_size, 1, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE, device=images.device
                    )
                    for bi in range(batch_size):
                        h, w = img_sizes[bi] if isinstance(img_sizes, torch.Tensor) else img_sizes[bi]
                        valid_img[bi, :, : int(h), : int(w)] = 1.0
                    mask_valid_mask = F.interpolate(
                        valid_img, size=(mask_h, mask_w), mode="bilinear", align_corners=False
                    )
                    mask_valid_mask = (mask_valid_mask > 0.5).float()

                    losses_mask_iter = criterion(
                        student_embedding=None,
                        teacher_embedding=None,
                        student_masks=student_masks_used,
                        teacher_masks=teacher_masks_used,
                        student_iou=student_iou,
                        teacher_iou=teacher_iou,
                        mask_valid_mask=mask_valid_mask,
                    )
                    # Accumulate (we'll average by decode_iters below)
                    mask_total = mask_total + losses_mask_iter["total_loss"]
                    for k, v in losses_mask_iter.items():
                        if k == "total_loss":
                            continue
                        if isinstance(v, torch.Tensor):
                            losses_mask_sum[k] = losses_mask_sum.get(k, 0.0) + v

                    # Iterative refinement: add points from disagreement between student and teacher.
                    if (
                        iter_i < decode_iters - 1
                        and _get_num_refine_points(config, images.device) > 0
                        and points is not None
                    ):
                        num_refine_pts = _get_num_refine_points(config, images.device)
                        if num_refine_pts <= 0:
                            continue
                        # Use the (single) distilled mask for refinement sampling.
                        # Shapes: (B, 1, H, W) -> (B, H, W)
                        s_logits = student_masks_used[:, 0].detach()
                        t_logits = teacher_masks_used[:, 0].detach()

                        new_points, new_labels, new_point_mask = sample_refinement_points(
                            student_mask_logits=s_logits,
                            teacher_mask_logits=t_logits,
                            valid_mask=mask_valid_mask,
                            num_points=num_refine_pts,
                            threshold=float(config.DISTILL.TEACHER_MASK_THRESHOLD),
                        )

                        # Concatenate while keeping right-padding property (SAM3 requires right-padded masks).
                        from sam3.model.geometry_encoders import concat_padded_sequences

                        if point_mask is None:
                            point_mask = torch.zeros(points.shape[0], points.shape[1], dtype=torch.bool, device=points.device)
                        if point_labels is None:
                            point_labels = torch.zeros(points.shape[0], points.shape[1], dtype=torch.long, device=points.device)

                        point_mask_prev = point_mask

                        # Points: (B,N,2) -> (N,B,2)
                        pts_seq = points.transpose(0, 1)
                        new_pts_seq = new_points.transpose(0, 1)
                        pts_seq, point_mask = concat_padded_sequences(
                            pts_seq, point_mask_prev, new_pts_seq, new_point_mask
                        )
                        points = pts_seq.transpose(0, 1)

                        # Labels: (B,N) -> (N,B)
                        lbl_seq = point_labels.transpose(0, 1)
                        new_lbl_seq = new_labels.transpose(0, 1)
                        lbl_seq, _ = concat_padded_sequences(
                            lbl_seq.unsqueeze(-1), point_mask_prev, new_lbl_seq.unsqueeze(-1), new_point_mask
                        )
                        point_labels = lbl_seq.squeeze(-1).transpose(0, 1)

                # Average mask loss terms across decode iterations
                mask_total = mask_total / float(decode_iters)
                for k in list(losses_mask_sum.keys()):
                    losses_mask_sum[k] = losses_mask_sum[k] / float(decode_iters)

            # Total loss = embedding loss (once) + averaged mask loss (if enabled)
            total_loss = embed_total + mask_total

            # Compile loss dict for logging/backprop
            losses = {"total_loss": total_loss}
            for k, v in losses_embed.items():
                if k != "total_loss":
                    losses[k] = v
            for k, v in losses_mask_sum.items():
                losses[k] = v
        
        # Scale loss for gradient accumulation
        loss = losses['total_loss'] / config.TRAIN.ACCUMULATION_STEPS
        
        # Backward
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(
            loss, optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0
        )
        
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )
        
        # NOTE: `torch.cuda.synchronize()` can significantly slow down training.
        # Only synchronize at log intervals (when we actually report timings/memory).
        if idx % config.PRINT_FREQ == 0:
            torch.cuda.synchronize()
        
        # Update meters
        loss_meter.update(losses['total_loss'].item(), batch_size)
        if grad_norm is not None:
            norm_meter.update(grad_norm)
        
        for k, v in losses.items():
            if k != 'total_loss' and isinstance(v, torch.Tensor):
                meters[k].update(v.item(), batch_size)
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Logging
        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            
            loss_str = ' '.join([f'{k} {v.val:.4f} ({v.avg:.4f})' 
                                for k, v in meters.items()])
            
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]  '
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}  '
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  '
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  '
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  '
                f'{loss_str}  mem {memory_used:.0f}MB'
            )
    
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, model, criterion, data_loader, logger):
    """Validate the model."""
    model.eval()
    
    loss_meter = AverageMeter()
    
    for batch in data_loader:
        images = batch['images'].cuda(non_blocking=True)
        teacher_embeddings = batch['teacher_embeddings'].cuda(non_blocking=True)
        img_sizes = batch['img_sizes']
        
        batch_size = images.shape[0]
        
        valid_mask = create_valid_mask(
            batch_size=batch_size,
            embed_size=config.DISTILL.EMBED_SIZE,
            img_sizes=img_sizes,
            img_size=config.DATA.IMG_SIZE,
            device=images.device,
        )
        
        with autocast(enabled=config.AMP_ENABLE):
            student_embeddings = model.module.forward_student(images) if dist.is_initialized() \
                else model.forward_student(images)
            
            losses = criterion(
                student_embedding=student_embeddings,
                teacher_embedding=teacher_embeddings,
                valid_mask=valid_mask,
            )
        
        loss_meter.update(losses['total_loss'].item(), batch_size)
    
    return loss_meter.avg


if __name__ == '__main__':
    args, config = parse_option()
    
    # Setup distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(config.LOCAL_RANK)
        dist.init_process_group('nccl')
    else:
        rank = 0
        world_size = 1
    
    # Set random seed
    seed = config.SEED + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    
    # Create output directory
    os.makedirs(config.OUTPUT, exist_ok=True)
    
    # Setup logger
    logger = create_logger(
        output_dir=config.OUTPUT,
        dist_rank=rank,
        name=f'geometry_finetune_{config.TAG}'
    )
    
    if rank == 0:
        logger.info(f"Config:\n{config}")
    
    main(config)
