import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import logging

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from stage2.model import EfficientSAM3Stage2
from stage2.data.sav_dataset import SAVVideoDataset
from sam3.model_builder import build_sam3_image_model
from sam2.training.utils.data_utils import BatchedVideoDatapoint
from sam2.training.loss_fns import dice_loss, sigmoid_focal_loss

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=rank)
        dist.barrier()
        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def train(args):
    rank, local_rank, world_size = setup_distributed()
    is_main_process = (rank == 0)
    
    # Config
    sam3_checkpoint = args.sam3_checkpoint
    save_dir = args.output_dir
    if is_main_process:
        os.makedirs(save_dir, exist_ok=True)
    
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    
    if is_main_process:
        logger.info(f"Loading SAM3 model from: {sam3_checkpoint}")
    
    # Load full SAM3 model to get the pretrained image encoder
    sam3_model = build_sam3_image_model(
        checkpoint_path=sam3_checkpoint,
        enable_text_encoder=False, 
        enable_vision_encoder=True,
    )
    image_encoder = sam3_model.backbone 
    
    if is_main_process:
        logger.info("Initializing EfficientSAM3Stage2...")
    
    model = EfficientSAM3Stage2(
        image_encoder=image_encoder,
        d_model=256,
        num_maskmem=7,
        image_size=1024,
        use_perceiver=True, # Enable Perceiver by default for Stage 2
        pool_size=1, # Disable pooling in EfficientMemoryAttention when using Perceiver
    )
    model.to(device)
    
    # Optimizer
    # Only train memory modules (Perceiver, MemoryEncoder, MemoryAttention)
    # Freeze backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main_process:
        logger.info(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    # Wrap in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.1)
    
    # Dataset
    if not args.data_path or not os.path.exists(args.data_path):
        if is_main_process:
            logger.error(f"Data path not found: {args.data_path}")
        raise FileNotFoundError(f"Data path not found: {args.data_path}")

    if is_main_process:
        logger.info(f"Loading SA-V dataset from {args.data_path}...")
    
    dataset = SAVVideoDataset(
        data_path=args.data_path,
        frames_per_video=8,
        img_size=1024,
    )

    def collate_fn(batch):
        # batch is a list of dicts
        # Stack frames: [B, T, C, H, W]
        frames = torch.stack([b['frames'] for b in batch], dim=0)
        # Stack masks: [B, T, num_objects, H, W]
        masks = torch.stack([b['masks'] for b in batch], dim=0)
        
        return {
            'img_batch': frames,
            'masks': masks,
        }

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=(sampler is None), 
        sampler=sampler,
        collate_fn=collate_fn, 
        num_workers=4
    )
    
    # Training Loop
    model.train()
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
            
        if is_main_process:
            logger.info(f"Epoch {epoch+1}/{args.epochs}")
            
        for i, batch in enumerate(dataloader):
            try:
                # batch is dict
                images = batch['img_batch'].to(device) # [B, T, 3, H, W]
                gt_masks = batch['masks'].to(device)   # [B, T, num_objects, H, W]
                
                B, T, _, H, W = images.shape
                
                # Forward pass
                # 1. Get image features for all frames
                image_batch_flat = images.flatten(0, 1) # [B*T, 3, H, W]
                
                if world_size > 1:
                    backbone_out = model.module.forward_image(image_batch_flat)
                else:
                    backbone_out = model.forward_image(image_batch_flat)
                
                # Prepare features for tracking
                features_per_frame = []
                pos_enc_per_frame = []
                
                num_levels = len(backbone_out["backbone_fpn"])
                for t in range(T):
                    feats_t = []
                    pos_t = []
                    for level in range(num_levels):
                        feat = backbone_out["backbone_fpn"][level]
                        pos = backbone_out["vision_pos_enc"][level]
                        
                        C_feat = feat.shape[1]
                        H_feat, W_feat = feat.shape[2], feat.shape[3]
                        
                        feat_reshaped = feat.view(B, T, C_feat, H_feat, W_feat)
                        pos_reshaped = pos.view(B, T, C_feat, H_feat, W_feat)
                        
                        feats_t.append(feat_reshaped[:, t]) 
                        pos_t.append(pos_reshaped[:, t])
                    
                    features_per_frame.append(feats_t)
                    pos_enc_per_frame.append(pos_t)
                
                flattened_features_per_frame = []
                flattened_pos_per_frame = []
                feat_sizes_per_frame = []
                
                for t in range(T):
                    feats_flat = []
                    pos_flat = []
                    sizes = []
                    for level in range(num_levels):
                        f = features_per_frame[t][level] 
                        p = pos_enc_per_frame[t][level]
                        
                        sizes.append((f.shape[2], f.shape[3]))
                        
                        f_flat = f.flatten(2).permute(2, 0, 1)
                        p_flat = p.flatten(2).permute(2, 0, 1)
                        
                        feats_flat.append(f_flat)
                        pos_flat.append(p_flat)
                    
                    flattened_features_per_frame.append(feats_flat)
                    flattened_pos_per_frame.append(pos_flat)
                    feat_sizes_per_frame.append(sizes)

                
                output_dict = {
                    "cond_frame_outputs": {}, 
                    "non_cond_frame_outputs": {}, 
                }
                
                total_loss = 0
                
                for t in range(T):
                    current_vision_feats = flattened_features_per_frame[t]
                    current_vision_pos_embeds = flattened_pos_per_frame[t]
                    feat_sizes = feat_sizes_per_frame[t]
                    
                    point_inputs = None
                    mask_inputs = None
                    
                    if t == 0:
                        # Initial frame: use GT mask as input
                        # gt_masks: [B, T, num_objects, H, W] -> [B, num_objects, H, W]
                        mask_inputs = gt_masks[:, t] 
                        is_init_cond_frame = True
                    else:
                        is_init_cond_frame = False
                    
                    if world_size > 1:
                        out = model.module.track_step(
                            frame_idx=t,
                            is_init_cond_frame=is_init_cond_frame,
                            current_vision_feats=current_vision_feats,
                            current_vision_pos_embeds=current_vision_pos_embeds,
                            feat_sizes=feat_sizes,
                            point_inputs=point_inputs,
                            mask_inputs=mask_inputs,
                            output_dict=output_dict,
                            num_frames=T,
                        )
                    else:
                        out = model.track_step(
                            frame_idx=t,
                            is_init_cond_frame=is_init_cond_frame,
                            current_vision_feats=current_vision_feats,
                            current_vision_pos_embeds=current_vision_pos_embeds,
                            feat_sizes=feat_sizes,
                            point_inputs=point_inputs,
                            mask_inputs=mask_inputs,
                            output_dict=output_dict,
                            num_frames=T,
                        )
                    
                    if is_init_cond_frame:
                        output_dict["cond_frame_outputs"][t] = out
                    else:
                        output_dict["non_cond_frame_outputs"][t] = out
                    
                    # Compute loss
                    pred_masks = out["pred_masks_high_res"] # [B, num_objects, H, W]
                    gt_mask_t = gt_masks[:, t]              # [B, num_objects, H, W]
                    
                    loss = dice_loss(pred_masks, gt_mask_t) + sigmoid_focal_loss(pred_masks, gt_mask_t)
                    total_loss += loss
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                if is_main_process and i % 10 == 0:
                    logger.info(f"Step {i}, Loss: {total_loss.item()}")
            except Exception as e:
                logger.error(f"Error in step {i}: {e}")
                continue
        
        # Save checkpoint
        if is_main_process:
            save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            if world_size > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    cleanup_distributed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3_checkpoint", type=str, default="sam3_checkpoints/sam3.pt")
    parser.add_argument("--data_path", type=str, default="data/sa-v/extracted_frames")
    parser.add_argument("--output_dir", type=str, default="output/stage2_checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    train(args)
