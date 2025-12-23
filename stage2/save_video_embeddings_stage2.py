#!/usr/bin/env python
# --------------------------------------------------------
# Stage 2: Pre-compute Video Embeddings from SAM3 Backbone
# --------------------------------------------------------

"""
Pre-compute and save trunk features (before FPN) for SA-V videos.

This script:
1. Loads SAM3's vision backbone
2. Processes video frames
3. Saves trunk features (before FPN) to disk

Features are saved at the trunk output level ([B, 1024, 72, 72] for 1008x1008 input)
to avoid repeated computation during training.

Supports multi-GPU distributed processing via torchrun.

Usage:
    # Single GPU
    python save_video_embeddings_stage2.py \
        --data_dir /path/to/sa-v/extracted_frames \
        --output_dir /path/to/embeddings \
        --batch_size 8

    # Multi-GPU (automatically distributes videos across GPUs)
    torchrun --nproc_per_node=4 save_video_embeddings_stage2.py \
        --data_dir /path/to/sa-v/extracted_frames \
        --output_dir /path/to/embeddings \
        --batch_size 8
"""

import os
import sys
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import only what we need to build the ViT backbone
from sam3.model.vitdet import ViT
from sam3.model_builder import download_ckpt_from_hf
from iopath.common.file_io import g_pathmgr


def setup_distributed():
    """Initialize distributed processing if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed processing."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def get_rank():
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size():
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


class VideoFrameDataset(Dataset):
    """
    Dataset for loading video frames from SA-V.
    
    Processes all frames in each video directory.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 1008,
        videos: list = None,
    ):
        """
        Args:
            data_dir: Path to SA-V extracted frames directory
            image_size: Size to resize images to
            videos: Optional list of video names to process
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Find all video directories
        if videos is not None:
            self.video_dirs = [self.data_dir / v for v in videos]
        else:
            self.video_dirs = sorted([
                d for d in self.data_dir.iterdir()
                if d.is_dir() and d.name.startswith('sav_')
            ])
        
        # Build frame index: list of (video_dir, frame_name)
        self.frames = []
        for video_dir in self.video_dirs:
            # Find all jpg/png frames
            frame_files = sorted([
                f for f in video_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
            ])
            for frame_file in frame_files:
                self.frames.append((video_dir, frame_file.name))
        
        # Transform
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"Found {len(self.video_dirs)} videos, {len(self.frames)} total frames")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        video_dir, frame_name = self.frames[idx]
        
        # Load image
        img_path = video_dir / frame_name
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        
        return {
            'image': img,
            'video_name': video_dir.name,
            'frame_name': frame_name,
        }


def collate_fn(batch):
    """Custom collate to preserve metadata."""
    images = torch.stack([b['image'] for b in batch])
    video_names = [b['video_name'] for b in batch]
    frame_names = [b['frame_name'] for b in batch]
    return {
        'images': images,
        'video_names': video_names,
        'frame_names': frame_names,
    }


class TrunkFeatureExtractor:
    """
    Extract trunk features from SAM3 backbone.
    
    Saves features BEFORE the FPN, so we only save one feature map
    per image rather than multi-scale outputs.
    
    This only loads the ViT trunk, not the full SAM3 model, to save memory.
    """
    
    def __init__(
        self,
        checkpoint_path: str = None,
        device: str = 'cuda',
    ):
        """
        Args:
            checkpoint_path: Path to SAM3 checkpoint (or None for HF download)
            device: Device to run on
        """
        self.device = device
        
        # Build only the ViT trunk (not the full model)
        print("Building ViT backbone...")
        self.trunk = ViT(
            img_size=1008,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=32,
            num_heads=16,
            mlp_ratio=4.625,
            norm_layer="LayerNorm",
            drop_path_rate=0.1,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
            compile_mode=None,
        )
        
        # Load checkpoint - only trunk weights
        print("Loading trunk weights from checkpoint...")
        if checkpoint_path is None:
            checkpoint_path = download_ckpt_from_hf()
        
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)
        
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]
        
        # Extract only trunk weights (filter by prefix)
        trunk_prefix = "detector.backbone.vision_backbone.trunk."
        trunk_weights = {}
        for k, v in ckpt.items():
            if k.startswith(trunk_prefix):
                new_key = k[len(trunk_prefix):]  # Remove prefix
                trunk_weights[new_key] = v
        
        print(f"Found {len(trunk_weights)} trunk parameters")
        missing, unexpected = self.trunk.load_state_dict(trunk_weights, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")
        
        # Move to device and freeze
        self.trunk = self.trunk.to(device)
        self.trunk.eval()
        for param in self.trunk.parameters():
            param.requires_grad = False
        
        print("Trunk loaded successfully")
        
        # Get output stride
        # SAM3 uses stride 14 for 1008 input -> 72x72 features
        self.backbone_stride = 14
        
        # Image normalization (SAM3 uses 0.5 mean/std)
        self.image_mean = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
        self.image_std = torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1)
    
    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract trunk features for a batch of images.
        
        Args:
            images: [B, C, H, W] - normalized images
            
        Returns:
            features: [B, 1024, H/14, W/14] - trunk features
        """
        images = images.to(self.device)
        
        # Forward through trunk only
        features = self.trunk(images)
        
        # trunk returns a list of features at different stages
        # We take the last (highest level) features
        if isinstance(features, (list, tuple)):
            features = features[-1]
        
        return features


def save_embeddings_for_video(
    extractor: TrunkFeatureExtractor,
    video_dir: Path,
    output_dir: Path,
    image_size: int = 1008,
    batch_size: int = 8,
    save_format: str = 'pt',
):
    """
    Save trunk embeddings for all frames in a video.
    
    Args:
        extractor: Feature extractor
        video_dir: Path to video directory
        output_dir: Output directory for embeddings
        image_size: Input image size
        batch_size: Batch size for processing
        save_format: 'pt' for torch, 'npy' for numpy
    """
    # Create output directory for this video
    video_output_dir = output_dir / video_dir.name
    video_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already processed
    metadata_file = video_output_dir / 'metadata.json'
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        if metadata.get('completed', False):
            print(f"Skipping {video_dir.name} (already processed)")
            return
    
    # Find all frames
    frame_files = sorted([
        f for f in video_dir.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])
    
    if len(frame_files) == 0:
        print(f"No frames found in {video_dir}")
        return
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
    # Process in batches
    num_batches = (len(frame_files) + batch_size - 1) // batch_size
    
    saved_files = []
    feature_shape = None
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(frame_files))
        batch_files = frame_files[start_idx:end_idx]
        
        # Load images
        images = []
        for frame_file in batch_files:
            img = Image.open(frame_file).convert('RGB')
            img = transform(img)
            images.append(img)
        
        images = torch.stack(images)
        
        # Extract features
        features = extractor.extract(images)
        features = features.cpu()
        
        if feature_shape is None:
            feature_shape = list(features.shape[1:])  # [C, H, W]
        
        # Save each frame's features
        for i, frame_file in enumerate(batch_files):
            frame_name = frame_file.stem
            
            if save_format == 'pt':
                output_path = video_output_dir / f"{frame_name}.pt"
                torch.save(features[i], output_path)
            else:
                output_path = video_output_dir / f"{frame_name}.npy"
                np.save(output_path, features[i].numpy())
            
            saved_files.append(frame_name)
    
    # Save metadata
    metadata = {
        'video_name': video_dir.name,
        'num_frames': len(frame_files),
        'feature_shape': feature_shape,
        'image_size': image_size,
        'backbone_stride': extractor.backbone_stride,
        'save_format': save_format,
        'frames': saved_files,
        'completed': True,
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute trunk embeddings for SA-V videos'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='Path to SA-V extracted frames directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for embeddings'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to SAM3 checkpoint (optional, downloads from HF if not provided)'
    )
    parser.add_argument(
        '--image_size',
        type=int,
        default=1008,
        help='Input image size (default: 1008)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
    )
    parser.add_argument(
        '--save_format',
        type=str,
        default='pt',
        choices=['pt', 'npy'],
        help='Format to save embeddings (default: pt)'
    )
    parser.add_argument(
        '--videos_file',
        type=str,
        default=None,
        help='Optional file with list of video names to process'
    )
    
    args = parser.parse_args()
    
    # Setup distributed processing
    rank, world_size, local_rank = setup_distributed()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    if is_main_process():
        print(f"Running with {world_size} GPU(s)")
    
    # Create output directory (only main process)
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Synchronize before continuing
    if dist.is_initialized():
        dist.barrier()
    
    # Initialize extractor on local GPU
    extractor = TrunkFeatureExtractor(
        checkpoint_path=args.checkpoint,
        device=device,
    )
    
    # Find video directories
    data_dir = Path(args.data_dir)
    
    if args.videos_file:
        with open(args.videos_file, 'r') as f:
            video_names = [line.strip() for line in f if line.strip()]
        all_video_dirs = [data_dir / v for v in video_names]
    else:
        all_video_dirs = sorted([
            d for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith('sav_')
        ])
    
    # Distribute videos across GPUs
    # Each GPU processes videos[rank::world_size]
    my_video_dirs = all_video_dirs[rank::world_size]
    
    if is_main_process():
        print(f"Total videos: {len(all_video_dirs)}")
    print(f"[GPU {rank}] Processing {len(my_video_dirs)} videos")
    
    # Process each video assigned to this GPU
    pbar = tqdm(my_video_dirs, desc=f"GPU {rank}", disable=not is_main_process())
    for video_dir in pbar:
        if not video_dir.exists():
            print(f"[GPU {rank}] Warning: {video_dir} does not exist, skipping")
            continue
        
        try:
            save_embeddings_for_video(
                extractor=extractor,
                video_dir=video_dir,
                output_dir=output_dir,
                image_size=args.image_size,
                batch_size=args.batch_size,
                save_format=args.save_format,
            )
        except Exception as e:
            print(f"[GPU {rank}] Error processing {video_dir.name}: {e}")
            continue
    
    # Synchronize before saving global metadata
    if dist.is_initialized():
        dist.barrier()
    
    # Save global metadata (only main process)
    if is_main_process():
        print("All GPUs finished processing!")
        
        global_metadata = {
            'data_dir': str(args.data_dir),
            'num_videos': len(all_video_dirs),
            'num_gpus': world_size,
            'image_size': args.image_size,
            'save_format': args.save_format,
            'backbone_stride': extractor.backbone_stride,
        }
        
        with open(output_dir / 'global_metadata.json', 'w') as f:
            json.dump(global_metadata, f, indent=2)
        
        print("Done!")
    
    # Cleanup
    cleanup_distributed()


if __name__ == '__main__':
    main()
