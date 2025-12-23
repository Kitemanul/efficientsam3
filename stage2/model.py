# --------------------------------------------------------
# Stage 2 Model Classes for Memory Distillation
# --------------------------------------------------------

"""
Model classes for Stage 2 memory bank training.

This module provides:
1. SAM3MemoryTeacher - Full SAM3 tracker for generating teacher outputs
2. SAM3MemoryStudent - Student model with efficient memory modules
3. Helper functions for building and loading models
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import torch
import torch.nn as nn
from iopath.common.file_io import g_pathmgr

from sam3.model.vitdet import ViT
from sam3.model_builder import download_ckpt_from_hf
from sam3.model.perceiver import EfficientSpatialPerceiver
from sam3.model.efficient_memory_attention import EfficientMemoryAttention
from sam3.model.position_encoding import PositionEmbeddingSine


class SAM3MemoryTeacher(nn.Module):
    """
    SAM3 Teacher model for memory distillation.
    
    This wraps the full SAM3 tracker and provides access to intermediate
    outputs needed for distillation (memory features, conditioned features).
    
    The entire model is frozen - no gradients are computed.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        image_size: int = 1008,
    ):
        super().__init__()
        self.image_size = image_size

        # Build only the ViT trunk (same as `stage2/save_video_embeddings_stage2.py`)
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

        checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        self._load_trunk_weights(checkpoint_path)

        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def _resolve_checkpoint_path(self, checkpoint_path: Optional[str]) -> str:
        """Pick a checkpoint path; prefer local `sam3_checkpoints/sam3.pt` if present."""
        if checkpoint_path:
            return checkpoint_path

        local = Path(__file__).resolve().parent.parent / "sam3_checkpoints" / "sam3.pt"
        if local.exists():
            return str(local)

        return download_ckpt_from_hf()

    def _load_trunk_weights(self, checkpoint_path: str) -> None:
        """Load only trunk weights from a SAM3 checkpoint into `self.trunk`."""
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)

        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        trunk_prefix = "detector.backbone.vision_backbone.trunk."
        trunk_weights = {
            k[len(trunk_prefix) :]: v for k, v in ckpt.items() if k.startswith(trunk_prefix)
        }
        if not trunk_weights:
            raise KeyError(
                f"No trunk weights found with prefix {trunk_prefix!r} in checkpoint {checkpoint_path!r}"
            )

        self.trunk.load_state_dict(trunk_weights, strict=False)
    
    @torch.no_grad()
    def extract_trunk_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract trunk features (before FPN) for a batch of images.
        
        Args:
            images: [B, C, H, W] - input images
            
        Returns:
            features: [B, 1024, H/14, W/14] - trunk features
        """
        out = self.trunk(images)
        if isinstance(out, (list, tuple)):
            out = out[-1]
        return out
    
    @torch.no_grad()
    def forward(
        self,
        frames: torch.Tensor,
        point_inputs: Optional[Dict] = None,
        mask_inputs: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Full forward pass through the tracker.
        
        This is used to get teacher outputs for distillation.
        
        Args:
            frames: [B, T, C, H, W] - video frames
            point_inputs: dict with point_coords and point_labels
            mask_inputs: [B, 1, H, W] - initial mask prompt
            
        Returns:
            dict with teacher outputs for distillation
        """
        B, T, C, H, W = frames.shape

        flat = frames.reshape(B * T, C, H, W)
        feats = self.extract_trunk_features(flat)  # [B*T, 1024, 72, 72]
        feats = feats.reshape(B, T, feats.shape[1], feats.shape[2], feats.shape[3])

        return {
            'trunk_features': feats,  # [B, T, 1024, 72, 72]
        }


class SAM3MemoryStudent(nn.Module):
    """
    Student model with efficient memory modules.
    
    This model uses:
    1. Pre-computed/frozen trunk features (from SAM3 or Stage 1 student)
    2. Trainable PerceiverResampler for memory compression
    3. Trainable EfficientMemoryAttention for fusing with current frame
    4. Frozen decoder for generating mask outputs
    
    Only the memory-related modules are trained.
    """
    
    def __init__(
        self,
        config,
        frozen_components: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.config = config
        
        # Feature dimensions
        self.mem_dim = config.MODEL.MEMORY_ENCODER.OUT_DIM
        self.hidden_dim = config.MODEL.MEMORY_ATTENTION.D_MODEL
        self.image_size = config.MODEL.TRACKER.IMAGE_SIZE
        self.backbone_stride = config.MODEL.TRACKER.BACKBONE_STRIDE
        self.feat_size = self.image_size // self.backbone_stride
        
        # Number of memory frames
        self.num_maskmem = config.MODEL.TRACKER.NUM_MASKMEM
        
        # Position encodings
        # NOTE: SAM3's PositionEmbeddingSine returns `num_pos_feats` channels.
        self.query_position_encoding = PositionEmbeddingSine(
            num_pos_feats=self.hidden_dim, normalize=True
        )
        self.memory_position_encoding = PositionEmbeddingSine(
            num_pos_feats=self.mem_dim, normalize=True
        )
        
        # Perceiver Resampler for memory compression (TRAINABLE)
        if config.MODEL.PERCEIVER.ENABLED:
            self.spatial_perceiver = EfficientSpatialPerceiver(
                dim=self.mem_dim,
                num_latents=config.MODEL.PERCEIVER.NUM_LATENTS,
                num_latents_2d=config.MODEL.PERCEIVER.NUM_LATENTS_2D,
                depth=config.MODEL.PERCEIVER.DEPTH,
                heads=config.MODEL.PERCEIVER.HEADS,
                dim_head=config.MODEL.PERCEIVER.DIM_HEAD,
                use_self_attn=config.MODEL.PERCEIVER.USE_SELF_ATTN,
                position_encoding=self.memory_position_encoding,
            )
        else:
            self.spatial_perceiver = None

        # Project memory tokens (C=mem_dim) into attention space (C=hidden_dim)
        self.mem_proj = nn.Linear(self.mem_dim, self.hidden_dim)
        self.mem_pos_proj = nn.Linear(self.mem_dim, self.hidden_dim)
        
        # Efficient Memory Attention (TRAINABLE)
        self.memory_attention = EfficientMemoryAttention(
            d_model=self.hidden_dim,
            dim_feedforward=config.MODEL.MEMORY_ATTENTION.DIM_FEEDFORWARD,
            num_heads=config.MODEL.MEMORY_ATTENTION.NUM_HEADS,
            num_layers=config.MODEL.MEMORY_ATTENTION.NUM_LAYERS,
            dropout=config.MODEL.MEMORY_ATTENTION.DROPOUT,
            pool_size=config.MODEL.MEMORY_ATTENTION.POOL_SIZE,
        )
        
        # Temporal position encoding for memories
        self.maskmem_tpos_enc = nn.Parameter(
            torch.zeros(self.num_maskmem, 1, 1, self.mem_dim)
        )
        nn.init.trunc_normal_(self.maskmem_tpos_enc, std=0.02)
        
        # No memory token (for first frame)
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.no_mem_pos_enc = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.no_mem_embed, std=0.02)
        nn.init.trunc_normal_(self.no_mem_pos_enc, std=0.02)
        
        # Store frozen components if provided
        self.frozen_components = frozen_components
    
    def _get_total_latents(self) -> int:
        """Get total number of latent tokens per memory frame."""
        if self.spatial_perceiver is not None:
            return (
                self.config.MODEL.PERCEIVER.NUM_LATENTS + 
                self.config.MODEL.PERCEIVER.NUM_LATENTS_2D
            )
        else:
            return self.feat_size * self.feat_size
    
    def compress_memory(
        self,
        maskmem_features: torch.Tensor,
        maskmem_pos_enc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress memory features using Perceiver.
        
        Args:
            maskmem_features: [B, C, H, W] - memory features from encoder
            maskmem_pos_enc: [B, C, H, W] - positional encoding
            
        Returns:
            compressed_features: [B, N, C] - compressed tokens
            compressed_pos_enc: [B, N, C] - positional encoding
        """
        if maskmem_pos_enc is None:
            maskmem_pos_enc = self.memory_position_encoding(maskmem_features).to(
                maskmem_features.dtype
            )

        if self.spatial_perceiver is not None:
            compressed, compressed_pos = self.spatial_perceiver(maskmem_features, maskmem_pos_enc)
            if compressed_pos is None:
                compressed_pos = torch.zeros_like(compressed)
            return compressed, compressed_pos
        else:
            # No compression - flatten to sequence
            B, C, H, W = maskmem_features.shape
            features = maskmem_features.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            pos_enc = maskmem_pos_enc.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            return features, pos_enc
    
    def prepare_memory(
        self,
        memory_list: list,
        memory_pos_list: list,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Prepare memory tokens for attention.
        
        Concatenates memories from multiple frames and adds
        temporal position encoding.
        
        Args:
            memory_list: list of [B, N, C] memory tensors
            memory_pos_list: list of [B, N, C] position tensors
            
        Returns:
            memory: [M, B, C] - concatenated memories (seq first)
            memory_pos: [M, B, C] - concatenated positions
            num_spatial: total number of spatial tokens
        """
        if len(memory_list) == 0:
            raise ValueError("prepare_memory() expects at least one memory item")
        
        all_memory = []
        all_pos = []

        # Use only the most recent NUM_MASKMEM frames
        memory_list = memory_list[-self.num_maskmem :]
        memory_pos_list = memory_pos_list[-self.num_maskmem :]

        for t_idx, (mem, pos) in enumerate(zip(memory_list, memory_pos_list)):
            # mem/pos: [B, N, mem_dim]
            if pos is None:
                pos = torch.zeros_like(mem)

            # Add temporal position encoding (in mem_dim space)
            tpos = self.maskmem_tpos_enc[t_idx].reshape(1, 1, self.mem_dim)  # [1,1,mem_dim]
            pos = pos + tpos
            
            # Convert to seq-first format: [B, N, C] -> [N, B, C]
            all_memory.append(mem.permute(1, 0, 2))
            all_pos.append(pos.permute(1, 0, 2))
        
        # Concatenate along sequence dimension
        memory = torch.cat(all_memory, dim=0)  # [M, B, C]
        memory_pos = torch.cat(all_pos, dim=0)  # [M, B, C]
        
        num_spatial = memory.shape[0]

        # Project to attention space
        memory = self.mem_proj(memory)
        memory_pos = self.mem_pos_proj(memory_pos)

        return memory, memory_pos, num_spatial
    
    def forward_memory_attention(
        self,
        current_features: torch.Tensor,
        current_pos_enc: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        num_spatial_mem: int,
    ) -> torch.Tensor:
        """
        Apply memory attention to fuse current features with memory.
        
        Args:
            current_features: [B, C, H, W] - current frame features
            current_pos_enc: [B, C, H, W] - current position encoding
            memory: [M, B, C] - memory tokens
            memory_pos: [M, B, C] - memory positions
            num_spatial_mem: number of spatial memory tokens
            
        Returns:
            conditioned_features: [B, C, H, W] - memory-conditioned features
        """
        B, C, H, W = current_features.shape
        
        # Flatten current features to sequence: [B, C, H, W] -> [HW, B, C]
        curr = current_features.flatten(2).permute(2, 0, 1)
        curr_pos = current_pos_enc.flatten(2).permute(2, 0, 1)
        
        # Apply efficient memory attention
        out = self.memory_attention(
            curr=curr,
            memory=memory,
            curr_pos=curr_pos,
            memory_pos=memory_pos,
            num_spatial_mem=num_spatial_mem,
            feat_size=(H, W),
        )
        
        # Reshape back: [HW, B, C] -> [B, C, H, W]
        out = out.permute(1, 2, 0).view(B, C, H, W)
        
        return out
    
    def forward(
        self,
        current_features: torch.Tensor,
        current_pos_enc: Optional[torch.Tensor] = None,
        memory_features: Optional[torch.Tensor] = None,
        memory_pos_enc: Optional[torch.Tensor] = None,
        current_mem_features: Optional[torch.Tensor] = None,
        current_mem_pos_enc: Optional[torch.Tensor] = None,
        is_first_frame: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            current_features: [B, C, H, W] - current frame backbone features
            current_pos_enc: [B, C, H, W] - position encoding
            memory_features: [B, T, C, H, W] - previous frame memory features
            memory_pos_enc: [B, T, C, H, W] - memory position encoding
            is_first_frame: whether this is the first frame (no memory)
            
        Returns:
            dict with:
                - conditioned_features: [B, C, H, W]
                - compressed_memory: [B, N, C] (if perceiver enabled)
        """
        B, C, H, W = current_features.shape

        if current_pos_enc is None:
            current_pos_enc = self.query_position_encoding(current_features).to(current_features.dtype)
        
        # Prepare memories
        if is_first_frame or memory_features is None:
            # First frame - no memory to attend to
            memory = self.no_mem_embed.expand(1, B, -1)
            memory_pos = self.no_mem_pos_enc.expand(1, B, -1)
            num_spatial = 0
        else:
            # Compress and prepare memories
            memory_list = []
            memory_pos_list = []
            
            num_frames = memory_features.shape[1]
            for t in range(num_frames):
                mem = memory_features[:, t]  # [B, C, H, W]
                pos = memory_pos_enc[:, t] if memory_pos_enc is not None else None
                
                # Compress with Perceiver
                compressed_mem, compressed_pos = self.compress_memory(mem, pos)
                memory_list.append(compressed_mem)
                memory_pos_list.append(compressed_pos)
            
            memory, memory_pos, num_spatial = self.prepare_memory(
                memory_list, memory_pos_list
            )
        
        # Apply memory attention
        conditioned_features = self.forward_memory_attention(
            current_features,
            current_pos_enc,
            memory,
            memory_pos,
            num_spatial,
        )
        
        # Optionally compress the current frame's memory feature map for storage
        if current_mem_features is not None:
            current_compressed, current_pos_compressed = self.compress_memory(
                current_mem_features, current_mem_pos_enc
            )
        else:
            current_compressed = torch.empty((B, 0, self.mem_dim), device=current_features.device)
            current_pos_compressed = torch.empty((B, 0, self.mem_dim), device=current_features.device)
        
        return {
            'conditioned_features': conditioned_features,
            'compressed_memory': current_compressed,
            'compressed_memory_pos': current_pos_compressed,
        }
    
    def get_trainable_params(self) -> list:
        """Get list of trainable parameters."""
        params = []
        
        # Perceiver parameters
        if self.spatial_perceiver is not None:
            params.extend(self.spatial_perceiver.parameters())
        
        # Memory attention parameters
        params.extend(self.memory_attention.parameters())

        # Memory projection parameters
        params.extend(self.mem_proj.parameters())
        params.extend(self.mem_pos_proj.parameters())
        
        # Temporal position encoding
        params.append(self.maskmem_tpos_enc)
        params.append(self.no_mem_embed)
        params.append(self.no_mem_pos_enc)
        
        return params
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {}
        
        if self.spatial_perceiver is not None:
            counts['perceiver'] = sum(
                p.numel() for p in self.spatial_perceiver.parameters()
            )
        
        counts['memory_attention'] = sum(
            p.numel() for p in self.memory_attention.parameters()
        )

        counts['memory_proj'] = sum(p.numel() for p in self.mem_proj.parameters()) + sum(
            p.numel() for p in self.mem_pos_proj.parameters()
        )
        
        counts['temporal_encoding'] = (
            self.maskmem_tpos_enc.numel() +
            self.no_mem_embed.numel() +
            self.no_mem_pos_enc.numel()
        )
        
        counts['total_trainable'] = sum(counts.values())
        
        return counts


def build_memory_teacher(config) -> SAM3MemoryTeacher:
    """Build teacher model for distillation."""
    return SAM3MemoryTeacher(
        checkpoint_path=config.MODEL.RESUME if config.MODEL.RESUME else None,
        image_size=config.MODEL.TRACKER.IMAGE_SIZE,
    )


def build_memory_student(config) -> SAM3MemoryStudent:
    """Build student model with efficient memory modules."""
    return SAM3MemoryStudent(config)


def load_student_checkpoint(
    model: SAM3MemoryStudent,
    checkpoint_path: str,
    strict: bool = False,
) -> Dict[str, Any]:
    """
    Load checkpoint into student model.
    
    Args:
        model: Student model
        checkpoint_path: Path to checkpoint file
        strict: Whether to require exact match
        
    Returns:
        Checkpoint dict with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Filter to only load memory-related weights
    memory_keys = [
        'spatial_perceiver', 'memory_attention',
        'maskmem_tpos_enc', 'no_mem_embed', 'no_mem_pos_enc'
    ]
    filtered_state = {
        k: v for k, v in state_dict.items()
        if any(k.startswith(prefix) for prefix in memory_keys)
    }
    
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    
    return {
        'checkpoint': checkpoint,
        'missing_keys': missing,
        'unexpected_keys': unexpected,
    }
