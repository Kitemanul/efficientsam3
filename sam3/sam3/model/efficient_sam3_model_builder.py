# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# EfficientSAM3 Stage 2: Model Builder
#
# Builder functions for EfficientSAM3 video tracking models.
# Follows SAM2/SAM3 patterns for model construction and checkpoint loading.

import os
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from sam3.model.efficient_sam3_tracker import (
    EfficientSam3TrackerBase,
    EfficientSam3TrackerPredictor,
)
from sam3.model.efficient_sam3_train import EfficientSam3Train


class SAM3VisionOnlyBackbone(nn.Module):
    """
    Wrapper for SAM3's vision backbone that excludes the language backbone.
    
    For SAM2-style training, we only need the vision backbone (463M params),
    not the language backbone (354M params). This wrapper provides the same
    forward_image() interface as SAM3VLBackbone but only uses vision_backbone.
    
    This reduces model size from 816M to 463M for the image encoder portion.
    """
    
    def __init__(self, vision_backbone, scalp: int = 0):
        super().__init__()
        self.vision_backbone = vision_backbone
        self.scalp = scalp
    
    def forward_image(self, samples: torch.Tensor):
        """Forward pass through vision backbone only.
        
        Returns the same format as SAM3VLBackbone.forward_image() but without
        any language-related outputs.
        """
        # Forward through vision backbone
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples
        )
        
        if self.scalp > 0:
            # Discard the lowest resolution features
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )
        
        sam2_output = None
        if sam2_features is not None and sam2_pos is not None:
            sam2_src = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }
        
        sam3_src = sam3_features[-1]
        output = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }
        
        return output


def build_efficient_sam3_tracker(
    # Backbone options
    backbone_type: str = "sam3",  # "sam3", "repvit", "tinyvit", "efficientvit"
    backbone_checkpoint: Optional[str] = None,
    sam3_checkpoint: Optional[str] = None,
    # Model architecture
    d_model: int = 256,
    num_maskmem: int = 7,
    image_size: int = 1008,
    backbone_stride: int = 14,
    # Memory attention args
    num_heads: int = 8,
    num_layers: int = 2,
    dim_feedforward: int = 1024,
    # Perceiver args
    use_perceiver: bool = True,
    perceiver_num_latents: int = 64,
    perceiver_depth: int = 2,
    perceiver_num_heads: int = 8,
    # SAM decoder args
    sam_mask_decoder_extra_args: Optional[Dict] = None,
    # Device and mode
    device: str = "cuda",
    mode: str = "eval",
    # Load pretrained SAM heads
    load_pretrained_sam_heads: bool = True,
    **kwargs,
) -> EfficientSam3TrackerBase:
    """
    Build EfficientSAM3 tracker for video segmentation.
    
    Args:
        backbone_type: Type of image encoder ("sam3", "repvit", "tinyvit", "efficientvit")
        backbone_checkpoint: Path to backbone checkpoint (for efficient backbones)
        sam3_checkpoint: Path to SAM3 checkpoint (for loading pretrained weights)
        d_model: Model dimension
        num_maskmem: Number of memory frames
        image_size: Input image size
        backbone_stride: Backbone feature stride
        num_heads: Number of attention heads in memory attention
        num_layers: Number of memory attention layers
        dim_feedforward: Feed-forward dimension
        use_perceiver: Whether to use Perceiver for memory compression
        perceiver_num_latents: Number of Perceiver latent tokens
        perceiver_depth: Perceiver depth
        perceiver_num_heads: Perceiver attention heads
        sam_mask_decoder_extra_args: Extra args for SAM mask decoder
        device: Device to load model on
        mode: "eval" or "train"
        load_pretrained_sam_heads: Whether to load pretrained SAM heads from SAM3
        
    Returns:
        EfficientSam3TrackerBase model
    """
    # 1. Build image encoder
    if backbone_type == "sam3":
        # Use SAM3's vision backbone only (not language backbone)
        from sam3.model_builder import build_sam3_video_model
        
        sam3_model = build_sam3_video_model(
            checkpoint_path=sam3_checkpoint,
            load_from_HF=sam3_checkpoint is None,
        )
        
        # Extract ONLY the vision backbone (463M params)
        # Exclude language backbone (354M params) - not needed for SAM2-style
        vl_backbone = sam3_model.detector.backbone  # SAM3VLBackbone
        vision_backbone = vl_backbone.vision_backbone  # Sam3DualViTDetNeck
        scalp = vl_backbone.scalp
        
        # Wrap vision backbone with SAM3VisionOnlyBackbone
        image_encoder = SAM3VisionOnlyBackbone(vision_backbone, scalp=scalp)
        sam3_tracker = sam3_model.tracker
        
        # Clean up detector to save memory
        del sam3_model.detector
        del vl_backbone.language_backbone
        del vl_backbone
        torch.cuda.empty_cache()
        
    elif backbone_type == "repvit":
        from sam3.backbones.repvit import build_repvit_backbone
        image_encoder = build_repvit_backbone(
            checkpoint=backbone_checkpoint,
            variant="m1",  # or "m2", "m3"
        )
        sam3_tracker = None
        
    elif backbone_type == "tinyvit":
        from sam3.backbones.tiny_vit import build_tinyvit_backbone
        image_encoder = build_tinyvit_backbone(
            checkpoint=backbone_checkpoint,
            variant="5m",  # or "11m", "21m"
        )
        sam3_tracker = None
        
    elif backbone_type == "efficientvit":
        from sam3.backbones.efficientvit import build_efficientvit_backbone
        image_encoder = build_efficientvit_backbone(
            checkpoint=backbone_checkpoint,
        )
        sam3_tracker = None
        
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")
    
    # 2. Build the tracker
    if sam_mask_decoder_extra_args is None:
        sam_mask_decoder_extra_args = {
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        }
    
    model = EfficientSam3TrackerBase(
        image_encoder=image_encoder,
        d_model=d_model,
        num_maskmem=num_maskmem,
        image_size=image_size,
        backbone_stride=backbone_stride,
        num_heads=num_heads,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        use_perceiver=use_perceiver,
        perceiver_num_latents=perceiver_num_latents,
        perceiver_depth=perceiver_depth,
        perceiver_num_heads=perceiver_num_heads,
        sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
        **kwargs,
    )
    
    # 3. Load pretrained SAM heads from SAM3 tracker
    if load_pretrained_sam_heads and sam3_tracker is not None:
        print("Loading pretrained SAM heads from SAM3 tracker...")
        _load_sam_heads_from_tracker(model, sam3_tracker)
        del sam3_tracker
        torch.cuda.empty_cache()
    
    # 4. Move to device and set mode
    model = model.to(device)
    if mode == "eval":
        model.eval()
    
    return model


def build_efficient_sam3_predictor(
    **kwargs,
) -> EfficientSam3TrackerPredictor:
    """
    Build EfficientSAM3 predictor for interactive video segmentation.
    
    This is the inference-focused model with methods like:
    - init_state()
    - add_new_points()
    - propagate_in_video()
    
    Args:
        **kwargs: Same arguments as build_efficient_sam3_tracker
        
    Returns:
        EfficientSam3TrackerPredictor model
    """
    # Get base model args
    backbone_type = kwargs.pop("backbone_type", "sam3")
    backbone_checkpoint = kwargs.pop("backbone_checkpoint", None)
    sam3_checkpoint = kwargs.pop("sam3_checkpoint", None)
    device = kwargs.pop("device", "cuda")
    mode = kwargs.pop("mode", "eval")
    load_pretrained_sam_heads = kwargs.pop("load_pretrained_sam_heads", True)
    
    # Build image encoder (same as tracker)
    if backbone_type == "sam3":
        from sam3.model_builder import build_sam3_video_model
        
        sam3_model = build_sam3_video_model(
            checkpoint_path=sam3_checkpoint,
            load_from_HF=sam3_checkpoint is None,
        )
        
        # Extract ONLY the vision backbone (463M params)
        # Exclude language backbone (354M params) - not needed for SAM2-style
        vl_backbone = sam3_model.detector.backbone  # SAM3VLBackbone
        vision_backbone = vl_backbone.vision_backbone  # Sam3DualViTDetNeck
        scalp = vl_backbone.scalp
        
        # Wrap vision backbone with SAM3VisionOnlyBackbone
        image_encoder = SAM3VisionOnlyBackbone(vision_backbone, scalp=scalp)
        sam3_tracker = sam3_model.tracker
        
        del sam3_model.detector
        del vl_backbone.language_backbone
        del vl_backbone
        torch.cuda.empty_cache()
    else:
        # Efficient backbones
        image_encoder = _build_efficient_backbone(backbone_type, backbone_checkpoint)
        sam3_tracker = None
    
    # Build predictor
    model = EfficientSam3TrackerPredictor(
        image_encoder=image_encoder,
        **kwargs,
    )
    
    # Load pretrained SAM heads
    if load_pretrained_sam_heads and sam3_tracker is not None:
        print("Loading pretrained SAM heads from SAM3 tracker...")
        _load_sam_heads_from_tracker(model, sam3_tracker)
        del sam3_tracker
        torch.cuda.empty_cache()
    
    model = model.to(device)
    if mode == "eval":
        model.eval()
    
    return model


def build_efficient_sam3_train(
    # Training-specific args
    freeze_image_encoder: bool = True,
    freeze_sam_heads: bool = True,
    prob_to_use_pt_input_for_train: float = 1.0,
    num_init_cond_frames_for_train: int = 1,
    **kwargs,
) -> EfficientSam3Train:
    """
    Build EfficientSAM3 training model.
    
    This model includes training-specific features:
    - Point/mask prompt sampling from GT
    - Iterative correction point sampling
    - Multi-frame conditioning
    
    For SAM2-style training with backbone_type="sam3":
    - Uses only the vision backbone from SAM3 (463M params)
    - Excludes the language backbone (354M params) - not needed for SAM2-style
    - Total model size ~478M instead of 823M
    
    Args:
        freeze_image_encoder: Whether to freeze image encoder
        freeze_sam_heads: Whether to freeze SAM decoder/prompt encoder
        prob_to_use_pt_input_for_train: Probability to use point input
        num_init_cond_frames_for_train: Number of initial conditioning frames
        **kwargs: Same arguments as build_efficient_sam3_tracker
        
    Returns:
        EfficientSam3Train model
    """
    # Get base model args
    backbone_type = kwargs.pop("backbone_type", "sam3")
    backbone_checkpoint = kwargs.pop("backbone_checkpoint", None)
    sam3_checkpoint = kwargs.pop("sam3_checkpoint", None)
    device = kwargs.pop("device", "cuda")
    load_pretrained_sam_heads = kwargs.pop("load_pretrained_sam_heads", True)
    
    # Build image encoder
    if backbone_type == "sam3":
        from sam3.model_builder import build_sam3_video_model
        
        sam3_model = build_sam3_video_model(
            checkpoint_path=sam3_checkpoint,
            load_from_HF=sam3_checkpoint is None,
        )
        
        # Extract ONLY the vision backbone (463M params)
        # Exclude language backbone (354M params) - not needed for SAM2-style training
        vl_backbone = sam3_model.detector.backbone  # SAM3VLBackbone
        vision_backbone = vl_backbone.vision_backbone  # Sam3DualViTDetNeck
        scalp = vl_backbone.scalp
        
        # Wrap vision backbone with SAM3VisionOnlyBackbone for forward_image() interface
        image_encoder = SAM3VisionOnlyBackbone(vision_backbone, scalp=scalp)
        
        # Keep tracker for loading SAM heads
        sam3_tracker = sam3_model.tracker
        
        # Clean up to save memory
        del sam3_model.detector
        del vl_backbone.language_backbone  # Explicitly delete language backbone
        del vl_backbone
        torch.cuda.empty_cache()
    else:
        image_encoder = _build_efficient_backbone(backbone_type, backbone_checkpoint)
        sam3_tracker = None
    
    # Build training model
    model = EfficientSam3Train(
        image_encoder=image_encoder,
        freeze_image_encoder=freeze_image_encoder,
        freeze_sam_heads=freeze_sam_heads,
        prob_to_use_pt_input_for_train=prob_to_use_pt_input_for_train,
        num_init_cond_frames_for_train=num_init_cond_frames_for_train,
        **kwargs,
    )
    
    # Load pretrained SAM heads
    if load_pretrained_sam_heads and sam3_tracker is not None:
        print("Loading pretrained SAM heads from SAM3 tracker...")
        _load_sam_heads_from_tracker(model, sam3_tracker)
        del sam3_tracker
        torch.cuda.empty_cache()
    
    model = model.to(device)
    model.train()
    # Keep frozen components in eval mode to avoid stochastic behavior (dropout/BN)
    if freeze_image_encoder and hasattr(model, "backbone"):
        model.backbone.eval()
    if freeze_sam_heads:
        for m in [
            getattr(model, "sam_mask_decoder", None),
            getattr(model, "sam_prompt_encoder", None),
            getattr(model, "obj_ptr_proj", None),
            getattr(model, "mask_downsample", None),
            getattr(model, "obj_ptr_tpos_proj", None),
        ]:
            if m is not None:
                m.eval()
    
    return model


def _build_efficient_backbone(backbone_type: str, checkpoint: Optional[str] = None):
    """Build efficient backbone (RepViT, TinyViT, EfficientViT)."""
    if backbone_type == "repvit":
        from sam3.backbones.repvit import build_repvit_backbone
        return build_repvit_backbone(checkpoint=checkpoint)
    elif backbone_type == "tinyvit":
        from sam3.backbones.tiny_vit import build_tinyvit_backbone
        return build_tinyvit_backbone(checkpoint=checkpoint)
    elif backbone_type == "efficientvit":
        from sam3.backbones.efficientvit import build_efficientvit_backbone
        return build_efficientvit_backbone(checkpoint=checkpoint)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def _load_sam_heads_from_tracker(model: EfficientSam3TrackerBase, sam3_tracker):
    """Load pretrained SAM heads from SAM3 tracker."""
    # Copy mask decoder weights
    model.sam_mask_decoder.load_state_dict(
        sam3_tracker.sam_mask_decoder.state_dict()
    )
    
    # Copy prompt encoder weights
    model.sam_prompt_encoder.load_state_dict(
        sam3_tracker.sam_prompt_encoder.state_dict()
    )
    
    # Copy object pointer projection
    model.obj_ptr_proj.load_state_dict(
        sam3_tracker.obj_ptr_proj.state_dict()
    )
    
    # Copy mask downsample convolution
    model.mask_downsample.load_state_dict(
        sam3_tracker.mask_downsample.state_dict()
    )
    
    # Copy embeddings
    model.no_obj_ptr.data.copy_(sam3_tracker.no_obj_ptr.data)
    model.no_obj_embed_spatial.data.copy_(sam3_tracker.no_obj_embed_spatial.data)
    model.no_mem_embed.data.copy_(sam3_tracker.no_mem_embed.data)
    model.no_mem_pos_enc.data.copy_(sam3_tracker.no_mem_pos_enc.data)
    model.maskmem_tpos_enc.data.copy_(sam3_tracker.maskmem_tpos_enc.data)
    model.obj_ptr_tpos_proj.load_state_dict(
        sam3_tracker.obj_ptr_tpos_proj.state_dict()
    )
    
    print("Successfully loaded pretrained SAM heads")


def load_efficient_sam3_checkpoint(
    model: EfficientSam3TrackerBase,
    checkpoint_path: str,
    strict: bool = True,
) -> None:
    """
    Load EfficientSAM3 checkpoint.
    
    Args:
        model: EfficientSam3TrackerBase model
        checkpoint_path: Path to checkpoint file
        strict: Whether to strictly enforce state dict matching
    """
    print(f"Loading checkpoint from {checkpoint_path}")
    
    if checkpoint_path.startswith("http"):
        checkpoint = torch.hub.load_state_dict_from_url(checkpoint_path, map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Handle different checkpoint formats
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    # Remove "module." prefix if present (from DDP training)
    state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print("Checkpoint loaded successfully")


def save_efficient_sam3_checkpoint(
    model: EfficientSam3TrackerBase,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: int = 0,
    **extra_state,
) -> None:
    """
    Save EfficientSAM3 checkpoint.
    
    Args:
        model: EfficientSam3TrackerBase model
        checkpoint_path: Path to save checkpoint
        optimizer: Optional optimizer to save
        epoch: Current epoch number
        **extra_state: Extra state to save
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # Get model state dict (handle DDP)
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    checkpoint = {
        "model": state_dict,
        "epoch": epoch,
        **extra_state,
    }
    
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")
