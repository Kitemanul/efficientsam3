# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# EfficientSAM3 Stage 2: Training Wrapper
#
# This module provides SAM2-style training for EfficientSAM3 video tracking.
# Follows the SAM2Train pattern: forward() -> prepare_prompt_inputs() -> forward_tracking()

import logging
from typing import Optional, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.efficient_sam3_tracker import EfficientSam3TrackerBase
from sam3.model.sam3_tracker_utils import get_next_point, sample_box_points

logger = logging.getLogger(__name__)


class EfficientSam3Train(EfficientSam3TrackerBase):
    """
    Training wrapper for EfficientSAM3 video tracker.
    
    Similar to SAM2Train, this class extends the base tracker with:
    1. forward() method that orchestrates training
    2. prepare_prompt_inputs() for sampling point/mask prompts from GT
    3. forward_tracking() for processing frames sequentially
    4. Iterative point correction sampling
    
    Key training features:
    - Point prompt sampling from ground truth masks
    - Box prompt sampling (optional)
    - Multi-frame conditioning
    - Iterative correction click sampling
    - Memory encoding for temporal consistency
    """
    
    def __init__(
        self,
        # Training-specific args
        prob_to_use_pt_input_for_train: float = 1.0,
        prob_to_use_pt_input_for_eval: float = 1.0,
        prob_to_use_box_input_for_train: float = 0.0,
        prob_to_use_box_input_for_eval: float = 0.0,
        num_frames_to_correct_for_train: int = 1,
        num_frames_to_correct_for_eval: int = 1,
        rand_frames_to_correct_for_train: bool = False,
        rand_frames_to_correct_for_eval: bool = False,
        num_init_cond_frames_for_train: int = 1,
        num_init_cond_frames_for_eval: int = 1,
        rand_init_cond_frames_for_train: bool = True,
        rand_init_cond_frames_for_eval: bool = False,
        add_all_frames_to_correct_as_cond: bool = True,
        num_correction_pt_per_frame: int = 7,
        pt_sampling_for_eval: str = "uniform",
        prob_to_sample_from_gt_for_train: float = 0.0,
        use_act_ckpt_iterative_pt_sampling: bool = False,
        forward_backbone_per_frame_for_eval: bool = False,
        freeze_image_encoder: bool = True,
        freeze_sam_heads: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # Point sampling settings
        self.prob_to_use_pt_input_for_train = prob_to_use_pt_input_for_train
        self.prob_to_use_box_input_for_train = prob_to_use_box_input_for_train
        self.prob_to_use_pt_input_for_eval = prob_to_use_pt_input_for_eval
        self.prob_to_use_box_input_for_eval = prob_to_use_box_input_for_eval
        
        # Correction frames
        self.num_frames_to_correct_for_train = num_frames_to_correct_for_train
        self.num_frames_to_correct_for_eval = num_frames_to_correct_for_eval
        self.rand_frames_to_correct_for_train = rand_frames_to_correct_for_train
        self.rand_frames_to_correct_for_eval = rand_frames_to_correct_for_eval
        
        # Initial conditioning frames
        self.num_init_cond_frames_for_train = num_init_cond_frames_for_train
        self.num_init_cond_frames_for_eval = num_init_cond_frames_for_eval
        self.rand_init_cond_frames_for_train = rand_init_cond_frames_for_train
        self.rand_init_cond_frames_for_eval = rand_init_cond_frames_for_eval
        
        # Other training settings
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.num_correction_pt_per_frame = num_correction_pt_per_frame
        self.pt_sampling_for_eval = pt_sampling_for_eval
        self.prob_to_sample_from_gt_for_train = prob_to_sample_from_gt_for_train
        self.use_act_ckpt_iterative_pt_sampling = use_act_ckpt_iterative_pt_sampling
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        
        # Random number generator (fixed seed for reproducibility)
        self.rng = np.random.default_rng(seed=42)
        
        # Freeze components
        if freeze_image_encoder and hasattr(self, 'backbone'):
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("Froze image encoder parameters")
        
        if freeze_sam_heads:
            for p in self.sam_mask_decoder.parameters():
                p.requires_grad = False
            for p in self.sam_prompt_encoder.parameters():
                p.requires_grad = False
            for p in self.obj_ptr_proj.parameters():
                p.requires_grad = False
            for p in self.mask_downsample.parameters():
                p.requires_grad = False
            logger.info("Froze SAM head parameters")
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (memory-related only)."""
        trainable_params = []
        
        # 1. Memory encoder (maskmem_backbone)
        trainable_params.extend(self.maskmem_backbone.parameters())
        
        # 2. Perceiver (if exists)
        if hasattr(self, 'spatial_perceiver') and self.spatial_perceiver is not None:
            trainable_params.extend(self.spatial_perceiver.parameters())
        
        # 3. Memory attention
        if hasattr(self, 'memory_attention') and self.memory_attention is not None:
            trainable_params.extend(self.memory_attention.parameters())
        
        # 4. Projection layers
        if hasattr(self, 'mem_proj'):
            trainable_params.extend(self.mem_proj.parameters())
        if hasattr(self, 'mem_pos_proj'):
            trainable_params.extend(self.mem_pos_proj.parameters())
        
        # 5. Temporal position encoding and embeddings
        trainable_params.append(self.maskmem_tpos_enc)
        trainable_params.append(self.no_mem_embed)
        trainable_params.append(self.no_mem_pos_enc)
        trainable_params.append(self.no_obj_embed_spatial)
        
        return trainable_params
    
    def forward(self, input_batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Forward pass for training.
        
        Args:
            input_batch: Dictionary containing:
                - flat_img_batch: [T*B, C, H, W] flattened images
                - masks: [T, B, H, W] ground truth masks
                - flat_obj_to_img_idx: mapping from frame to image indices
                - num_frames: number of frames
                
        Returns:
            List of per-frame outputs containing pred_masks and other tensors
        """
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # Precompute image features on all frames
            backbone_out = self.forward_image(input_batch["flat_img_batch"])
        else:
            # Defer image feature computation
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}
        
        # Prepare prompt inputs (sample points from GT masks)
        backbone_out = self.prepare_prompt_inputs(backbone_out, input_batch)
        
        # Forward tracking on each frame
        previous_stages_out = self.forward_tracking(backbone_out, input_batch)
        
        return previous_stages_out
    
    def prepare_prompt_inputs(
        self,
        backbone_out: Dict[str, Any],
        input_batch: Dict[str, Any],
        start_frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Prepare point/mask prompts from ground truth masks.
        
        Args:
            backbone_out: Backbone features
            input_batch: Input batch with GT masks
            start_frame_idx: Starting frame index
            
        Returns:
            Updated backbone_out with prompt inputs
        """
        # Load GT masks
        gt_masks_per_frame = {}
        for t in range(input_batch["num_frames"]):
            if isinstance(input_batch["masks"], dict):
                mask_t = input_batch["masks"][t]
                # Mask should be [B, 1, H, W] - don't unsqueeze if already 4D
                if mask_t.dim() == 3:
                    mask_t = mask_t.unsqueeze(1)
                gt_masks_per_frame[t] = mask_t
            else:
                mask_t = input_batch["masks"][t]
                if mask_t.dim() == 3:
                    mask_t = mask_t.unsqueeze(1)
                gt_masks_per_frame[t] = mask_t
        
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
        num_frames = input_batch["num_frames"]
        backbone_out["num_frames"] = num_frames
        
        # Training vs eval settings
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval
        
        # Handle single-frame case
        if num_frames == 1:
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1
        
        # Decide whether to use point input
        use_pt_input = self.rng.random() < prob_to_use_pt_input
        
        # Select initial conditioning frames
        if rand_init_cond_frames and num_init_cond_frames > 1:
            num_init_cond_frames = self.rng.integers(1, num_init_cond_frames + 1)
        
        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]
        else:
            # Uniformly sample conditioning frames
            init_cond_frames = [start_frame_idx]
            other_frames = list(range(start_frame_idx + 1, num_frames))
            if len(other_frames) > 0:
                extra_frames = self.rng.choice(
                    other_frames,
                    size=min(num_init_cond_frames - 1, len(other_frames)),
                    replace=False,
                ).tolist()
                init_cond_frames.extend(sorted(extra_frames))
        
        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = [
            t for t in range(num_frames) if t not in init_cond_frames
        ]
        
        # Sample prompts on initial conditioning frames
        backbone_out["point_inputs_per_frame"] = {}
        backbone_out["mask_inputs_per_frame"] = {}
        
        if use_pt_input:
            for t in init_cond_frames:
                gt_mask = gt_masks_per_frame[t]
                
                # Use box or point input
                use_box_input = self.rng.random() < prob_to_use_box_input
                if use_box_input:
                    points, labels = sample_box_points(gt_mask)
                else:
                    points, labels = get_next_point(
                        gt_masks=gt_mask,
                        pred_masks=None,
                        method="uniform" if self.training else self.pt_sampling_for_eval,
                    )
                
                backbone_out["point_inputs_per_frame"][t] = {
                    "point_coords": points,
                    "point_labels": labels,
                }
        else:
            # Use mask input
            for t in init_cond_frames:
                backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
        
        # Select frames for correction clicks
        if use_pt_input and num_frames_to_correct > 1:
            correction_candidates = backbone_out["frames_not_in_init_cond"]
            if rand_frames_to_correct:
                num_to_select = min(
                    num_frames_to_correct - len(init_cond_frames),
                    len(correction_candidates)
                )
                if num_to_select > 0:
                    frames_to_add_correction_pt = self.rng.choice(
                        correction_candidates,
                        size=num_to_select,
                        replace=False,
                    ).tolist()
                else:
                    frames_to_add_correction_pt = []
            else:
                frames_to_add_correction_pt = correction_candidates[:num_frames_to_correct - len(init_cond_frames)]
        else:
            frames_to_add_correction_pt = []
        
        backbone_out["frames_to_add_correction_pt"] = frames_to_add_correction_pt
        
        return backbone_out
    
    def forward_tracking(
        self,
        backbone_out: Dict[str, Any],
        input_batch: Dict[str, Any],
        return_dict: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Forward video tracking on each frame.
        
        Args:
            backbone_out: Backbone features and prompts
            input_batch: Input batch
            return_dict: Whether to return as dict or list
            
        Returns:
            List of per-frame outputs
        """
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        
        if img_feats_already_computed:
            # Prepare backbone features
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)
        
        # Get tracking parameters
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
        
        # Processing order: conditioning frames first, then rest
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]
        
        output_dict = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        for stage_id in processing_order:
            # Get image indices for this frame
            if "flat_obj_to_img_idx" in input_batch:
                img_ids = input_batch["flat_obj_to_img_idx"][stage_id]
            else:
                img_ids = torch.tensor([stage_id], device=self.device)
            
            if img_feats_already_computed:
                # Retrieve precomputed features
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Compute features on the fly
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input_batch["flat_img_batch"], img_ids
                )
            
            # Get prompts for this frame
            point_inputs = backbone_out["point_inputs_per_frame"].get(stage_id, None)
            mask_inputs = backbone_out["mask_inputs_per_frame"].get(stage_id, None)
            
            # Run tracking step
            current_out = self.track_step(
                frame_idx=stage_id,
                is_init_cond_frame=stage_id in init_cond_frames,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
                gt_masks=backbone_out["gt_masks_per_frame"].get(stage_id, None),
                frames_to_add_correction_pt=frames_to_add_correction_pt,
            )
            
            # Store output
            add_output_as_cond_frame = stage_id in init_cond_frames or (
                self.add_all_frames_to_correct_as_cond
                and stage_id in frames_to_add_correction_pt
            )
            if add_output_as_cond_frame:
                output_dict["cond_frame_outputs"][stage_id] = current_out
            else:
                output_dict["non_cond_frame_outputs"][stage_id] = current_out
        
        if return_dict:
            return output_dict
        
        # Convert to list for loss function
        all_frame_outputs = {}
        all_frame_outputs.update(output_dict["cond_frame_outputs"])
        all_frame_outputs.update(output_dict["non_cond_frame_outputs"])
        all_frame_outputs = [all_frame_outputs[t] for t in range(num_frames)]
        
        # Remove obj_ptr to make DDP happy with activation checkpointing
        all_frame_outputs = [
            {k: v for k, v in d.items() if k != "obj_ptr"} for d in all_frame_outputs
        ]
        
        return all_frame_outputs
    
    def track_step(
        self,
        frame_idx: int,
        is_init_cond_frame: bool,
        current_vision_feats: List[torch.Tensor],
        current_vision_pos_embeds: List[torch.Tensor],
        feat_sizes: List[tuple],
        point_inputs: Optional[Dict] = None,
        mask_inputs: Optional[torch.Tensor] = None,
        output_dict: Optional[Dict] = None,
        num_frames: int = 1,
        track_in_reverse: bool = False,
        run_mem_encoder: bool = True,
        prev_sam_mask_logits: Optional[torch.Tensor] = None,
        gt_masks: Optional[torch.Tensor] = None,
        frames_to_add_correction_pt: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Single tracking step on a frame.
        
        This method:
        1. Fuses current features with memory from past frames
        2. Runs SAM decoder to get masks
        3. Optionally samples correction points
        4. Encodes output into memory
        
        Returns:
            Dictionary with pred_masks, ious, object_score_logits, etc.
        """
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []
        
        # Get image for memory encoding
        # (would come from input_batch in full implementation)
        image = None
        
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        # Use actual channel dimensions from the tensors (not self.hidden_dim)
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        
        # Fuse with memory from past frames (only pass last feature level)
        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats[-1:],  # Only last level
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
        )
        
        # Decide whether to use multimask output
        if point_inputs is not None:
            num_pts = point_inputs["point_labels"].size(1)
            multimask_output = (
                self.multimask_output_in_sam
                and (is_init_cond_frame or self.multimask_output_for_tracking)
                and self.multimask_min_pt_num <= num_pts <= self.multimask_max_pt_num
            )
        else:
            multimask_output = False
        
        # Run SAM heads
        if mask_inputs is not None and self._use_mask_as_output:
            # Directly use mask input as output
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = self._use_mask_as_output(
                pix_feat_with_mem, high_res_features, mask_inputs
            )
        else:
            # Use SAM decoder
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = self._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
                gt_masks=gt_masks,
            )
        
        # Build output dict
        current_out = {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "ious": ious,
            "object_score_logits": object_score_logits,
            "obj_ptr": obj_ptr,
        }
        
        # Encode memory for future frames
        if run_mem_encoder:
            is_mask_from_pts = point_inputs is not None
            maskmem_features, maskmem_pos_enc = self._encode_new_memory(
                image=image,
                current_vision_feats=current_vision_feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=object_score_logits,
                is_mask_from_pts=is_mask_from_pts,
                output_dict=output_dict,
                is_init_cond_frame=is_init_cond_frame,
            )
            current_out["maskmem_features"] = maskmem_features
            current_out["maskmem_pos_enc"] = maskmem_pos_enc
        
        return current_out
