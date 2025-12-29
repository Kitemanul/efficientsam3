# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from sam3.model.sam3_tracker_base import Sam3TrackerBase
from sam3.model.efficient_memory_attention import EfficientMemoryAttention
from sam3.model.memory import (
    SimpleMaskEncoder,
    SimpleMaskDownSampler,
    SimpleFuser,
    CXBlock,
)
from sam3.model.position_encoding import PositionEmbeddingSine
from sam3.model.perceiver import PerceiverResampler

class EfficientSAM3Stage2(Sam3TrackerBase):
    def __init__(
        self,
        image_encoder,
        # Memory Attention args
        d_model: int = 256,
        dim_feedforward: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        pool_size: int = 2,
        pos_enc_at_input: bool = True,
        # Memory Encoder args
        mask_in_chans: int = 1,
        # Perceiver args
        use_perceiver: bool = True,
        perceiver_num_latents: int = 64,
        perceiver_depth: int = 2,
        perceiver_num_heads: int = 8,
        perceiver_head_dim: int = 64,
        perceiver_ff_mult: int = 4,
        # SAM2Base args
        num_maskmem: int = 7,
        image_size: int = 1024,
        backbone_stride: int = 16,
        sigmoid_scale_for_mem_enc: float = 1.0,
        sigmoid_bias_for_mem_enc: float = 0.0,
        binarize_mask_from_pts_for_mem_enc: bool = False,
        use_mask_input_as_output_without_sam: bool = False,
        max_cond_frames_in_attn: int = -1,
        directly_add_no_mem_embed: bool = False,
        use_high_res_features_in_sam: bool = False,
        multimask_output_in_sam: bool = False,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        multimask_output_for_tracking: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        memory_temporal_stride_for_eval: int = 1,
        non_overlap_masks_for_mem_enc: bool = False,
        use_obj_ptrs_in_encoder: bool = False,
        max_obj_ptrs_in_encoder: int = 16,
        add_tpos_enc_to_obj_ptrs: bool = True,
        proj_tpos_enc_in_obj_ptrs: bool = False,
        use_signed_tpos_enc_to_obj_ptrs: bool = False,
        only_obj_ptrs_in_the_past_for_eval: bool = False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        fixed_no_obj_ptr: bool = False,
        soft_no_obj_ptr: bool = False,
        use_mlp_for_obj_ptr_proj: bool = False,
        no_obj_embed_spatial: bool = False,
        sam_mask_decoder_extra_args: dict = None,
        compile_image_encoder: bool = False,
        # Extra args
        **kwargs,
    ):
        # 1. Build Memory Attention
        memory_attention = EfficientMemoryAttention(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            pool_size=pool_size,
            pos_enc_at_input=pos_enc_at_input,
        )

        # 2. Build Memory Encoder
        # We need to construct the components for SimpleMaskEncoder
        # Assuming out_dim=64 for memory encoder as in model_builder
        mem_dim = 64 
        
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=mem_dim, 
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=image_size,
        )
        
        mask_downsampler = SimpleMaskDownSampler(
            kernel_size=3, stride=2, padding=1, interpol_size=None 
        )

        cx_block_layer = CXBlock(
            dim=d_model, 
            kernel_size=7,
            padding=3,
            layer_scale_init_value=1.0e-06,
            use_dwconv=True,
        )

        fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)

        memory_encoder = SimpleMaskEncoder(
            out_dim=mem_dim, 
            position_encoding=position_encoding,
            mask_downsampler=mask_downsampler,
            fuser=fuser,
            in_dim=d_model,
        )

        # 3. Initialize SAM2Base (Sam3TrackerBase)
        super().__init__(
            backbone=image_encoder,
            transformer=memory_attention,
            maskmem_backbone=memory_encoder,
            num_maskmem=num_maskmem,
            image_size=image_size,
            backbone_stride=backbone_stride,
            max_cond_frames_in_attn=max_cond_frames_in_attn,
            multimask_output_in_sam=multimask_output_in_sam,
            multimask_min_pt_num=multimask_min_pt_num,
            multimask_max_pt_num=multimask_max_pt_num,
            multimask_output_for_tracking=multimask_output_for_tracking,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
        )
        
        # 4. Build Perceiver (Optional)
        if use_perceiver:
            self.spatial_perceiver = PerceiverResampler(
                dim=mem_dim, # memory_encoder out_dim
                num_latents=perceiver_num_latents,
                depth=perceiver_depth,
                num_heads=perceiver_num_heads,
                head_dim=perceiver_head_dim,
                ff_mult=perceiver_ff_mult,
            )
        else:
            self.spatial_perceiver = None

        # Set other attributes
        self.use_high_res_features_in_sam = use_high_res_features_in_sam
        self.directly_add_no_mem_embed = directly_add_no_mem_embed
        self.sigmoid_scale_for_mem_enc = sigmoid_scale_for_mem_enc
        self.sigmoid_bias_for_mem_enc = sigmoid_bias_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = binarize_mask_from_pts_for_mem_enc
        self.use_mask_input_as_output_without_sam = use_mask_input_as_output_without_sam
        self.iou_prediction_use_sigmoid = iou_prediction_use_sigmoid
        self.memory_temporal_stride_for_eval = memory_temporal_stride_for_eval
        self.non_overlap_masks_for_mem_enc = non_overlap_masks_for_mem_enc
        self.use_obj_ptrs_in_encoder = use_obj_ptrs_in_encoder
        self.add_tpos_enc_to_obj_ptrs = add_tpos_enc_to_obj_ptrs
        self.proj_tpos_enc_in_obj_ptrs = proj_tpos_enc_in_obj_ptrs
        self.use_signed_tpos_enc_to_obj_ptrs = use_signed_tpos_enc_to_obj_ptrs
        self.only_obj_ptrs_in_the_past_for_eval = only_obj_ptrs_in_the_past_for_eval
        
        # Compilation
        if compile_image_encoder:
             self.backbone.forward = torch.compile(
                self.backbone.forward,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )

    def forward_image(self, img_batch):
        """Get the image feature on the input batch."""
        # Override to use SAM3 features directly
        backbone_out_raw = self.backbone.forward_image(img_batch)
        
        # Use the main features (SAM3 features)
        backbone_out = {
            "backbone_fpn": backbone_out_raw["backbone_fpn"],
            "vision_pos_enc": backbone_out_raw["vision_pos_enc"],
        }
        
        # precompute projected level 0 and level 1 features in SAM decoder
        # to avoid running it again on every SAM click
        if self.use_high_res_features_in_sam:
            backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
            
        return backbone_out

    def _encode_new_memory(
        self,
        image,
        current_vision_feats,
        feat_sizes,
        pred_masks_high_res,
        object_score_logits,
        is_mask_from_pts,
        output_dict=None,
        is_init_cond_frame=False,
    ):
        """Encode the current image and its prediction into a memory feature."""
        # Get initial memory features from parent (SimpleMaskEncoder)
        maskmem_features, maskmem_pos_enc = super()._encode_new_memory(
            image,
            current_vision_feats,
            feat_sizes,
            pred_masks_high_res,
            object_score_logits,
            is_mask_from_pts,
            output_dict,
            is_init_cond_frame,
        )

        # Apply Perceiver Resampler if enabled
        if self.spatial_perceiver is not None:
            # maskmem_features: [B, C, H, W]
            # maskmem_pos_enc: [B, C, H, W]
            
            # Perceiver expects [B, C, H, W] input and returns [B, N, C]
            maskmem_features, maskmem_pos_enc = self.spatial_perceiver(
                maskmem_features, maskmem_pos_enc
            )
            
            # Sam3TrackerBase supports [B, N, C] features (it permutes them to [N, B, C])
            # So we can return them directly.
                
        return maskmem_features, maskmem_pos_enc

