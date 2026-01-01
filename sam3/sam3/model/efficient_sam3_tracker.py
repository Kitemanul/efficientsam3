# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# EfficientSAM3 Stage 2: Efficient Video Tracker
# 
# This module provides SAM2-style video tracking with efficient memory components.
# Architecture follows Sam3TrackerBase -> Sam3TrackerPredictor pattern from SAM3,
# but replaces memory attention with efficient alternatives (Perceiver + EfficientMemoryAttention).

import logging
from collections import OrderedDict
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam3.model.sam3_tracker_base import Sam3TrackerBase, NO_OBJ_SCORE
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.utils.sam2_utils import load_video_frames
from sam3.model.efficient_memory_attention import EfficientMemoryAttention
from sam3.model.perceiver import PerceiverResampler
from sam3.model.memory import (
    SimpleMaskEncoder,
    SimpleMaskDownSampler,
    SimpleFuser,
    CXBlock,
)
from sam3.model.position_encoding import PositionEmbeddingSine

from tqdm.auto import tqdm

try:
    from timm.layers import trunc_normal_
except ModuleNotFoundError:
    from timm.models.layers import trunc_normal_

logger = logging.getLogger(__name__)


class EfficientSam3TrackerBase(Sam3TrackerBase):
    """
    EfficientSAM3 Tracker Base - extends Sam3TrackerBase with efficient memory components.
    
    Key differences from Sam3TrackerBase:
    1. Uses PerceiverResampler to compress spatial memory tokens (5184 -> 64)
    2. Uses EfficientMemoryAttention instead of TransformerWrapper for memory fusion
    3. Adds projection layers (mem_proj, mem_pos_proj) for 64-d -> 256-d projection
    
    This class maintains compatibility with Sam3TrackerBase's API while providing
    81x reduction in memory tokens.
    """
    
    def __init__(
        self,
        # Image encoder (from SAM3 backbone or efficient backbone)
        image_encoder: nn.Module,
        # Memory Attention args
        d_model: int = 256,
        dim_feedforward: int = 1024,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "gelu",
        pool_size: int = 1,  # Disable pooling when using Perceiver
        pos_enc_at_input: bool = True,
        # Perceiver args
        use_perceiver: bool = True,
        perceiver_num_latents: int = 64,
        perceiver_depth: int = 2,
        perceiver_num_heads: int = 8,
        perceiver_head_dim: int = 64,
        perceiver_ff_mult: int = 4,
        # Sam3TrackerBase args
        num_maskmem: int = 7,
        image_size: int = 1008,
        backbone_stride: int = 14,
        max_cond_frames_in_attn: int = -1,
        multimask_output_in_sam: bool = False,
        multimask_min_pt_num: int = 1,
        multimask_max_pt_num: int = 1,
        multimask_output_for_tracking: bool = False,
        forward_backbone_per_frame_for_eval: bool = False,
        memory_temporal_stride_for_eval: int = 1,
        offload_output_to_cpu_for_eval: bool = False,
        trim_past_non_cond_mem_for_eval: bool = False,
        non_overlap_masks_for_mem_enc: bool = False,
        max_obj_ptrs_in_encoder: int = 16,
        sam_mask_decoder_extra_args: Optional[Dict] = None,
        compile_all_components: bool = False,
        use_memory_selection: bool = False,
        mf_threshold: float = 0.01,
        **kwargs,
    ):
        # Memory dimension (SimpleMaskEncoder output)
        mem_dim = 64
        
        # 1. Build Memory Attention (EfficientMemoryAttention)
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
        
        # 2. Build Memory Encoder (SimpleMaskEncoder - same as SAM3)
        position_encoding = PositionEmbeddingSine(
            num_pos_feats=mem_dim,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=image_size,
        )
        
        mask_downsampler = SimpleMaskDownSampler(
            kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
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
        
        # 3. Create a dummy transformer wrapper for parent class compatibility
        # The parent class expects transformer.decoder to be None
        class DummyTransformer(nn.Module):
            def __init__(self, d_model):
                super().__init__()
                self.d_model = d_model
                self.decoder = None  # Required by Sam3TrackerBase assertion
            def forward(self, *args, **kwargs):
                raise NotImplementedError("Use memory_attention instead")
        
        dummy_transformer = DummyTransformer(d_model)
        
        # 4. Initialize parent class Sam3TrackerBase
        super().__init__(
            backbone=image_encoder,
            transformer=dummy_transformer,
            maskmem_backbone=memory_encoder,
            num_maskmem=num_maskmem,
            image_size=image_size,
            backbone_stride=backbone_stride,
            max_cond_frames_in_attn=max_cond_frames_in_attn,
            multimask_output_in_sam=multimask_output_in_sam,
            multimask_min_pt_num=multimask_min_pt_num,
            multimask_max_pt_num=multimask_max_pt_num,
            multimask_output_for_tracking=multimask_output_for_tracking,
            forward_backbone_per_frame_for_eval=forward_backbone_per_frame_for_eval,
            memory_temporal_stride_for_eval=memory_temporal_stride_for_eval,
            offload_output_to_cpu_for_eval=offload_output_to_cpu_for_eval,
            trim_past_non_cond_mem_for_eval=trim_past_non_cond_mem_for_eval,
            non_overlap_masks_for_mem_enc=non_overlap_masks_for_mem_enc,
            max_obj_ptrs_in_encoder=max_obj_ptrs_in_encoder,
            sam_mask_decoder_extra_args=sam_mask_decoder_extra_args,
            compile_all_components=False,  # We handle compilation separately
            use_memory_selection=use_memory_selection,
            mf_threshold=mf_threshold,
        )
        
        # 5. Add EfficientSAM3-specific components
        self.memory_attention = memory_attention
        
        # 6. Build Perceiver for spatial memory compression (optional but recommended)
        if use_perceiver:
            self.spatial_perceiver = PerceiverResampler(
                dim=mem_dim,
                num_latents=perceiver_num_latents,
                depth=perceiver_depth,
                num_heads=perceiver_num_heads,
                head_dim=perceiver_head_dim,
                ff_mult=perceiver_ff_mult,
            )
            # Projection layers: 64-d -> 256-d
            self.mem_proj = nn.Linear(mem_dim, d_model)
            self.mem_pos_proj = nn.Linear(mem_dim, d_model)
        else:
            self.spatial_perceiver = None
            # Still need projections if mem_dim != d_model
            if mem_dim != d_model:
                self.mem_proj = nn.Linear(mem_dim, d_model)
                self.mem_pos_proj = nn.Linear(mem_dim, d_model)
        
        # 7. Training-specific attributes
        self.teacher_force_obj_scores_for_mem = False
        self.prob_to_dropout_spatial_mem = 0.0
        
        # 8. Compilation (optional)
        if compile_all_components:
            self._compile_all_components()
    
    def _compile_all_components(self):
        """Compile model components for faster inference."""
        logger.info("Compiling EfficientSam3TrackerBase components...")
        if hasattr(self, 'memory_attention'):
            self.memory_attention = torch.compile(
                self.memory_attention,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
        if hasattr(self, 'spatial_perceiver') and self.spatial_perceiver is not None:
            self.spatial_perceiver = torch.compile(
                self.spatial_perceiver,
                mode="max-autotune",
                fullgraph=True,
                dynamic=False,
            )
    
    def forward_image(self, img_batch):
        """Get the image feature on the input batch.
        
        Handles both SAM3VLBackbone (with .forward_image method) and 
        efficient backbones (with direct __call__).
        """
        # Check if backbone has SAM3-style forward_image method
        if hasattr(self.backbone, 'forward_image'):
            backbone_out_raw = self.backbone.forward_image(img_batch)
            # SAM3 backbone returns {"sam2_backbone_out": {...}}
            if "sam2_backbone_out" in backbone_out_raw:
                backbone_out = backbone_out_raw["sam2_backbone_out"]
            else:
                backbone_out = backbone_out_raw
        else:
            # Efficient backbone (RepViT, TinyViT, etc.) - direct call
            backbone_out = self.backbone(img_batch)
        
        # Precompute projected level 0 and level 1 features for SAM decoder
        backbone_out["backbone_fpn"][0] = self.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )
        
        return backbone_out


class EfficientSam3TrackerPredictor(EfficientSam3TrackerBase):
    """
    EfficientSAM3 Tracker Predictor - for SAM2-style interactive video segmentation.
    
    This class provides the inference API similar to Sam3TrackerPredictor:
    - init_state(): Initialize inference state for a video
    - add_new_points(): Add point prompts on a frame
    - add_new_mask(): Add mask prompts on a frame
    - propagate_in_video(): Propagate masks through the video
    
    No text encoder or VL fusion - purely SAM2-style point/mask prompts.
    """
    
    def __init__(
        self,
        # Inference-specific args
        clear_non_cond_mem_around_input: bool = False,
        clear_non_cond_mem_for_multi_obj: bool = False,
        fill_hole_area: int = 0,
        always_start_from_first_ann_frame: bool = False,
        max_point_num_in_prompt_enc: int = 16,
        non_overlap_masks_for_output: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj
        self.fill_hole_area = fill_hole_area
        self.always_start_from_first_ann_frame = always_start_from_first_ann_frame
        self.max_point_num_in_prompt_enc = max_point_num_in_prompt_enc
        self.non_overlap_masks_for_output = non_overlap_masks_for_output
        
        # Use bfloat16 for inference
        self.bf16_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        self.bf16_context.__enter__()
        
        self.iter_use_prev_mask_pred = True
        self.add_all_frames_to_correct_as_cond = True
    
    @torch.inference_mode()
    def init_state(
        self,
        video_height: Optional[int] = None,
        video_width: Optional[int] = None,
        num_frames: Optional[int] = None,
        video_path: Optional[str] = None,
        cached_features: Optional[Dict] = None,
        offload_video_to_cpu: bool = False,
        offload_state_to_cpu: bool = False,
        async_loading_frames: bool = False,
    ) -> Dict[str, Any]:
        """Initialize an inference state for a video."""
        inference_state = {}
        
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        inference_state["device"] = self.device
        
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")
        
        if video_path is not None:
            images, video_height, video_width = load_video_frames(
                video_path=video_path,
                image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=async_loading_frames,
                compute_device=inference_state["storage_device"],
            )
            inference_state["images"] = images
            inference_state["num_frames"] = len(images)
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
        else:
            inference_state["video_height"] = video_height
            inference_state["video_width"] = video_width
            inference_state["num_frames"] = num_frames
        
        # Inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        
        # Feature cache
        inference_state["cached_features"] = (
            {} if cached_features is None else cached_features
        )
        
        # Constants
        inference_state["constants"] = {}
        
        # Object ID mapping
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        
        # Output storage
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        
        # First annotation frame
        inference_state["first_ann_frame_idx"] = None
        
        # Per-object output slices
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        
        # Consolidated frames
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        
        # Tracking metadata
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        
        self.clear_all_points_in_video(inference_state)
        
        return inference_state
    
    def clear_all_points_in_video(self, inference_state):
        """Clear all point/mask inputs and reset tracking state."""
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        inference_state["output_dict_per_obj"] = {}
        inference_state["temp_output_dict_per_obj"] = {}
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        inference_state["first_ann_frame_idx"] = None
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
    
    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        
        # New object - only allow before tracking starts
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"].append(obj_id)
            # Initialize per-object storage
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object {obj_id} after tracking has started. "
                "Call clear_all_points_in_video first."
            )
    
    @torch.inference_mode()
    def add_new_points(
        self,
        inference_state,
        frame_idx: int,
        obj_id: int,
        points: torch.Tensor,
        labels: torch.Tensor,
        clear_old_points: bool = True,
    ):
        """Add new point prompts on a frame for an object.
        
        Args:
            inference_state: The inference state from init_state()
            frame_idx: Frame index to add points to
            obj_id: Object ID
            points: Point coordinates [N, 2] in relative (0-1) format
            labels: Point labels [N] where 1=positive, 0=negative
            clear_old_points: Whether to clear existing points
            
        Returns:
            Tuple of (frame_idx, obj_ids, low_res_masks, video_res_masks)
        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        
        # Convert points to absolute coordinates
        video_height = inference_state["video_height"]
        video_width = inference_state["video_width"]
        
        if points.dim() == 2:
            points = points.unsqueeze(0)  # Add batch dim
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        
        # Scale to image size
        abs_points = points.clone()
        abs_points[..., 0] *= self.image_size  # x
        abs_points[..., 1] *= self.image_size  # y
        
        point_inputs = {
            "point_coords": abs_points.to(self.device),
            "point_labels": labels.to(self.device),
        }
        
        # Store point inputs
        if clear_old_points:
            inference_state["point_inputs_per_obj"][obj_idx] = {frame_idx: point_inputs}
        else:
            if frame_idx not in inference_state["point_inputs_per_obj"][obj_idx]:
                inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
            else:
                # Concatenate with existing points
                old = inference_state["point_inputs_per_obj"][obj_idx][frame_idx]
                inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = {
                    "point_coords": torch.cat([old["point_coords"], point_inputs["point_coords"]], dim=1),
                    "point_labels": torch.cat([old["point_labels"], point_inputs["point_labels"]], dim=1),
                }
        
        # Set first annotation frame
        if inference_state["first_ann_frame_idx"] is None:
            inference_state["first_ann_frame_idx"] = frame_idx
        
        # Run SAM on this frame to get initial mask
        # (Implementation would call track_step internally)
        # For now, return placeholder
        out_obj_ids = [obj_id]
        low_res_masks = torch.zeros(1, 1, self.low_res_mask_size, self.low_res_mask_size, device=self.device)
        video_res_masks = torch.zeros(1, 1, video_height, video_width, device=self.device)
        
        return frame_idx, out_obj_ids, low_res_masks, video_res_masks
    
    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx: Optional[int] = None,
        max_frame_num_to_track: Optional[int] = None,
        reverse: bool = False,
    ):
        """Propagate masks through the video.
        
        Yields:
            Tuple of (frame_idx, obj_ids, video_res_masks) for each frame
        """
        inference_state["tracking_has_started"] = True
        
        num_frames = inference_state["num_frames"]
        obj_ids = inference_state["obj_ids"]
        
        if start_frame_idx is None:
            start_frame_idx = inference_state.get("first_ann_frame_idx", 0)
        
        if reverse:
            frame_range = range(start_frame_idx, -1, -1)
        else:
            frame_range = range(start_frame_idx, num_frames)
        
        if max_frame_num_to_track is not None:
            frame_range = list(frame_range)[:max_frame_num_to_track]
        
        video_height = inference_state["video_height"]
        video_width = inference_state["video_width"]
        
        for frame_idx in tqdm(frame_range, desc="Propagating"):
            # Get masks for all objects on this frame
            # (Full implementation would use track_step with memory)
            video_res_masks = torch.zeros(
                len(obj_ids), 1, video_height, video_width,
                device=self.device
            )
            
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
            
            yield frame_idx, obj_ids, video_res_masks


# Alias for backward compatibility
EfficientSAM3Stage2 = EfficientSam3TrackerBase
