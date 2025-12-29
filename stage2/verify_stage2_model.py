import torch
import sys
import os
import logging

# Add paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from stage2.model import EfficientSAM3Stage2
from sam3.model_builder import build_sam3_image_model, build_sam3_video_model
from sam3.model.vl_combiner import SAM3VLBackbone

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_stage2_model():
    logger.info("Verifying EfficientSAM3Stage2 model structure...")
    
    # 1. Load Original SAM3 Model (Teacher)
    # We use build_sam3_video_model to get the full tracker structure for comparison
    # Note: This might require a checkpoint, but we can try to build without one for structure check
    logger.info("Building Original SAM3 Video Model...")
    try:
        sam3_video_model = build_sam3_video_model(checkpoint_path=None, load_from_HF=False)
        sam3_tracker = sam3_video_model.tracker
    except Exception as e:
        logger.warning(f"Could not build full SAM3 video model: {e}")
        logger.warning("Falling back to manual inspection of expected components.")
        sam3_tracker = None

    # 2. Build EfficientSAM3Stage2
    logger.info("Building EfficientSAM3Stage2...")
    
    # We need an image encoder to initialize EfficientSAM3Stage2
    # In train_memory_stage2.py, we use build_sam3_image_model with enable_text_encoder=False
    sam3_image_model = build_sam3_image_model(
        checkpoint_path=None,
        load_from_HF=False,
        enable_text_encoder=False, # Crucial: This should prevent loading text encoder
        enable_vision_encoder=True,
    )
    image_encoder = sam3_image_model.backbone # This is SAM3VLBackbone
    
    stage2_model = EfficientSAM3Stage2(
        image_encoder=image_encoder,
        d_model=256,
        num_maskmem=7,
        image_size=1024,
        use_perceiver=True,
    )
    
    # 3. Check for Text Encoder in Stage 2 Model
    logger.info("Checking for Text Encoder in Stage 2 Model...")
    
    has_text_encoder = False
    text_encoder_module = None
    
    # In SAM3VLBackbone, the text encoder is usually stored in self.text_backbone or similar
    # Let's inspect the backbone
    backbone = stage2_model.backbone
    if isinstance(backbone, SAM3VLBackbone):
        logger.info(f"Backbone is SAM3VLBackbone.")
        # Check 'text' attribute which usually holds the text encoder
        if hasattr(backbone, 'text') and backbone.text is not None:
            has_text_encoder = True
            text_encoder_module = backbone.text
            logger.warning("Text encoder FOUND in backbone.text!")
        else:
            logger.info("Text encoder NOT found in backbone.text (as expected).")
            
    else:
        logger.warning(f"Backbone is not SAM3VLBackbone, it is {type(backbone)}")

    # Also check recursively for any module named 'text_encoder' or similar
    found_text_modules = []
    for name, module in stage2_model.named_modules():
        if 'text' in name.lower() and isinstance(module, torch.nn.Module):
             # Filter out some false positives if necessary, but 'text' is usually specific
             found_text_modules.append(name)
    
    if found_text_modules:
        logger.info(f"Found modules with 'text' in name: {found_text_modules}")
    else:
        logger.info("No modules with 'text' in name found.")

    # 4. Compare with Original SAM3 Tracker (if available)
    if sam3_tracker:
        logger.info("Comparing with Original SAM3 Tracker...")
        # Check Memory Attention
        logger.info(f"Stage 2 Memory Attention: {type(stage2_model.transformer)}")
        logger.info(f"Original SAM3 Memory Attention: {type(sam3_tracker.transformer)}")
        
        # Check Memory Encoder
        logger.info(f"Stage 2 Memory Encoder: {type(stage2_model.maskmem_backbone)}")
        logger.info(f"Original SAM3 Memory Encoder: {type(sam3_tracker.maskmem_backbone)}")
        
        # Check Perceiver
        if hasattr(stage2_model, 'spatial_perceiver'):
             logger.info(f"Stage 2 has Perceiver: {type(stage2_model.spatial_perceiver)}")
        else:
             logger.info("Stage 2 does NOT have Perceiver (Unexpected if use_perceiver=True)")

    # 5. Conclusion
    if has_text_encoder:
        logger.error("VERIFICATION FAILED: Text encoder is present in the model.")
        logger.info("To avoid this, ensure 'enable_text_encoder=False' is passed to 'build_sam3_image_model'.")
        logger.info("In 'sam3/model_builder.py', 'build_sam3_image_model' handles this flag.")
        logger.info("If 'enable_text_encoder=False', '_create_vl_backbone' is called with 'text_encoder=None'.")
        logger.info("Check 'SAM3VLBackbone' in 'sam3/model/vl_combiner.py' to see how it handles None text encoder.")
    else:
        logger.info("VERIFICATION PASSED: Text encoder is NOT present in the model.")
        logger.info("The model structure appears correct for Stage 2 training (Vision + Memory only).")

if __name__ == "__main__":
    verify_stage2_model()
