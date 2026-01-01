# --------------------------------------------------------
# EfficientSAM3 Stage 2: Efficient Memory Bank Distillation
# --------------------------------------------------------
"""
Stage 2 focuses on making SAM3's memory bank more efficient by:
1. Implementing PerceiverResampler to compress memory tokens (EdgeTAM-style)
2. Adding EfficientMemoryAttention with pooling (EfficientTAM-style)
3. Reducing memory attention layers from 4 to 2

This allows significant speedup in video object tracking while
maintaining quality comparable to the full SAM3 tracker.

Architecture:
    Sam3TrackerBase (original SAM3)
        └── EfficientSam3TrackerBase (efficient memory)
                ├── EfficientSam3TrackerPredictor (inference)
                └── EfficientSam3Train (training)

Key modules are defined in:
    - sam3/sam3/model/efficient_sam3_tracker.py
    - sam3/sam3/model/efficient_sam3_train.py
    - sam3/sam3/model/efficient_sam3_model_builder.py
"""

# Re-export from sam3.model for convenience
from sam3.model import (
    EfficientSam3TrackerBase,
    EfficientSam3TrackerPredictor,
    EfficientSam3Train,
    build_efficient_sam3_tracker,
    build_efficient_sam3_predictor,
    build_efficient_sam3_train,
    EfficientMemoryAttention,
    PerceiverResampler,
)

__all__ = [
    "EfficientSam3TrackerBase",
    "EfficientSam3TrackerPredictor",
    "EfficientSam3Train",
    "build_efficient_sam3_tracker",
    "build_efficient_sam3_predictor",
    "build_efficient_sam3_train",
    "EfficientMemoryAttention",
    "PerceiverResampler",
]
